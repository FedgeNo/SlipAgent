"""Whole-diff comments use ordinary user input with retained artifact references."""

import io
import asyncio
from importlib import import_module

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.application.current import set_app

from slipagent.artifacts import ArtifactStore, ReadArtifactTool
from slipagent.agent import Agent
from slipagent.context import ContextError
from slipagent.diffcomments import attach_reference, current_references, diff_reference, read_reference, REFERENCE_MARKER
from slipagent.inputqueue import input_queue
from slipagent.sessions import SessionJournal
from slipagent.terminal import TerminalUI
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolCall, content_text
from test_agent import StubClient, RecordingTool, completion

run_copy = import_module('test-runtime').run_copy


def saved_diff(store, step=2, call='edit'):
    return store.save('edit file.py', '--- file.py\n+++ file.py\n@@ -1 +1 @@\n-old_referenced_content\n+new\n',
                      kind='diff', step_id=step, call_id=call)


def click_comment(ui):
    handler = next(part[2] for part in ui._viewer_header() if part[1] == 'Comment on Diff')
    handler(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))


@pytest.fixture
async def ui(tmp_path):
    store = ArtifactStore(100_000)
    store.use_directory(tmp_path / 'artifacts')
    with create_pipe_input() as pipe:
        terminal = TerminalUI(lambda width: 'status', io.StringIO(), input=pipe, output=DummyOutput())
        terminal.__dict__['_artifact_store'] = store
        try:
            yield terminal
        finally:
            terminal.close()
            await store.aclose()


async def test_saved_diff_has_comment_link_but_output_does_not(ui):
    store = ui._artifact_store
    output = store.save('command', 'output', kind='command_output', step_id=1, call_id='a')
    diff = store.save('edit file.py', '--- file.py\n+++ file.py\n@@ -1 +1 @@\n-old\n+new\n',
                      kind='diff', step_id=2, call_id='b')
    await ui.open_artifact(output['record_id'])
    assert 'Comment on Diff' not in fragment_list_to_text(ui._viewer_header())
    await ui.open_artifact(diff['record_id'])
    assert 'Comment on Diff' in fragment_list_to_text(ui._viewer_header())


async def test_comment_preserves_draft_and_queues_original_diff_not_latest(ui):
    first = saved_diff(ui._artifact_store)
    second = saved_diff(ui._artifact_store, step=3, call='later-edit')
    ui.input.text = 'Undo this'
    ui.input.buffer.cursor_position = 4
    ui.set_working(True)
    await ui.open_artifact(first['record_id'])
    click_comment(ui)
    assert ui.input.text == 'Undo this' and ui.input.buffer.cursor_position == 4
    assert ui.app.layout.current_control is ui.input.control
    assert ui._lines.empty() and ui.working
    assert str(first['record_id']) in ui._prompt_label()
    with set_app(ui.app):
        content = ui.input.control.create_content(100, 5)
    assert f"[Diff {first['record_id']}]" in fragment_list_to_text(content.get_line(0))
    assert 'Remove Diff Reference' in fragment_list_to_text(ui._activity())
    await ui.open_artifact(second['record_id'])
    ui.close_viewer()
    ui.input.buffer.validate_and_handle()
    message = await ui.read_line()
    assert message.startswith('Undo this' + REFERENCE_MARKER)
    assert read_reference(message) == diff_reference(first)
    assert ui._prompt_label() == '> ' and not ui.input.text
    assert ui.working


async def test_detach_keeps_draft_and_next_turn_is_ordinary(ui):
    record = saved_diff(ui._artifact_store)
    await ui.open_artifact(record['record_id'])
    click_comment(ui)
    ui.input.text = 'Escape it properly'
    handler = next(part[2] for part in ui._activity() if part[1] == 'Remove Diff Reference')
    handler(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
    assert ui.input.text == 'Escape it properly'
    ui.input.buffer.validate_and_handle()
    assert await ui.read_line() == 'Escape it properly'


async def test_blank_and_command_input_do_not_consume_reference(ui):
    record = saved_diff(ui._artifact_store)
    await ui.open_artifact(record['record_id'])
    click_comment(ui)
    for text in ['', '/model', '/key dummy-key']:
        ui.input.text = text
        ui.input.buffer.validate_and_handle()
        assert await ui.read_line() == text
        assert str(record['record_id']) in ui._prompt_label()
    ui.clear_transcript()
    assert ui._prompt_label() == '> '


async def test_secret_prompt_is_isolated_from_diff_reference(ui):
    record = saved_diff(ui._artifact_store)
    await ui.open_artifact(record['record_id'])
    click_comment(ui)
    ui.input.text = 'Draft comment'
    ask = asyncio.create_task(ui.ask('API key: ', password=True))
    await asyncio.sleep(0)
    ui.input.text = 'dummy-secret'
    ui.input.buffer.validate_and_handle()
    assert await ask == 'dummy-secret'
    assert ui.input.text == 'Draft comment'
    assert str(record['record_id']) in ui._prompt_label()


async def test_references_follow_normal_queue_and_scope_system_prompt(ui):
    record = saved_diff(ui._artifact_store)
    comment = attach_reference('Undo this', diff_reference(record))
    client = StubClient([
        completion(tool_calls=[ToolCall('work', 'record', {'value': 'initial'})]),
        completion(tool_calls=[ToolCall('read', 'read_artifact', {'record_id': record['record_id'], 'field': 'output'})]),
        completion('Reviewed the referenced diff.'), completion('Ordinary follow-up.'),
    ])
    registry = ToolRegistry([RecordingTool(), ReadArtifactTool(ui._artifact_store)])
    agent = Agent(client, registry, 'test', planning=False)
    events = []
    agent.on_event = events.append
    queued = False
    async def boundary():
        nonlocal queued
        if len(client.calls) == 1 and not queued:
            queued = True
            agent.enqueue(comment)
    agent.on_boundary = boundary
    try:
        await agent.run('Start work')
        await agent.wait_for_compaction()
        assert 'Current Diff Comments' not in content_text(client.calls[0]['messages'][0].content)
        for call in client.calls[1:]:
            system = content_text(call['messages'][0].content)
            assert '# Current Diff Comments' in system and str(record['record_id']) in system
            assert 'whole diff attached to that comment' in system
        assert 'old_referenced_content' in content_text(client.calls[2]['messages'][-1].content)
        assert [event.text for event in events if event.kind == 'user_message_sent'] == [comment]
        assert not agent.pending and not agent.queued_messages
        await agent.run('Now do something else')
        await agent.wait_for_compaction()
        assert 'Current Diff Comments' not in content_text(client.calls[-1]['messages'][0].content)
    finally:
        await registry.aclose()


async def test_discarded_comment_never_enters_system_prompt():
    agent = Agent(StubClient([]), ToolRegistry(), 'test', planning=False)
    reference = {'record_id': 123, 'step_id': 2, 'call_id': 'edit', 'command': 'edit file.py'}
    agent.enqueue(attach_reference('Undo this', reference))
    queue = input_queue(agent)
    assert queue.discard(queue.snapshot()[0][0])
    context = await agent._queued_context(agent.registry.specs(), 1)
    assert 'Current Diff Comments' not in content_text(context[0].content)


async def test_pending_reference_survives_resume_and_context_failure(tmp_path, monkeypatch):
    store = ArtifactStore(100_000)
    original = Agent(StubClient([]), ToolRegistry([ReadArtifactTool(store)], services={'artifact_store': store}), 'test', planning=False)
    journal = SessionJournal(str(tmp_path), tmp_path / 'saved')
    original.registry.services['session_journal'] = journal
    journal.begin(original)
    record = saved_diff(store)
    comment = attach_reference('Escape it properly', diff_reference(record))
    original.enqueue(comment)
    data = journal.load(journal.session_id)
    restored_store = ArtifactStore(100_000)
    registry = ToolRegistry([ReadArtifactTool(restored_store)], services={'artifact_store': restored_store})
    client = StubClient([completion('Handled the saved diff.')])
    restored = Agent(client, registry, 'test', planning=False)
    journal.restore(restored, data)
    try:
        assert restored.pending == [comment]
        assert restored_store.read(record['record_id'])['output'].startswith('--- file.py')
        async def fail(*args, **kwargs):
            raise ContextError('input too large')
        with monkeypatch.context() as failure:
            failure.setattr(type(restored), '_context_view', fail)
            with pytest.raises(ContextError):
                await restored.run('')
        assert list(restored.queued_messages.values()) == [comment]
        await restored.run('')
        await restored.wait_for_compaction()
        system = content_text(client.calls[0]['messages'][0].content)
        assert 'Current Diff Comments' in system and str(record['record_id']) in system
    finally:
        await original.registry.aclose()
        await registry.aclose()


def test_only_current_user_group_supplies_references():
    one = {'record_id': 1, 'step_id': 2, 'call_id': 'first', 'command': 'edit one'}
    two = {'record_id': 2, 'step_id': 3, 'call_id': 'second', 'command': 'edit two'}
    first = Message.user(attach_reference('Undo this', one))
    second = Message.user(attach_reference('Escape it', two))
    messages = [first, second, Message.assistant('Continuing'), Message.tool_result('tool', 'output')]
    assert current_references(messages) == [one, two]
    assert current_references([first, Message.assistant('Done'), Message.user('Unrelated')]) == []
    assert current_references([Message.user('Normal'), Message.tool_result('tool', first.content)]) == []


@pytest.mark.parametrize('data', ['{}', '[]', 'invalid', '{"record_id":true}', '[' * 2000 + ']' * 2000])
def test_invalid_reference_does_not_activate_prompt(data):
    assert read_reference('Comment' + REFERENCE_MARKER + data) is None


def test_reload_preserves_attached_draft_and_reference(run_copy):
    run_copy('''
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from slipagent.terminal import TerminalUI
        from slipagent.diffcomments import read_reference
        store = registry.resources.artifacts
        record = store.save('edit file.py', '--- file.py\\n+++ file.py\\n-old\\n+new', kind='diff', step_id=1, call_id='edit')
        with create_pipe_input() as pipe:
            terminal = TerminalUI(lambda columns: 'status', sink, input=pipe, output=DummyOutput())
            terminal.__dict__['_artifact_store'] = store
            renderer.terminal = terminal
            try:
                await terminal.open_artifact(record['record_id'])
                terminal._comment_on_diff(terminal._viewer.panel.reference)
                terminal.input.text = 'Undo this'
                for generation in range(1, 3):
                    frame.request()
                    await frame.checkpoint()
                    assert frame.generation == generation, sink.getvalue()
                    assert terminal.input.text == 'Undo this'
                    assert str(record['record_id']) in terminal._prompt_label()
                terminal.input.buffer.validate_and_handle()
                message = await terminal.read_line()
                assert read_reference(message)['record_id'] == record['record_id']
                agent.enqueue(message)
                context = await agent._queued_context(registry.specs(), 1)
                assert '# Current Diff Comments' in content_text(context[0].content)
            finally:
                terminal.close()
                renderer.terminal = None
    ''')


def test_reload_refreshes_rendered_diff_header(run_copy):
    run_copy('''
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.formatted_text import fragment_list_to_text
        from prompt_toolkit.application.current import set_app
        from slipagent.terminal import TerminalUI
        store = registry.resources.artifacts
        record = store.save('edit file.py', '--- file.py\\n+++ file.py\\n-old\\n+new', kind='diff', step_id=1, call_id='edit')
        with create_pipe_input() as pipe:
            terminal = TerminalUI(lambda columns: 'status', sink, input=pipe, output=DummyOutput())
            terminal.__dict__['_artifact_store'] = store
            renderer.terminal = terminal
            try:
                await terminal.open_artifact(record['record_id'])
                edit('terminal.py', '"Comment on Diff"', '"Comment on Saved Diff"')
                await frame.checkpoint()
                assert frame.generation == 1, sink.getvalue()
                header = terminal._viewer.container.children[0].content
                with set_app(terminal.app):
                    content = header.create_content(100, 1)
                assert 'Comment on Saved Diff' in fragment_list_to_text(content.get_line(0))
            finally:
                terminal.close()
                renderer.terminal = None
    ''')
