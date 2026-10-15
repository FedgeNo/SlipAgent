"""Pending input can be inspected/discarded without changing accepted work."""

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from slipagent import cli
from slipagent.agent import Agent
from slipagent.inputqueue import input_queue
from slipagent.sessions import SessionError, SessionJournal
from slipagent.terminal import TerminalUI
from slipagent.tools.base import ToolRegistry
from slipagent.types import content_text
from test_agent import StubClient, completion, unpack_context
from test_cli_e2e import metadata_server


@pytest.mark.parametrize('drained', [False, True])
async def test_discard_duplicate_preserves_history_task_and_resume(tmp_path, drained):
    agent = Agent(StubClient([completion('Accepted answer')]), ToolRegistry(), 'test')
    journal = SessionJournal(str(tmp_path), tmp_path / 'saved')
    agent.registry.services['session_journal'] = journal
    await agent.run('Accepted request')
    await agent.wait_for_compaction()
    accepted = list(agent.messages)
    task = agent.history.task
    for text in ['same', 'keep middle', 'same']:
        agent.enqueue(text)
    queue = input_queue(agent)
    entries = queue.snapshot()
    if drained:
        agent._drain_pending()
        agent.history.sync(agent.messages)
    assert queue.discard(entries[2][0])
    assert queue.snapshot() == entries[:2]
    assert agent.messages[:len(accepted)] == accepted
    assert agent.history.task is task
    assert task.sources[1] == ['Accepted request']
    assert all(prompts.count('same') <= 1 for prompts in task.sources.values())
    restored = Agent(StubClient([completion('Continued')]), ToolRegistry(), 'test')
    journal.restore(restored, journal.load(journal.session_id))
    assert [text for _, text in input_queue(restored).snapshot()] == ['same', 'keep middle']
    await restored.run('')
    await restored.wait_for_compaction()
    user = [m.content for m in unpack_context(restored.client.calls[0]['messages']) if m.role == 'user']
    assert user.count('same') == 1
    assert 'keep middle' in user


async def test_discard_unsent_input_in_recorded_failure_keeps_step_metadata(tmp_path):
    agent = Agent(StubClient([completion('Accepted answer')]), ToolRegistry(), 'test')
    journal = SessionJournal(str(tmp_path), tmp_path / 'saved')
    agent.registry.services['session_journal'] = journal
    await agent.run('Accepted request')
    await agent.wait_for_compaction()
    agent.enqueue('Discard this unsent correction')
    agent._drain_pending()
    agent._record_request_failure({'outcome': 'request_error', 'error': 'Failed before submission',
                                   'response_excerpt': '', 'diagnostic_attempt': None}, agent.usage)
    failed = agent.history.steps[-1]
    summary = failed.summary
    failure = failed.request_failure
    queue = input_queue(agent)
    assert queue.discard(queue.snapshot()[0][0])
    assert agent.history.steps[-1] is failed
    assert failed.summary == summary and failed.request_failure == failure
    assert failed.request == 'Accepted request'
    assert failed.parts()['prompt'] == 'Accepted request'
    assert 'Discard this unsent correction' not in content_text(failed.context_messages()[0].content)
    raw = json.loads((agent.history.store.directory / '00000002.raw.json').read_text())
    assert raw['request'] == raw['parts']['prompt'] == 'Accepted request'
    assert raw['model']['user_prompts'] == []
    assert all(message['role'] != 'user' for message in raw['messages'])
    assert 'Discard this unsent correction' not in str(agent.history.task.sources)
    data = journal.load(journal.session_id)
    assert len(data['history'].steps) == 2
    assert data['history'].steps[-1].request_failure == failure
    assert data['history'].task.sources == {1: ['Accepted request']}


@pytest.mark.parametrize('drained', [False, True])
async def test_discard_event_is_atomic_without_following_state_event(tmp_path, drained):
    agent = Agent(StubClient([]), ToolRegistry(), 'test')
    journal = SessionJournal(str(tmp_path), tmp_path / 'saved')
    agent.registry.services['session_journal'] = journal
    agent.enqueue('first')
    agent.enqueue('second')
    if drained:
        agent._drain_pending()
    queue = input_queue(agent)
    assert queue.discard(queue.snapshot()[0][0])
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert events[-1]['type'] == 'discard_input'
    data = journal.load(journal.session_id)
    restored = Agent(StubClient([]), ToolRegistry(), 'test')
    journal.restore(restored, data)
    assert [text for _, text in input_queue(restored).snapshot()] == ['second']


async def test_failed_journal_discard_retains_input(tmp_path, monkeypatch):
    agent = Agent(StubClient([]), ToolRegistry(), 'test')
    journal = SessionJournal(str(tmp_path), tmp_path / 'saved')
    agent.registry.services['session_journal'] = journal
    agent.enqueue('Keep on save failure')
    queue = input_queue(agent)
    entries = queue.snapshot()

    def fail(kind, **data):
        raise SessionError('Test save failure')

    monkeypatch.setattr(journal, '_append', fail)
    with pytest.raises(SessionError, match='Test save failure'):
        queue.discard(entries[0][0])
    assert queue.snapshot() == entries
    assert agent.pending == ['Keep on save failure']


async def test_discard_during_context_preparation_rebuilds_before_submission(monkeypatch):
    agent = Agent(StubClient([completion('Done')]), ToolRegistry(), 'test')
    agent.enqueue('Remove before sending')
    entered, release = asyncio.Event(), asyncio.Event()
    original = type(agent)._context_view
    builds = []

    async def context(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        builds.append(result)
        if len(builds) == 1:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(type(agent), '_context_view', context)
    task = asyncio.create_task(agent.run('Accepted request'))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        queue = input_queue(agent)
        assert queue.discard(queue.snapshot()[0][0])
        release.set()
        assert await asyncio.wait_for(task, 2) == 'Done'
        await agent.wait_for_compaction()
        assert len(builds) == 2
        sent = unpack_context(agent.client.calls[0]['messages'])
        assert not any(message.content == 'Remove before sending' for message in sent)
        assert agent.history.task.sources == {1: ['Accepted request']}
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_old_selection_cannot_discard_new_identical_input():
    agent = Agent(StubClient([]), ToolRegistry(), 'test')
    agent.enqueue('same')
    queue = input_queue(agent)
    old = queue.snapshot()[0][0]
    agent._drain_pending()
    agent._queued_messages_sent()
    agent.enqueue('same')
    assert not queue.discard(old)
    assert agent.pending == ['same']
    assert agent.messages[-1].content == 'same'


async def test_old_selection_cannot_discard_input_after_session_reset():
    agent = Agent(StubClient([]), ToolRegistry(), 'test')
    agent.enqueue('same')
    queue = input_queue(agent)
    old = queue.snapshot()[0][0]
    agent.reset()
    agent.enqueue('same')
    assert not queue.discard(old)
    assert agent.pending == ['same']


async def test_discard_after_context_preparation_invalidates_request(monkeypatch):
    agent = Agent(StubClient([completion('Done')]), ToolRegistry(), 'test')
    agent.enqueue('Remove from prepared request')
    original = type(agent)._visible_specs

    def visible_specs(self, specs):
        queue = input_queue(self)
        if queue.prepared is not None and queue.snapshot():
            assert queue.discard(queue.snapshot()[0][0])
        return original(self, specs)

    monkeypatch.setattr(type(agent), '_visible_specs', visible_specs)
    assert await agent.run('Accepted request') == 'Done'
    await agent.wait_for_compaction()
    assert not input_queue(agent).snapshot()
    sent = unpack_context(agent.client.calls[0]['messages'])
    assert not any(message.content == 'Remove from prepared request' for message in sent)
    assert agent.history.task.sources == {1: ['Accepted request']}


@pytest.mark.parametrize('target', ['accepted', 'pending-out-of-range'])
async def test_journal_rejects_discards_outside_unsent_queue(tmp_path, target):
    agent = Agent(StubClient([completion('Accepted answer')]), ToolRegistry(), 'test')
    journal = SessionJournal(str(tmp_path), tmp_path / 'saved')
    agent.registry.services['session_journal'] = journal
    await agent.run('Accepted request')
    await agent.wait_for_compaction()
    agent.enqueue('Pending')
    index = next(i for i, message in enumerate(agent.messages) if message.role == 'user') if target == 'accepted' else None
    journal._append('discard_input', index=index, pending_index=None if target == 'accepted' else 99)
    with pytest.raises(SessionError, match='discard'):
        journal.load(journal.session_id)


async def test_queue_command_plain_output_and_task_guard(tmp_path, metadata_server):
    session = await cli.build_session(cli.build_parser().parse_args(['--no-mcp', '-w', str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    session.agent.client = StubClient([completion('Answer')])
    try:
        await session.agent.run('Current task')
        await session.agent.wait_for_compaction()
        task = session.agent.history.task
        session.agent.enqueue('full first line\nfull last line')
        session.agent.enqueue('keep this')
        session.agent._drain_pending()
        await cli._handle_command(session, '/queue')
        assert 'full first line\nfull last line' in session.renderer.stream.getvalue()
        await cli._handle_command(session, '/task new')
        assert session.agent.history.task is task
        await cli._handle_command(session, '/queue discard 1')
        assert [text for _, text in input_queue(session.agent).snapshot()] == ['keep this']
        await cli._handle_command(session, '/queue 1')
        assert session.renderer.stream.getvalue().endswith('keep this\n')
        for argument in ['0', '-1', 'discard', 'discard 1 2', '９', '9' * 5000]:
            await cli._handle_command(session, '/queue ' + argument)
        assert session.agent.pending or session.agent.queued_messages
        await cli._handle_command(session, '/queue discard 1')
        assert not session.agent.pending and not session.agent.queued_messages
        assert session.agent.history.task is task
        assert task.sources == {1: ['Current task']}
        await cli._handle_command(session, '/task new')
        assert session.agent.history.task is not task
    finally:
        await cli._shutdown(session)


@pytest.mark.parametrize('action', ['view', 'discard', None, 'delivered'])
async def test_interactive_queue_inspects_full_text_or_discards_selected(tmp_path, metadata_server, action):
    session = await cli.build_session(cli.build_parser().parse_args(['--no-mcp', '-w', str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    calls, viewed = [], []
    session.agent.enqueue('First\nmessage')
    session.agent.enqueue('Second\nmessage')

    async def choose(title, options):
        calls.append((title, options))
        if len(calls) == 1:
            if action == 'delivered':
                session.agent._drain_pending()
                session.agent._queued_messages_sent()
                session.agent.enqueue('Second\nmessage')
            return options[1][0]
        return 'discard' if action == 'delivered' else action

    terminal = SimpleNamespace(choose=choose, open_viewer=lambda *args: viewed.append(args),
                               write=lambda text, **kwargs: session.renderer.stream.write(text + '\n'),
                               set_title=lambda title: None)
    session.renderer.terminal = terminal
    # Queue events need the real renderer separately tested below; this command
    # fixture observes full messages and mutation without running an application.
    session.agent.on_event = None
    try:
        await cli._handle_command(session, '/queue')
        assert calls[0][0] == 'Pending Messages'
        assert 'Second\nmessage' in calls[0][1][1][1]
        if action == 'view':
            assert viewed == [('Pending Message', 'Second\nmessage')]
        elif action == 'discard':
            assert session.agent.pending == ['First\nmessage']
        elif action == 'delivered':
            assert session.agent.pending == ['Second\nmessage']
            assert 'nothing discarded' in session.renderer.stream.getvalue()
        else:
            assert session.agent.pending == ['First\nmessage', 'Second\nmessage']
    finally:
        session.renderer.terminal = None
        await cli._shutdown(session)


async def test_discard_refreshes_queued_previews_without_adding_to_feed():
    sink = io.StringIO()
    with create_pipe_input() as pipe:
        terminal = TerminalUI(lambda width: 'status', sink, input=pipe, output=DummyOutput())
        renderer = cli.Renderer(cli.Style(False), sink, verbose=False)
        renderer.terminal = terminal
        agent = Agent(StubClient([]), ToolRegistry(), 'test', on_event=renderer.handle)
        try:
            for text in ['same', 'middle', 'same']:
                agent.enqueue(text)
            queue = input_queue(agent)
            assert queue.discard(queue.snapshot()[2][0])
            assert terminal._queued_prompts == ['same', 'middle']
            assert sink.getvalue() == ''
            agent._drain_pending()
            agent._queued_messages_sent()
            assert terminal._queued_prompts == []
        finally:
            terminal.close()
