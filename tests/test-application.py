"""Shared session controls work without importing or constructing a CLI."""

import asyncio
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest

from slipagent.agent import Agent
from slipagent.application import application
from slipagent.events import AgentEvent
from slipagent.sessionstate import Session
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry
from test_agent import StubClient, completion


def session_for(workspace, replies=()):
    registry = ToolRegistry()
    registry.resources.workspace = workspace
    client = StubClient(list(replies))
    client.aclose = AsyncMock()
    agent = Agent(client, registry, 'test/model')
    return Session(agent=agent, registry=registry, client=client, workspace=workspace,
                   api_key='test', base_url='http://unused.invalid', http_referer=None, app_title='test')


def test_session_factory_and_controller_import_without_cli_or_terminal(tmp_path):
    script = '''
import asyncio, sys
from pathlib import Path
from slipagent.application import application
from slipagent.config import Config
from slipagent.sessionfactory import create_session, SessionOptions

async def main():
    session = await create_session(Config(api_key="test", workspace=Path(sys.argv[1])), SessionOptions())
    assert "slipagent.cli" not in sys.modules
    assert "slipagent.terminal" not in sys.modules
    assert not hasattr(session, "renderer")
    await application(session).close()

asyncio.run(main())
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


async def test_interfaces_share_one_engine_and_subscribe_without_copying_history(workspace):
    session = session_for(workspace, [completion('Shared answer')])
    app = application(session)
    first, second, states = [], [], []
    app.subscribe(first.append)
    unsubscribe = application(session).subscribe(second.append)
    app.states.subscribe(states.append)
    try:
        assert application(session) is app
        assert await app.run('One prompt') == 'Shared answer'
        assert any(event.kind == 'assistant_text' for event in first)
        assert len(first) == len(second)
        assert all(left is right for left, right in zip(first, second))
        assert any(state.running for state in states) and not states[-1].running
        unsubscribe()
        count = len(second)
        session.agent._emit(AgentEvent(kind='notice', text='Only one listener remains'))
        assert len(second) == count and first[-1].text == 'Only one listener remains'
        assert sum(message.role == 'user' for message in session.agent.messages) == 1
    finally:
        await app.close()


async def test_detaching_waiter_keeps_run_alive_and_blocks_duplicate_start(workspace):
    session = session_for(workspace, [completion('Finished')])
    app = application(session)
    entered, release = asyncio.Event(), asyncio.Event()
    original = session.client.chat
    async def wait(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)
    session.client.chat = wait
    task = app.start('Work')
    try:
        with pytest.raises(RuntimeError, match='settle'):
            app.start('Duplicate')
        await asyncio.wait_for(entered.wait(), 5)
        waiter = asyncio.create_task(app.wait())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, 5) == 'Finished'
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await app.close()


async def test_shared_queue_identity_discard_and_busy_mutation_guard(workspace):
    session = session_for(workspace)
    app = application(session)
    try:
        app.submit('First')
        app.submit('Second')
        identifier, _ = app.queued_input()[0]
        assert app.discard_input(identifier)
        assert [text for _, text in app.queued_input()] == ['Second']
        session.agent.running = True
        with pytest.raises(RuntimeError):
            app.set_danger(True)
        assert not workspace.access.danger
        session.agent.running = False
        app.set_planning(False)
        assert not session.agent.planning and 'update_plan' not in session.registry
    finally:
        session.agent.running = False
        await app.close()


async def test_interrupt_and_close_drain_same_owned_work_once(workspace, caplog):
    session = session_for(workspace)
    app = application(session)
    entered, cancelled = asyncio.Event(), asyncio.Event()
    async def wait(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    session.client.chat = wait
    task = app.start('Wait')
    try:
        await asyncio.wait_for(entered.wait(), 5)
        def failed_display():
            raise RuntimeError('interface detached')
        assert app.interrupt(on_finished=failed_display)
        assert not app.interrupt()
        await asyncio.wait_for(app.wait_for_interrupt(), 5)
        assert cancelled.is_set() and task.cancelled()
        assert 'Interrupt completion display failed' in caplog.text
        await asyncio.gather(app.close(), app.close())
        session.client.aclose.assert_awaited_once()
        with pytest.raises(RuntimeError, match='closed'):
            app.start('After close')
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await app.close()


async def test_session_rename_reset_resume_share_existing_journal(workspace, tmp_path):
    session = session_for(workspace, [completion('Saved answer')])
    journal = SessionJournal(str(workspace.root), tmp_path / 'sessions')
    session.registry.resources.journal = journal
    app = application(session)
    try:
        await app.run('Remember this')
        identifier = journal.session_id
        assert await app.rename('Shared session') == 'Shared session | SlipAgent'
        assert any(row['id'] == identifier and row['title'] == 'Shared session | SlipAgent' for row in await app.saved_sessions())
        await app.reset()
        assert not any(message.role == 'user' for message in session.agent.messages)
        await app.resume(identifier)
        assert any(message.content == 'Remember this' for message in session.agent.messages)
        assert session.agent.history.steps
        assert journal.session_id == identifier
    finally:
        await app.close()


async def test_editor_frontend_needs_no_terminal_objects(workspace):
    from slipagent.useredits import UserEdits
    session = session_for(workspace)
    opened = []
    class Editor:
        active = False
        async def open(self, path, line):
            self.active = True
            opened.append((path, line))
        async def close_for_switch(self): return True
        async def shutdown(self): self.active = False
        def notify(self, text): pass
        def clear_pending_notice(self): pass
        def recorded(self, path, artifact, error): pass
    editor = Editor()
    edits = UserEdits(session, editor)
    session.registry.services['user_edits'] = edits
    path = workspace.root / 'file.txt'
    session.agent.running = True
    await edits.request(path, 3)
    assert opened == []
    session.agent.running = False
    boundary = asyncio.create_task(edits.boundary())
    try:
        async with asyncio.timeout(5):
            while not opened:
                await asyncio.sleep(0)
        assert opened == [(path, 3)] and not boundary.done()
        edits.record(path, '+change')
        assert session.registry.context_notes['user_edits'][str(path)]['diff'] == '+change'
        edits.close()
        await boundary
    finally:
        edits.close()
        boundary.cancel()
        await asyncio.gather(boundary, return_exceptions=True)
        await application(session).close()


async def test_failed_interface_listener_does_not_interrupt_other_interface(workspace, caplog):
    session = session_for(workspace, [completion('Still delivered')])
    app = application(session)
    received = []
    def broken(event):
        raise RuntimeError('display failed')
    app.subscribe(broken)
    app.subscribe(received.append)
    try:
        assert await app.run('Work') == 'Still delivered'
        assert any(event.kind == 'assistant_text' and event.text == 'Still delivered' for event in received)
        assert 'Detached a failing application event listener' in caplog.text
    finally:
        await app.close()


async def test_cli_adapter_and_another_interface_both_receive_session_notices(workspace):
    import io
    from slipagent.cli import Renderer, Session as CLISession, Style
    core = session_for(workspace)
    renderer = Renderer(Style(False), io.StringIO(), False)
    session = CLISession(agent=core.agent, registry=core.registry, client=core.client, workspace=workspace,
                         renderer=renderer, api_key='test', base_url='http://unused.invalid',
                         http_referer=None, app_title='test')
    app = application(session)
    other = []
    app.subscribe(other.append)
    try:
        session.notice('Shared provider notice')
        assert 'Shared provider notice' in renderer.stream.getvalue()
        assert any(event.kind == 'notice' and event.text == 'Shared provider notice' for event in other)
    finally:
        await app.close()


async def test_shared_fork_and_deletion_preserve_saved_sessions_without_replaying_tools(workspace, tmp_path):
    from slipagent.sessions import SessionError
    session = session_for(workspace, [completion('Saved')])
    journal = SessionJournal(str(workspace.root), tmp_path / 'sessions')
    session.registry.resources.journal = journal
    app = application(session)
    try:
        await app.run('Original request')
        original = journal.session_id
        forked = await app.fork()
        assert original != forked
        assert {row['id'] for row in await app.saved_sessions()} == {original, forked}
        await app.rename_saved(original, 'Original branch')
        await app.delete_saved(original)
        with pytest.raises(SessionError):
            journal.load(original)
        assert len(session.client.calls) == 1
        assert any(message.content == 'Original request' for message in session.agent.messages)
        await app.delete_current()
        assert await app.saved_sessions() == []
    finally:
        await app.close()


async def test_second_interface_cannot_start_while_a_session_restore_is_in_progress(workspace, tmp_path, monkeypatch):
    import threading
    session = session_for(workspace, [completion('Saved')])
    journal = SessionJournal(str(workspace.root), tmp_path / 'sessions')
    session.registry.resources.journal = journal
    app = application(session)
    await app.run('Original')
    original = journal.load
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    def gated(identifier):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original(identifier)
    monkeypatch.setattr(journal, 'load', gated)
    restoring = asyncio.create_task(app.resume(journal.session_id))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(RuntimeError, match='settle'):
            application(session).start('Conflicting request')
        release.set()
        await restoring
        assert not app.state().changing
    finally:
        release.set()
        restoring.cancel()
        await asyncio.gather(restoring, return_exceptions=True)
        await app.close()
