"""A slow disk delays the agent before a tool starts, never the event loop."""

import asyncio
import os
import threading
import subprocess
import sys
import pytest

from slipagent.agent import Agent
from slipagent.sessions import SessionError, SessionJournal
from slipagent.tools.base import ToolRegistry
from slipagent.types import ToolCall
from test_agent import StubClient, RecordingTool, completion


def test_flush_settles_finished_writer_before_its_scheduled_callback(tmp_path):
    # A subprocess deadline also catches an event-loop spin that asyncio timeouts cannot.
    script = '''
import asyncio, sys
from pathlib import Path
from slipagent.agent import Agent
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry

async def main():
    root = Path(sys.argv[1])
    agent = Agent(None, ToolRegistry(), "test")
    journal = SessionJournal(str(root), root / "saved")
    journal.begin(agent)
    finished = asyncio.create_task(asyncio.sleep(0))
    await finished
    journal._writer = finished
    asyncio.get_running_loop().call_soon(journal._writer_finished, finished)
    await journal.flush()
    assert journal._writer is None

asyncio.run(main())
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def attach(agent, workspace, tmp_path):
    journal = SessionJournal(str(workspace.root), tmp_path / "saved")
    agent.registry.services["session_journal"] = journal
    journal.begin(agent)
    return journal


async def test_records_do_not_sync_on_the_event_loop(workspace, tmp_path, monkeypatch):
    loop_thread = threading.get_ident()
    syncing = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda descriptor: (syncing.append(threading.get_ident()), real(descriptor)))
    agent = Agent(StubClient([completion("Done")]), ToolRegistry(), "test")
    journal = await asyncio.to_thread(attach, agent, workspace, tmp_path)
    await agent.run("Write records")
    assert loop_thread not in syncing
    await journal.flush()
    assert syncing and loop_thread not in syncing


async def test_flush_waits_for_the_disk_while_the_loop_stays_responsive(workspace, tmp_path, monkeypatch):
    agent = Agent(StubClient([completion("Done")]), ToolRegistry(), "test")
    journal = attach(agent, workspace, tmp_path)
    journal._append("state", note="pending")
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    real = os.fsync

    def sync(descriptor):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return real(descriptor)

    monkeypatch.setattr(os, 'fsync', sync)
    flushing = asyncio.create_task(journal.flush())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        tick = asyncio.Event()
        loop.call_soon(tick.set)
        await asyncio.wait_for(tick.wait(), 5)
        assert not flushing.done()
    finally:
        release.set()
        await asyncio.wait_for(flushing, 5)


async def test_failed_sync_marks_the_journal_unavailable(workspace, tmp_path, monkeypatch):
    agent = Agent(StubClient([completion("Done")]), ToolRegistry(), "test")
    journal = attach(agent, workspace, tmp_path)
    journal._append("state", note="pending")

    def fail(descriptor):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fail)
    try:
        await journal.flush()
    except SessionError as exc:
        assert "disk full" in str(exc)
    else:
        raise AssertionError("flush must report the failed sync")
    assert journal.failed


@pytest.mark.parametrize('cancel_at', ['journal', 'tool'])
async def test_cancel_at_tool_start_durability_boundary_never_replays(workspace, tmp_path, monkeypatch, cancel_at):
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()

    class GatedTool(RecordingTool):
        async def run(self, **kwargs):
            if cancel_at == 'tool':
                self.seen.append(kwargs)
                entered.set()
                await asyncio.Event().wait()
            return await super().run(**kwargs)

    tool = GatedTool()
    agent = Agent(StubClient([completion('Run', [ToolCall('call', 'record', {'value': 'once'})])]),
                  ToolRegistry([tool]), 'test')
    journal = attach(agent, workspace, tmp_path)
    sync = journal._sync
    blocked = False

    def gated_sync():
        nonlocal blocked
        if cancel_at == 'journal' and not blocked and '"tool_start"' in journal.path.read_text():
            blocked = True
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        sync()

    monkeypatch.setattr(journal, '_sync', gated_sync)
    task = asyncio.create_task(agent.run('Perform one action'))
    restored = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        expected = [] if cancel_at == 'journal' else [{'value': 'once'}]
        assert tool.seen == expected
        task.cancel()
        if cancel_at == 'journal':
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        await journal.flush()
        restored = Agent(StubClient([completion('Recovered')]), ToolRegistry([tool]), 'test')
        journal.restore(restored, journal.load(journal.session_id))
        assert await restored.run('Continue') == 'Recovered'
        assert tool.seen == expected
        calls = [call.id for message in restored.messages for call in message.tool_calls or []]
        results = [message.tool_call_id for message in restored.messages if message.role == 'tool']
        assert results == calls and len(results) == 1
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.registry.aclose()
        if restored is not None:
            await restored.registry.aclose()
