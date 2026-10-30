"""A slow disk delays the agent before a tool starts, never the event loop."""

import asyncio
import os
import threading
import time

from slipagent.agent import Agent
from slipagent.sessions import SessionError, SessionJournal
from slipagent.tools.base import ToolRegistry
from test_agent import StubClient, completion


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
    release = threading.Event()
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda descriptor: (release.wait(5), real(descriptor)))
    flushing = asyncio.create_task(journal.flush())
    started = time.monotonic()
    ticks = 0
    while not flushing.done() and ticks < 5:
        await asyncio.sleep(0.01)
        ticks += 1
    assert ticks == 5 and not flushing.done()
    assert time.monotonic() - started < 2
    release.set()
    await asyncio.wait_for(flushing, 5)
    assert not journal._unsynced


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
