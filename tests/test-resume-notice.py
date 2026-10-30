"""Resume announces loading before compaction, journal I/O, or state restoration."""

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from slipagent.cli import LOADING_CONVERSATION_NOTICE, Style
from slipagent.commands import CommandContext, ResumeCommand, SessionsCommand
from slipagent.sessions import SessionError, SessionJournal
from slipagent.terminal import TerminalUI


@pytest.mark.parametrize("entry", ["picker", "id", "sessions"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_loading_notice_precedes_all_resume_work_and_cleans_up(entry, outcome):
    terminal = Mock(spec=TerminalUI)
    terminal.status_line = SimpleNamespace(notification="")
    def notify(text):
        terminal.status_line.notification = text
    terminal.notify.side_effect = notify
    async def announce(text):
        terminal.notify(text)
    terminal.notify_before_work = AsyncMock(side_effect=announce)
    terminal.choose = AsyncMock(side_effect=["saved", "resume"] if entry == "sessions" else ["saved"])
    entered, release = asyncio.Event(), asyncio.Event()
    async def compaction():
        assert terminal.status_line.notification == LOADING_CONVERSATION_NOTICE
        entered.set()
        await release.wait()
    agent = SimpleNamespace(wait_for_compaction=compaction, messages=[], pending=[], queued_messages={},
                            history=SimpleNamespace(steps=[]))
    renderer = SimpleNamespace(terminal=terminal, restore_transcript=AsyncMock())
    session = SimpleNamespace(agent=agent, renderer=renderer, registry=SimpleNamespace(resources=SimpleNamespace(jobs=None)))
    journal = Mock(spec=SessionJournal)
    journal.session_id = "saved"
    journal.listing.return_value = [{"id": "saved", "title": "Saved Session", "current": "False"}]
    if outcome == "failure":
        journal.load.side_effect = SessionError("Unreadable saved session")
    context = CommandContext(session, Style(False), io.StringIO())
    command = SessionsCommand() if entry == "sessions" else ResumeCommand()
    task = asyncio.create_task(command.execute_saved(context, "saved" if entry == "id" else "", journal))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        terminal.notify_before_work.assert_awaited_once_with(LOADING_CONVERSATION_NOTICE)
        journal.load.assert_not_called()
        journal.restore.assert_not_called()
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            await task
        if outcome == "success":
            journal.load.assert_called_once_with("saved")
            renderer.restore_transcript.assert_awaited_once()
        else:
            assert terminal.status_line.notification == ""
            renderer.restore_transcript.assert_not_awaited()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
