"""Editable sessions wait for tool boundaries and deliver explicit user diffs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from slipagent.agent import Agent
from slipagent.tools.base import ToolRegistry
from slipagent.types import decode_json_content
from slipagent.useredits import UserEdits
from slipagent.sessions import SessionJournal
from test_agent import StubClient, RecordingTool, completion
from slipagent.types import ToolCall


async def test_edit_request_waits_for_batch_boundary_and_preserves_background_jobs(tmp_path):
    terminal = SimpleNamespace(_editor_active=False, notify=Mock())
    async def opened(path, *, line):
        terminal._editor_active = True
    terminal._open_editor_now = AsyncMock(side_effect=opened)
    jobs = SimpleNamespace(stop_all=AsyncMock())
    session = SimpleNamespace(agent=SimpleNamespace(running=True), registry=ToolRegistry(), reloader=None)
    session.registry.services["command_jobs"] = jobs
    edits = UserEdits(session, terminal)
    await edits.request(tmp_path / "a", 12)
    terminal._open_editor_now.assert_not_awaited()
    boundary = asyncio.create_task(edits.boundary())
    await asyncio.sleep(.02)
    jobs.stop_all.assert_not_awaited()
    terminal._open_editor_now.assert_awaited_once_with(tmp_path / "a", line=12)
    assert not boundary.done()
    edits.close()
    await boundary


async def test_opening_notification_clears_at_boundary_and_preserves_newer_events(tmp_path):
    terminal = SimpleNamespace(_editor_active=False, _notification="")
    terminal.notify = lambda text: setattr(terminal, "_notification", text)
    terminal._open_editor_now = AsyncMock()
    session = SimpleNamespace(agent=SimpleNamespace(running=True), reloader=None)
    edits = UserEdits(session, terminal)
    await edits.request(tmp_path / "a")
    assert terminal._notification == "Opening editor after this turn"
    await edits.boundary()
    assert terminal._notification == ""
    await edits.request(tmp_path / "b")
    terminal.notify("New event")
    edits.close()
    assert terminal._notification == "New event"


async def test_agent_finishes_all_tools_before_opening_editor_and_waits_before_next_request(tmp_path):
    tool = RecordingTool()
    client = StubClient([completion(tool_calls=[ToolCall("one", "record", {"value": "first"}), ToolCall("two", "record", {"value": "second"})]), completion("done")])
    agent = Agent(client, ToolRegistry([tool]), "test/model")
    terminal = SimpleNamespace(_editor_active=False, notify=Mock(), _touched_items={}, write=Mock())
    async def opened(path, *, line):
        assert tool.seen == [{"value": "first"}, {"value": "second"}]
        terminal._editor_active = True
    terminal._open_editor_now = AsyncMock(side_effect=opened)
    session = SimpleNamespace(agent=agent, registry=agent.registry, reloader=None)
    edits = UserEdits(session, terminal)
    agent.on_boundary = edits.checkpoint
    def event(event):
        if event.kind == "tool_start" and event.tool_call.id == "one":
            edits.pending = tmp_path / "file", 1
            edits.released.clear()
    agent.on_event = event
    run = asyncio.create_task(agent.run("Do the work"))
    try:
        async with asyncio.timeout(3):
            while not terminal._editor_active:
                await asyncio.sleep(.01)
        assert len(client.calls) == 1
        edits.record(tmp_path / "file", "+user edit")
        terminal._editor_active = False
        edits.close()
        assert await run == "done"
        payload = decode_json_content(client.calls[-1]["messages"][-1].to_api()["content"])
        assert payload["user_messages"] == []
        assert payload["user_edits"][0]["diff"] == "+user edit"
        assert {call["call_id"] for call in payload["history"][-1]["tool_calls"]} == {"one", "two"}
    finally:
        edits.close()
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)


async def test_saved_user_edit_survives_resume_and_is_distinct_from_user_message(tmp_path):
    client = StubClient([completion("done")])
    agent = Agent(client, ToolRegistry(), "test/model")
    journal = SessionJournal(str(tmp_path), tmp_path / "sessions")
    agent.registry.services["session_journal"] = journal
    terminal = SimpleNamespace(_touched_items={}, write_artifact_reference=Mock(), notify=Mock(), write=Mock())
    session = SimpleNamespace(agent=agent, registry=agent.registry)
    edits = UserEdits(session, terminal)
    path = tmp_path / "file.txt"
    edits.record(path, "@@ -1 +1 @@\n-old\n+user\n", "edit1")
    data = journal.load(journal.session_id)
    restored = Agent(client, ToolRegistry(), "test/model")
    journal.restore(restored, data)
    await restored.run("Continue")
    payload = decode_json_content(client.calls[-1]["messages"][-1].to_api()["content"])
    assert payload["user_messages"] == ["Continue"]
    assert payload["user_edits"][0]["path"] == str(path)
    assert payload["user_edits"][0]["source"] == "user"
    assert "+user" in payload["user_edits"][0]["diff"]
    edits.record(path, "", "edit1")
    assert agent.registry.context_notes["user_edits"]["edit1"]["reverted_to_editor_baseline"]


async def test_cancelling_held_agent_does_not_wait_again_in_final_boundary(tmp_path):
    client = StubClient([])
    agent = Agent(client, ToolRegistry(), "test/model")
    terminal = SimpleNamespace(_editor_active=True, notify=Mock())
    session = SimpleNamespace(agent=agent, registry=agent.registry, reloader=None)
    edits = UserEdits(session, terminal)
    edits.released.clear()
    agent.on_boundary = edits.checkpoint
    task = asyncio.create_task(agent.run("Work"))
    await asyncio.sleep(.02)
    assert not task.done()
    task.cancel()
    async with asyncio.timeout(3):
        await asyncio.gather(task, return_exceptions=True)
    assert not agent.running
    assert not client.calls
    assert not edits.released.is_set()
    edits.close()
