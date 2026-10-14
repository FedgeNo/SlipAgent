"""Editor holds block all agent batches while independent background work continues."""

import asyncio
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from slipagent.agent import Agent
from slipagent.jobs import CommandJobs
from slipagent.tools.base import ToolRegistry
from slipagent.tools.output import CommandArchive
from slipagent.tools.shell import RunCommandTool
from slipagent.types import ToolCall, content_text
from slipagent.useredits import UserEdits
from test_agent import RecordingTool, StubClient, completion


def editor(agent):
    opened = asyncio.Event()
    waiting = asyncio.Event()
    terminal = SimpleNamespace(_editor_active=False, notify=Mock())
    async def open_editor(path, *, line):
        terminal._editor_active = True
        opened.set()
    terminal._open_editor_now = open_editor
    session = SimpleNamespace(agent=agent, registry=agent.registry, reloader=None)
    edits = UserEdits(session, terminal)
    async def checkpoint():
        waiting.set()
        await edits.checkpoint()
    agent.on_boundary = checkpoint
    return edits, terminal, opened, waiting


async def test_starting_agent_with_editor_open_cannot_request_or_execute_tools(tmp_path):
    tool = RecordingTool()
    client = StubClient([completion(tool_calls=[ToolCall("after", "record", {"value": "after close"})]), completion("done")])
    agent = Agent(client, ToolRegistry([tool]), "test/model")
    edits, terminal, opened, waiting = editor(agent)
    opening = asyncio.create_task(edits.request(tmp_path / "file"))
    run = None
    try:
        await asyncio.wait_for(opened.wait(), 2)
        run = asyncio.create_task(agent.run("Work"))
        await asyncio.wait_for(waiting.wait(), 2)
        assert not run.done() and not client.calls and not tool.seen
        assert not edits.released.is_set()
        terminal._editor_active = False
        edits.close()
        assert await asyncio.wait_for(run, 2) == "done"
        assert tool.seen == [{"value": "after close"}]
    finally:
        edits.close()
        if run is not None and not run.done():
            run.cancel()
        await asyncio.gather(opening, *([run] if run is not None else []), return_exceptions=True)
        await agent.wait_for_compaction()


async def test_hold_blocks_next_tool_batch_but_allows_history_compaction(tmp_path):
    summary_started, summary_release = asyncio.Event(), asyncio.Event()
    class Client(StubClient):
        async def chat(self, **kwargs):
            if content_text(kwargs["messages"][0].content).lstrip().startswith("# Summarizing the Previous Turn"):
                summary_started.set()
                await summary_release.wait()
            return await super().chat(**kwargs)
    tool = RecordingTool()
    client = Client([
        completion(tool_calls=[ToolCall("one", "record", {"value": "first"}), ToolCall("two", "record", {"value": "second"})]),
        completion(tool_calls=[ToolCall("three", "record", {"value": "after close"})]),
        completion("done"),
    ])
    agent = Agent(client, ToolRegistry([tool]), "test/model")
    edits, terminal, opened, _ = editor(agent)
    def event(event):
        if event.kind == "tool_start" and event.tool_call.id == "one":
            edits.pending = tmp_path / "file", 1
            edits.hold()
    agent.on_event = event
    run = asyncio.create_task(agent.run("Work"))
    try:
        await asyncio.wait_for(opened.wait(), 2)
        await asyncio.wait_for(summary_started.wait(), 2)
        assert tool.seen == [{"value": "first"}, {"value": "second"}]
        summary_release.set()
        await asyncio.wait_for(agent.wait_for_compaction(), 2)
        assert agent.history.steps[0].summary
        assert client.summary_calls and all(not call.get("tools") for call in client.summary_calls)
        assert len(client.calls) == 1 and not run.done()
        assert terminal._editor_active and not edits.released.is_set()
        assert tool.seen == [{"value": "first"}, {"value": "second"}]
        terminal._editor_active = False
        edits.close()
        assert await asyncio.wait_for(run, 2) == "done"
        assert tool.seen[-1] == {"value": "after close"}
    finally:
        summary_release.set()
        edits.close()
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await agent.wait_for_compaction()


async def test_running_command_survives_editor_open_and_can_finish_while_agent_held(workspace):
    archive, jobs = CommandArchive(100_000), CommandJobs()
    tool, client = RecordingTool(), StubClient([completion("done")])
    registry = ToolRegistry([tool], services={"command_jobs": jobs})
    agent = Agent(client, registry, "test/model")
    edits, terminal, opened, waiting = editor(agent)
    ready, finish = workspace.root / "ready", workspace.root / "finish"
    code = "from pathlib import Path\nimport time\nPath('ready').touch()\nwhile not Path('finish').exists():\n    time.sleep(.01)\nprint('finished')"
    opening = run = None
    try:
        result = await RunCommandTool(workspace, archive, jobs).run(shlex.join([sys.executable, "-u", "-c", code]), background=True)
        assert not result.is_error
        job = jobs.jobs[result.content["job_id"]]
        async with asyncio.timeout(3):
            while not ready.exists():
                await asyncio.sleep(.01)
        opening = asyncio.create_task(edits.request(workspace.root / "file"))
        await asyncio.wait_for(opened.wait(), 2)
        run = asyncio.create_task(agent.run("Work"))
        await asyncio.wait_for(waiting.wait(), 2)
        assert jobs.active and job.state == "running" and not job.task.done()
        finish.touch()
        await asyncio.wait_for(asyncio.shield(job.task), 2)
        assert job.state == "completed" and job.log.returncode == 0
        assert not client.calls and not tool.seen and not run.done()
        assert terminal._editor_active and not edits.released.is_set()
        terminal._editor_active = False
        edits.close()
        assert await asyncio.wait_for(run, 2) == "done"
    finally:
        finish.touch()
        edits.close()
        if run is not None and not run.done():
            run.cancel()
        await asyncio.gather(*(task for task in (opening, run) if task is not None), return_exceptions=True)
        await jobs.aclose()
        await archive.aclose()
        await agent.wait_for_compaction()
