"""Escape ends the step without discarding results or launching another step."""

import asyncio
import io
import json
import shlex
import sys

import pytest

from slipagent import cli
from slipagent.tools.base import Tool, ToolResult
from test_agent import StubClient, call, completion
from test_cli_e2e import metadata_server


@pytest.mark.parametrize("phase", ["request", "tools"])
async def test_interrupt_preserves_results_and_prevents_continuation(tmp_path, monkeypatch, metadata_server, phase):
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    ran = []

    class BlockingClient(StubClient):
        async def chat(self, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    class ProbeTool(Tool):
        name = "probe_interrupt"
        description = "Test interruption."
        parameters = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}

        async def run(self, value):
            ran.append(value)
            if value == "blocked":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return ToolResult.ok("Completed " + value)

    session.registry.register(ProbeTool())
    calls = [call("probe_interrupt", value=value) for value in ["done", "blocked", "unstarted"]]
    for index, tool_call in enumerate(calls):
        tool_call.id = str(index)
    client = BlockingClient([]) if phase == "request" else StubClient([completion(tool_calls=calls), completion("Continued.")])
    session.agent.client = client

    async def read(*arguments):
        await asyncio.Event().wait()

    monkeypatch.setattr(cli, "_read_line", read)
    step = asyncio.create_task(cli._run_request(session, "Inspect", session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        session.agent.enqueue("Queued follow-up")
        await cli._handle_command(session, "/danger")
        cli._request_interrupt(session)
        cleanup = session.extensions["interrupt_task"]
        cli._request_interrupt(session)
        assert session.extensions["interrupt_task"] is cleanup
        assert await asyncio.wait_for(step, 3) is False
        assert cancelled.is_set()
        assert not session.agent.running
        assert session.agent.pending == ["Queued follow-up"]
        assert not session.workspace.access.danger
        assert not session.extensions.get("deferred_commands")
        assert not session.registry.services["step_compactor"].jobs
        assert not client.summary_calls
        if phase == "tools":
            assert len(client.calls) == 1
            assert ran == ["done", "blocked"]
            results = [message for message in session.agent.messages if message.role == "tool"]
            assert len(results) == 3
            assert "Completed done" in results[0].content
            assert "interrupted" in results[1].content
            assert "not run" in results[2].content
            assert session.agent.history.steps
        assert "Interrupted." in session.renderer.stream.getvalue()
    finally:
        step.cancel()
        await asyncio.gather(step, return_exceptions=True)
        await cli._shutdown(session)


async def test_interrupt_cancels_inflight_interactive_command(tmp_path, monkeypatch, metadata_server):
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def command(*arguments):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(cli, "_handle_command", command)
    task = asyncio.create_task(cli._run_interactive_command(session, "/model example"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        cli._request_interrupt(session)
        assert await asyncio.wait_for(task, 2) is False
        assert cancelled.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli._shutdown(session)


async def test_interrupt_stops_background_processes_summaries_and_commands(tmp_path, metadata_server):
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    try:
        command = shlex.join([sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"])
        result = await session.registry.invoke("run_command", {"command": command, "background": True})
        assert not result.is_error, result.content
        job_id = json.loads(result.content.splitlines()[0])["job_id"]
        jobs = session.registry.services["command_jobs"]
        job = jobs.jobs[job_id]
        for _ in range(200):
            output = await session.registry.invoke("read_command_output", {"log_id": job_id})
            if "ready" in output.content:
                break
            await asyncio.sleep(.01)
        assert "ready" in output.content
        summary = asyncio.create_task(asyncio.Event().wait())
        compactor = session.registry.services["step_compactor"]
        compactor.jobs.add(summary)
        command_task = asyncio.create_task(asyncio.Event().wait())
        session.extensions["command_tasks"] = {command_task}
        cli._request_interrupt(session)
        await asyncio.wait_for(cli._wait_for_interrupt(session), 3)
        assert job.state == "stopped"
        assert job.log.finished and not jobs.active
        assert summary.cancelled() and command_task.cancelled()
        assert not compactor.jobs
    finally:
        await cli._shutdown(session)


async def test_interrupt_during_deferred_command_discards_remaining_commands(tmp_path, monkeypatch, metadata_server):
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    entered = asyncio.Event()
    ran = []

    async def command(session, line):
        ran.append(line)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(cli, "_handle_command", command)
    session.extensions["deferred_commands"] = ["/model example", "/danger"]
    session.extensions["resume_after_commands"] = True
    task = asyncio.create_task(cli._apply_deferred_commands(session))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        cli._request_interrupt(session)
        assert await asyncio.wait_for(task, 2) == (False, False)
        assert ran == ["/model example"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli._shutdown(session)


async def test_interrupt_as_turn_finishes_does_not_launch_pending_prompt(tmp_path, monkeypatch, metadata_server):
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    runs = []

    async def run(agent, prompt, **kwargs):
        runs.append(prompt)
        agent.enqueue("Pending correction")
        # Escape arrives after completion, before the controller handles it.
        asyncio.get_running_loop().call_soon(cli._request_interrupt, session)
        return "Finished"

    async def read(*arguments):
        await asyncio.Event().wait()

    monkeypatch.setattr(type(session.agent), "run", run)
    monkeypatch.setattr(cli, "_read_line", read)
    try:
        assert await asyncio.wait_for(cli._run_request(session, "Inspect", session.renderer.style), 2) is False
        assert runs == ["Inspect"]
        assert session.agent.pending == ["Pending correction"]
    finally:
        await cli._shutdown(session)
