import argparse
import asyncio
import io

import pytest

from slipagent import cli
from slipagent.agent import Agent
from slipagent.tools.base import ToolRegistry
from slipagent.workspace import Workspace
from test_agent import RecordingTool, StubClient, call, completion


@pytest.mark.parametrize("stop", [False, True])
async def test_answer_only_followup_runs_once_unless_stopped(tmp_path, monkeypatch, stop):
    from slipagent.tools.answer import AnswerTool
    from slipagent.types import ToolCall
    session, started, release = session_with_paused_request(tmp_path)
    session.registry.register(AnswerTool())
    session.client.responses = [completion(tool_calls=[ToolCall("a", "answer", {"text": "First done"})]), completion("Followup done")]
    queue = asyncio.Queue()
    monkeypatch.setattr(cli.Repl, "read_line", lambda *args: queue.get())
    task = asyncio.create_task(cli.Repl(session).run_request("first task", session.renderer.style))
    try:
        await started.wait()
        session.agent.enqueue("queued followup")
        if stop:
            session.agent.request_stop()
        release.set()
        await asyncio.wait_for(task, 3)
        assert len(session.client.calls) == (1 if stop else 2)
        assert session.agent.has_queued_input is stop
        assert sum(message.content == "queued followup" for message in session.agent.messages) == 1
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()


async def test_stopped_one_shot_returns_failure(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    agent = SimpleNamespace(run=AsyncMock(return_value="Stopped after repeated actions."),
                            wait_for_compaction=AsyncMock(), stopped=True)
    renderer = cli.Renderer(cli.Style(False), io.StringIO(), False)
    renderer.output_markdown = True
    monkeypatch.setattr(cli.Repl, "shutdown", AsyncMock())
    assert await cli.Launcher(argparse.Namespace()).run_one_shot(SimpleNamespace(agent=agent, renderer=renderer), "task") == 1


async def test_queued_command_pause_is_labelled_a_pause_not_a_stop(tmp_path, monkeypatch):
    from types import SimpleNamespace
    session, started, release = session_with_paused_request(tmp_path)
    queue = asyncio.Queue()
    monkeypatch.setattr(cli.Repl, "read_line", lambda *args: queue.get())
    labels = []
    task = asyncio.create_task(cli.Repl(session).run_request("go", session.renderer.style))
    try:
        await asyncio.wait_for(started.wait(), 3)
        session.renderer.terminal = SimpleNamespace(set_working=lambda working, **flags: labels.append(flags))
        # A queued setting halts the run only to apply it, then work continues.
        cli.CommandRunner(session).queue("/plugins", "plugins")
        assert session.agent.stop_requested and labels[-1] == {"stopping": False, "pausing": True}
    finally:
        session.renderer.terminal = None
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()


def session_with_paused_request(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()
    class Client(StubClient):
        async def chat(self, **kwargs):
            started.set()
            await release.wait()
            return await super().chat(**kwargs)
        async def aclose(self):
            pass
    client = Client([completion(tool_calls=[call(value="read")]), completion("done"), completion("followup")])
    registry = ToolRegistry([RecordingTool()])
    agent = Agent(client, registry, "test")
    renderer = cli.Renderer(cli.Style(False), io.StringIO(), False)
    session = cli.Session(agent, registry, client, renderer, Workspace(tmp_path), "test", "http://unused.invalid", None, "test")
    return session, started, release


async def test_slow_key_status_runs_during_batch_and_run_can_be_interrupted(tmp_path, monkeypatch):
    session, started, release = session_with_paused_request(tmp_path)
    queue = asyncio.Queue()
    catalog_started = asyncio.Event()
    async def read(*args):
        return await queue.get()
    async def catalog(*args):
        catalog_started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(cli.Repl, "read_line", read)
    monkeypatch.setattr(cli.KeyEntry, "command", catalog)
    task = asyncio.create_task(cli.Repl(session).run_request("go", session.renderer.style))
    try:
        await started.wait()
        queue.put_nowait("/key status")
        # A read-only command starts at once and leaves the paused request running.
        await asyncio.wait_for(catalog_started.wait(), 1)
        assert not session.agent.stop_requested
        command_task, = session.work.command_tasks
        # Interrupting the paused run also cancels the slow command.
        cli.Repl(session).request_interrupt()
        await asyncio.wait_for(task, 1)
        await cli.Repl(session).wait_for_interrupt()
        assert session.client.calls == []
        assert session.agent.stopped
        assert command_task.cancelled()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()


@pytest.mark.parametrize("exit_command", ["/quit", None])
async def test_shutdown_does_not_start_a_queued_run(tmp_path, monkeypatch, exit_command):
    session, started, release = session_with_paused_request(tmp_path)
    session.client.responses = [completion("first done"), completion("should not run")]
    queue = asyncio.Queue()
    async def read(*args):
        return await queue.get()
    monkeypatch.setattr(cli.Repl, "read_line", read)
    task = asyncio.create_task(cli.Repl(session).run_request("first task", session.renderer.style))
    try:
        await started.wait()
        queue.put_nowait("queued followup")
        queue.put_nowait(exit_command)
        await asyncio.sleep(.03)
        release.set()
        assert await asyncio.wait_for(task, 1)
        assert len(session.client.calls) == 1
        assert session.agent.pending == ["queued followup"]
        assert "1 queued message was not run because the session is exiting" in session.renderer.stream.getvalue()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()


@pytest.mark.parametrize("line", ["/reset", "next task"])
async def test_input_received_during_deferred_commands_survives_run_end(tmp_path, monkeypatch, line):
    session, started, release = session_with_paused_request(tmp_path)
    session.client.responses = [completion("done")]
    queue = asyncio.Queue()
    applying, applied, consumed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_read = cli.Repl.read_line

    async def read(*args):
        value = await queue.get()
        consumed.set()
        return value

    async def apply(*args, **kwargs):
        applying.set()
        await applied.wait()
        return False, False

    monkeypatch.setattr(cli.Repl, "read_line", read)
    monkeypatch.setattr(cli.CommandRunner, "apply_deferred", apply)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))
    task = asyncio.create_task(cli.Repl(session).run_request("go", session.renderer.style))
    try:
        await asyncio.wait_for(started.wait(), 1)
        release.set()
        await asyncio.wait_for(applying.wait(), 1)
        queue.put_nowait(line)
        await asyncio.wait_for(consumed.wait(), 1)
        applied.set()
        assert not await asyncio.wait_for(task, 1)
        assert await asyncio.wait_for(original_read(cli.Repl(session), session.renderer.style), 1) == line
        assert await asyncio.wait_for(original_read(cli.Repl(session), session.renderer.style), 1) is None
    finally:
        applied.set()
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()


async def test_quit_stops_rate_limit_wait_without_another_request(tmp_path, monkeypatch):
    import httpx
    from slipagent.capabilities import ModelCapabilities
    from slipagent.openrouter import OpenRouterClient

    requests = []
    waiting = asyncio.Event()
    queue = asyncio.Queue()

    def handle(request):
        requests.append(request)
        return httpx.Response(429, headers={"retry-after": "30"}, json={"error": {"message": "Rate limited"}})

    async def read(*args):
        return await queue.get()

    monkeypatch.setattr(cli.Repl, "read_line", read)
    client = OpenRouterClient("test", transport=httpx.MockTransport(handle))
    client.cache_capabilities("test", ModelCapabilities({}, [{"tag": "stub", "supported_parameters": ["tools"], "context_length": 1000000}]))
    registry = ToolRegistry()
    renderer = cli.Renderer(cli.Style(False), io.StringIO(), False)
    agent = Agent(client, registry, "test")
    backoff = client._backoff
    def observe_backoff(attempt, retry_after=None):
        waiting.set()
        return backoff(attempt, retry_after)
    monkeypatch.setattr(client, "_backoff", observe_backoff)
    session = cli.Session(agent, registry, client, renderer, Workspace(tmp_path), "test", client.base_url, None, "test")
    task = asyncio.create_task(cli.Repl(session).run_request("go", renderer.style))
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        queue.put_nowait("/quit")
        assert await asyncio.wait_for(asyncio.shield(task), 1)
        assert len(requests) == 1
        assert agent.stopped
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()
