import asyncio
import io

import pytest

from slipagent import cli
from slipagent.agent import Agent
from slipagent.tools.base import ToolRegistry
from slipagent.workspace import Workspace
from test_agent import RecordingTool, StubClient, call, completion


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


async def test_slow_key_status_does_not_block_stop(tmp_path, monkeypatch):
    session, started, release = session_with_paused_request(tmp_path)
    queue = asyncio.Queue()
    catalog_started = asyncio.Event()
    async def read(*args):
        return await queue.get()
    async def catalog(*args):
        catalog_started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(cli, "_read_line", read)
    monkeypatch.setattr(cli, "_key_command", catalog)
    task = asyncio.create_task(cli._run_request(session, "go", session.renderer.style))
    try:
        await started.wait()
        queue.put_nowait("/key status")
        await asyncio.wait_for(catalog_started.wait(), 1)
        queue.put_nowait("/stop")
        async with asyncio.timeout(.5):
            while not session.agent.stop_requested:
                await asyncio.sleep(.005)
        release.set()
        await asyncio.wait_for(task, 1)
        assert len(session.client.calls) == 1
        assert session.agent.stopped
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli._shutdown(session)


@pytest.mark.parametrize("exit_command", ["/quit", None])
async def test_shutdown_does_not_start_a_queued_run(tmp_path, monkeypatch, exit_command):
    session, started, release = session_with_paused_request(tmp_path)
    session.client.responses = [completion("first done"), completion("should not run")]
    queue = asyncio.Queue()
    async def read(*args):
        return await queue.get()
    monkeypatch.setattr(cli, "_read_line", read)
    task = asyncio.create_task(cli._run_request(session, "first task", session.renderer.style))
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
        await cli._shutdown(session)


@pytest.mark.parametrize("line", ["/reset", "next task"])
async def test_input_received_during_deferred_commands_survives_run_end(tmp_path, monkeypatch, line):
    session, started, release = session_with_paused_request(tmp_path)
    session.client.responses = [completion("done")]
    queue = asyncio.Queue()
    applying, applied, consumed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_read = cli._read_line

    async def read(*args):
        value = await queue.get()
        consumed.set()
        return value

    async def apply(*args, **kwargs):
        applying.set()
        await applied.wait()
        return False, False

    monkeypatch.setattr(cli, "_read_line", read)
    monkeypatch.setattr(cli, "_apply_deferred_commands", apply)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))
    task = asyncio.create_task(cli._run_request(session, "go", session.renderer.style))
    try:
        await asyncio.wait_for(started.wait(), 1)
        release.set()
        await asyncio.wait_for(applying.wait(), 1)
        queue.put_nowait(line)
        await asyncio.wait_for(consumed.wait(), 1)
        applied.set()
        assert not await asyncio.wait_for(task, 1)
        assert await asyncio.wait_for(original_read(session, session.renderer.style), 1) == line
        assert await asyncio.wait_for(original_read(session, session.renderer.style), 1) is None
    finally:
        applied.set()
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli._shutdown(session)


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

    monkeypatch.setattr(cli, "_read_line", read)
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
    task = asyncio.create_task(cli._run_request(session, "go", renderer.style))
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        queue.put_nowait("/quit")
        assert await asyncio.wait_for(asyncio.shield(task), 1)
        assert len(requests) == 1
        assert agent.stopped
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli._shutdown(session)
