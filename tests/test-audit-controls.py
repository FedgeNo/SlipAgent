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


async def test_slow_catalog_does_not_block_stop(tmp_path, monkeypatch):
    session, started, release = session_with_paused_request(tmp_path)
    queue = asyncio.Queue()
    catalog_started = asyncio.Event()
    async def read(*args):
        return await queue.get()
    async def catalog(*args):
        catalog_started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(cli, "_read_line", read)
    monkeypatch.setattr(cli, "_models_command", catalog)
    task = asyncio.create_task(cli._run_turn(session, "go", session.renderer.style))
    try:
        await started.wait()
        queue.put_nowait("/models")
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
    task = asyncio.create_task(cli._run_turn(session, "first task", session.renderer.style))
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
