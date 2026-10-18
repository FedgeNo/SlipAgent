"""Startup stays visible and independent services overlap without early requests."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest

from slipagent import cli
from slipagent.agent import Agent
from slipagent.environment import ProjectEnvironment
from slipagent.mcp import MCPManager, ServerSpec
from slipagent.repomap import RepositoryMap
from slipagent.nvidia import ULTRA_MODEL
from slipagent.providers import create_client


@pytest.mark.parametrize("mcp_status", ["connected", "error"])
async def test_startup_reports_failed_metadata_and_mcp_outcomes(tmp_path, monkeypatch, mcp_status):
    import io
    from unittest.mock import AsyncMock
    client = SimpleNamespace(aclose=AsyncMock(), model_capabilities=AsyncMock(return_value=None),
                             key_info=AsyncMock(return_value=None))
    monkeypatch.setattr(cli, "create_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(ProjectEnvironment, "snapshot", AsyncMock(return_value={}))
    monkeypatch.setattr(RepositoryMap, "snapshot", AsyncMock(return_value=""))
    args = cli.build_parser().parse_args([
        "--workspace", str(tmp_path), "--api-key", "dummy", "--no-session", "--no-reload", "--no-mcp",
        "--model", "test/manual",
    ])
    session = await cli.Launcher(args).build_session(defer_startup=True)
    session.renderer.stream = io.StringIO()
    session.mcp = SimpleNamespace(connect_all=AsyncMock(return_value=[SimpleNamespace(status=mcp_status)]),
                                  aclose=AsyncMock())
    try:
        await cli.Launcher(args).initialize(session, cli.Config.from_env(api_key="dummy", workspace=tmp_path), announce=True)
        output = session.renderer.stream.getvalue()
        assert "Model capability check: failed." in output
        assert "Finished fetching the model catalog" not in output
        assert f"MCP: {int(mcp_status == 'connected')} connected; {int(mcp_status == 'error')} failed." in output
        assert "Finished connecting to MCP" not in output
    finally:
        await cli.Repl(session).shutdown()


def test_startup_reports_missing_current_directory(monkeypatch, capsys):
    from pathlib import Path

    def missing_directory(cls):
        raise FileNotFoundError(2, "No such file or directory")

    async def dispatch(*args):
        pytest.fail("Startup must stop before initializing services")

    monkeypatch.setattr(Path, "cwd", classmethod(missing_directory))
    monkeypatch.setattr(cli.Launcher, "dispatch", dispatch)
    assert cli.main(["-p", "test", "--no-mcp", "--no-session"]) == 2
    error = capsys.readouterr().err
    assert "current directory is unavailable" in error
    assert "cd" in error
    assert "Traceback" not in error


@pytest.mark.parametrize("cancel", [False, True])
async def test_deferred_startup_overlaps_services_before_context(tmp_path, monkeypatch, cancel):
    entered = set()
    release = asyncio.Event()
    context = []

    async def phase(name):
        entered.add(name)
        await release.wait()

    async def metadata(*args, **kwargs):
        await phase("metadata")
        return object()

    async def quota():
        await phase("quota")
        return None

    async def environment(self):
        await phase("environment")
        return {}

    async def repository(self, *args):
        await phase("repository")
        return ""

    async def mcp(self, **kwargs):
        await phase("mcp")
        return []

    async def preview(self, *args, **kwargs):
        assert release.is_set()
        context.append(True)
        return []

    async def close():
        pass

    client = SimpleNamespace(model_capabilities=metadata, key_info=quota, aclose=close)
    monkeypatch.setattr(cli, "create_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(ProjectEnvironment, "snapshot", environment)
    monkeypatch.setattr(RepositoryMap, "snapshot", repository)
    monkeypatch.setattr(MCPManager, "connect_all", mcp)
    monkeypatch.setattr(Agent, "_context_view", preview)
    args = cli.build_parser().parse_args([
        "--workspace", str(tmp_path), "--api-key", "dummy", "--no-session", "--no-reload",
        "--model", "test/manual",
    ])
    session = await cli.Launcher(args).build_session(defer_startup=True)
    assert not entered
    task = asyncio.create_task(session.extensions.pop("startup")())
    try:
        async with asyncio.timeout(2):
            while len(entered) < 5:
                await asyncio.sleep(0)
        assert not context
        if cancel:
            task.cancel()
            await asyncio.sleep(.01)
            assert not task.done(), "startup must drain its in-flight repository worker"
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not context
        else:
            await task
            assert context == [True]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()


async def test_mcp_startup_has_four_connection_limit(tmp_path, monkeypatch):
    manager = MCPManager(tmp_path)
    specs = {str(i): ServerSpec(name=str(i), command="unused") for i in range(7)}
    entered = []
    completed = []
    release = asyncio.Event()
    active = 0
    maximum = 0

    async def connect(name, spec):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        entered.append(name)
        try:
            await release.wait()
            return SimpleNamespace(spec=spec)
        finally:
            active -= 1

    monkeypatch.setattr(manager, "configured", lambda: specs)
    monkeypatch.setattr(manager, "connect", connect)
    task = asyncio.create_task(manager.connect_all(on_state=completed.append))
    try:
        async with asyncio.timeout(2):
            while len(entered) < 4:
                await asyncio.sleep(0)
        assert len(entered) == 4
        release.set()
        states = await task
        assert [state.spec.name for state in states] == sorted(specs)
        assert len(completed) == 7
        assert maximum == 4
    finally:
        release.set()
        await task


async def test_cancel_mcp_startup_drains_connections(tmp_path, monkeypatch):
    manager = MCPManager(tmp_path)
    specs = {str(i): ServerSpec(name=str(i), command="unused") for i in range(7)}
    entered, drained = [], []

    async def connect(name, spec):
        entered.append(name)
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            drained.append(name)

    monkeypatch.setattr(manager, "configured", lambda: specs)
    monkeypatch.setattr(manager, "connect", connect)
    task = asyncio.create_task(manager.connect_all())
    async with asyncio.timeout(2):
        while len(entered) < 4:
            await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(drained) == sorted(entered)
    assert len(entered) == 4


@pytest.mark.parametrize("provider", ["openrouter", "nvidia", "kilo"])
async def test_startup_fetches_each_catalog_once(tmp_path, monkeypatch, provider):
    requests = []

    def respond(request):
        requests.append(request)
        properties = {"id": ULTRA_MODEL, "context_length": 1_000_000,
                      "supported_parameters": ["tools", "max_tokens"]}
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [properties]})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json={"data": {"endpoints": [properties]}})
        return httpx.Response(200, json={"data": {"label": "dummy", "usage": 0}})

    def client(name, **options):
        return create_client(name, **options, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(cli, "create_client", client)
    args = cli.build_parser().parse_args([
        "--workspace", str(tmp_path), "--api-key", "dummy", "--provider", provider,
        "--model", ULTRA_MODEL, "--no-session", "--no-reload", "--no-mcp",
    ])
    session = await cli.Launcher(args).build_session()
    try:
        assert sum(request.url.path.endswith("/models") for request in requests) == 1
        assert sum(request.url.path.endswith("/endpoints") for request in requests) == (provider == "openrouter")
        assert not any(request.method == "POST" for request in requests)
    finally:
        await cli.Repl(session).shutdown()


async def test_cli_limits_worker_threads_to_four(monkeypatch):
    entered = asyncio.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = maximum = 0
    loop = asyncio.get_running_loop()

    def work():
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active >= 4:
                loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(3)
        finally:
            with lock:
                active -= 1

    async def build(*args, **kwargs):
        tasks = [asyncio.create_task(asyncio.to_thread(work)) for _ in range(8)]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.sleep(.02)
            assert maximum == 4
        finally:
            release.set()
            await asyncio.gather(*tasks)
        return object()

    async def repl(session):
        return 0

    monkeypatch.setattr(cli.Launcher, "build_session", build)
    monkeypatch.setattr(cli.Repl, "run", repl)
    assert await cli.Launcher(cli.build_parser().parse_args([])).dispatch(None) == 0


async def test_deferred_startup_failure_closes_session(tmp_path, monkeypatch, capsys):
    closed = []

    async def close():
        closed.append(True)

    monkeypatch.setattr(cli, "create_client", lambda *args, **kwargs: SimpleNamespace(aclose=close))
    args = cli.build_parser().parse_args([
        "--workspace", str(tmp_path), "--api-key", "dummy", "--no-session", "--no-reload", "--no-mcp",
    ])
    session = await cli.Launcher(args).build_session(defer_startup=True)

    async def fail():
        raise cli.APIConfigError("startup unavailable")

    session.extensions["startup"] = fail
    assert await cli.Repl(session).run() == 2
    assert closed == [True]
    assert "Startup failed: startup unavailable" in capsys.readouterr().err
