"""MCP wire, deadline, and cleanup regressions; no remote services."""

import asyncio
import json
import sys

import pytest

from slipagent.agent import Agent
from slipagent.mcp import MCPClient, MCPError, MCPManager, ServerSpec
from slipagent.tools.base import ToolRegistry
from test_mcp import stub_spec


@pytest.mark.parametrize("mode", ["legacy", "modern"])
async def test_server_instructions_reach_context_and_disappear_on_disconnect(tmp_path, mode):
    registry = ToolRegistry()
    manager = MCPManager(tmp_path, registry=registry)
    state = await manager.connect("server", stub_spec(mode=mode))
    try:
        assert state.status == "connected", state.detail
        agent = Agent(client=object(), registry=registry, model="test")
        view = await agent._context_view(registry.specs(), 1)
        assert "Use echo before boom." in "\n".join(m.content or "" for m in view)
        assert state.client.server_info["name"] == "stub"
        await manager.disconnect("server")
        view = await agent._context_view(registry.specs(), 1)
        assert "Use echo before boom." not in "\n".join(m.content or "" for m in view)
    finally:
        await manager.aclose()


@pytest.mark.parametrize("content", [[], [{"type": "text", "text": '{"value": "unique-result"}'}]])
async def test_structured_content_is_retained_once(content):
    client = MCPClient(stub_spec())
    async def respond(*args, **kwargs):
        return {"result": {"content": content, "structuredContent": {"value": "unique-result"}, "isError": True}}
    client._request = respond
    result = await client.call_tool("echo", {})
    assert result.is_error
    assert result.content.count("unique-result") == 1


@pytest.mark.parametrize("cwd", [None, "sub"])
async def test_server_cwd_is_anchored_to_workspace(tmp_path, cwd):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root if cwd is None else root / cwd
    target.mkdir(exist_ok=True)
    from pathlib import Path
    server = target / "server.py"
    server.write_text(Path(stub_spec().args[0]).read_text())
    manager = MCPManager(root, registry=ToolRegistry())
    try:
        state = await manager.connect("local", ServerSpec("local", sys.executable, ["server.py"], cwd=cwd))
        assert state.status == "connected", state.detail
    finally:
        await manager.aclose()


async def test_cancel_during_tool_discovery_reaps_server(tmp_path, monkeypatch):
    reached = asyncio.Event()
    original = MCPClient.list_tools
    async def wait_forever(self):
        reached.set()
        await asyncio.Event().wait()
        return await original(self)
    monkeypatch.setattr(MCPClient, "list_tools", wait_forever)
    manager = MCPManager(tmp_path, registry=ToolRegistry())
    task = asyncio.create_task(manager.connect("server", stub_spec()))
    await asyncio.wait_for(reached.wait(), 3)
    client = manager.servers["server"].client
    process = client._process
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.connected
        assert process.returncode is not None
        assert not manager.registry.names
    finally:
        await manager.aclose()


async def test_deadline_includes_a_blocked_pipe_write(tmp_path):
    server = tmp_path / "blocked.py"
    server.write_text("import time\ntime.sleep(60)\n")
    client = MCPClient(ServerSpec("blocked", sys.executable, [str(server)]), timeout=.05)
    client._process = await asyncio.create_subprocess_exec(
        sys.executable, str(server), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    try:
        with pytest.raises(MCPError, match="within|timed out"):
            await asyncio.wait_for(client._request("tools/call", {"body": "x" * 3_000_000}, None), 1)
        assert not client.connected
    finally:
        await client.disconnect()


async def test_malformed_tool_schema_is_rejected_during_discovery():
    client = MCPClient(stub_spec())
    async def response(*args, **kwargs):
        return {"result": {"tools": [{"name": "bad", "inputSchema": {"type": "object", "properties": []}}]}}
    client._request = response
    with pytest.raises(MCPError, match="schema|Schema"):
        await client.list_tools()
