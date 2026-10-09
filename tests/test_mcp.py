"""MCP client: config parsing, protocol negotiation, and tool exposure.

Protocol and registry tests drive a subprocess via `mcp_stub_server.py`,
exercising JSON-RPC framing, negotiation, and teardown without network access.
Configuration and content tests exercise their helpers directly.
"""

from __future__ import annotations

from slipagent.types import content_text

import json
import sys
from pathlib import Path

import pytest

from slipagent.mcp import (
    MCPClient,
    MCPError,
    MCPManager,
    MCPTool,
    ServerSpec,
    config_path,
    flatten_content,
    load_servers,
    save_servers,
)
from slipagent.tools.base import ToolRegistry

STUB = str(Path(__file__).parent / "mcp_stub_server.py")


def stub_spec(name: str = "stub", mode: str = "legacy") -> ServerSpec:
    return ServerSpec(name=name, command=sys.executable, args=[STUB, mode])


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_missing_config_is_not_an_error(tmp_path: Path) -> None:
    assert load_servers(tmp_path) == {}
    assert config_path(tmp_path) == tmp_path / ".mcp.json"


def test_config_round_trips(tmp_path: Path) -> None:
    save_servers(tmp_path, {"db": ServerSpec(name="db", command="npx", args=["-y", "srv"])})

    loaded = load_servers(tmp_path)

    assert loaded["db"].command == "npx"
    assert loaded["db"].args == ["-y", "srv"]


def test_save_preserves_unrelated_keys(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(json.dumps({"other": 1}), encoding="utf-8")

    save_servers(tmp_path, {"db": ServerSpec(name="db", command="npx")})

    assert json.loads((tmp_path / ".mcp.json").read_text())["other"] == 1


def test_malformed_config_raises(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(MCPError):
        load_servers(tmp_path)


def test_server_without_command_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"broken": {}}}), encoding="utf-8"
    )

    with pytest.raises(MCPError, match="command"):
        load_servers(tmp_path)


# --------------------------------------------------------------------------- #
# Content flattening
# --------------------------------------------------------------------------- #


def test_text_content_is_forwarded() -> None:
    assert flatten_content([{"type": "text", "text": "hello"}]) == "hello"


def test_binary_content_is_summarised_not_dumped() -> None:
    """Base64 payloads would flood the context without informing the model."""
    flat = flatten_content([{"type": "image", "data": "A" * 5000, "mimeType": "image/png"}])

    assert "image" in flat
    assert "omitted" in flat
    assert len(flat) < 200


def test_empty_content_gets_a_placeholder() -> None:
    assert flatten_content([]) == "(no content returned)"


# --------------------------------------------------------------------------- #
# Protocol negotiation
# --------------------------------------------------------------------------- #


async def test_modern_server_uses_per_request_metadata() -> None:
    client = MCPClient(stub_spec(mode="modern"))
    await client.connect()
    try:
        assert client.era == "modern"
        assert client.protocol_version == "2026-07-28"
    finally:
        await client.disconnect()


async def test_legacy_server_falls_back_to_initialize() -> None:
    client = MCPClient(stub_spec(mode="legacy"))
    await client.connect()
    try:
        assert client.era == "legacy"
        assert client.protocol_version == "2025-11-25"
    finally:
        await client.disconnect()


async def test_version_mismatch_retries_a_supported_version() -> None:
    """A modern server that rejects our version names one it does support."""
    client = MCPClient(stub_spec(mode="modern-old"))
    await client.connect()
    try:
        assert client.era == "modern"
        assert client.protocol_version == "2025-11-25"
    finally:
        await client.disconnect()


async def test_missing_binary_reports_a_clear_error() -> None:
    spec = ServerSpec(name="ghost", command="/nonexistent/binary-mcp")
    client = MCPClient(spec)

    with pytest.raises(MCPError, match="could not start"):
        await client.connect()


async def test_disconnect_is_safe_to_call_twice() -> None:
    client = MCPClient(stub_spec())
    await client.connect()

    await client.disconnect()
    await client.disconnect()

    assert not client.connected


# --------------------------------------------------------------------------- #
# Tool listing and calling
# --------------------------------------------------------------------------- #


async def test_lists_and_calls_tools() -> None:
    client = MCPClient(stub_spec())
    await client.connect()
    try:
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == ["echo", "boom"]

        result = await client.call_tool("echo", {"message": "hi"})

        assert result.content == "echo: hi"
        assert not result.is_error
    finally:
        await client.disconnect()


async def test_large_tool_result_survives_mcp_framing() -> None:
    client = MCPClient(stub_spec())
    await client.connect()
    try:
        text = "x" * 9_000_000 + " END"
        result = await client.call_tool("echo", {"message": text})
        assert not result.is_error
        assert result.content == "echo: " + text
    finally:
        await client.disconnect()


async def test_server_reported_error_is_an_error_result() -> None:
    client = MCPClient(stub_spec())
    await client.connect()
    try:
        result = await client.call_tool("boom", {})
        assert result.is_error
    finally:
        await client.disconnect()


async def test_calling_an_unknown_tool_is_an_error_not_a_crash() -> None:
    client = MCPClient(stub_spec())
    await client.connect()
    try:
        result = await client.call_tool("nope", {})

        assert result.is_error
        assert "nope" in content_text(result.content)
    finally:
        await client.disconnect()


# --------------------------------------------------------------------------- #
# Registry integration
# --------------------------------------------------------------------------- #


async def test_tools_are_qualified_with_the_server_name() -> None:
    """Two servers may both expose `search`, so names must be disambiguated."""
    registry = ToolRegistry()
    manager = MCPManager(".", registry=registry)
    state = await manager.connect("stub", stub_spec())

    assert state.status == "connected"
    assert "stub__echo" in registry
    await manager.aclose()


async def test_remote_schema_is_advertised_to_the_model() -> None:
    registry = ToolRegistry()
    manager = MCPManager(".", registry=registry)
    await manager.connect("stub", stub_spec())
    try:
        spec = next(s for s in registry.specs() if s.name == "stub__echo")

        assert spec.parameters["required"] == ["message"]
        assert "MCP:stub" in spec.description
    finally:
        await manager.aclose()


async def test_disconnect_removes_the_tools_it_contributed() -> None:
    registry = ToolRegistry()
    manager = MCPManager(".", registry=registry)
    await manager.connect("stub", stub_spec())

    await manager.disconnect("stub")

    assert "stub__echo" not in registry
    assert not [name for name in registry.names if name.startswith("stub__")]


async def test_invoking_through_the_registry_returns_the_remote_text() -> None:
    registry = ToolRegistry()
    manager = MCPManager(".", registry=registry)
    await manager.connect("stub", stub_spec())
    try:
        result = await registry.invoke("stub__echo", '{"message": "hi"}')

        assert result.content == "echo: hi"
        assert not result.is_error
    finally:
        await manager.aclose()


async def test_calling_a_disconnected_tool_reports_clearly() -> None:
    """A tool that outlives its server must fail loudly, not hang or crash."""
    registry = ToolRegistry()
    manager = MCPManager(".", registry=registry)
    await manager.connect("stub", stub_spec())
    tool = registry.get("stub__echo")
    assert isinstance(tool, MCPTool)

    await manager.disconnect("stub")

    result = await tool.run(message="hi")
    assert result.is_error
    assert "not connected" in content_text(result.content)
    await manager.aclose()


async def test_a_failing_server_does_not_stop_the_others() -> None:
    registry = ToolRegistry()
    manager = MCPManager(".", registry=registry)
    bad = await manager.connect("ghost", ServerSpec(name="ghost", command="/nope/binary"))
    good = await manager.connect("stub", stub_spec())

    assert bad.status == "error"
    assert bad.detail
    assert good.status == "connected"
    assert "stub__echo" in registry
    await manager.aclose()


async def test_connect_all_reports_per_server_status(tmp_path: Path) -> None:
    save_servers(
        tmp_path,
        {
            "ghost": ServerSpec(name="ghost", command="/nope/binary"),
            "stub": stub_spec(),
        },
    )
    manager = MCPManager(tmp_path, registry=ToolRegistry())

    states = {state.spec.name: state for state in await manager.connect_all()}

    assert states["ghost"].status == "error"
    assert states["stub"].status == "connected"
    await manager.aclose()


def test_save_refuses_to_overwrite_malformed_config(tmp_path: Path) -> None:
    target = tmp_path / ".mcp.json"
    target.write_text("{broken")
    manager = MCPManager(tmp_path)
    with pytest.raises(MCPError):
        manager.add("new", stub_spec())
    assert target.read_text() == "{broken"


@pytest.mark.parametrize("env", [{"TOKEN": 1}, {"TOKEN": None}])
def test_server_rejects_nonstring_environment(env: dict) -> None:
    with pytest.raises(MCPError, match="env"):
        ServerSpec.from_json("bad", {"command": "test", "env": env})


async def test_server_initiated_request_is_declined_regardless_of_id(tmp_path: Path) -> None:
    server = tmp_path / "server.py"
    server.write_text('''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request.get('method') == 'server/discover':
        print(json.dumps({'jsonrpc': '2.0', 'id': 'server-request', 'method': 'roots/list'}), flush=True)
        reply = json.loads(sys.stdin.readline())
        if reply.get('id') == 'server-request' and 'error' in reply:
            print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': {'supportedVersions': ['2026-07-28']}}), flush=True)
''')
    client = MCPClient(ServerSpec(name="requester", command=sys.executable, args=[str(server)]), timeout=.3)
    try:
        await client.connect()
        assert client.era == "modern"
    finally:
        await client.disconnect()


async def test_failed_connection_reaps_server(tmp_path: Path) -> None:
    server = tmp_path / "bad-server.py"
    server.write_text('''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if 'id' in request:
        print(json.dumps({'id': request['id'], 'error': {'code': -32601}}), flush=True)
''')
    client = MCPClient(ServerSpec(name="bad", command=sys.executable, args=[str(server)]))
    with pytest.raises(MCPError):
        await client.connect()
    try:
        assert not client.connected
    finally:
        await client.disconnect()


async def test_client_can_reconnect_and_call_tools() -> None:
    client = MCPClient(stub_spec())
    try:
        await client.connect()
        await client.disconnect()
        await client.connect()
        result = await client.call_tool("echo", {"message": "reconnected"})
        assert result.content == "echo: reconnected"
    finally:
        await client.disconnect()


async def test_parallel_calls_receive_their_own_results() -> None:
    import asyncio
    client = MCPClient(stub_spec())
    try:
        await client.connect()
        results = await asyncio.gather(*[client.call_tool("echo", {"message": str(i)}) for i in range(5)])
        assert [r.content for r in results] == [f"echo: {i}" for i in range(5)]
    finally:
        await client.disconnect()


def test_saved_mcp_credentials_are_owner_only(tmp_path: Path) -> None:
    path = save_servers(tmp_path, {"stub": ServerSpec(name="stub", command="test", env={"TOKEN": "secret"})})
    assert path.stat().st_mode & 0o777 == 0o600


async def test_version_negotiation_retries_discovery() -> None:
    class NegotiatingClient(MCPClient):
        requests = []
        async def _request(self, method, params, meta, **kwargs):
            self.requests.append(meta)
            if len(self.requests) == 1:
                return {"error": {"code": -32022, "data": {"supported": ["2025-11-25"]}}}
            return {"result": {"supportedVersions": ["2025-11-25"],
                               "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "negotiated"}},
                               "capabilities": {"tools": {}}}}
    client = NegotiatingClient(stub_spec())
    assert await client._try_modern()
    assert len(client.requests) == 2
    assert client.server_info["name"] == "negotiated"


async def test_large_stderr_does_not_block_server(tmp_path, monkeypatch) -> None:
    import slipagent.mcp as mcp
    monkeypatch.setattr(mcp, "MAX_LINE_BYTES", 1024)
    monkeypatch.setattr(mcp, "PROBE_TIMEOUT", .3)
    server = tmp_path / "verbose-server.py"
    server.write_text('''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request.get('method') == 'server/discover':
        sys.stderr.write('x' * 500000 + '\\n')
        sys.stderr.flush()
        print(json.dumps({'id': request['id'], 'result': {'supportedVersions': ['2026-07-28']}}), flush=True)
''')
    client = MCPClient(ServerSpec(name="verbose", command=sys.executable, args=[str(server)]), timeout=.3)
    try:
        await client.connect()
        assert client.era == "modern"
    finally:
        await client.disconnect()
