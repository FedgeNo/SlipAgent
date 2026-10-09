"""MCP client: expose tools from Model Context Protocol servers.

Speaks JSON-RPC 2.0 over a server's stdio transport, one message per line, as
required by the spec. A connected server's tools are wrapped in `MCPTool` and
registered in the normal `ToolRegistry`, so the model sees them alongside the
built-ins with no special-casing anywhere in the agent loop.

The client implements two negotiation paths:

  modern (2026-07-28+) is stateless. Every request carries its version in
  `params._meta`, and a client probes with `server/discover` to learn what a server
  supports. An `UnsupportedProtocolVersionError` (-32022) means "modern, wrong
  version" and is retried against a mutually supported version; any other
  probe failure triggers the legacy handshake.

  legacy (<= 2025-11-25) requires an `initialize` handshake followed by a
  `notifications/initialized` notification before any other traffic.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import __version__
from .tools.base import Tool, ToolRegistry, ToolResult, validate_schema_shape
from .config import save_private_text
from .lifecycle import finish_cleanup

# Protocol versions this client knows how to speak, newest first.
MODERN_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOL_VERSIONS = (
    MODERN_PROTOCOL_VERSION,
    LEGACY_PROTOCOL_VERSION,
    "2025-06-18",
    "2024-11-05",
)

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

CLIENT_INFO = {"name": "slipagent", "version": __version__}

DEFAULT_CONFIG_NAME = ".mcp.json"
DEFAULT_TIMEOUT = 30.0
PROBE_TIMEOUT = 10.0
# The spec forbids newlines inside a message, but a large tool result can still
# be big, so the line reader gets a generous ceiling rather than the 64 KiB
# asyncio default.
MAX_LINE_BYTES = 256_000_000
STDERR_TAIL_LINES = 20

# JSON-RPC error codes we care about.
ERR_METHOD_NOT_FOUND = -32601
ERR_UNSUPPORTED_VERSION = -32022

# Disambiguates tools from different servers that share a name, per the spec's
# guidance for clients that aggregate multiple servers.
TOOL_SEPARATOR = "__"


class MCPError(Exception):
    """Raised when an MCP server cannot be reached or misbehaves."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ServerSpec:
    """One configured MCP server: a command to launch over stdio."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    description: str = ""

    @classmethod
    def from_json(cls, name: str, raw: Any) -> ServerSpec:
        if not isinstance(raw, dict):
            raise MCPError(f"server '{name}' must be a JSON object")
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            raise MCPError(f"server '{name}' is missing a 'command' string")
        args = raw.get("args") or []
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise MCPError(f"server '{name}' has a non-string 'args' list")
        env = raw.get("env")
        if env is not None and (
            not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
        ):
            raise MCPError(f"server '{name}' has a malformed 'env' object")
        cwd = raw.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise MCPError(f"server '{name}' has a non-string 'cwd'")
        return cls(
            name=name,
            command=command.strip(),
            args=list(args),
            env=env,
            cwd=cwd,
            description=str(raw.get("description") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"command": self.command, "args": self.args}
        if self.env:
            payload["env"] = self.env
        if self.cwd:
            payload["cwd"] = self.cwd
        if self.description:
            payload["description"] = self.description
        return payload

    def command_line(self) -> str:
        parts = [self.command, *self.args]
        return " ".join(parts)


def config_path(workspace: Path | str = ".") -> Path:
    """Where a workspace keeps its MCP server list."""
    return Path(workspace).expanduser() / DEFAULT_CONFIG_NAME


def load_servers(workspace: Path | str = ".") -> dict[str, ServerSpec]:
    """Read `.mcp.json`. A missing file means no servers, not an error."""
    path = config_path(workspace)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError) as exc:
        raise MCPError(f"could not read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MCPError(f"{path} must contain a JSON object")

    servers = raw.get("mcpServers")
    if servers is None:
        servers = {}
    if not isinstance(servers, dict):
        raise MCPError(f"{path}: 'mcpServers' must be a JSON object")
    return {name: ServerSpec.from_json(name, entry) for name, entry in servers.items()}


def save_servers(workspace: Path | str, servers: dict[str, ServerSpec]) -> Path:
    """Write `.mcp.json`, preserving unrelated keys already in the file."""
    path = config_path(workspace)
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8") or "{}")
            if not isinstance(loaded, dict):
                raise MCPError(f"{path} must contain a JSON object")
            existing = loaded
        except (OSError, ValueError) as exc:
            raise MCPError(f"could not read {path}: {exc}") from exc

    existing["mcpServers"] = {name: spec.to_json() for name, spec in servers.items()}
    save_private_text(path, json.dumps(existing, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Tool metadata
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MCPToolInfo:
    """A tool as described by a server's `tools/list`."""

    name: str
    description: str
    input_schema: dict[str, Any]


def flatten_content(items: Iterable[Any]) -> str:
    """Render a tool result's `content` array as plain text for the model.

    Only text is forwarded verbatim. Binary payloads are summarised rather than
    base64-dumped, since a megabyte of base64 would crowd out the conversation
    without telling the model anything it can act on.
    """
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text") or ""))
        elif kind in ("image", "audio"):
            mime = item.get("mimeType") or "application/octet-stream"
            data = item.get("data") or ""
            parts.append(f"[{kind} omitted: {mime}, {len(data)} base64 chars]")
        elif kind == "resource_link":
            uri = item.get("uri") or item.get("name") or "resource"
            parts.append(f"[resource: {uri}]")
        elif kind == "resource":
            resource = item.get("resource")
            if isinstance(resource, dict):
                if "text" in resource:
                    parts.append(str(resource.get("text") or ""))
                else:
                    parts.append(f"[resource: {resource.get('uri', 'unknown')}]")
        else:
            parts.append(f"[{kind or 'unknown'} content omitted]")
    text = "\n".join(part for part in parts if part)
    return text or "(no content returned)"


# --------------------------------------------------------------------------- #
# The stdio client
# --------------------------------------------------------------------------- #


class MCPClient:
    """A JSON-RPC connection to one MCP server subprocess."""

    def __init__(self, spec: ServerSpec, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.spec = spec
        self.timeout = timeout
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.instructions = ""
        self.tools: list[MCPToolInfo] = []
        self.era: str = ""

        self._process: asyncio.subprocess.Process | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._stderr_tail: list[str] = []
        self._pump: asyncio.Task[None] | None = None
        self._stderr_pump: asyncio.Task[None] | None = None
        self._next_id = 0
        self._request_lock = asyncio.Lock()
        self._disconnect_task: asyncio.Task[None] | None = None

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def connected(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def connect(self) -> None:
        """Launch the server and negotiate a protocol version."""
        if self._disconnect_task is not None:
            await finish_cleanup(self._disconnect_task)
            self._disconnect_task = None
        if self.connected:
            return
        self._queue = asyncio.Queue()
        self._stderr_tail.clear()
        self.protocol_version = None
        self.era = ""
        self.instructions = ""
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.spec.command,
                *self.spec.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.spec.cwd,
                env={**os.environ, **(self.spec.env or {})},
                limit=MAX_LINE_BYTES,
                start_new_session=os.name == "posix",
            )
        except (OSError, ValueError) as exc:
            raise MCPError(
                f"could not start '{self.spec.name}' via "
                f"`{self.spec.command_line()}`: {exc}"
            ) from exc

        self._pump = asyncio.create_task(self._read_stdout())
        self._stderr_pump = asyncio.create_task(self._read_stderr())
        try:
            await self._negotiate()
        except BaseException:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Retain cleanup ownership until the server and pumps have settled."""
        if self._disconnect_task is None:
            self._disconnect_task = asyncio.create_task(self._disconnect())
        await finish_cleanup(self._disconnect_task)

    async def _disconnect(self) -> None:
        """Close stdin, then escalate to signals if the server lingers."""
        process = self._process
        try:
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    await self._terminate(process)
        finally:
            self._process = None
            tasks = [task for task in (self._pump, self._stderr_pump) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._pump = self._stderr_pump = None

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """Terminate, then kill if needed; target the process group on POSIX."""
        for force in (False, True):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                elif force:
                    process.kill()
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
                return
            except asyncio.TimeoutError:
                continue
        raise MCPError(f"could not stop '{self.spec.name}'")

    async def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        while True:
            try:
                line = await process.stdout.readline()
            except (asyncio.LimitOverrunError, ValueError):
                self._queue.put_nowait(
                    {
                        "__slipagent_error__": (
                            "server sent a line longer than the "
                            f"{MAX_LINE_BYTES} byte limit"
                        )
                    }
                )
                break
            except (OSError, asyncio.CancelledError):
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                self._queue.put_nowait(
                    {"__slipagent_error__": f"server wrote non-JSON to stdout: {text[:200]}"}
                )
                continue
            if isinstance(message, dict):
                self._queue.put_nowait(message)
        # stdout closed: unblock anything still waiting.
        self._queue.put_nowait({"__slipagent_eof__": True})

    async def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        pending = ""
        truncated = False
        while True:
            try:
                chunk = await process.stderr.read(4096)
            except (OSError, asyncio.CancelledError):
                break
            if not chunk:
                if pending:
                    self._stderr_tail.append(pending + ("…" if truncated else ""))
                    del self._stderr_tail[:-STDERR_TAIL_LINES]
                break
            parts = chunk.decode("utf-8", errors="replace").split("\n")
            for index, part in enumerate(parts):
                remaining = 2000 - len(pending)
                pending += part[:remaining]
                truncated = truncated or len(part) > remaining
                if index < len(parts) - 1:
                    self._stderr_tail.append(pending.rstrip() + ("…" if truncated else ""))
                    del self._stderr_tail[:-STDERR_TAIL_LINES]
                    pending = ""
                    truncated = False

    # -- protocol ----------------------------------------------------------- #

    def _meta(self, version: str) -> dict[str, Any]:
        return {
            META_PROTOCOL_VERSION: version,
            META_CLIENT_INFO: CLIENT_INFO,
            META_CLIENT_CAPABILITIES: {},
        }

    async def _negotiate(self) -> None:
        """Detect the server's era, then settle on a shared version."""
        if await self._try_modern():
            return
        await self._initialize_legacy()

    async def _try_modern(self) -> bool:
        """Probe `server/discover`. False means "assume legacy"."""
        try:
            response = await self._request(
                "server/discover", {}, self._meta(MODERN_PROTOCOL_VERSION),
                timeout=PROBE_TIMEOUT,
            )
        except MCPError:
            return False

        version = MODERN_PROTOCOL_VERSION
        if "error" in response:
            error = response.get("error") or {}
            code = error.get("code") if isinstance(error, dict) else None
            if code == ERR_UNSUPPORTED_VERSION:
                # A modern server that wants a different version. Pick one it
                # advertises rather than falling back to a legacy handshake.
                data = error.get("data") if isinstance(error, dict) else None
                supported = data.get("supported", []) if isinstance(data, dict) else []
                if not isinstance(supported, list):
                    raise MCPError(f"'{self.spec.name}' returned malformed supported versions")
                chosen = next(
                    (v for v in SUPPORTED_PROTOCOL_VERSIONS if v in supported), None
                )
                if chosen is None:
                    raise MCPError(
                        f"'{self.spec.name}' speaks only unsupported MCP "
                        f"version(s): {', '.join(map(str, supported)) or 'unknown'}"
                    )
                response = await self._request("server/discover", {}, self._meta(chosen), timeout=PROBE_TIMEOUT)
                if "error" in response:
                    raise MCPError(self._describe_error("server/discover", response))
                version = chosen
            else:
                return False

        result = response.get("result") or {}
        if not isinstance(result, dict):
            raise MCPError("server/discover returned no result object")
        supported = result.get("supportedVersions")
        if not isinstance(supported, list) or not all(isinstance(item, str) for item in supported):
            raise MCPError("server/discover returned malformed supportedVersions")
        chosen = next((item for item in SUPPORTED_PROTOCOL_VERSIONS if item in supported), None)
        if chosen is None:
            raise MCPError(f"'{self.spec.name}' advertises no supported protocol version")
        self.protocol_version = version if version in supported else chosen
        metadata = result.get("_meta", {})
        self.server_info = metadata.get(META_SERVER_INFO, {}) if isinstance(metadata, dict) else {}
        self._server_details(result)
        self.era = "modern"
        return True

    def _server_details(self, result: dict[str, Any]) -> None:
        capabilities = result.get("capabilities", {})
        instructions = result.get("instructions", "")
        if not isinstance(capabilities, dict) or not isinstance(instructions, str):
            raise MCPError(f"'{self.spec.name}' returned malformed capabilities or instructions")
        self.capabilities = capabilities
        self.instructions = instructions

    async def _initialize_legacy(self) -> None:
        response = await self._request(
            "initialize",
            {
                "protocolVersion": LEGACY_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
            None,
        )
        if "error" in response:
            raise MCPError(self._describe_error("initialize", response))

        result = response.get("result") or {}
        if isinstance(result, dict):
            version = result.get("protocolVersion")
            self.protocol_version = version if isinstance(version, str) else None
            self.server_info = result.get("serverInfo") or {}
            self._server_details(result)

        if self.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            reported = self.protocol_version or "unknown"
            raise MCPError(
                f"'{self.spec.name}' negotiated unsupported MCP version {reported}"
            )
        self.era = "legacy"
        await self._notify("notifications/initialized")

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None,
        meta: dict[str, Any] | None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        # One reader consumes the response queue. Serialize requests so waiting
        # for one ID cannot discard another caller's response as unrelated.
        async with self._request_lock:
            return await self._send_request(method, params, meta, timeout=timeout)

    async def _send_request(
        self, method: str, params: dict[str, Any] | None, meta: dict[str, Any] | None,
        *, timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one request and wait for the response with the matching id."""
        process = self._process
        if process is None or process.stdin is None:
            raise MCPError(f"'{self.spec.name}' is not running")

        self._next_id += 1
        request_id = self._next_id
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None or meta is not None:
            payload["params"] = dict(params or {})
        if meta is not None:
            payload["params"]["_meta"] = meta

        sent = False
        try:
            # The deadline includes backpressure while writing. Otherwise a
            # non-reading server can hang forever before response waiting starts.
            async with asyncio.timeout(timeout if timeout is not None else self.timeout):
                process.stdin.write(
                    (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
                )
                await process.stdin.drain()
                sent = True
                return await self._await_response(request_id, method)
        except asyncio.TimeoutError:
            if sent:
                await self._cancel(request_id, method)
            else:
                await self._abort_connection()
            raise MCPError(
                f"'{self.spec.name}' did not answer {method} within "
                f"{timeout if timeout is not None else self.timeout:g}s. "
                "The operation may have partial effects; inspect its state before retrying."
            ) from None
        except asyncio.CancelledError:
            if sent:
                await self._cancel(request_id, method)
            else:
                await self._abort_connection()
            raise
        except (OSError, RuntimeError) as exc:
            raise MCPError(
                f"lost the connection to '{self.spec.name}': {exc}"
                + self._stderr_hint()
            ) from exc

    async def _abort_connection(self) -> None:
        """Discard a blocked write buffer and reap a connection that cannot drain."""
        process = self._process
        if process is not None:
            if process.stdin is not None:
                process.stdin.transport.abort()
            await self._terminate(process)
        await self.disconnect()

    async def _await_response(self, request_id: int, method: str) -> dict[str, Any]:
        while True:
            message = await self._queue.get()
            if "__slipagent_error__" in message:
                raise MCPError(str(message["__slipagent_error__"]))
            if "__slipagent_eof__" in message:
                raise MCPError(
                    f"'{self.spec.name}' exited before answering {method}"
                    f"{self._stderr_hint()}"
                )

            if "id" not in message:
                continue  # an unrelated notification
            if "method" in message:
                # A server-initiated request. We support no client features, so
                # decline rather than leaving it waiting.
                await self._decline(message)
                continue
            if message.get("id") != request_id:
                continue
            return message

    async def _decline(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        reply = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": ERR_METHOD_NOT_FOUND,
                "message": f"slipagent does not implement {message.get('method')}",
            },
        }
        try:
            process.stdin.write((json.dumps(reply) + "\n").encode("utf-8"))
            await process.stdin.drain()
        except (OSError, RuntimeError):
            pass

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        try:
            async with asyncio.timeout(min(1.0, self.timeout)):
                process.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
                await process.stdin.drain()
        except asyncio.TimeoutError:
            await self._abort_connection()
        except (OSError, RuntimeError):
            pass

    async def _cancel(self, request_id: int, method: str) -> None:
        await self._notify("notifications/cancelled", {"requestId": request_id, "reason": f"timeout on {method}"})

    # -- operations --------------------------------------------------------- #

    async def list_tools(self) -> list[MCPToolInfo]:
        """Fetch every tool the server exposes, following pagination cursors."""
        tools: list[MCPToolInfo] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            response = await self._request("tools/list", params, self._request_meta())
            if "error" in response:
                raise MCPError(self._describe_error("tools/list", response))

            result = response.get("result") or {}
            if not isinstance(result, dict):
                break
            for entry in result.get("tools") or []:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    continue
                schema = entry.get("inputSchema")
                try:
                    if not isinstance(schema, dict) or schema.get("type") != "object":
                        raise ValueError("inputSchema must describe an object")
                    validate_schema_shape(schema)
                except (ValueError, RecursionError) as exc:
                    raise MCPError(f"Invalid schema for MCP tool '{name}': {exc}") from exc
                tools.append(
                    MCPToolInfo(
                        name=name,
                        description=str(entry.get("description") or ""),
                        input_schema=schema,
                    )
                )
            cursor = result.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
        self.tools = tools
        return tools

    def _request_meta(self) -> dict[str, Any] | None:
        """Attach per-request metadata only on the modern negotiation path."""
        if self.era != "modern" or self.protocol_version is None:
            return None
        return self._meta(self.protocol_version)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        response = await self._request(
            "tools/call", {"name": name, "arguments": arguments}, self._request_meta()
        )
        if "error" in response:
            return ToolResult.error(
                f"MCP tool '{name}' failed: {self._describe_error('tools/call', response)}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            return ToolResult.error(f"MCP tool '{name}' returned no result object")
        text = flatten_content(result.get("content") or [])
        structured = result.get("structuredContent")
        if structured is not None:
            if not isinstance(structured, dict):
                return ToolResult.error(f"MCP tool '{name}' returned malformed structuredContent")
            # Servers often repeat the same JSON in a text block for legacy
            # clients. Keep other commentary, but do not duplicate that object.
            remaining = []
            for item in result.get("content") or []:
                if isinstance(item, dict) and item.get("type") == "text":
                    try:
                        if json.loads(item.get("text", "")) == structured:
                            continue
                    except (ValueError, TypeError):
                        pass
                remaining.append(item)
            return ToolResult({"content": flatten_content(remaining) if remaining else "", "structuredContent": structured},
                              bool(result.get("isError")))
        is_error = bool(result.get("isError"))
        return ToolResult(text, is_error)

    # -- diagnostics -------------------------------------------------------- #

    def _describe_error(self, method: str, response: dict[str, Any]) -> str:
        error = response.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "no message")
            code = error.get("code")
            return f"{message} (code={code})" if code is not None else message
        return f"{method} failed: {error or 'no error detail'}"

    def _stderr_hint(self) -> str:
        if not self._stderr_tail:
            return ""
        tail = " | ".join(line for line in self._stderr_tail[-3:] if line)
        return f"; server stderr: {tail}" if tail else ""


# --------------------------------------------------------------------------- #
# Tool adapter
# --------------------------------------------------------------------------- #


class MCPTool(Tool):
    """Exposes one remote MCP tool through the normal `Tool` interface."""

    strict_arguments = False

    def __init__(self, client: MCPClient, info: MCPToolInfo, qualified_name: str) -> None:
        self.client = client
        self.info = info
        self.name = qualified_name
        self.description = (
            f"[MCP:{client.spec.name}] {info.description or info.name}".strip()
        )
        self.parameters = info.input_schema or {"type": "object"}

    async def run(self, **kwargs: Any) -> ToolResult:
        if not self.client.connected:
            return ToolResult.error(
                f"MCP server '{self.client.spec.name}' is not connected. "
                f"Reconnect it with /mcp."
            )
        try:
            return await self.client.call_tool(self.info.name, kwargs)
        except MCPError as exc:
            return ToolResult.error(f"MCP call failed: {exc}")


# --------------------------------------------------------------------------- #
# Session-level manager
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ServerState:
    """Connection status and tools for one configured server."""

    spec: ServerSpec
    client: MCPClient
    status: str = "disconnected"
    detail: str = ""
    qualified: list[str] = field(default_factory=list)

    def line(self) -> str:
        name = f"{self.spec.name} ({self.spec.command_line()})"
        return f"{name}: {self.status}" + (f" — {self.detail}" if self.detail else "")


class MCPManager:
    """Owns the MCP clients for a session and the tools they contribute.

    Tools are added to and removed from the shared `ToolRegistry` as servers
    connect and disconnect, so the model sees the current set on its next step
    without the session being rebuilt.
    """

    def __init__(self, workspace: Path | str = ".", *, registry: ToolRegistry | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.registry = registry
        self.servers: dict[str, ServerState] = {}

    # -- config ------------------------------------------------------------- #

    def configured(self) -> dict[str, ServerSpec]:
        return load_servers(self.workspace)

    def config_error(self) -> str | None:
        """Return a message if `.mcp.json` is unreadable, else None."""
        try:
            load_servers(self.workspace)
        except MCPError as exc:
            return str(exc)
        return None

    def add(self, name: str, spec: ServerSpec) -> None:
        servers = self.configured()
        servers[name] = spec
        save_servers(self.workspace, servers)

    def remove(self, name: str) -> bool:
        servers = self.configured()
        if name not in servers:
            return False
        del servers[name]
        save_servers(self.workspace, servers)
        return True

    # -- connection --------------------------------------------------------- #

    async def connect(self, name: str, spec: ServerSpec) -> ServerState:
        """Connect one server and register its tools."""
        await self.disconnect(name)
        cwd = Path(spec.cwd).expanduser() if spec.cwd is not None else self.workspace
        if not cwd.is_absolute():
            cwd = self.workspace / cwd
        client = MCPClient(replace(spec, cwd=str(cwd)))
        state = ServerState(spec=spec, client=client)
        self.servers[name] = state

        try:
            await client.connect()
            tools = await client.list_tools()
            state.qualified = self._register(name, client, tools, state)
        except BaseException as exc:
            await self.disconnect(name)
            state.status = "error"
            state.detail = str(exc)
            if isinstance(exc, MCPError):
                return state
            raise

        state.status = "connected"
        version = client.protocol_version or "unknown"
        collision = state.detail
        state.detail = f"MCP {version} ({client.era}), {len(state.qualified)} tool(s), cwd: {client.spec.cwd}"
        if collision:
            state.detail += f"; {collision}"
        return state

    def _register(
        self,
        name: str,
        client: MCPClient,
        tools: list[MCPToolInfo],
        state: ServerState,
    ) -> list[str]:
        if self.registry is None:
            return [f"{name}{TOOL_SEPARATOR}{info.name}" for info in tools]
        registered = state.qualified
        for info in tools:
            qualified = f"{name}{TOOL_SEPARATOR}{info.name}"
            if qualified in self.registry:
                state.detail = f"skipped '{qualified}': name collides with another tool"
                continue
            self.registry.register(MCPTool(client, info, qualified))
            registered.append(qualified)
        return registered

    async def connect_all(self) -> list[ServerState]:
        """Connect every configured server, reporting failures per server."""
        states: list[ServerState] = []
        for name, spec in sorted(self.configured().items()):
            states.append(await self.connect(name, spec))
        return states

    async def disconnect(self, name: str) -> None:
        """Close a server and remove any tools it contributed."""
        state = self.servers.get(name)
        if state is None:
            return
        for qualified in state.qualified:
            if self.registry is not None:
                self.registry.unregister(qualified)
        state.qualified = []
        state.status = "disconnected"
        await state.client.disconnect()

    async def aclose(self) -> None:
        for name in list(self.servers):
            await self.disconnect(name)
