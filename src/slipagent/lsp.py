"""Optional stdio LSP navigation using explicitly configured server commands."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .lifecycle import Lifetime, finish_cleanup
from .tools.base import Tool, ToolResult
from .tools.shell import kill_and_drain
from .workspace import Workspace, WorkspaceError

MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_DOCUMENT_BYTES = 2_000_000
OPERATIONS = {"definition": "definitionProvider", "references": "referencesProvider",
              "implementation": "implementationProvider", "hover": "hoverProvider"}


def validate_servers(servers: Any) -> None:
    if not isinstance(servers, dict):
        raise ValueError("language_servers must be an object keyed by server name")
    for name, value in servers.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, dict):
            raise ValueError("Each language server needs a name and configuration object")
        if value.keys() - {"command", "extensions", "language_id", "initialization_options", "settings"}:
            raise ValueError(f"Unknown language server setting in {name}")
        command = value.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part.strip() or "\0" in part for part in command):
            raise ValueError(f"language_servers.{name}.command must be a nonempty argv array")
        extensions = value.get("extensions")
        if not isinstance(extensions, list) or not extensions or any(not isinstance(ext, str) or not ext.startswith(".") or "/" in ext or "\\" in ext for ext in extensions):
            raise ValueError(f"language_servers.{name}.extensions must list dot-prefixed file extensions")
        if not isinstance(value.get("language_id"), str) or not value["language_id"].strip():
            raise ValueError(f"language_servers.{name}.language_id is required")
        if not isinstance(value.get("settings", {}), dict):
            raise ValueError(f"language_servers.{name}.settings must be an object")


class LSPError(Exception):
    """A configured server cannot fulfill the requested navigation operation."""


class LanguageServer:
    """Own one subprocess and framed RPC stream; never execute server edits."""
    def __init__(self, workspace: Workspace, config: dict[str, Any]) -> None:
        self.workspace, self.config = workspace, config
        self.process: asyncio.subprocess.Process | None = None
        self.pending: dict[int, asyncio.Future[Any]] = {}
        self.sequence = 0
        self.capabilities: dict[str, Any] = {}
        self.initialized = False
        self.encoding = "utf-16"
        self.stderr = ""
        self.write_lock = asyncio.Lock()
        self.lifetime = Lifetime("language server")
        self.closing: asyncio.Task[None] | None = None

    async def start(self) -> None:
        command = list(self.config["command"])
        command[0] = os.path.expanduser(command[0])
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            *command, cwd=self.workspace.root, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix", limit=16384,
        ))
        try:
            self.process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            async def cleanup() -> None:
                await kill_and_drain(await spawn)
            await finish_cleanup(asyncio.create_task(cleanup()))
            raise
        process = self.process
        self.lifetime.defer(lambda: kill_and_drain(process))
        self.lifetime.spawn(self._read())
        self.lifetime.spawn(self._stderr())
        root = self.workspace.root.as_uri()
        result = await self.request("initialize", {
            "processId": os.getpid(), "clientInfo": {"name": "SlipAgent"},
            "rootUri": root, "workspaceFolders": [{"uri": root, "name": self.workspace.root.name}],
            "initializationOptions": self.config.get("initialization_options"),
            "capabilities": {
                "general": {"positionEncodings": ["utf-16"]},
                "workspace": {"configuration": True, "workspaceFolders": True, "applyEdit": False},
                "textDocument": {"definition": {"linkSupport": True}, "implementation": {"linkSupport": True},
                                 "hover": {"contentFormat": ["plaintext", "markdown"]}},
            },
        })
        if not isinstance(result, dict) or not isinstance(result.get("capabilities"), dict):
            raise LSPError("Server returned invalid initialize capabilities")
        self.capabilities = result["capabilities"]
        self.encoding = self.capabilities.get("positionEncoding", "utf-16")
        if self.encoding != "utf-16":
            raise LSPError(f"Server selected unsupported position encoding {self.encoding}")
        await self.notify("initialized", {})
        self.initialized = True
        await self.notify("workspace/didChangeConfiguration", {"settings": self.config.get("settings", {})})

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None or self.process.returncode is not None:
            raise LSPError("Language server is not running")
        raw = json.dumps({"jsonrpc": "2.0", **payload}, ensure_ascii=False).encode("utf-8")
        async with asyncio.timeout(5):
            async with self.write_lock:
                self.process.stdin.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw)
                await self.process.stdin.drain()

    async def notify(self, method: str, params: Any) -> None:
        await self._send({"method": method, "params": params})

    async def request(self, method: str, params: Any, timeout: float = 30) -> Any:
        self.sequence += 1
        key = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[key] = future
        try:
            async with asyncio.timeout(timeout):
                await self._send({"id": key, "method": method, "params": params})
                return await future
        except (TimeoutError, asyncio.CancelledError):
            try:
                await asyncio.wait_for(self.notify("$/cancelRequest", {"id": key}), 1)
            except (OSError, LSPError, TimeoutError):
                pass
            raise
        finally:
            self.pending.pop(key, None)
            if not future.done():
                future.cancel()

    async def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                header = await self.process.stdout.readuntil(b"\r\n\r\n")
                lengths = [line.split(b":", 1)[1].strip() for line in header.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")]
                if len(lengths) != 1 or not lengths[0].isdigit() or not 0 < int(lengths[0]) <= MAX_MESSAGE_BYTES:
                    raise LSPError("Invalid or oversized LSP Content-Length")
                raw = await self.process.stdout.readexactly(int(lengths[0]))
                message = json.loads(raw)
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    raise LSPError("Invalid LSP JSON-RPC message")
                if "method" in message:
                    if "id" in message:
                        await self._answer(message)
                    continue
                key = message.get("id")
                future = self.pending.get(key) if type(key) is int else None
                if future is not None and not future.done():
                    if "error" in message:
                        future.set_exception(LSPError(str(message["error"])))
                    elif "result" in message:
                        future.set_result(message["result"])
                    else:
                        future.set_exception(LSPError("LSP response lacks result or error"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(LSPError(f"Language server connection failed: {exc}; stderr: {self.stderr}"))

    async def _answer(self, message: dict[str, Any]) -> None:
        method = message["method"]
        result: Any = None
        if method == "workspace/configuration":
            result = []
            for item in (message.get("params") or {}).get("items", []):
                value = self.config.get("settings", {})
                for part in item.get("section", "").split(".") if item.get("section") else []:
                    value = value.get(part) if isinstance(value, dict) else None
                result.append(value)
        elif method == "workspace/workspaceFolders":
            result = [{"uri": self.workspace.root.as_uri(), "name": self.workspace.root.name}]
        elif method == "workspace/applyEdit":
            result = {"applied": False, "failureReason": "SlipAgent LSP tools provide navigation only"}
        elif method != "window/showMessageRequest":
            await self._send({"id": message["id"], "error": {"code": -32601, "message": "Unsupported client method"}})
            return
        await self._send({"id": message["id"], "result": result})

    async def _stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while chunk := await self.process.stderr.read(4096):
            self.stderr = (self.stderr + chunk.decode("utf-8", errors="replace"))[-8000:]

    async def _close(self) -> None:
        try:
            if self.initialized and self.process is not None and self.process.returncode is None:
                try:
                    await self.request("shutdown", None, timeout=1)
                    await asyncio.wait_for(self.notify("exit", None), 1)
                    await asyncio.wait_for(self.process.wait(), 1)
                except (OSError, LSPError, TimeoutError):
                    pass
        finally:
            await self.lifetime.aclose()

    async def aclose(self) -> None:
        if self.closing is None:
            self.closing = asyncio.create_task(self._close())
        await finish_cleanup(self.closing)


def _document(path: Path) -> str:
    if not path.is_file():
        raise LSPError("Navigation requires an existing regular file")
    with path.open("rb") as source:
        raw = source.read(MAX_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_DOCUMENT_BYTES or b"\0" in raw:
        raise LSPError("Navigation requires a UTF-8 text file no larger than 2 MB")
    return raw.decode("utf-8")


class LanguageServers:
    """Serialize document queries and keep configured clients across reloads."""
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.clients: dict[str, LanguageServer] = {}
        self.lock = asyncio.Lock()
        self.files: dict[str, tuple[int, int]] | None = None
        self.closing: asyncio.Task[None] | None = None

    async def _reconcile(self, settings: dict[str, Any]) -> None:
        for name, client in tuple(self.clients.items()):
            if settings.get(name) != client.config:
                await client.aclose()
                del self.clients[name]

    async def refresh(self) -> None:
        from .environment import load_project_options
        async with self.lock:
            if self.closing is not None:
                raise LSPError("Language server service is closing or closed")
            await self._reconcile(load_project_options(self.workspace).get("language_servers", {}))

    def _file_stamps(self) -> dict[str, tuple[int, int]]:
        files = {}
        for path in self.workspace.iter_files():
            try:
                target = self.workspace.resolve(path)
                stat = target.stat()
                files[target.as_uri()] = stat.st_mtime_ns, stat.st_size
            except FileNotFoundError:
                continue
        return files

    async def _sync_files(self) -> None:
        current = await asyncio.to_thread(self._file_stamps)
        changes = []
        if self.files is not None:
            for uri in sorted(self.files.keys() | current.keys()):
                if current.get(uri) != self.files.get(uri):
                    changes.append({"uri": uri, "type": 1 if uri not in self.files else 3 if uri not in current else 2})
        for name, client in tuple(self.clients.items()):
            if client.initialized:
                try:
                    for start in range(0, len(changes), 1000):
                        await client.notify("workspace/didChangeWatchedFiles", {"changes": changes[start:start + 1000]})
                except (OSError, LSPError, TimeoutError):
                    await client.aclose()
                    del self.clients[name]
        self.files = current

    async def query(self, operation: str, path: str, line: int, column: int, server: str | None, offset: int) -> ToolResult:
        from .environment import load_project_options
        async with self.lock:
            if self.closing is not None:
                raise LSPError("Language server service is closing or closed")
            settings = load_project_options(self.workspace).get("language_servers", {})
            await self._reconcile(settings)
            target = self.workspace.resolve(path)
            matches = [name for name, config in settings.items() if target.suffix.casefold() in
                       [ext.casefold() for ext in config["extensions"]] and (server is None or server == name)]
            if len(matches) != 1:
                return ToolResult.error("Select one configured language server for this file using server. "
                                        f"Matches: {matches}. Configure language_servers in .slipagent/project.json; no servers are installed automatically.")
            name = matches[0]
            config = settings[name]
            text = _document(target)
            lines = text.split("\n")
            if not 1 <= line <= len(lines) or not 1 <= column <= len(lines[line - 1].rstrip("\r")) + 1:
                return ToolResult.error("line and column must be 1-based positions within the file; columns count Unicode characters")
            await self._sync_files()
            client = self.clients.get(name)
            if client is not None and (client.config != config or client.process is None or client.process.returncode is not None):
                await client.aclose()
                del self.clients[name]
                client = None
            if client is None:
                client = LanguageServer(self.workspace, config)
                self.clients[name] = client
            try:
                if not client.initialized:
                    await client.start()
                supported = client.capabilities.get(OPERATIONS[operation])
                if supported is None or supported is False:
                    return ToolResult.error(f"Language server {name} does not advertise {operation} support")
                uri = target.as_uri()
                # Close after each query so the next open always supplies current
                # disk contents, including edits by a shell or another contributor.
                await client.notify("textDocument/didOpen", {"textDocument": {
                    "uri": uri, "languageId": config["language_id"], "version": 1, "text": text,
                }})
                params: dict[str, Any] = {"textDocument": {"uri": uri}, "position": {
                    "line": line - 1, "character": len(lines[line - 1][:column - 1].encode("utf-16-le")) // 2,
                }}
                if operation == "references":
                    params["context"] = {"includeDeclaration": True}
                response = await client.request("textDocument/" + operation, params)
                await client.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
                return self._render(operation, response, offset)
            except BaseException:
                await client.aclose()
                self.clients.pop(name, None)
                raise

    def _render(self, operation: str, response: Any, offset: int) -> ToolResult:
        if operation == "hover":
            if response is None:
                return ToolResult.ok("No hover information")
            if not isinstance(response, dict) or "contents" not in response:
                raise LSPError("Malformed hover response")
            contents = response["contents"]
            parts = contents if isinstance(contents, list) else [contents]
            if any(not isinstance(part, str) and (not isinstance(part, dict) or not isinstance(part.get("value"), str)) for part in parts):
                raise LSPError("Malformed hover contents")
            text = "\n\n".join(part if isinstance(part, str) else part["value"] for part in parts)
            return ToolResult.ok(text[:16000] + ("\n… [hover truncated at 16000 characters]" if len(text) > 16000 else ""))
        locations = response if isinstance(response, list) else [response] if response is not None else []
        result, excluded = [], 0
        for location in locations:
            if not isinstance(location, dict):
                raise LSPError("Malformed navigation location")
            uri = location.get("uri", location.get("targetUri"))
            span = location.get("range", location.get("targetSelectionRange"))
            if not isinstance(uri, str) or not isinstance(span, dict):
                raise LSPError("Navigation location lacks URI or range")
            parsed = urlsplit(uri)
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
                excluded += 1
                continue
            decoded = unquote(parsed.path)
            if os.name == "nt" and len(decoded) > 2 and decoded[0] == "/" and decoded[2] == ":":
                decoded = decoded[1:]
            try:
                target = self.workspace.resolve(decoded)
            except WorkspaceError:
                excluded += 1
                continue
            start = span.get("start")
            if not isinstance(start, dict) or any(type(start.get(key)) is not int or start[key] < 0 for key in ("line", "character")):
                raise LSPError("Navigation location has an invalid position")
            result.append({"path": self.workspace.relative(target), "line": start["line"] + 1,
                           "column_utf16": start["character"] + 1})
        return ToolResult.ok(json.dumps({"locations": result[offset:offset + 100], "total": len(result),
                                        "excluded_outside_workspace": excluded,
                                        "next_offset": offset + 100 if offset + 100 < len(result) else None,
                                        "position_units": "1-based lines and UTF-16 columns"}, ensure_ascii=False))

    async def aclose(self) -> None:
        async def close() -> None:
            async with self.lock:
                clients, self.clients = list(self.clients.values()), {}
                await asyncio.gather(*(client.aclose() for client in clients))
        if self.closing is None:
            self.closing = asyncio.create_task(close())
        await finish_cleanup(self.closing)


class NavigateCodeTool(Tool):
    name = "navigate_code"
    instruction_path = "path"
    description = (
        "Use an explicitly configured language server for definition, references, implementation, or hover. "
        "Input line/column are 1-based Unicode character positions. Results label UTF-16 columns explicitly. "
        "Only workspace locations are returned; use offset to page past 100 results. "
        "Requires language_servers in .slipagent/project.json and an already installed stdio server."
    )
    parameters = {"type": "object", "properties": {
        "operation": {"type": "string", "enum": list(OPERATIONS)}, "path": {"type": "string"},
        "line": {"type": "integer", "minimum": 1}, "column": {"type": "integer", "minimum": 1},
        "server": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
    }, "required": ["operation", "path", "line", "column"]}

    def __init__(self, servers: LanguageServers) -> None:
        self.servers = servers

    async def run(self, operation: str, path: str, line: int, column: int, server: str | None = None, offset: int = 0) -> ToolResult:
        return await self.servers.query(operation, path, line, column, server, offset)
