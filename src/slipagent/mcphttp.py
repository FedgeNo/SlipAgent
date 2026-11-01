"""Bounded, asynchronous Streamable HTTP transport for the shared MCP client."""

from __future__ import annotations

import asyncio
import codecs
import json
import os
import re
import sys
from typing import Any

import httpx

from .lifecycle import finish_cleanup
from .mcp import CLIENT_INFO, DEFAULT_TIMEOUT, MCPClient, MCPError, MCPTool, MCPToolInfo, ServerSpec
from .tools.base import ToolResult
from .tools.blocking import run_blocking

HTTP_VERSIONS = ('2025-11-25', '2025-06-18')
MAX_HTTP_BYTES = 8 * 1024 * 1024
MAX_HTTP_TOOLS = 1000
MAX_SSE_EVENTS = 1024
VALIDATION_TIMEOUT = 5.0


def _decode_json(data: bytes | str) -> dict[str, Any]:
    try:
        message = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MCPError('Malformed HTTP MCP JSON response') from exc
    if not isinstance(message, dict) or message.get('jsonrpc') != '2.0':
        raise MCPError('HTTP MCP response must be a JSON-RPC 2.0 object')
    return message


def _encode_json(payload: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MCPError('Cannot encode HTTP MCP request') from exc
    if len(encoded) > MAX_HTTP_BYTES:
        raise MCPError('HTTP MCP request exceeds the 8 MiB limit')
    return encoded


class SSEMessages:
    """Incremental UTF-8 and SSE framing, used only by an owned worker."""

    def __init__(self) -> None:
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.buffer = ''
        self.data: list[str] = []
        self.events = 0

    def feed(self, chunk: bytes, final: bool = False) -> list[dict[str, Any]]:
        try:
            self.buffer += self.decoder.decode(chunk, final=final)
        except UnicodeError as exc:
            raise MCPError('HTTP MCP event stream is not UTF-8') from exc
        messages = []
        start = 0
        for match in re.finditer(r'\r\n|\r|\n', self.buffer):
            if not final and match.group() == '\r' and match.end() == len(self.buffer):
                break
            line = self.buffer[start:match.start()]
            start = match.end()
            if not line:
                self.events += 1
                if self.events > MAX_SSE_EVENTS:
                    raise MCPError('HTTP MCP event stream exceeds the 1024-event limit')
                value = '\n'.join(self.data)
                self.data.clear()
                if value.strip():
                    messages.append(_decode_json(value))
            elif line.startswith('data:'):
                value = line[5:]
                self.data.append(value[1:] if value.startswith(' ') else value)
        self.buffer = self.buffer[start:]
        return messages


class MCPHTTPClient(MCPClient):
    """Reuse MCP tools/results while keeping HTTP I/O and parsing off the UI."""

    def __init__(self, spec: ServerSpec, *, timeout: float = DEFAULT_TIMEOUT,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        super().__init__(spec, timeout=timeout)
        self._transport = transport
        self._http: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._closing = False
        self._needs_initialize = False
        self._active: set[asyncio.Task[Any]] = set()
        self._tool_count = 0

    @property
    def connected(self) -> bool:
        return self._http is not None and not self._http.is_closed and not self._closing

    async def connect(self) -> None:
        if self._disconnect_task is not None:
            await finish_cleanup(self._disconnect_task)
            self._disconnect_task = None
        if self.connected:
            return
        self._closing = False
        self._session_id = None
        self.protocol_version = None
        self._http = httpx.AsyncClient(transport=self._transport, follow_redirects=False,
                                      timeout=httpx.Timeout(None, connect=min(10, self.timeout), pool=min(10, self.timeout)),
                                      limits=httpx.Limits(max_connections=4, max_keepalive_connections=2))
        try:
            async with asyncio.timeout(self.timeout):
                await self._handshake()
        except BaseException as exc:
            await self.disconnect()
            if isinstance(exc, TimeoutError):
                raise MCPError('HTTP MCP initialization exceeded its deadline') from None
            raise

    async def _handshake(self) -> None:
        self._session_id = None
        self.protocol_version = None
        response = await self._exchange('initialize', {'protocolVersion': HTTP_VERSIONS[0],
                                        'capabilities': {}, 'clientInfo': CLIENT_INFO})
        result = response.get('result')
        if not isinstance(result, dict) or result.get('protocolVersion') not in HTTP_VERSIONS:
            raise MCPError('HTTP MCP initialization did not negotiate a supported Streamable HTTP version')
        self.protocol_version = result['protocolVersion']
        self.server_info = result.get('serverInfo', {})
        instructions = result.get('instructions', '')
        if (not isinstance(self.server_info, dict) or not isinstance(instructions, str)
                or len(instructions) > 256 * 1024):
            raise MCPError('HTTP MCP server metadata is malformed or its instructions exceed 262144 characters')
        self._server_details(result)
        self.era = 'legacy'
        await self._notify('notifications/initialized')
        self._needs_initialize = False

    def _headers(self) -> dict[str, str]:
        headers = {**(self.spec.headers or {}), 'Accept': 'application/json, text/event-stream',
                   'Content-Type': 'application/json', 'Accept-Encoding': 'identity'}
        if self.protocol_version is not None:
            headers['MCP-Protocol-Version'] = self.protocol_version
        if self._session_id is not None:
            headers['MCP-Session-Id'] = self._session_id
        return headers

    async def _request(self, method: str, params: dict[str, Any] | None,
                       meta: dict[str, Any] | None, *, timeout: float | None = None) -> dict[str, Any]:
        task = asyncio.current_task()
        assert task is not None
        self._active.add(task)
        request_id: int | None = None
        try:
            limit = timeout if timeout is not None else self.timeout
            async with asyncio.timeout(limit):
                async with self._request_lock:
                    if self._needs_initialize:
                        try:
                            async with asyncio.timeout(self.timeout):
                                await self._handshake()
                        except TimeoutError:
                            raise MCPError('HTTP MCP session reinitialization exceeded its deadline; no tool was sent') from None
                    request_id = self._next_id + 1
                    return await self._exchange(method, params)
        except (TimeoutError, asyncio.CancelledError) as exc:
            if request_id is not None and not self._closing:
                await finish_cleanup(asyncio.create_task(self._cancel_http(request_id)))
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise MCPError(f"HTTP MCP {method} exceeded its {limit:g}s deadline. "
                           'The operation may have partial effects; inspect its state before retrying.') from None
        finally:
            self._active.discard(task)

    async def _exchange(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        self._next_id += 1
        payload = {'jsonrpc': '2.0', 'id': self._next_id, 'method': method, 'params': params or {}}
        result = await self._post(payload, self._next_id)
        assert result is not None
        return result

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        async with asyncio.timeout(min(self.timeout, 5)):
            await self._post({'jsonrpc': '2.0', 'method': method, 'params': params or {}}, None)

    async def _cancel_http(self, request_id: int) -> None:
        try:
            async with asyncio.timeout(min(1, self.timeout)):
                await self._notify('notifications/cancelled', {'requestId': request_id})
        except (MCPError, TimeoutError):
            pass

    async def _post(self, payload: dict[str, Any], request_id: int | None) -> dict[str, Any] | None:
        client = self._http
        if not self.connected or client is None or self.spec.url is None:
            raise MCPError('HTTP MCP connection is closed')
        content = await run_blocking(_encode_json, payload)
        try:
            async with client.stream('POST', self.spec.url, headers=self._headers(), content=content) as response:
                if 300 <= response.status_code < 400:
                    raise MCPError('HTTP MCP redirects are refused; configure the final endpoint URL')
                if response.status_code == 404 and self._session_id is not None:
                    self._session_id = None
                    self._needs_initialize = True
                    raise MCPError('HTTP MCP session expired. The next operation will initialize a new session; '
                                   'this operation was not replayed and may have partial effects.')
                if not 200 <= response.status_code < 300:
                    raise MCPError(f'HTTP MCP returned status {response.status_code}; '
                                   'the operation may have partial effects and was not replayed')
                if request_id is None:
                    if response.status_code != 202:
                        raise MCPError('HTTP MCP notifications and responses require status 202')
                    return None
                if payload.get('method') == 'initialize':
                    session_id = response.headers.get('mcp-session-id')
                    if session_id is not None:
                        if not session_id or len(session_id) > 4096 or any(not 33 <= ord(c) <= 126 for c in session_id):
                            raise MCPError('HTTP MCP returned an invalid session ID')
                        self._session_id = session_id
                kind = response.headers.get('content-type', '').split(';', 1)[0].strip().lower()
                if kind not in ('application/json', 'text/event-stream'):
                    raise MCPError('HTTP MCP response must be application/json or text/event-stream')
                if response.headers.get('content-encoding', 'identity').lower() != 'identity':
                    raise MCPError('HTTP MCP requires uncompressed responses to enforce its byte limit')
                data = bytearray()
                parser = SSEMessages() if kind == 'text/event-stream' else None
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_HTTP_BYTES:
                        raise MCPError('HTTP MCP response exceeds the 8 MiB limit')
                    if parser is None:
                        data.extend(chunk)
                    else:
                        messages = await run_blocking(parser.feed, chunk)
                        for message in messages:
                            result = await self._message(message, request_id)
                            if result is not None:
                                return result
                    await asyncio.sleep(0)
                if parser is None:
                    result = await self._message(await run_blocking(_decode_json, bytes(data)), request_id)
                    if result is None:
                        raise MCPError('HTTP MCP JSON body did not contain the requested response')
                    return result
                for message in await run_blocking(parser.feed, b'', True):
                    result = await self._message(message, request_id)
                    if result is not None:
                        return result
                raise MCPError('HTTP MCP stream ended before its response; '
                               'the operation may have partial effects and was not replayed')
        except httpx.HTTPError as exc:
            # Exception strings can contain URLs, query credentials, or headers.
            raise MCPError(f'HTTP MCP transport failed ({type(exc).__name__}); '
                           'the operation may have partial effects and was not replayed') from None

    async def _message(self, message: dict[str, Any], request_id: int) -> dict[str, Any] | None:
        if 'method' in message:
            if 'id' in message:
                async with asyncio.timeout(min(1, self.timeout)):
                    await self._post({'jsonrpc': '2.0', 'id': message['id'], 'error': {
                        'code': -32601, 'message': 'SlipAgent does not implement server-initiated requests'}}, None)
            return None
        if type(message.get('id')) is not int or message['id'] != request_id:
            raise MCPError('HTTP MCP returned a mismatched response ID')
        if ('result' in message) == ('error' in message):
            raise MCPError('HTTP MCP response needs exactly one result or error')
        return message

    async def list_tools(self) -> list[MCPToolInfo]:
        self._tool_count = 0
        try:
            async with asyncio.timeout(self.timeout):
                return await super().list_tools()
        except TimeoutError:
            raise MCPError('HTTP MCP tool listing exceeded its deadline') from None

    async def _tool_page(self, result: dict[str, Any]) -> list[MCPToolInfo]:
        self._tool_count += len(result['tools'])
        if self._tool_count > MAX_HTTP_TOOLS:
            raise MCPError('HTTP MCP catalog exceeds the 1000-tool limit')
        return await run_blocking(self._parse_http_page, result)

    def _parse_http_page(self, result: dict[str, Any]) -> list[MCPToolInfo]:
        for entry in result['tools']:
            if not isinstance(entry, dict):
                continue
            if (not isinstance(entry.get('name'), str) or len(entry['name']) > 256
                    or not isinstance(entry.get('description', ''), str)
                    or len(entry.get('description', '')) > 65536):
                raise MCPError('HTTP MCP tool metadata exceeds its name/description limits or is malformed')
            pending = [(entry.get('inputSchema'), 0)]
            nodes = 0
            while pending:
                item, depth = pending.pop()
                nodes += 1
                if nodes > 4096 or depth > 64:
                    raise MCPError('HTTP MCP tool schema exceeds its 4096-node or 64-level limit')
                if isinstance(item, dict):
                    if any(len(key) > 1024 for key in item):
                        raise MCPError('HTTP MCP schema keys exceed the 1024-character limit')
                    pending.extend((value, depth + 1) for value in item.values())
                elif isinstance(item, list):
                    pending.extend((value, depth + 1) for value in item)
        return self._parse_tool_page(result)

    async def _format_tool_result(self, name: str, response: dict[str, Any]) -> ToolResult:
        try:
            return await run_blocking(self._tool_result, name, response)
        except (TypeError, ValueError, RecursionError) as exc:
            raise MCPError('Malformed HTTP MCP tool content') from exc

    async def _disconnect(self) -> None:
        self._closing = True
        tasks = list(self._active)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        client = self._http
        try:
            if client is not None:
                if self._session_id is not None and self.spec.url is not None:
                    try:
                        async with asyncio.timeout(1):
                            async with client.stream('DELETE', self.spec.url, headers=self._headers()):
                                pass
                    except (httpx.HTTPError, TimeoutError):
                        pass
                async with asyncio.timeout(2):
                    await client.aclose()
        except TimeoutError:
            raise MCPError('HTTP MCP client cleanup exceeded its deadline') from None
        finally:
            self._http = None
            self._session_id = None


class MCPHTTPTool(MCPTool):
    """Server-provided schemas cannot run expensive regexes on the UI thread."""

    async def validate_call(self, arguments: dict[str, Any]) -> str | None:
        from .config import command_environment
        from .tools.shell import kill_and_drain

        process = None
        spawn = None
        settled = False
        tasks: list[asyncio.Task[Any]] = []
        try:
            payload = await run_blocking(_encode_json, {'schema': self.parameters, 'arguments': arguments})
            spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                sys.executable, '-c', 'from slipagent.mcphttp import validate_input_worker; validate_input_worker()',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                env=command_environment(), start_new_session=os.name == 'posix'))
            async with asyncio.timeout(VALIDATION_TIMEOUT):
                process = await asyncio.shield(spawn)
                assert process.stdin is not None and process.stdout is not None

                async def write() -> None:
                    assert process is not None and process.stdin is not None
                    process.stdin.write(payload)
                    await process.stdin.drain()
                    process.stdin.close()

                async def read() -> bytes:
                    assert process is not None and process.stdout is not None
                    try:
                        await process.stdout.readexactly(MAX_HTTP_BYTES + 1)
                    except asyncio.IncompleteReadError as exc:
                        return exc.partial
                    raise MCPError('HTTP MCP argument validation output exceeds its byte limit')

                writer = asyncio.create_task(write())
                reader = asyncio.create_task(read())
                tasks = [writer, reader]
                _, data = await asyncio.gather(writer, reader)
                await process.wait()
                settled = True
                if process.returncode != 0:
                    return f'HTTP MCP argument validator exited with code {process.returncode}; no tool was sent.'
                result = await run_blocking(json.loads, data)
                problem = result['problem']
                if problem is None:
                    arguments.clear()
                    arguments.update(result['arguments'])
                return str(problem) if problem is not None else None
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError, MCPError, TimeoutError, KeyError, TypeError):
            return 'HTTP MCP argument validation failed or exceeded its deadline; no tool was sent.'
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if spawn is not None and not settled:
                async def reap() -> None:
                    try:
                        child = process if process is not None else await spawn
                    except (OSError, ValueError):
                        return
                    await kill_and_drain(child)
                await finish_cleanup(asyncio.create_task(reap()))


def validate_input_worker() -> None:
    """One bounded-input validator process; its parent owns deadline and reaping."""
    from .tools.base import validate_arguments

    if os.name == 'posix':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = 512 * 1024 * 1024
        if soft != resource.RLIM_INFINITY:
            limit = min(limit, soft)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    raw = sys.stdin.buffer.read(MAX_HTTP_BYTES + 1)
    if len(raw) > MAX_HTTP_BYTES:
        raise ValueError('Validation input exceeds its byte limit')
    value = json.loads(raw)
    arguments = value['arguments']
    problem = validate_arguments(value['schema'], arguments, allow_unknown=True)
    result = json.dumps({'problem': problem, 'arguments': arguments}, ensure_ascii=False).encode('utf-8')
    if len(result) > MAX_HTTP_BYTES:
        raise ValueError('Validation output exceeds its byte limit')
    sys.stdout.buffer.write(result)
