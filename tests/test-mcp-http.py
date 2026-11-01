"""HTTP MCP failures remain bounded and leave the event loop responsive."""

import asyncio
import json
import threading

import httpx
import pytest

from slipagent import mcphttp
from slipagent.commands import McpCommand
from slipagent.mcp import MCPError, MCPManager, MCPTool, ServerSpec, load_servers, save_servers
from slipagent.mcphttp import MCPHTTPClient, MCPHTTPTool
from slipagent.tools.base import ToolRegistry


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks=(), *, stall=False):
        self.chunks = chunks
        self.stall = stall
        self.closed = False
        self.started = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        for chunk in self.chunks:
            yield chunk
            await asyncio.sleep(0)
        if self.stall:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class Router:
    def __init__(self):
        self.calls = []
        self.initialized = 0
        self.tool_reply = None

    async def handle(self, request):
        if request.method == 'DELETE':
            self.calls.append((request, None))
            return httpx.Response(204)
        body = json.loads(request.content)
        self.calls.append((request, body))
        method = body.get('method')
        if method == 'initialize':
            self.initialized += 1
            return httpx.Response(200, headers={'mcp-session-id': f'session-{self.initialized}'}, json={
                'jsonrpc': '2.0', 'id': body['id'], 'result': {'protocolVersion': '2025-11-25',
                'serverInfo': {'name': 'fixture', 'version': '1'}, 'capabilities': {}, 'instructions': 'Use echo.'}})
        if method == 'tools/list':
            return httpx.Response(200, json={'jsonrpc': '2.0', 'id': body['id'], 'result': {'tools': [
                {'name': 'echo', 'description': 'Echo text.', 'inputSchema': {'type': 'object', 'properties': {}}}]}})
        if method == 'tools/call':
            if self.tool_reply is not None:
                return await self.tool_reply(request, body)
            return httpx.Response(200, json={'jsonrpc': '2.0', 'id': body['id'], 'result': {
                'content': [{'type': 'text', 'text': 'hello'}]}})
        return httpx.Response(202)


def client_for(router, *, timeout=1):
    return MCPHTTPClient(ServerSpec.from_json('remote', {'url': 'https://fixture.test/mcp',
                         'headers': {'Authorization': 'Bearer dummy-private-token'}, 'timeout': timeout}),
                         timeout=timeout, transport=httpx.MockTransport(router.handle))


def test_http_configuration_round_trip_and_private_display(tmp_path):
    spec = ServerSpec.from_json('remote', {'url': 'https://fixture.test/mcp?key=private-query',
                               'headers': {'Authorization': 'Bearer dummy-private-token'}})
    save_servers(tmp_path, {'remote': spec})
    assert load_servers(tmp_path)['remote'].to_json() == spec.to_json()
    assert spec.command_line() == 'https://fixture.test/mcp'
    assert 'dummy-private-token' not in repr(spec)
    action, parsed, _ = McpCommand().split('add remote https://fixture.test/mcp')
    assert action == 'add' and parsed.url == 'https://fixture.test/mcp'


@pytest.mark.parametrize('raw', [
    {'url': 'file:///tmp/mcp'}, {'url': 'https://user:password@fixture.test/mcp'},
    {'url': 'https://fixture.test/#fragment'}, {'url': 'https://fixture.test', 'command': 'python'},
    {'url': 'https://fixture.test', 'headers': {'MCP-Session-Id': 'override'}},
    {'url': 'https://fixture.test', 'headers': {'Authorization': 'bad\r\nheader'}},
    {'command': 'python', 'headers': {'Authorization': 'secret'}},
])
def test_invalid_http_configuration_is_rejected(raw):
    with pytest.raises(MCPError):
        ServerSpec.from_json('remote', raw)


async def test_json_http_session_headers_registration_and_cleanup(workspace, monkeypatch):
    router = Router()
    registry = ToolRegistry()
    original = MCPHTTPClient
    monkeypatch.setattr(mcphttp, 'MCPHTTPClient', lambda spec: original(spec, transport=httpx.MockTransport(router.handle)))
    manager = MCPManager(workspace.root, registry=registry)
    try:
        state = await manager.connect('remote', ServerSpec.from_json('remote', {'url': 'https://fixture.test/mcp',
                                      'headers': {'Authorization': 'Bearer dummy-private-token'}}))
        assert state.status == 'connected'
        assert state.client.instructions == 'Use echo.'
        assert isinstance(registry.get('remote__echo'), MCPTool)
        result = await registry.invoke('remote__echo', {})
        assert not result.is_error and result.content == 'hello'
        for request, body in router.calls[1:]:
            assert request.headers['mcp-session-id'] == 'session-1'
            assert request.headers['mcp-protocol-version'] == '2025-11-25'
            assert request.headers['authorization'] == 'Bearer dummy-private-token'
        await manager.disconnect('remote')
        assert registry.get('remote__echo') is None
        assert router.calls[-1][0].method == 'DELETE'
        assert not state.client.connected
    finally:
        await manager.aclose()
        await registry.aclose()


async def test_sse_returns_matching_response_without_waiting_for_eof():
    router = Router()
    stream = None

    async def response(request, body):
        nonlocal stream
        messages = [
            {'jsonrpc': '2.0', 'method': 'notifications/progress', 'params': {}},
            {'jsonrpc': '2.0', 'id': 'server-question', 'method': 'sampling/createMessage', 'params': {}},
            {'jsonrpc': '2.0', 'id': body['id'], 'result': {'content': [{'type': 'text', 'text': 'héllo'}]}},
        ]
        data = b'id: priming\r\ndata:\r\n\r\n' + b''.join(
            ('data: ' + json.dumps(message, ensure_ascii=False) + '\r\n\r\n').encode() for message in messages)
        stream = Stream([data[i:i + 3] for i in range(0, len(data), 3)], stall=True)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    router.tool_reply = response
    client = client_for(router)
    try:
        await client.connect()
        assert (await asyncio.wait_for(client.call_tool('echo', {}), 5)).content == 'héllo'
        assert stream.closed
        declines = [body for _, body in router.calls if body and body.get('id') == 'server-question']
        assert declines[0]['error']['code'] == -32601
    finally:
        await client.disconnect()


@pytest.mark.parametrize('phase', ['headers', 'body', 'heartbeat'])
async def test_stalled_requests_time_out_without_replay_or_blocking_ui(phase):
    router = Router()
    entered = asyncio.Event()
    stream = None

    async def stalled(request, body):
        nonlocal stream
        entered.set()
        if phase == 'headers':
            await asyncio.Event().wait()
        stream = Stream([b': heartbeat\n\n'] if phase == 'heartbeat' else [], stall=True)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    router.tool_reply = stalled
    client = client_for(router, timeout=.2)
    task = None
    try:
        await client.connect()
        task = asyncio.create_task(client.call_tool('echo', {'mutation': 'once'}))
        await asyncio.wait_for(entered.wait(), 5)
        ui_tick = asyncio.Event()
        asyncio.get_running_loop().call_soon(ui_tick.set)
        await asyncio.wait_for(ui_tick.wait(), 1)
        with pytest.raises(MCPError, match='deadline'):
            await asyncio.wait_for(task, 5)
        assert len([body for _, body in router.calls if body and body.get('method') == 'tools/call']) == 1
        if stream is not None:
            assert stream.closed
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await client.disconnect()


async def test_disconnect_cancels_inflight_request_and_releases_response():
    router = Router()
    stream = Stream(stall=True)

    async def response(request, body):
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    router.tool_reply = response
    client = client_for(router)
    await client.connect()
    task = asyncio.create_task(client.call_tool('echo', {}))
    await asyncio.wait_for(stream.started.wait(), 5)
    await asyncio.wait_for(client.disconnect(), 5)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed and not client.connected and not client._active


@pytest.mark.parametrize('kind', ['invalid_json', 'wrong_id', 'oversized', 'compressed', 'broken_stream'])
async def test_malformed_and_lost_responses_become_tool_errors(kind, monkeypatch):
    router = Router()

    async def response(request, body):
        if kind == 'wrong_id':
            return httpx.Response(200, json={'jsonrpc': '2.0', 'id': 999999, 'result': {}})
        if kind == 'invalid_json':
            return httpx.Response(200, headers={'content-type': 'application/json'}, content=b'not JSON')
        if kind == 'compressed':
            return httpx.Response(200, headers={'content-type': 'application/json', 'content-encoding': 'gzip'}, stream=Stream(stall=True))
        if kind == 'broken_stream':
            raise httpx.ReadError('secret URL and authorization must not be exposed')
        return httpx.Response(200, headers={'content-type': 'application/json'}, content=b'x' * 1025)

    client = client_for(router)
    try:
        await client.connect()
        tools = await client.list_tools()
        router.tool_reply = response
        monkeypatch.setattr(mcphttp, 'MAX_HTTP_BYTES', 1024)
        result = await MCPTool(client, tools[0], 'remote__echo').run()
        assert result.is_error
        assert 'secret URL' not in result.content
        assert len([body for _, body in router.calls if body and body.get('method') == 'tools/call']) == 1
    finally:
        await client.disconnect()


async def test_expired_session_reinitializes_before_next_explicit_call_without_replaying():
    router = Router()

    async def response(request, body):
        if body['params']['arguments'].get('first'):
            return httpx.Response(404)
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': body['id'], 'result': {'content': []}})

    router.tool_reply = response
    client = client_for(router)
    try:
        await client.connect()
        with pytest.raises(MCPError, match='session expired'):
            await client.call_tool('echo', {'first': True})
        await client.call_tool('echo', {'second': True})
        assert router.initialized == 2
        arguments = [body['params']['arguments'] for _, body in router.calls if body and body.get('method') == 'tools/call']
        assert arguments == [{'first': True}, {'second': True}]
    finally:
        await client.disconnect()


async def test_redirects_never_forward_credentials():
    requests = []

    def redirect(request):
        requests.append(request)
        return httpx.Response(307, headers={'location': 'https://other.test/mcp'})

    client = MCPHTTPClient(ServerSpec.from_json('remote', {'url': 'https://fixture.test/mcp',
                           'headers': {'Authorization': 'Bearer dummy-secret'}}),
                           transport=httpx.MockTransport(redirect))
    with pytest.raises(MCPError, match='redirect'):
        await client.connect()
    assert len(requests) == 1 and not client.connected


async def test_json_work_runs_in_owned_worker_while_ui_can_progress(monkeypatch):
    router = Router()
    client = client_for(router, timeout=5)
    await client.connect()
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    original = mcphttp._decode_json
    threads = []

    def decode(data):
        threads.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original(data)

    monkeypatch.setattr(mcphttp, '_decode_json', decode)
    task = asyncio.create_task(client.call_tool('echo', {}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert threading.get_ident() not in threads
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
        await client.disconnect()


async def test_pathological_remote_schema_cannot_block_ui_or_send_tool(monkeypatch):
    from slipagent.mcp import MCPToolInfo
    from slipagent.types import ToolCall
    from slipagent.agent import Agent
    from test_agent import StubClient, completion

    router = Router()
    client = client_for(router)
    await client.connect()
    schema = {'type': 'object', 'properties': {'text': {'type': 'string', 'pattern': '(a+)+$'}}}
    tool = MCPHTTPTool(client, MCPToolInfo('echo', '', schema), 'remote__echo')
    registry = ToolRegistry([tool])
    agent = Agent(StubClient([completion('Check', [ToolCall('call', tool.name, {'text': 'a' * 40 + '!'})]),
                              completion('Validation rejected')]), registry, 'test')
    monkeypatch.setattr(mcphttp, 'VALIDATION_TIMEOUT', 1.0)
    try:
        task = asyncio.create_task(agent.run('Check this tool'))
        ticks = 0
        while not task.done():
            ticks += 1
            await asyncio.sleep(.02)
        assert await task == 'Validation rejected'
        assert ticks > 1
        results = [message.content for message in agent.messages if message.role == 'tool']
        assert 'validation' in str(results).lower() and 'deadline' in str(results)
        assert not any(body and body.get('method') == 'tools/call' for _, body in router.calls)
    finally:
        await registry.aclose()
        await client.disconnect()


async def test_isolated_argument_validation_preserves_normalization_and_schema_errors():
    from slipagent.mcp import MCPToolInfo

    router = Router()
    client = client_for(router)
    schema = {'type': 'object', 'properties': {'count': {'type': 'integer', 'minimum': 1}}, 'required': ['count']}
    tool = MCPHTTPTool(client, MCPToolInfo('echo', '', schema), 'remote__echo')
    arguments = {'count': 2.0}
    assert await tool.validate_call(arguments) is None
    assert type(arguments['count']) is int
    assert await tool.validate_call({'count': 0}) is not None


@pytest.mark.parametrize('method', ['initialize', 'notifications/initialized'])
async def test_stalled_initialization_is_a_contained_connection_error(method, monkeypatch, workspace):
    router = Router()
    original_handle = router.handle

    async def handle(request):
        if request.method == 'POST' and json.loads(request.content).get('method') == method:
            await asyncio.Event().wait()
        return await original_handle(request)

    original_client = MCPHTTPClient
    monkeypatch.setattr(mcphttp, 'MCPHTTPClient', lambda spec: original_client(
        spec, timeout=.2, transport=httpx.MockTransport(handle)))
    registry = ToolRegistry()
    manager = MCPManager(workspace.root, registry=registry)
    try:
        state = await asyncio.wait_for(manager.connect('remote', ServerSpec(url='https://fixture.test/mcp', name='remote')), 5)
        assert state.status == 'error' and 'deadline' in state.detail
        assert not state.client.connected and not registry.names
    finally:
        await manager.aclose()
        await registry.aclose()


async def test_cancelled_validation_reaps_spawned_process(monkeypatch):
    from slipagent.mcp import MCPToolInfo

    router = Router()
    client = client_for(router)
    schema = {'type': 'object', 'properties': {'text': {'type': 'string', 'pattern': '(a+)+$'}}}
    tool = MCPHTTPTool(client, MCPToolInfo('echo', '', schema), 'remote__echo')
    entered = asyncio.Event()
    children = []
    original_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        children.append(process)
        entered.set()
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    task = asyncio.create_task(tool.validate_call({'text': 'a' * 40 + '!'}))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert children and all(child.returncode is not None for child in children)


async def test_failed_validator_spawn_is_a_validation_error(monkeypatch):
    from slipagent.mcp import MCPToolInfo

    async def fail(*args, **kwargs):
        raise OSError('fixture spawn failure')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fail)
    client = client_for(Router())
    tool = MCPHTTPTool(client, MCPToolInfo('echo', '', {'type': 'object'}), 'remote__echo')
    assert 'no tool was sent' in await tool.validate_call({})


@pytest.mark.parametrize('field,value', [('instructions', 7), ('serverInfo', 'invalid'),
                                        ('instructions', 'x' * 262145)])
async def test_invalid_initialization_metadata_is_contained(field, value):
    router = Router()
    original = router.handle

    async def handle(request):
        response = await original(request)
        if request.method == 'POST' and json.loads(request.content).get('method') == 'initialize':
            body = response.json()
            body['result'][field] = value
            return httpx.Response(200, json=body)
        return response

    client = MCPHTTPClient(ServerSpec(name='remote', url='https://fixture.test/mcp'),
                           transport=httpx.MockTransport(handle))
    with pytest.raises(MCPError, match='metadata'):
        await client.connect()
    assert not client.connected


@pytest.mark.parametrize('kind', ['name', 'description', 'depth', 'key', 'nodes'])
async def test_oversized_tool_metadata_is_rejected_before_registration(kind):
    client = client_for(Router())
    entry = {'name': 'echo', 'description': '', 'inputSchema': {'type': 'object'}}
    if kind == 'name':
        entry['name'] = 'x' * 257
    elif kind == 'description':
        entry['description'] = 'x' * 65537
    elif kind == 'key':
        entry['inputSchema']['properties'] = {'x' * 1025: {'type': 'string'}}
    elif kind == 'nodes':
        entry['inputSchema']['enum'] = list(range(4097))
    else:
        child = entry['inputSchema']
        for _ in range(65):
            child['properties'] = {'child': {'type': 'object'}}
            child = child['properties']['child']
    with pytest.raises(MCPError, match='limit'):
        await client._tool_page({'tools': [entry]})


async def test_streamable_http_over_real_loopback_socket():
    router = Router()
    handlers = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            header = (await reader.readuntil(b'\r\n\r\n')).decode('ascii')
            lines = header.split('\r\n')
            method, path, _ = lines[0].split(' ')
            headers = dict(line.split(': ', 1) for line in lines[1:] if ': ' in line)
            length = int(next((value for name, value in headers.items() if name.lower() == 'content-length'), '0'))
            body = await reader.readexactly(length)
            response = await router.handle(httpx.Request(method, 'http://fixture' + path, headers=headers, content=body))
            content = response.content
            wire_headers = ''.join(f'{key}: {value}\r\n' for key, value in response.headers.items()
                                   if key.lower() not in {'content-length', 'connection'})
            writer.write((f'HTTP/1.1 {response.status_code} OK\r\n{wire_headers}'
                          f'Content-Length: {len(content)}\r\nConnection: close\r\n\r\n').encode() + content)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    client = MCPHTTPClient(ServerSpec(name='loopback', url=f'http://127.0.0.1:{port}/mcp'),
                           transport=httpx.AsyncHTTPTransport(trust_env=False))
    try:
        await client.connect()
        assert (await client.list_tools())[0].name == 'echo'
        assert (await client.call_tool('echo', {})).content == 'hello'
    finally:
        await client.disconnect()
        server.close()
        await server.wait_closed()
        for task in list(handlers):
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
