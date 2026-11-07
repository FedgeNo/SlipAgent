"""Local endpoints share the engine, enforce OS access and bound their lifetime."""

import asyncio
import importlib
import json
import os
import socket
from pathlib import Path
import uuid

import pytest

from slipagent.application import application
from slipagent.controlclient import request, instances
from slipagent.events import AgentEvent
from slipagent.localcontrol import start_control, encode, QUEUE_SIZE
from slipagent import localipc
from slipagent.types import Message

session_for = importlib.import_module('test-application').session_for


@pytest.mark.skipif(not hasattr(socket, 'SO_PEERCRED'), reason='Linux peer credentials')
def test_peer_credentials_keep_unsigned_uids():
    import struct
    from types import SimpleNamespace
    peer = SimpleNamespace(getsockopt=lambda *args: struct.pack('iII', 123, 4000000000, 4000000001))
    writer = SimpleNamespace(get_extra_info=lambda name: peer)
    assert localipc.peer_identity(writer) == '4000000000'


@pytest.fixture
async def controlled(workspace):
    session = session_for(workspace)
    host = await start_control(session)
    try:
        yield session, host
    finally:
        await application(session).close()


async def test_discovery_and_multiple_instances_share_no_execution_state(controlled, workspace):
    session, host = controlled
    second = session_for(workspace)
    other = await start_control(second)
    try:
        listed = await instances()
        assert {item['instanceId'] for item in listed} == {host.instance, other.instance}
        assert (await request(host.endpoint, 'submit', {'text': 'First'}))['result']['accepted']
        assert [text for _, text in application(session).queued_input()] == ['First']
        assert application(second).queued_input() == []
        assert await start_control(session) is host
        assert not session.agent.running
    finally:
        await application(second).close()
    assert {item['instanceId'] for item in await instances()} == {host.instance}


async def test_reconnect_request_ids_prevent_duplicate_submissions(controlled):
    session, host = controlled
    identifier = uuid.uuid4().hex
    first = await request(host.endpoint, 'submit', {'text': 'Only once'}, request_id=identifier)
    second = await request(host.endpoint, 'submit', {'text': 'Only once'}, request_id=identifier)
    assert first == second
    assert len(application(session).queued_input()) == 1
    conflict = await request(host.endpoint, 'submit', {'text': 'Changed'}, request_id=identifier)
    assert 'error' in conflict and len(application(session).queued_input()) == 1


async def test_disconnected_waiter_does_not_repeat_an_inflight_action(controlled, monkeypatch):
    session, host = controlled
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def rename(text):
        calls.append(text)
        entered.set()
        await release.wait()
        return text
    monkeypatch.setattr(application(session), 'rename', rename)
    reader, writer = await localipc.connect(host.endpoint)
    writer.write(encode({'jsonrpc': '2.0', 'id': 'inflight', 'method': 'rename', 'params': {'text': 'Once'}}))
    await writer.drain()
    await asyncio.wait_for(entered.wait(), 2)
    writer.close()
    await writer.wait_closed()
    retried = asyncio.create_task(request(host.endpoint, 'rename', {'text': 'Once'}, request_id='inflight'))
    release.set()
    assert (await retried)['result'] == 'Once'
    assert calls == ['Once']


async def test_concurrent_startup_and_shutdown_leave_one_endpoint(workspace):
    session = session_for(workspace)
    try:
        first, second = await asyncio.gather(start_control(session), start_control(session))
        assert first is second
        await asyncio.gather(application(session).close(), application(session).close())
        assert await instances() == []
        with pytest.raises(RuntimeError, match='closed'):
            await start_control(session)
    finally:
        await application(session).close()


@pytest.mark.skipif(os.name == 'nt', reason='Unix socket identity')
async def test_cleanup_does_not_delete_a_replacement_file(controlled):
    session, host = controlled
    endpoint = Path(host.endpoint)
    endpoint.unlink()
    endpoint.write_text('Keep this replacement')
    await application(session).close()
    assert endpoint.read_text() == 'Keep this replacement'


@pytest.mark.skipif(os.name != 'posix' or not hasattr(os, 'geteuid') or os.geteuid() != 0,
                    reason='Requires root to exercise a real second Unix user')
async def test_other_unix_user_is_allowed_only_in_danger_mode(controlled):
    import sys
    from test_cli_e2e import cli_environment
    session, host = controlled
    script = '''import json, os, socket, sys
os.setgroups([])
os.setgid(65534)
os.setuid(65534)
connection = socket.socket(socket.AF_UNIX)
try:
    connection.connect(sys.argv[1])
except PermissionError:
    print('denied', flush=True)
    sys.exit(0)
connection.sendall(b'{"jsonrpc":"2.0","id":"foreign","method":"status"}\\n')
stream = connection.makefile('rb')
print(json.loads(stream.readline())['result']['danger'], flush=True)
sys.stdin.readline()
print('closed' if stream.readline() == b'' else 'still connected', flush=True)
'''
    async def child():
        return await asyncio.create_subprocess_exec(sys.executable, '-c', script, host.endpoint,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=cli_environment(), start_new_session=True)
    denied = await child()
    stdout, stderr = await asyncio.wait_for(denied.communicate(), 5)
    assert denied.returncode == 0 and stdout.strip() == b'denied', stderr
    application(session).set_danger(True)
    accepted = await child()
    try:
        assert await asyncio.wait_for(accepted.stdout.readline(), 5) == b'True\n'
        application(session).set_danger(False)
        accepted.stdin.write(b'continue\n')
        await accepted.stdin.drain()
        stdout, stderr = await asyncio.wait_for(accepted.communicate(), 5)
        assert accepted.returncode == 0 and stdout.strip() == b'closed', stderr
    finally:
        if accepted.returncode is None:
            accepted.kill()
            await accepted.communicate()


@pytest.mark.parametrize('method,params', [('submit', {'text': ''}), ('submit', {'text': ' ', 'bogus': True}),
    ('setting', {'setting': 'danger', 'enabled': 'false'}), ('setting', {'setting': 'unknown', 'enabled': True}),
    ('status', {'bogus': 1}), ('missing', {}), ('history', {'stepId': 0})])
async def test_invalid_actions_leave_session_unchanged(controlled, method, params):
    session, host = controlled
    result = await request(host.endpoint, method, params)
    assert 'error' in result
    assert not application(session).queued_input()
    assert not session.workspace.access.danger


async def test_session_guard_and_queue_discard(controlled):
    session, host = controlled
    refused = await request(host.endpoint, 'submit', {'text': 'No'}, session_id='another-session')
    assert 'error' in refused and not application(session).queued_input()
    await request(host.endpoint, 'submit', {'text': 'Keep'})
    key = application(session).queued_input()[0][0]
    assert (await request(host.endpoint, 'discard', {'identifier': key}))['result']
    assert not application(session).queued_input()


async def test_history_session_guard_is_rechecked_after_async_read(controlled, monkeypatch):
    from slipagent.sessions import SessionJournal
    session, host = controlled
    journal = SessionJournal(str(session.workspace.root), session.workspace.root / 'test-sessions')
    session.registry.resources.journal = journal
    journal.begin(session.agent)
    entered, release = asyncio.Event(), asyncio.Event()
    previous = session.agent.session_id
    async def read(method, params):
        entered.set()
        await release.wait()
        return {'text': 'Previous session page'}
    monkeypatch.setattr(host, 'execute', read)
    task = asyncio.create_task(request(host.endpoint, 'history', {'stepId': 1}, session_id=previous))
    await asyncio.wait_for(entered.wait(), 2)
    await application(session).reset()
    release.set()
    response = await task
    assert 'error' in response and 'result' not in response


async def test_queue_plan_and_saved_session_pages(controlled, monkeypatch):
    session, host = controlled
    await request(host.endpoint, 'submit', {'text': 'queued text'})
    queue = (await request(host.endpoint, 'queue'))['result']
    assert queue[0]['text'] == 'queued text' and queue[0]['identifier']
    plan = (await request(host.endpoint, 'plan'))['result']
    assert 'No working plan' in json.loads(plan['text'])['status']
    async def saved():
        return [{'id': str(index)} for index in range(3)]
    monkeypatch.setattr(application(session), 'saved_sessions', saved)
    first = (await request(host.endpoint, 'sessions', {'limit': 2}))['result']
    assert first == {'items': [{'id': '0'}, {'id': '1'}], 'nextOffset': 2}
    last = (await request(host.endpoint, 'sessions', {'offset': 2, 'limit': 2}))['result']
    assert last == {'items': [{'id': '2'}], 'nextOffset': None}


async def test_client_command_needs_no_provider_key_and_lists_same_instances(controlled, monkeypatch):
    import sys
    from test_cli_e2e import cli_environment
    session, host = controlled
    env = cli_environment()
    env.pop('OPENROUTER_API_KEY')
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'slipagent.cli', 'control', 'list',
        env=env, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
    assert process.returncode == 0, stderr
    assert json.loads(stdout)[0]['instanceId'] == host.instance


async def test_permission_failure_closes_public_listener_and_keeps_previous_mode(controlled, monkeypatch):
    session, host = controlled
    def fail(danger):
        raise OSError('Permission change failed')
    monkeypatch.setattr(host.listener, 'set_access', fail)
    with pytest.raises(OSError, match='Permission change failed'):
        application(session).set_danger(True)
    assert not session.workspace.access.danger
    async with asyncio.timeout(3):
        while host.closing is None:
            await asyncio.sleep(0)
        await host.aclose()
    assert await instances() == []


async def test_malformed_json_and_subscriptions_report_errors_without_mutation(controlled):
    session, host = controlled
    reader, writer = await localipc.connect(host.endpoint)
    try:
        writer.write(b'not json\n')
        await writer.drain()
        assert json.loads(await reader.readline())['error']['code'] == -32600
        writer.write(encode({'jsonrpc': '2.0', 'id': 'watch', 'method': 'subscribe', 'params': {'extra': True}}))
        await writer.drain()
        assert 'error' in json.loads(await reader.readline())
        assert not application(session).queued_input()
    finally:
        writer.close()
        await writer.wait_closed()


async def test_subscription_streams_events_and_closes_with_application(controlled):
    session, host = controlled
    reader, writer = await localipc.connect(host.endpoint)
    writer.write(encode({'jsonrpc': '2.0', 'id': 'watch', 'method': 'subscribe'}))
    await writer.drain()
    assert json.loads(await reader.readline())['result']['instanceId'] == host.instance
    session.agent._emit(AgentEvent(kind='phase', text='Reading project'))
    event = json.loads(await reader.readline())
    assert event['method'] == 'event' and event['params']['kind'] == 'phase'
    session.agent._emit(AgentEvent(kind='context', text='secret request body'))
    session.agent._emit(AgentEvent(kind='notice', text='Visible'))
    assert json.loads(await reader.readline())['params']['text'] == 'Visible'
    await application(session).close()
    assert await asyncio.wait_for(reader.read(), 2) == b''
    writer.close()
    await writer.wait_closed()
    if os.name != 'nt':
        assert not Path(host.endpoint).exists()


@pytest.mark.skipif(os.name == 'nt', reason='Unix socket filesystem permissions')
async def test_permissions_and_peer_identity_follow_danger_mode(controlled, monkeypatch):
    session, host = controlled
    assert Path(host.endpoint).stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(localipc, 'peer_identity', lambda writer: 'foreign-user')
    reader, writer = await localipc.connect(host.endpoint)
    assert await asyncio.wait_for(reader.read(), 2) == b''
    writer.close()
    await writer.wait_closed()
    application(session).set_danger(True)
    assert Path(host.endpoint).stat().st_mode & 0o777 == 0o666
    reader, writer = await localipc.connect(host.endpoint)
    writer.write(encode({'jsonrpc': '2.0', 'id': 'yes', 'method': 'subscribe'}))
    await writer.drain()
    assert json.loads(await reader.readline())['result']['danger']
    application(session).set_danger(False)
    assert Path(host.endpoint).stat().st_mode & 0o777 == 0o600
    assert await asyncio.wait_for(reader.read(), 2) == b''
    writer.close()
    await writer.wait_closed()


async def test_bounded_file_backed_original_history_pages(controlled):
    session, host = controlled
    text = 'Exact original λ\\n\n' * 5000
    session.agent.messages.extend([Message.user('Read'), Message.assistant(text)])
    session.agent.history.sync(session.agent.messages)
    session.agent._persist()
    restored, offset = '', 0
    while True:
        result = await request(host.endpoint, 'history', {'stepId': 1, 'offset': offset, 'limit': 4096})
        page = result['result']
        assert len(page['text']) <= 4096
        restored += page['text']
        offset = page['nextOffset']
        if offset is None:
            break
    assert json.loads(restored)['response'] == text
    assert session.agent.history.steps[0].parts()['response'] == text


async def test_oversized_input_disconnects_without_submission(controlled):
    session, host = controlled
    reader, writer = await localipc.connect(host.endpoint)
    writer.write(b'x' * (localipc.LIMIT + 1) + b'\n')
    await writer.drain()
    assert await asyncio.wait_for(reader.read(), 2) == b''
    assert not application(session).queued_input()
    writer.close()
    await writer.wait_closed()


async def test_slow_subscriber_is_disconnected_without_blocking_agent(controlled):
    session, host = controlled
    reader, writer = await localipc.connect(host.endpoint)
    writer.write(encode({'jsonrpc': '2.0', 'id': 'watch', 'method': 'subscribe'}))
    await writer.drain()
    await reader.readline()
    for _ in range(QUEUE_SIZE + 1):
        session.agent._emit(AgentEvent(kind='notice', text='Progress'))
    assert await asyncio.wait_for(reader.read(), 2) == b''
    writer.close()
    await writer.wait_closed()


async def test_local_submission_uses_existing_cli_driver_once(tmp_path, monkeypatch):
    import io
    from slipagent import cli
    from test_cli_e2e import StubOpenRouter, text_step
    with StubOpenRouter([text_step('Controlled response')]) as stub:
        args = cli.build_parser().parse_args(['--no-mcp', '--no-reload', '--model', 'stub/one',
            '--base-url', stub.base_url, '-w', str(tmp_path)])
        session = await cli.Launcher(args).build_session()
        session.renderer.stream = io.StringIO()
        async def read(*args):
            await asyncio.Event().wait()
        monkeypatch.setattr(cli.Repl, 'read_line', read)
        task = asyncio.create_task(cli.Repl(session).run())
        try:
            async with asyncio.timeout(10):
                while 'local_control' not in session.extensions:
                    await asyncio.sleep(.01)
                host = session.extensions['local_control']
                await request(host.endpoint, 'submit', {'text': 'One request'})
                while not session.agent.history.steps:
                    await asyncio.sleep(.01)
            assert len(stub.requests) == 1
            assert session.agent.history.steps[0].parts()['response'] == 'Controlled response'
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_no_control_flag_disables_the_repl_endpoint(tmp_path, monkeypatch):
    import io
    from slipagent import cli
    from test_cli_e2e import StubOpenRouter, text_step
    with StubOpenRouter([text_step('Unused')]) as stub:
        args = cli.build_parser().parse_args(['--no-control', '--no-mcp', '--no-reload', '--model', 'stub/one',
            '--base-url', stub.base_url, '-w', str(tmp_path)])
        session = await cli.Launcher(args).build_session()
        session.renderer.stream = io.StringIO()
        async def read(*args):
            return None
        monkeypatch.setattr(cli.Repl, 'read_line', read)
        await cli.Repl(session).run()
        assert 'Local control:' not in session.renderer.stream.getvalue()
        assert await instances() == []
        assert not stub.requests


async def test_context_pages_serve_the_latest_outgoing_request_of_this_conversation(controlled):
    from slipagent.requestbody import RequestBody
    session, host = controlled
    missing = await request(host.endpoint, 'context')
    assert 'No working request' in missing['error']['message']
    text = 'Exact λ request ' * 2000
    body = RequestBody({'model': 'test/model', 'messages': [{'role': 'system', 'content': 'rules'},
                                                            {'role': 'user', 'content': text}]})
    session.agent._emit(AgentEvent(kind='context', step=4, request_body=body))
    restored, offset = '', 0
    while True:
        page = (await request(host.endpoint, 'context', {'offset': offset, 'limit': 4096}))['result']
        assert page['stepId'] == 4 and page['model'] == 'test/model'
        assert page['estimatedTokens'] == body.estimate and page['sentAt'].endswith('+00:00')
        assert page['attempt'] is None  # No request diagnostics in this session.
        assert len(page['text']) <= 4096
        restored += page['text']
        offset = page['nextOffset']
        if offset is None:
            break
    assert json.loads(restored)['messages'][1]['content'] == text
    session.agent._emit(AgentEvent(kind='context', step=5, text=json.dumps({'model': 'text/form', 'messages': []})))
    page = (await request(host.endpoint, 'context'))['result']
    assert page['stepId'] == 5 and page['model'] == 'text/form' and page['estimatedTokens'] is None
    assert json.loads(page['text']) == {'model': 'text/form', 'messages': []}
    await application(session).reset()
    stale = await request(host.endpoint, 'context')
    assert 'No working request' in stale['error']['message']
    # A host whose event handler never retained the body serves the newest diagnostics attempt.
    from slipagent.diagnostics import RequestDiagnostics
    diagnostics = RequestDiagnostics()
    diagnostics.use_directory(Path(session.workspace.root) / 'requests')
    session.registry.resources.diagnostics = diagnostics
    attempt = diagnostics.begin(RequestBody({'model': 'logged/model', 'messages': [{'role': 'user', 'content': 'λ'}]}),
                                step=1, step_id=1)
    page = (await request(host.endpoint, 'context'))['result']
    assert page['attempt'] == attempt and page['stepId'] == 1 and page['model'] == 'logged/model'
    assert page['estimatedTokens'] > 0 and json.loads(page['text'])['messages'][0]['content'] == 'λ'
