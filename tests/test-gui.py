"""The browser observes the existing engine through an authenticated bounded host."""

import asyncio
import json
import signal
import socket
from unittest.mock import AsyncMock

import httpx
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from slipagent import gui, guitransport
from slipagent.application import application
from slipagent.agent import Agent
from slipagent.events import AgentEvent
from slipagent.sessionstate import Session
from slipagent.tools.base import ToolRegistry, ToolResult
from slipagent.types import ToolCall
from test_agent import StubClient


def session_for(workspace):
    registry = ToolRegistry()
    registry.resources.workspace = workspace
    client = StubClient([])
    client.aclose = AsyncMock()
    return Session(agent=Agent(client, registry, 'test/model'), registry=registry,
                   client=client, workspace=workspace, api_key='test',
                   base_url='http://unused.invalid', http_referer=None, app_title='test')


async def receive(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), 5))


async def authenticated(host, client_id=''):
    ws = await connect(f'ws://127.0.0.1:{host.port}/api/events', origin=f'http://127.0.0.1:{host.port}')
    await ws.send(json.dumps({'type': 'authenticate', 'token': host.token, 'clientId': client_id}))
    return ws


async def test_http_auth_host_origin_and_static_allowlist(workspace):
    session = session_for(workspace)
    host = gui.GuiHost(session)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=host.app), base_url='http://localhost:9000') as client:
        assert (await client.get('/api/snapshot')).status_code == 401
        headers = {'Authorization': 'Bearer ' + host.token}
        response = await client.get('/api/snapshot', headers=headers)
        assert response.status_code == 200
        assert response.json()['state']['workspace'] == str(workspace.root)
        assert response.headers['cache-control'] == 'no-store'
        assert (await client.get('/api/snapshot', headers={**headers, 'Origin': 'https://attacker.invalid'})).status_code == 403
        assert (await client.get('/api/snapshot', headers={**headers, 'Host': 'attacker.invalid'})).status_code == 403
        assert (await client.get('/api/snapshot', headers={**headers, 'Origin': 'http://localhost:9000'})).status_code == 200
        for path in ('/', '/app.js', '/app.css', '/third-party-notices.txt'):
            response = await client.get(path)
            assert response.status_code == 200
            assert host.token not in response.text
        for path in ('/docs', '/openapi.json', '/.env', '/agents/gui-implementation-plan.md'):
            assert (await client.get(path)).status_code == 404
    await host.aclose()
    await application(session).close()


async def test_authenticated_live_events_reconnect_and_shutdown(workspace):
    session = session_for(workspace)
    app = application(session)
    handlers = [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)]
    host, concurrent = await asyncio.gather(gui.start_gui(session), gui.start_gui(session))
    try:
        assert host is concurrent
        assert handlers == [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)]
        assert host is await gui.start_gui(session)
        assert host.socket.getsockname()[0] == '127.0.0.1'
        assert host.token not in host.url.split('#')[0]
        first, second = await authenticated(host), await authenticated(host)
        try:
            a, b = await receive(first), await receive(second)
            assert a == b and a['type'] == 'snapshot'
            session.agent._emit(AgentEvent(kind='notice', text='Shared observation'))
            a, b = await receive(first), await receive(second)
            assert a == b and a['event']['text'] == 'Shared observation'
            assert a['sequence'] > 0
        finally:
            await first.close()
            await second.close()
        session.agent._emit(AgentEvent(kind='notice', text='Missed while detached'))
        third = await authenticated(host)
        try:
            snapshot = await receive(third)
            assert snapshot['type'] == 'snapshot' and snapshot['resync']
            assert snapshot['epoch'] == a['epoch'] and snapshot['sequence'] > a['sequence']
            await app.close()
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(third.recv(), 5)
        finally:
            await third.close()
        assert host.task.done() and not host.transport.clients
    finally:
        await app.close()


@pytest.mark.parametrize('mode', ['token', 'origin', 'malformed', 'timeout', 'binary'])
async def test_websocket_refuses_unauthorized_clients(workspace, monkeypatch, mode):
    session = session_for(workspace)
    monkeypatch.setattr(gui, 'AUTH_TIMEOUT', .1)
    host = await gui.start_gui(session)
    try:
        if mode == 'origin':
            from websockets.exceptions import InvalidStatus
            with pytest.raises(InvalidStatus):
                async with connect(f'ws://127.0.0.1:{host.port}/api/events', origin='http://attacker.invalid'):
                    pytest.fail('Accepted foreign origin')
        else:
            async with connect(f'ws://127.0.0.1:{host.port}/api/events', origin=f'http://127.0.0.1:{host.port}') as ws:
                if mode == 'binary':
                    await ws.send(b'not a text authentication message')
                elif mode != 'timeout':
                    await ws.send('{' if mode == 'malformed' else json.dumps({'type': 'authenticate', 'token': 'wrong'}))
                with pytest.raises(ConnectionClosed) as failure:
                    await asyncio.wait_for(ws.recv(), 5)
                assert failure.value.rcvd.code == 1008
        assert not host.transport.clients
    finally:
        await application(session).close()


async def test_payloads_are_bounded_and_slow_client_is_evicted(workspace, monkeypatch):
    session = session_for(workspace)
    monkeypatch.setattr(guitransport, 'QUEUE_SIZE', 2)
    host = gui.GuiHost(session)
    transport = host.transport
    slow, fast = transport.attach(), transport.attach()
    slow.get_nowait()
    fast.get_nowait()
    try:
        for index in range(3):
            session.agent._emit(AgentEvent(kind='tool_end', text='x' * 100_000,
                                          tool_call=ToolCall('1', 'read_file', {'secret': 'not exported'}),
                                          result=ToolResult.ok('large original result')))
            event = fast.get_nowait()['event']
            assert len(event['text']) == guitransport.TEXT_LIMIT + 1
            assert event['text'].endswith('…') and event['truncated']
            assert 'result' not in event and 'arguments' not in event['toolCall']
        assert slow.get_nowait() is None
        assert slow not in transport.clients and fast in transport.clients
        session.agent._emit(AgentEvent(kind='context', text='request secret'))
        assert fast.empty()
        await host.aclose()
        assert not application(session).state().closed
        assert fast.get_nowait() is None
    finally:
        await application(session).close()


async def test_detaching_browser_does_not_cancel_active_run(workspace, monkeypatch):
    session = session_for(workspace)
    started, release = asyncio.Event(), asyncio.Event()
    async def run(*args, **kwargs):
        started.set()
        await release.wait()
        return 'Done'
    monkeypatch.setattr(type(session.agent), 'run', run)
    app = application(session)
    host = await gui.start_gui(session)
    task = app.start('Work')
    try:
        await asyncio.wait_for(started.wait(), 5)
        ws = await authenticated(host)
        assert (await receive(ws))['state']['running']
        await ws.close()
        await host.aclose()
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, 5) == 'Done'
    finally:
        await app.close()


async def test_failed_bind_cleans_subscriptions(workspace):
    session = session_for(workspace)
    app = application(session)
    listeners = len(app.events.listeners)
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        with pytest.raises(OSError):
            await gui.start_gui(session, port=occupied.getsockname()[1])
    assert len(app.events.listeners) == listeners
    assert 'gui_host' not in session.registry.services
    await app.close()


async def test_connection_limit_and_cancelled_close_preserve_cleanup(workspace, monkeypatch):
    session = session_for(workspace)
    monkeypatch.setattr(gui, 'MAX_CLIENTS', 1)
    host = await gui.start_gui(session)
    ws = await authenticated(host)
    try:
        await receive(ws)
        from websockets.exceptions import InvalidStatus
        with pytest.raises(InvalidStatus):
            await authenticated(host)
        waiter = asyncio.create_task(host.aclose())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert host.task.done() and not host.transport.clients
        assert not application(session).state().closed
        await host.aclose()
    finally:
        await ws.close()
        await application(session).close()


async def test_observer_cannot_execute_actions(workspace):
    session = session_for(workspace)
    host = await gui.start_gui(session)
    try:
        ws = await authenticated(host)
        try:
            await receive(ws)
            await ws.send(json.dumps({'type': 'start', 'prompt': 'Do not execute'}))
            with pytest.raises(ConnectionClosed) as failure:
                await asyncio.wait_for(ws.recv(), 5)
            assert failure.value.rcvd.code == 1008
            assert not application(session).busy and not session.agent.has_queued_input
        finally:
            await ws.close()
    finally:
        await application(session).close()


async def test_oversized_websocket_input_is_rejected(workspace):
    session = session_for(workspace)
    host = await gui.start_gui(session)
    try:
        ws = await authenticated(host)
        try:
            await receive(ws)
            await ws.send('x' * 5000)
            with pytest.raises(ConnectionClosed) as failure:
                await asyncio.wait_for(ws.recv(), 5)
            assert failure.value.rcvd.code == 1009
        finally:
            await ws.close()
    finally:
        await application(session).close()


async def test_gui_command_reuses_host_and_parser_exposes_launch_options(workspace):
    import io
    from slipagent.cli import Style, build_parser
    from slipagent.commands import CommandContext, GuiCommand
    session = session_for(workspace)
    out = io.StringIO()
    try:
        args = build_parser().parse_args(['--gui', '--gui-port', '9999'])
        assert args.gui and args.gui_port == 9999
        command = GuiCommand()
        assert command.background('') and not command.changes_session('')
        await command.run(CommandContext(session, Style(False), out), '')
        host = session.registry.services['gui_host']
        await command.run(CommandContext(session, Style(False), out), '')
        assert out.getvalue().count(host.url) == 2
        assert out.getvalue().count(f'ssh -N -L {host.port}:127.0.0.1:{host.port} YOUR_SSH_HOST') == 2
        assert 'on your own computer' in out.getvalue()
        assert 'same Browser link' in out.getvalue()
    finally:
        await application(session).close()


def test_gui_launch_flag_starts_after_initialization_and_exits_cleanly(tmp_path):
    from test_cli_e2e import run_repl_commands
    result = run_repl_commands(tmp_path, ['/quit'], '--gui', '--no-mcp')
    assert result.returncode == 0, result.stderr
    assert 'Browser: http://127.0.0.1:' in result.stderr
    import re
    port = re.search(r'Browser: http://127\.0\.0\.1:(\d+)', result.stderr)[1]
    assert f'ssh -N -L {port}:127.0.0.1:{port} YOUR_SSH_HOST' in result.stderr
    assert 'on your own computer' in result.stderr
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('arguments', [['--gui', 'one shot'], ['--gui-port', '-1'], ['--gui-port', '65536']])
async def test_invalid_gui_launch_is_rejected_before_building_session(monkeypatch, arguments):
    from slipagent.cli import Launcher, build_parser
    build = AsyncMock(side_effect=AssertionError('Constructed session for invalid arguments'))
    monkeypatch.setattr(Launcher, 'build_session', build)
    args = build_parser().parse_args(arguments)
    assert await Launcher(args).dispatch('one shot' if args.prompt else None) == 2
    build.assert_not_called()


async def test_all_originals_remain_browsable_after_model_compaction(workspace, tmp_path, monkeypatch):
    from slipagent.context import CompletedStep
    from slipagent.guibrowser import GuiBrowser
    from slipagent.sources import FileText
    from slipagent.turnstore import TurnStore
    from slipagent.types import Message
    session = session_for(workspace)
    original = 'Original beginning ' + 'x' * 200_000 + ' original ending'
    reply = Message.assistant(original)
    reply.reasoning_model = 'old/model'
    reply.response_provider = 'openrouter'
    messages = [Message.user('Very old prompt'), reply]
    step = CompletedStep(1, 'Very old prompt', messages)
    store = TurnStore(tmp_path / 'turns')
    store.archive(step, messages)
    step.summary = 'A compressed summary must not replace originals in the GUI'
    session.agent.history.steps.append(step)
    session.agent.messages.extend(step.messages)
    browser = GuiBrowser(session)
    monkeypatch.setattr(FileText, 'read', lambda self: pytest.fail('Materialized a full archived field'))
    try:
        first = browser.detail('/turns/1/response')
        last = browser.detail('/turns/1/response', len(original) - 30)
        assert first['text'] == original[:16384] and first['more']
        assert last['text'] == original[-30:] and not last['more']
        assert browser.children('/turns')['items'][0]['id'] == '/turns/1'
        assert browser.feed(0, 10)['items'][-1]['text'] == original[:4096]
        assert browser.feed(0, 10)['items'][-1]['source'] == {'model': 'old/model', 'provider': 'openrouter'}
        assert browser.value(f'/messages/{len(session.agent.messages) - 1}')['response_source'] == {
            'model': 'old/model', 'provider': 'openrouter'}
    finally:
        store.close()
        await application(session).close()


async def test_structured_actions_use_shared_input_and_reject_stale_sessions(workspace):
    session = session_for(workspace)
    host = gui.GuiHost(session)
    headers = {'Authorization': 'Bearer ' + host.token, 'Origin': 'http://localhost'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=host.app), base_url='http://localhost', headers=headers) as client:
        command = {'sessionId': session.agent.session_id, 'action': 'submit', 'text': 'From browser'}
        assert (await client.post('/api/actions', json=command)).status_code == 200
        assert application(session).input_ready.is_set()
        assert application(session).queued_input()[0][1] == 'From browser'
        assert not application(session).busy
        assert (await client.post('/api/actions', json={**command, 'sessionId': 'obsolete'})).status_code == 409
        assert (await client.post('/api/actions', json={**command, 'action': 'arbitrary_method'})).status_code == 422
        assert (await client.post('/api/actions', json=command, headers={'Origin': 'http://foreign.invalid'})).status_code == 403
        assert (await client.get('/api/children?node=/settings')).status_code == 200
        assert (await client.get('/api/controls')).json()['queue'][0][1] == 'From browser'
    await host.aclose()
    await application(session).close()


async def test_browser_submission_wakes_cli_without_another_runner(workspace, monkeypatch):
    import io
    from slipagent import cli
    from test_agent import completion
    base = session_for(workspace)
    base.client.responses = [completion('From the shared engine')]
    session = cli.Session(agent=base.agent, registry=base.registry, workspace=workspace,
                          client=base.client, api_key='test', base_url='http://unused.invalid',
                          http_referer=None, app_title='test', renderer=cli.Renderer(cli.Style(False), io.StringIO(), False))
    async def read(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(cli.Repl, 'read_line', read)
    repl = cli.Repl(session)
    waiter = asyncio.create_task(repl.read_idle_input(session.renderer.style))
    await asyncio.sleep(0)
    application(session).submit('Browser prompt')
    try:
        assert await asyncio.wait_for(waiter, 5) == (None, True)
        assert not await repl.run_request('', session.renderer.style)
        assert len(session.client.calls) == 1
        assert any(message.content == 'Browser prompt' for message in session.agent.messages)
    finally:
        await application(session).close()


async def test_browser_editor_waits_records_saves_and_releases_on_disconnect(workspace):
    session = session_for(workspace)
    path = workspace.root / 'edit.txt'
    path.write_text('Before\n')
    host = await gui.start_gui(session)
    ws = await authenticated(host, 'editing-tab')
    await receive(ws)
    boundary = None
    try:
        session.agent.running = True
        await host.editor.begin(path, 'editing-tab')
        await asyncio.sleep(0)
        assert host.editor.file is None
        assert not host.editor.coordinator.released.is_set()
        boundary = asyncio.create_task(host.editor.coordinator.boundary())
        async with asyncio.timeout(5):
            while not host.editor.active:
                await asyncio.sleep(.01)
        assert (await host.editor.page('editing-tab', 0))['text'] == 'Before\n'
        await host.editor.update('editing-tab', 0, 'After\n', save=True)
        assert path.read_text() == 'After\n'
        assert any(entry['path'] == str(path) and '+After' in entry['diff']
                   for entry in session.registry.context_notes['user_edits'].values())
        await ws.close()
        async with asyncio.timeout(5):
            await boundary
            while host.editor.client_id:
                await asyncio.sleep(.01)
        assert not host.editor.active and host.editor.coordinator.released.is_set()
    finally:
        session.agent.running = False
        await ws.close()
        await application(session).close()
        if boundary is not None:
            await boundary


async def test_browser_editor_prevents_session_changes_and_detects_disk_conflicts(workspace):
    from slipagent.editor import EditConflict
    session = session_for(workspace)
    path = workspace.root / 'edit.txt'
    path.write_text('Original\n')
    host = await gui.start_gui(session)
    try:
        await host.editor.begin(path, 'editing-tab')
        async with asyncio.timeout(5):
            while not host.editor.active:
                await asyncio.sleep(.01)
        with pytest.raises(RuntimeError, match='Close the current editor'):
            await application(session).reset()
        path.write_text('Changed outside the editor\n')
        with pytest.raises(EditConflict):
            await host.editor.update('editing-tab', 0, 'Do not overwrite\n', save=True)
        assert path.read_text() == 'Changed outside the editor\n'
        await host.editor.aclose()
        assert host.editor.coordinator.released.is_set()
    finally:
        await application(session).close()


async def test_gui_search_reads_old_originals_instead_of_context_summaries(workspace):
    from slipagent.guibrowser import GuiBrowser
    from slipagent.types import Message
    session = session_for(workspace)
    session.agent.messages.extend([Message.user('A far older needle is still here'),
                                   Message.assistant('Recent output without the matching word')])
    browser = GuiBrowser(session)
    result = browser.search('NEEDLE')
    assert len(result['items']) == 1
    match = result['items'][0]
    assert match['match'] == 'needle' and match['index'] < len(session.agent.messages) - 1
    assert 'far older' in match['before'] and result['nextCursor'] is None
    await application(session).close()


async def test_model_skill_and_retained_output_hierarchies_use_existing_owners(workspace):
    from slipagent.artifacts import ArtifactStore
    from slipagent.guibrowser import GuiBrowser
    from slipagent.types import ModelInfo
    session = session_for(workspace)
    session._catalog = [ModelInfo('vendor/model', provider='groq')]
    package = workspace.root / '.slipagent/skills/browser-fixture'
    package.mkdir(parents=True)
    (package / 'SKILL.md').write_text('---\nname: browser-fixture\ndescription: GUI test skill\n---\nKeep the original instructions.\n')
    store = ArtifactStore(1024 * 1024)
    session.registry.resources.artifacts = store
    reference = store.save('Saved complete output', 'beginning ' + 'x' * 100_000 + ' ending', kind='output')
    browser = GuiBrowser(session)
    try:
        models = browser.children('/models/groq')['items']
        assert models[0]['preview'] == 'free'
        assert browser.value(models[0]['id'])['id'] == 'vendor/model'
        skill = next(item for item in browser.children('/skills')['items'] if item['label'] == 'browser-fixture')
        await application(session).select_skill(skill['actionValue'])
        assert session.registry.services['selected_skill']['instructions'] == 'Keep the original instructions.'
        assert browser.value(skill['id'])['name'] == 'browser-fixture'
        output = browser.children('/artifacts')['items'][0]
        assert output['id'] == f"/artifacts/{reference['record_id']}"
        assert browser.detail(output['id'] + '/output')['more']
        assert browser.detail(output['id'] + '/output', 100_000)['text'].endswith(' ending')
    finally:
        await application(session).close()


def test_session_metadata_pages_do_not_load_message_bodies(workspace, tmp_path, monkeypatch):
    import os
    import tracemalloc
    from slipagent.sessions import SessionJournal
    journal = SessionJournal(str(workspace.root), tmp_path / 'sessions')
    monkeypatch.setattr(journal, '_first_prompt', lambda path: pytest.fail('Read a prompt for metadata listing'))
    for number in range(3):
        path = journal.directory / (f'{number:032x}.jsonl')
        header = {'type': 'header', 'project': journal.project, 'created': '2026-11-02T00:00:00+00:00',
                  'title': f'Session {number}'}
        path.write_text(json.dumps(header) + '\n' + json.dumps({'message': 'x' * 2_000_000}) + '\n')
        os.utime(path, ns=(number + 1, number + 1))
    tracemalloc.start()
    try:
        first = journal.listing_page(limit=2)
        second = journal.listing_page(first['nextCursor'], limit=2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert [row['title'] for row in first['items'] + second['items']] == ['Session 2', 'Session 1', 'Session 0']
    assert second['nextCursor'] is None
    assert peak < 512 * 1024


async def test_active_session_hierarchy_keeps_parent_links(workspace):
    from slipagent.context import CompletedStep
    from slipagent.guibrowser import GuiBrowser
    from slipagent.types import Message
    session = session_for(workspace)
    session.agent.history.steps.append(CompletedStep(1, 'Original prompt', [Message.user('Original prompt'), Message.assistant('Original reply')]))
    browser = GuiBrowser(session)
    root = f'/sessions/{session.agent.session_id}'
    try:
        facets = browser.children(root)['items']
        turns = next(item for item in facets if item['label'] == 'Turns')
        assert turns['id'] == root + '/turns'
        page = browser.children(turns['id'])
        assert page['parent'] == root
        assert page['items'][0]['id'] == root + '/turns/1'
        assert browser.detail(root + '/turns/1/response')['text'] == 'Original reply'
    finally:
        await application(session).close()


async def test_cancelled_browser_save_still_records_receipt_and_restores_port(workspace, monkeypatch):
    import threading
    from types import SimpleNamespace
    from slipagent.editor import FileEditor
    from slipagent.useredits import UserEdits
    session = session_for(workspace)
    original_port = SimpleNamespace(active=False, notify=lambda text: None, clear_pending_notice=lambda: None,
                                    shutdown=AsyncMock(), recorded=lambda *args: None)
    coordinator = UserEdits(session, original_port)
    session.registry.resources.user_edits = coordinator
    path = workspace.root / 'save.txt'
    path.write_text('Before\n')
    host = await gui.start_gui(session)
    started, release = threading.Event(), threading.Event()
    save = FileEditor.save
    def delayed_save(file, **kwargs):
        started.set()
        assert release.wait(5)
        return save(file, **kwargs)
    monkeypatch.setattr(FileEditor, 'save', delayed_save)
    try:
        await host.editor.begin(path, 'editing-tab')
        async with asyncio.timeout(5):
            while not host.editor.active:
                await asyncio.sleep(.01)
        assert coordinator.editor is host.editor
        waiter = asyncio.create_task(host.editor.update('editing-tab', 0, 'After\n', save=True))
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 5), 6)
        waiter.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert path.read_text() == 'After\n'
        assert any('+After' in entry['diff'] for entry in session.registry.context_notes['user_edits'].values())
        await host.editor.aclose()
        assert coordinator.editor is original_port and coordinator.released.is_set()
    finally:
        release.set()
        await application(session).close()


async def test_failed_host_can_restart_without_leaking_the_old_listener(workspace):
    session = session_for(workspace)
    host = await gui.start_gui(session)
    port = host.port
    host.task.cancel()
    await asyncio.gather(host.task, return_exceptions=True)
    try:
        replacement = await gui.start_gui(session, port=port)
        assert replacement is not host and host.socket.fileno() == -1
        assert replacement.port == port
    finally:
        await application(session).close()


async def test_journal_identity_change_is_published_before_next_live_event(workspace):
    session = session_for(workspace)
    host = await gui.start_gui(session)
    ws = await authenticated(host)
    try:
        old = (await receive(ws))['state']['sessionId']
        session.agent.session_id = 'new-journal-identity'
        session.agent.running = True
        session.agent._emit(AgentEvent(kind='step_start', step=1))
        state = await receive(ws)
        assert state['type'] == 'state'
        assert state['state']['sessionId'] != old and state['state']['running']
        assert (await receive(ws))['event']['kind'] == 'step_start'
    finally:
        session.agent.running = False
        await ws.close()
        await application(session).close()
