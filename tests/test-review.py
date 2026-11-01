"""Dedicated reviews enforce read-only execution and validate navigable findings."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from slipagent.agent import Agent
from slipagent.review import review_mode, show_review
from slipagent.tools import build_default_registry
from test_agent import StubClient, completion, call
from test_cli_e2e import metadata_server


def report(**changes):
    return {'status': 'completed', 'summary': 'One defect.', 'coverage': 'example.txt and its caller; tests not executed.',
            'findings': [{'severity': 'high', 'title': 'Empty input fails', 'path': 'example.txt', 'line': 1,
                          'evidence': 'An empty input reaches indexing and raises rather than returning an empty result.',
                          'uncertainty': 'Runtime behavior inferred from source.'}], **changes}


@pytest.fixture
def review_agent(workspace):
    (workspace.root / 'example.txt').write_text('first\nsecond\n')
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([]), registry, 'test/model')
    return agent


async def close(agent):
    await agent.wait_for_compaction()
    await agent.registry.aclose()


@pytest.mark.parametrize('native', [False, True])
async def test_review_uses_regular_history_and_structured_answer_with_safe_tools(review_agent, monkeypatch, native):
    agent = review_agent
    from slipagent.tools.discovery import tool_catalog
    catalog = tool_catalog(agent.registry)
    catalog.select(agent.registry.specs(), [])
    monkeypatch.setattr(Agent, '_uses_native_tools', lambda self, capabilities, **kwargs: native)
    agent.client.responses = [completion(tool_calls=[call('read_file', path='example.txt')]),
                              completion(tool_calls=[call('answer', **report())])]
    original = {name: agent.registry.get(name) for name in agent.registry.names}
    try:
        with review_mode(agent, 'working changes') as state:
            text = await agent.run('Review working changes')
            assert state.report['status'] == 'completed'
            assert 'example.txt:1' in text
            assert not state.stale('example.txt')
            for request in agent.client.calls:
                if native:
                    names = {spec.name for spec in request['tools']}
                    assert names == {'read_file', 'list_dir', 'glob', 'grep', 'git_status', 'git_diff', 'git_log', 'answer'}
                else:
                    assert request.get('tools') is None
                assert 'dedicated code review' in str(request['messages'][0].content)
                assert '# Additional Tool Catalog' not in str(request['messages'][0].content)
            result = next(message.content['content'] for message in agent.messages
                          if message.role == 'tool' and message.content['tool'] == 'answer')
            assert result['review']['findings'][0]['line'] == 1
        assert all(agent.registry.get(name) is tool for name, tool in original.items())
        assert agent.registry.services['last_review'] is state
        assert agent.registry.services['tool_catalog'] is catalog
    finally:
        await close(agent)


async def test_review_rejects_mixed_mutating_batch_before_reading_and_restores_profile(review_agent):
    agent = review_agent
    agent.client.responses = [completion(tool_calls=[call('write_file', path='example.txt', content='bad'),
                                                    call('read_file', path='example.txt')]), completion('Stopped')]
    selected = {'name': 'another skill'}
    agent.registry.services['selected_skill'] = selected
    agent.exposed_tools = ('read_file',)
    try:
        with review_mode(agent, 'scope') as state:
            assert 'run_command' not in agent.registry and 'run_python' not in agent.registry
            assert 'load_tools' not in agent.registry
            assert agent.registry.services['selected_skill'] is selected
            await agent.run('Review')
            assert state.files == {} and state.report is None
            assert 'incomplete' in state.text()
        assert agent.registry.services['selected_skill'] is selected
        assert agent.exposed_tools == ('read_file',)
        assert (agent.registry.resources.workspace.root / 'example.txt').read_text() == 'first\nsecond\n'
        assert 'write_file' in agent.registry
    finally:
        await close(agent)


@pytest.mark.parametrize('change', [
    {'severity': 'imaginary'}, {'line': 0}, {'line': 50}, {'path': 'missing.txt'}, {'uncertainty': ''},
])
async def test_review_rejects_invalid_findings(review_agent, change):
    agent = review_agent
    try:
        with review_mode(agent, 'scope') as state:
            await agent.registry.invoke('read_file', {'path': 'example.txt'})
            submitted = report()
            submitted['findings'][0].update(change)
            result = await agent.registry.invoke('answer', submitted)
            assert result.is_error and state.report is None
    finally:
        await close(agent)


async def test_review_requires_source_read_and_marks_changed_files_before_navigation(review_agent):
    agent = review_agent
    try:
        with review_mode(agent, 'scope') as state:
            assert (await agent.registry.invoke('answer', report())).is_error
            await agent.registry.invoke('read_file', {'path': 'example.txt'})
            assert not (await agent.registry.invoke('answer', report())).is_error
            (state.workspace.root / 'example.txt').write_text('changed\nsecond\n')
            assert '[file changed' in state.text()
            terminal = SimpleNamespace(choose=AsyncMock(return_value='0'), _read_file=AsyncMock(), notify=Mock())
            await show_review(terminal, state)
            terminal.notify.assert_called_once()
            terminal._read_file.assert_awaited_once_with(str(state.workspace.root / 'example.txt'), line=1)
    finally:
        await close(agent)


@pytest.mark.parametrize('status', ['completed', 'incomplete'])
async def test_review_accepts_empty_findings_without_claiming_verified_correctness(review_agent, status):
    agent = review_agent
    try:
        with review_mode(agent, 'scope') as state:
            result = await agent.registry.invoke('answer', report(status=status, findings=[]))
            assert not result.is_error
            assert f'Review {status}' in state.text()
            terminal = SimpleNamespace(open_viewer=Mock())
            await show_review(terminal, state)
            terminal.open_viewer.assert_called_once()
    finally:
        await close(agent)


async def test_review_restores_tools_when_cancelled(review_agent):
    agent = review_agent
    entered = asyncio.Event()
    async def blocked(**kwargs):
        entered.set()
        await asyncio.Event().wait()
    agent.client.chat = blocked
    original = agent.registry.get('write_file')
    async def run():
        with review_mode(agent, 'scope'):
            await agent.run('Review')
    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.registry.get('write_file') is original
        assert 'active_review' not in agent.registry.services
        assert agent.registry.services['last_review'].report is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await close(agent)


@pytest.mark.parametrize('outcome', ['completed', 'prose', 'interrupted', 'failure', 'queued_input'])
async def test_repl_review_restores_profile_before_deferred_commands(tmp_path, monkeypatch, metadata_server, outcome):
    import io
    from slipagent import cli

    session = await cli.Launcher(cli.build_parser().parse_args([
        '--no-mcp', '--model', 'stub/model', '-w', str(tmp_path),
    ])).build_session()
    session.renderer.stream = io.StringIO()
    entered, release = asyncio.Event(), asyncio.Event()
    class GatedClient(StubClient):
        async def chat(self, **kwargs):
            entered.set()
            await release.wait()
            if outcome == 'failure':
                raise RuntimeError('provider failed')
            return await super().chat(**kwargs)
    responses = ([completion(tool_calls=[call('answer', **report(findings=[]))])]
                 if outcome in {'completed', 'queued_input'} else [completion('Partial review')])
    if outcome == 'queued_input':
        responses += [completion(tool_calls=[call('write_file', path='followup.txt', content='requested follow-up')]),
                      completion('Follow-up finished')]
    session.agent.client = GatedClient(responses)
    async def read(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(cli.Repl, 'read_line', read)
    original = session.registry.get('write_file')
    applied = []
    original_apply = cli.CommandRunner.apply_deferred
    async def checked_apply(self, **kwargs):
        assert 'active_review' not in self.session.registry.services
        assert self.session.registry.get('write_file') is original
        applied.append(True)
        return await original_apply(self, **kwargs)
    monkeypatch.setattr(cli.CommandRunner, 'apply_deferred', checked_apply)
    task = asyncio.create_task(cli.Repl(session).run_review('working changes', session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if outcome == 'interrupted':
            session.agent.request_stop()
        if outcome == 'queued_input':
            session.agent.enqueue('Now write followup.txt as requested.')
        elif outcome != 'failure':
            session.work.deferred_commands = ['/danger on']
        release.set()
        if outcome == 'failure':
            with pytest.raises(RuntimeError, match='provider failed'):
                await asyncio.wait_for(task, 5)
        else:
            assert await asyncio.wait_for(task, 5) is False
            assert applied
            assert session.workspace.access.danger == (outcome != 'queued_input')
        state = session.registry.services['last_review']
        if outcome in {'completed', 'queued_input'}:
            assert state.report['status'] == 'completed'
        else:
            assert 'incomplete' in state.text()
        assert session.registry.get('write_file') is original
        assert not session.reloader or session.reloader.busy == 0
        records = session.registry.resources.artifacts.records
        saved = next(record for record in records.values() if record['kind'] == 'review')
        assert saved['record_id'] > 0
        if outcome == 'queued_input':
            assert (tmp_path / 'followup.txt').read_text() == 'requested follow-up'
            assert len(session.agent.client.calls) == 3
        if outcome == 'completed':
            assert session.renderer.stream.getvalue().count('Review completed') == 1
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()


async def test_review_command_waits_for_current_run_before_installing_profile(tmp_path, monkeypatch, metadata_server):
    import io
    from slipagent import cli

    session = await cli.Launcher(cli.build_parser().parse_args([
        '--no-mcp', '--model', 'stub/model', '-w', str(tmp_path),
    ])).build_session()
    session.renderer.stream = io.StringIO()
    entered, release = asyncio.Event(), asyncio.Event()
    class GatedClient(StubClient):
        async def chat(self, **kwargs):
            if not self.calls:
                entered.set()
                await release.wait()
            return await super().chat(**kwargs)
    client = GatedClient([completion('Earlier task settled.'),
                          completion(tool_calls=[call('answer', **report(findings=[]))])])
    session.agent.client = client
    async def read(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(cli.Repl, 'read_line', read)
    original = session.registry.get('write_file')
    task = asyncio.create_task(cli.Repl(session).run_request('Earlier task', session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await cli.CommandRunner(session).handle('/review working changes')
        assert session.registry.get('write_file') is original
        assert 'active_review' not in session.registry.services
        release.set()
        await asyncio.wait_for(task, 5)
        assert len(client.calls) == 1
        scope = session.extensions.pop('pending_review')
        assert scope == 'working changes'
        assert not await cli.Repl(session).run_review(scope, session.renderer.style)
        assert session.registry.services['last_review'].report['status'] == 'completed'
        assert session.registry.get('write_file') is original
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()
