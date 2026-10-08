"""Reload tests edit disposable source copies in isolated interpreter processes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from offline.sitecustomize import GUARD_DIRECTORY


SETUP = '''
import asyncio
import io
import json
import os
from pathlib import Path
import re
import httpx
import slipagent.cli as cli
import slipagent.agent as agent_module
import slipagent.context as context_module
import slipagent.openrouter as router
from slipagent.runtime import RuntimeFrame
from slipagent.tools import build_default_registry
from slipagent.tools.base import Tool, ToolResult
from slipagent.types import Completion, Message, ToolCall, Usage
from slipagent.workspace import Workspace

async def main():
    workspace = Workspace(Path.cwd())
    requests = []
    responses = []
    async def transport(request):
        if request.method == 'POST':
            body = json.loads(request.content)
            if body['messages'][0]['content'].lstrip().startswith('Summarize one completed SlipAgent step'):
                return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'Completed demo step.'}, 'finish_reason': 'stop'}]})
        requests.append(request)
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'test/model', 'context_length': 1000000}]})
        if request.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {'is_free_tier': True}})
        body = json.loads(request.content)
        completion = responses.pop(0)
        system = ' '.join(m.get('content') or '' for m in body['messages'] if m['role'] == 'system')
        current_record = json.loads(body['messages'][-1]['content'])
        step_id = current_record['step_id']
        content = completion.get('content') or ''
        completion['content'] = content or 'Requesting tools.'
        return httpx.Response(200, json={'model': 'test/model', 'choices': [{'message': completion, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15, 'cost': .01}})
    client = router.OpenRouterClient('test-key', transport=httpx.MockTransport(transport))
    registry = build_default_registry(workspace)
    sink = io.StringIO()
    renderer = cli.Renderer(cli.Style(False), sink, False)
    prompt = agent_module.build_system_prompt(str(workspace.root), project_instructions='PINNED PROJECT GUIDANCE')
    agent = agent_module.Agent(client, registry, 'test/model', system_prompt=prompt, on_event=lambda event: renderer.handle(event))
    session = cli.Session(agent, registry, client, renderer, workspace, 'test-key', client.base_url, None, 'test')
    frame = RuntimeFrame(session, cli)
    session.reloader = frame
    agent.on_boundary = lambda: frame.checkpoint(boundary=True)
    root = frame.root
    def edit(filename, before, after):
        path = root / ('prompts/' + filename if filename.endswith('.txt') else filename)
        source = path.read_text()
        assert before in source, (filename, before)
        path.write_text(source.replace(before, after))
    try:
__BODY__
    finally:
        await cli._shutdown(session)

asyncio.run(main())
'''


@pytest.fixture
def run_copy(tmp_path: Path):
    package_root = tmp_path / "source"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "src" / "slipagent", package_root / "slipagent",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    project = tmp_path / "project"
    shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", package_root / "slipagent" / "prompts")
    project.mkdir()

    def run(body: str) -> None:
        script = SETUP.replace("__BODY__", textwrap.indent(textwrap.dedent(body).strip(), "        "))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([GUARD_DIRECTORY, str(package_root)])}
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=project, env=env,
            capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run


def test_prompt_resource_reloads_and_rejects_invalid_template(run_copy):
    run_copy('''
        edit('system-prompt.txt', 'You are SlipAgent,', 'You are the updated SlipAgent,')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        accepted = agent.system_prompt
        assert 'You are the updated SlipAgent,' in accepted
        edit('system-prompt.txt', '{workspace}', '{unknown_placeholder}')
        await frame.checkpoint()
        assert frame.generation == 1 and agent.system_prompt == accepted
        try:
            cli.build_system_prompt(str(workspace.root))
        except KeyError:
            pass
        else:
            raise AssertionError('Invalid prompt placeholders must be reported')
        assert 'Reload rejected' in sink.getvalue()
        edit('system-prompt.txt', '{unknown_placeholder}', '{workspace}')
        edit('system-prompt.txt', 'You are the updated SlipAgent,', 'You are the recovered SlipAgent,')
        await frame.checkpoint()
        assert frame.generation == 2, sink.getvalue()
        assert 'You are the recovered SlipAgent,' in agent.system_prompt
        assert agent.messages[0].content == agent.system_prompt
    ''')


def test_initialized_guidance_survives_component_reload(run_copy):
    run_copy('''
        await cli._handle_command(session, '/init')
        guidance = (workspace.root / 'AGENTS.md').read_text()
        assert guidance in (await agent._context_view(registry.specs(), 1))[0].content
        edit('system-prompt.txt', 'You are SlipAgent,', 'You are the updated SlipAgent,')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert 'You are the updated SlipAgent,' in agent.system_prompt
        assert guidance in (await agent._context_view(registry.specs(), 1))[0].content
        assert agent.messages[0].content == agent.system_prompt
        agent.reset()
        assert guidance in (await agent._context_view(registry.specs(), 1))[0].content
    ''')


def test_file_checkpoints_and_cycle_guard_survive_behavior_reload(run_copy):
    run_copy('''
        from slipagent.checkpoints import current_checkpoint
        from slipagent.tools.files import _atomic_write
        path = workspace.root / 'file.txt'
        path.write_text('original')
        checkpoints = agent._checkpoints()
        checkpoints.begin(1)
        token = current_checkpoint.set(checkpoints)
        try:
            _atomic_write(workspace, path, 'first edit')
        finally:
            current_checkpoint.reset(token)
            checkpoints.finish()
        checkpoint_id = checkpoints.listing()[0]['id']
        guard = agent._loop_guard()
        guard.observe([(ToolCall('one', 'read_file', {'path': 'file.txt'}), ToolResult.ok('same'))], registry, 1)
        edit('checkpoints.py', 'Unknown or already restored file checkpoint.', 'Unknown or restored file checkpoint.')
        edit('progress.py', 'History is preserved.', 'History remains preserved.')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert agent._checkpoints() is checkpoints
        assert agent._loop_guard() is guard
        assert guard.recent
        checkpoints.rewind(checkpoint_id)
        assert path.read_text() == 'original'
        responses.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'write', 'type': 'function', 'function': {'name': 'write_file', 'arguments': '{"path":"file.txt","content":"second edit"}'}}]}, {'role': 'assistant', 'content': 'Done'}])
        assert await agent.run('edit file') == 'Done'
        assert len(checkpoints.listing()) == 1
        checkpoints.rewind(checkpoints.listing()[0]['id'])
        assert path.read_text() == 'original'
    ''')


def test_background_jobs_and_request_diagnostics_survive_reload_and_rejection(run_copy):
    run_copy('''
        import shlex
        import sys
        jobs = registry.services['command_jobs']
        diagnostics = registry.services['request_diagnostics']
        request = json.dumps({'model': 'original/model', 'messages': [{'role': 'user', 'content': 'original input'}]})
        attempt = diagnostics.begin(request, step=1, step_id=1)
        diagnostics.finish(attempt, 'request_error', response='partial failed response')
        command = shlex.join([sys.executable, '-c', 'import time; print("running", flush=True); time.sleep(60)'])
        result = await registry.invoke('run_command', {'command': command, 'background': True})
        key = result.content['job_id']
        edit('system-prompt.txt', 'You are SlipAgent,', 'You are the updated SlipAgent,')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert registry.services['command_jobs'] is jobs and jobs.active
        assert registry.services['request_diagnostics'] is diagnostics
        assert request in diagnostics.read(attempt)
        edit('system-prompt.txt', 'You are the updated SlipAgent,', 'You are SlipAgent,')
        (root / 'broken.py').write_text('invalid Python !!!')
        await frame.checkpoint()
        assert frame.generation == 1 and jobs.active
        stopped = await registry.invoke('command_jobs', {'action': 'stop', 'job_id': key})
        assert stopped.content['state'] == 'stopped'
        assert not jobs.active
        assert 'partial failed response' in diagnostics.read(attempt)
    ''')


def test_background_memory_service_and_task_tool_survive_reload_and_migrate(run_copy):
    run_copy('''
        # Simulate a live Agent created before the new service/tool existed.
        registry.services.pop('step_compactor')
        edit('system-prompt.txt', 'You are SlipAgent,', 'You are the updated SlipAgent,')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        responses.append({'role': 'assistant', 'content': 'finished'})
        assert await agent.run('Inspect the project.') == 'finished'
        await agent.wait_for_compaction()
        compactor = registry.services['step_compactor']
        assert agent.history.steps[0].summary == 'Completed demo step.'
        edit('background-summary-prompt.txt', 'Keep short summaries short', 'Keep brief summaries brief')
        await frame.checkpoint()
        assert frame.generation == 2, sink.getvalue()
        assert registry.services['step_compactor'] is compactor
        responses.append({'role': 'assistant', 'content': 'continued'})
        assert await agent.run('Continue.') == 'continued'
        await agent.wait_for_compaction()
        assert agent.history.steps[1].summary == 'Completed demo step.'
        agent.reset()
        assert not compactor.jobs
    ''')


def test_context_view_and_snapshot_survive_component_reload(run_copy):
    run_copy('''
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        with create_pipe_input() as pipe:
            terminal = cli.TerminalUI(lambda columns: 'status', sink, input=pipe, output=DummyOutput())
            renderer.terminal = terminal
            terminal.input.buffer.text = 'draft'
            responses.append({'role': 'assistant', 'content': 'finished'})
            assert await agent.run('Inspect this task.') == 'finished'
            blocks = terminal._context_blocks
            assert 'Inspect this task.' in ' '.join(text for style, text in blocks)
            assert 'PINNED PROJECT GUIDANCE' in ' '.join(text for style, text in blocks)
            terminal.context_visible = True
            context_window = terminal.context_window
            edit('cli.py', '"  current model: ', '"  updated model: ')
            await frame.checkpoint()
            assert frame.generation == 1, ''.join(terminal._raw_output)
            assert terminal.context_visible
            assert terminal.context_window is context_window
            assert terminal._context_blocks is blocks
            assert terminal.input.buffer.text == 'draft'
            renderer.terminal = None
    ''')


def test_reload_updates_behavior_preserving_live_session_and_class_identities(run_copy):
    run_copy('''
        from slipagent.tools.web import _TextExtractor
        old_agent_type = type(agent)
        old_error_type = router.OpenRouterAPIError
        history = agent.history
        registry_identity = id(registry)
        client_identity = id(client._client)
        session_identity = id(session)
        session.free_calls = 87
        session.quota_checked_at = 123
        session._catalog = []
        agent.usage = Usage(total_tokens=15, cost=.02)
        agent.history.digest = 'saved overview'
        session.extensions['component-data'] = {'counter': 7}
        responses.append({'role': 'assistant', 'content': 'first answer'})
        assert await agent.run('first question') == 'first answer'
        agent.pending.append('queued input')
        original_messages = list(agent.messages[1:])
        original_id = agent.session_id
        old_web_client = registry.get('fetch_page')._client
        edit('cli.py', '"  current model: ', '"  chosen model: ')
        edit('agent.py', 'text = record.text.strip()', 'text = "new behavior: " + record.text.strip()')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert type(agent) is old_agent_type
        assert router.OpenRouterAPIError is old_error_type
        assert id(session) == session_identity
        assert agent.history is history and history.digest == 'saved overview'
        assert agent.messages[1:] == original_messages
        assert agent.pending == ['queued input']
        assert agent.session_id == original_id
        assert id(registry) == registry_identity and id(client._client) == client_identity
        assert session.free_calls == 87 and session.quota_checked_at == 123 and session._catalog == []
        assert session.extensions['component-data'] == {'counter': 7}
        assert old_web_client.is_closed
        assert registry.get('recall_history').history is history
        await cli._handle_command(session, '/model')
        assert 'chosen model: test/model' in sink.getvalue()
        responses.append({'role': 'assistant', 'content': 'second answer'})
        assert await agent.run('second question') == 'new behavior: second answer'
        # Patched zero-argument super() and imported exception aliases still work.
        error = router.OpenRouterAPIError('failure', 400)
        assert str(error) == 'failure' and isinstance(error, old_error_type)
        extractor = _TextExtractor()
        extractor.feed('<p>sample</p>')
        assert 'sample' in extractor.text()
        # A second generation must also update existing classes and prompt suffixes.
        edit('agent.py', 'new behavior: ', 'third version: ')
        await frame.checkpoint()
        assert frame.generation == 2, sink.getvalue()
        assert agent.system_prompt.count('PINNED PROJECT GUIDANCE') == 1
        assert agent.usage.total_tokens == 45
    ''')


@pytest.mark.parametrize("bad_source", ["this is not valid Python !!!", "import missing_reload_dependency"])
def test_invalid_edit_keeps_previous_version_and_recovers_on_next_edit(run_copy, bad_source):
    run_copy(f'''
        path = root / 'cli.py'
        original = path.read_text()
        path.write_text(original + '\\n' + {bad_source!r} + '\\n')
        await frame.checkpoint()
        assert frame.generation == 0
        assert 'previous code remains active' in sink.getvalue()
        context = await agent._context_view(registry.specs(), 1)
        assert 'Reload rejected' in context[0].content
        assert 'previous code remains active' in context[0].content
        await cli._handle_command(session, '/model')
        assert 'current model: test/model' in sink.getvalue()
        messages = sink.getvalue().count('Reload rejected')
        await frame.checkpoint()
        assert sink.getvalue().count('Reload rejected') == messages
        path.write_text(original.replace('"  current model: ', '"  changed model: '))
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        await cli._handle_command(session, '/model')
        assert 'changed model: test/model' in sink.getvalue()
    ''')


def test_danger_mode_survives_component_reload(run_copy):
    run_copy('''
        await cli._handle_command(session, '/danger')
        edit('cli.py', '"  current model: ', '"  updated model: ')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert workspace.access.danger
        outside = Path.cwd().parent / 'outside-reload.txt'
        outside.write_text('external fixture')
        result = await registry.invoke('read_file', {'path': str(outside)})
        assert not result.is_error and 'external fixture' in result.content, result.content
        await cli._handle_command(session, '/danger off')
        assert not workspace.access.danger
        assert (await registry.invoke('read_file', {'path': str(outside)})).is_error
    ''')


def test_hashes_detect_same_size_same_timestamp_edits_and_manual_reload(run_copy):
    run_copy('''
        path = root / 'cli.py'
        timestamp = path.stat().st_mtime_ns
        size = path.stat().st_size
        edit('cli.py', '"  current model: ', '"  updated model: ')
        assert path.stat().st_size == size
        os.utime(path, ns=(timestamp, timestamp))
        await cli._handle_command(session, '/reload')
        assert frame.generation == 2, sink.getvalue()  # pre-command edit + forced reload
        await cli._handle_command(session, '/model')
        assert 'updated model: test/model' in sink.getvalue()
        await cli._handle_command(session, '/reload')
        assert frame.generation == 3
    ''')


def test_search_worker_uses_active_snapshot_after_rejected_source_edit(run_copy):
    run_copy('''
        Path('sample.txt').write_text('needle in a haystack')
        path = root / 'tools' / 'grep.py'
        original = path.read_text()
        path.write_text('invalid Python !!!')
        await frame.checkpoint()
        assert frame.generation == 0
        result = await registry.invoke('grep', {'pattern': 'needle'})
        assert not result.is_error and 'sample.txt:1:' in result.content, result.content
        path.write_text(original.replace('matches.append([number, shown])', 'matches.append([number, "UPDATED " + shown])'))
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        result = await registry.invoke('grep', {'pattern': 'needle'})
        assert 'UPDATED needle' in result.content, result.content
    ''')


def test_edit_waits_for_complete_tool_batch_and_next_step_uses_new_behavior(run_copy):
    run_copy('''
        started = asyncio.Event()
        release = asyncio.Event()
        class WaitTool(Tool):
            name = 'wait'
            description = 'wait'
            async def run(self):
                started.set()
                await release.wait()
                return ToolResult.ok('wait complete')
        remote = WaitTool()
        registry.register(remote)
        responses.extend([
            {'role': 'assistant', 'content': 'starting', 'tool_calls': [
                {'id': 'one', 'type': 'function', 'function': {'name': 'wait', 'arguments': '{}'}},
                {'id': 'two', 'type': 'function', 'function': {'name': 'list_dir', 'arguments': '{}'}},
            ]},
            {'role': 'assistant', 'content': 'finished'},
        ])
        task = asyncio.create_task(agent.run('run both'))
        await asyncio.wait_for(started.wait(), 3)
        edit('agent.py', 'text = record.text.strip()', 'text = "reloaded: " + record.text.strip()')
        edit('tools/navigate.py', 'is empty.', 'EMPTY AFTER RELOAD.')
        await frame.checkpoint()
        await cli._handle_command(session, '/reload')
        assert frame.generation == 0
        release.set()
        assert await task == 'reloaded: finished', sink.getvalue()
        assert frame.generation == 1
        assert registry.get('wait') is remote
        roles = [message.role for message in agent.messages]
        assert roles == ['system', 'user', 'assistant', 'tool', 'tool', 'assistant']
        assert agent.messages[3].content['content'] == 'wait complete'
        assert agent.messages[4].tool_call_id == 'two'
        assert 'is empty.' in agent.messages[4].content['content']
        assert 'EMPTY AFTER RELOAD.' in (await registry.invoke('list_dir', {})).content
        assert len([r for r in requests if r.method == 'POST']) == 2
    ''')


def test_stop_with_pending_reload_does_not_take_another_model_turn(run_copy):
    run_copy('''
        started = asyncio.Event()
        release = asyncio.Event()
        class WaitTool(Tool):
            name = 'wait'
            description = 'wait'
            async def run(self):
                started.set()
                await release.wait()
                return ToolResult.ok('saved result')
        registry.register(WaitTool())
        responses.append({'role': 'assistant', 'content': 'starting', 'tool_calls': [
            {'id': 'one', 'type': 'function', 'function': {'name': 'wait', 'arguments': '{}'}},
        ]})
        task = asyncio.create_task(agent.run('start'))
        await asyncio.wait_for(started.wait(), 3)
        edit('cli.py', '"  current model: ', '"  updated model: ')
        await cli._handle_command(session, '/reload')
        await cli._handle_command(session, '/stop')
        release.set()
        await task
        assert agent.stopped and not agent.running
        assert agent.messages[-1].content['content'] == 'saved result'
        assert frame.generation == 1, sink.getvalue()
        assert len([r for r in requests if r.method == 'POST']) == 1
    ''')


def test_stop_during_final_reload_prevents_queued_followup(run_copy):
    run_copy('''
        closing = asyncio.Event()
        release = asyncio.Event()
        old_tool = registry.get('fetch_page')
        original_close = old_tool.aclose
        async def delayed_close():
            closing.set()
            await release.wait()
            await original_close()
        old_tool.aclose = delayed_close
        original_handle = renderer.handle
        def handle(event):
            original_handle(event)
            if event.kind == 'assistant_text':
                edit('cli.py', '"  current model: ', '"  updated model: ')
                agent.enqueue('queued followup')
        renderer.handle = handle
        responses.append({'role': 'assistant', 'content': 'done'})
        task = asyncio.create_task(agent.run('start'))
        await asyncio.wait_for(closing.wait(), 3)
        assert agent.running
        await cli._handle_command(session, '/stop')
        release.set()
        assert await task == 'done'
        assert agent.stopped and not agent.running
        assert agent.pending == ['queued followup']
        assert len([r for r in requests if r.method == 'POST']) == 1
    ''')


def test_cli_module_entry_supports_reload_and_disabled_mode(run_copy):
    run_copy('''
        import subprocess
        import sys
        import textwrap
        script = textwrap.dedent("""
        import runpy
        import sys
        import slipagent.openrouter as router
        from slipagent.capabilities import ModelCapabilities
        from slipagent.config import DEFAULT_MODEL
        from slipagent.types import KeyInfo, ModelInfo
        async def quota(self):
            return KeyInfo.from_api({'is_free_tier': True})
        async def models(self, **kwargs):
            return [ModelInfo(DEFAULT_MODEL, context_length=1000000)]
        async def capabilities(self, model, **kwargs):
            return ModelCapabilities({'id': model, 'context_length': 1000000}, [{
                'tag': 'stub', 'supported_parameters': ['tools'], 'context_length': 1000000,
            }])
        router.OpenRouterClient.key_info = quota
        router.OpenRouterClient.list_models = models
        router.OpenRouterClient.model_capabilities = capabilities
        sys.argv = ['slipagent', '--api-key', 'test', '--no-mcp', '--no-color'] + sys.argv[1:]
        runpy.run_module('slipagent.cli', run_name='__main__', alter_sys=True)
        """)
        env = {**os.environ, 'SLIPAGENT_NO_DOTENV': '1'}
        result = subprocess.run([sys.executable, '-c', script], input='/reload\\n/model\\n/quit\\n', text=True, capture_output=True, env=env, timeout=10)
        assert result.returncode == 0, result.stderr
        assert 'Applied component reload 1' in result.stderr, result.stderr
        assert 'current model:' in result.stderr
        result = subprocess.run([sys.executable, '-c', script, '--no-reload'], input='/reload\\n/quit\\n', text=True, capture_output=True, env=env, timeout=10)
        assert result.returncode == 0, result.stderr
        assert 'live reload is disabled' in result.stderr
        assert 'Applied component reload' not in result.stderr
    ''')


def test_idle_watcher_applies_edits_without_user_input(run_copy):
    run_copy('''
        watcher = asyncio.create_task(frame.watch())
        try:
            edit('cli.py', '"  current model: ', '"  updated model: ')
            async with asyncio.timeout(4):
                while frame.generation == 0:
                    await asyncio.sleep(.05)
            await cli._handle_command(session, '/model')
            assert 'updated model: test/model' in sink.getvalue()
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
    ''')


def test_frame_and_state_layout_edits_require_restart(run_copy):
    run_copy('''
        path = root / 'agent.py'
        original = path.read_text()
        edit('agent.py', '    model: str', '    model: str\\n    extra_state: str = "new"')
        await frame.checkpoint()
        assert frame.generation == 0
        assert 'state layout changed for Agent' in sink.getvalue()
        path.write_text(original)
        edit('runtime.py', 'POLL_INTERVAL = .5', 'POLL_INTERVAL = .6')
        await frame.checkpoint()
        assert frame.generation == 0
        assert 'Restart required for frame/contract edits: runtime.py' in sink.getvalue()
        await cli._handle_command(session, '/model')
        assert 'current model: test/model' in sink.getvalue()
    ''')


def test_terminal_rebuild_preserves_draft_history_scroll_and_input_queue(run_copy):
    run_copy('''
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.output import DummyOutput
        from slipagent.terminal import TerminalUI
        with create_pipe_input() as pipe:
            terminal = TerminalUI(lambda columns: 'status', sink, input=pipe, output=DummyOutput())
            renderer.terminal = terminal
            terminal.input.buffer.text = 'draft text'
            terminal._input_label = 'you › '
            terminal.input.buffer.cursor_position = 4
            terminal.input.buffer.history.append_string('previous task')
            terminal.write('stored transcript')
            terminal.transcript._following = False
            terminal.transcript.vertical_scroll = 3
            terminal.transcript._pending_scroll = -2
            terminal._lines.put_nowait('queued prompt')
            app = terminal.app
            input_buffer = terminal.input.buffer
            transcript = terminal.transcript
            edit('terminal.py', '"Ready"', '"Live Ready"')
            await frame.checkpoint()
            assert frame.generation == 1, sink.getvalue()
            assert terminal.app is app
            assert terminal.input.buffer is input_buffer and terminal.transcript is transcript
            assert input_buffer.text == 'draft text' and input_buffer.cursor_position == 4
            assert input_buffer.history.get_strings() == ['previous task']
            assert terminal._raw_output[0] == 'stored transcript'
            assert transcript.vertical_scroll == 3 and not transcript._following
            assert transcript._pending_scroll == -2
            assert await terminal.read_line() == 'queued prompt'
            assert terminal._activity()[0][1] == 'Live Ready'
            assert terminal._input_label == '> '
            # Callbacks installed before the reload now dispatch to new methods.
            assert any(
                'Live Ready' in str(window.content.create_content(80, 1).get_line(0))
                for window in app.layout.find_all_windows()
                if isinstance(window.content, FormattedTextControl)
            )
            renderer.terminal = None
    ''')


def test_failed_terminal_build_rolls_back_entire_candidate(run_copy):
    run_copy('''
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from slipagent.terminal import TerminalUI
        with create_pipe_input() as pipe:
            terminal = TerminalUI(lambda columns: 'status', sink, input=pipe, output=DummyOutput())
            renderer.terminal = terminal
            original_tool = registry.get('fetch_page')
            original_prompt = agent.system_prompt
            original_layout = terminal.app.layout
            original_bindings = terminal.app.key_bindings
            original_style = terminal.app.style
            original_height = terminal.input.window.height
            edit('cli.py', '"  current model: ', '"  updated model: ')
            edit('terminal.py', '    def _bindings(self) -> KeyBindings:', '    def _bindings(self) -> KeyBindings:\\n        self.input.window.height = Dimension.exact(1)\\n        raise ValueError("broken layout")')
            await frame.checkpoint()
            assert frame.generation == 0
            assert 'broken layout' in ''.join(terminal._raw_output)
            assert registry.get('fetch_page') is original_tool
            assert not original_tool._client.is_closed
            assert agent.system_prompt == original_prompt
            assert terminal.app.layout is original_layout
            assert terminal.app.key_bindings is original_bindings
            assert terminal.app.style is original_style
            assert terminal.input.window.height is original_height
            renderer.terminal = None
            await cli._handle_command(session, '/model')
            assert 'current model: test/model' in sink.getvalue()
            assert terminal._activity()[0][1] == 'Ready'
            renderer.terminal = None
    ''')


def test_new_tool_modules_reload_and_close_with_their_own_helpers(run_copy):
    run_copy('''
        import textwrap
        (root / 'cleanup.py').write_text('def close(tool):\\n    tool.closed = True\\n')
        (root / 'tools' / 'extra.py').write_text(textwrap.dedent("""
        from .base import Tool, ToolResult
        class ExtraTool(Tool):
            name = 'extra'
            description = 'new component'
            def __init__(self):
                self.closed = False
            async def run(self):
                return ToolResult.ok('new tool')
            async def aclose(self):
                if self.closed:
                    return
                from ..cleanup import close
                close(self)
        """))
        edit('tools/__init__.py', 'from .base import Tool,', 'from .extra import ExtraTool\\nfrom .base import Tool,')
        edit('tools/__init__.py', '*file_tools(workspace),', 'ExtraTool(),\\n            *file_tools(workspace),')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        old_extra = registry.get('extra')
        assert (await registry.invoke('extra', {})).content == 'new tool'
        assert not old_extra.closed
        frame.request()
        await frame.checkpoint()
        assert frame.generation == 2, sink.getvalue()
        assert old_extra.closed
        assert 'Could not close' not in sink.getvalue()
        new_extra = registry.get('extra')
        await cli._shutdown(session)
        assert new_extra.closed
    ''')


def test_rebuilt_tools_can_change_resource_fields_without_leaking_old_clients(run_copy):
    run_copy('''
        old_fetch = registry.get('fetch_page')._client
        old_search = registry.get('web_search')._client
        edit('tools/web.py', 'self._client', 'self._http')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert old_fetch.is_closed and old_search.is_closed
        new_fetch = registry.get('fetch_page')._http
        assert not new_fetch.is_closed
        await cli._shutdown(session)
        assert new_fetch.is_closed
    ''')


def test_running_terminal_accepts_input_after_component_reload(run_copy):
    run_copy('''
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from slipagent.terminal import TerminalUI
        with create_pipe_input() as pipe:
            terminal = TerminalUI(lambda columns: 'status', sink, input=pipe, output=DummyOutput())
            renderer.terminal = terminal
            app = terminal.app
            task = asyncio.create_task(terminal.run())
            try:
                async with asyncio.timeout(3):
                    while not app.is_running:
                        await asyncio.sleep(.02)
                pipe.send_text('draft')
                async with asyncio.timeout(3):
                    while terminal.input.buffer.text != 'draft':
                        await asyncio.sleep(.02)
                edit('terminal.py', 'self._lines.put_nowait(buffer.text)', 'self._lines.put_nowait("live:" + buffer.text)')
                await frame.checkpoint()
                assert frame.generation == 1, ''.join(terminal._raw_output)
                assert terminal.app is app and app.is_running
                assert terminal.input.buffer.text == 'draft'
                pipe.send_text('\\r')
                assert await asyncio.wait_for(terminal.read_line(), 3) == 'live:draft'
            finally:
                terminal.close()
                await task
                renderer.terminal = None
    ''')


def test_keyboard_stays_responsive_during_threaded_staging_and_busy_commit_defers(run_copy):
    run_copy('''
        import threading
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from slipagent.terminal import TerminalUI
        started = threading.Event()
        release = threading.Event()
        main_thread = threading.get_ident()
        stage = frame._stage
        def paused_stage(sources, prefix):
            assert threading.get_ident() != main_thread
            started.set()
            assert release.wait(3)
            return stage(sources, prefix)
        frame._stage = paused_stage
        with create_pipe_input() as pipe:
            terminal = TerminalUI(lambda columns: 'status', sink, input=pipe, output=DummyOutput())
            renderer.terminal = terminal
            app_task = asyncio.create_task(terminal.run())
            reload_task = None
            try:
                async with asyncio.timeout(3):
                    while not terminal.app.is_running:
                        await asyncio.sleep(.01)
                edit('cli.py', '"  current model: ', '"  threaded model: ')
                reload_task = asyncio.create_task(frame.checkpoint())
                assert await asyncio.to_thread(started.wait, 3)
                pipe.send_text('typing during reload')
                async with asyncio.timeout(2):
                    while terminal.input.buffer.text != 'typing during reload':
                        await asyncio.sleep(.01)
                assert not reload_task.done()
                frame.busy += 1
                release.set()
                await reload_task
                assert frame.generation == 0
                frame.busy -= 1
                await frame.checkpoint()
                assert frame.generation == 1, sink.getvalue()
                assert terminal.input.buffer.text == 'typing during reload'
            finally:
                release.set()
                if reload_task is not None:
                    await asyncio.gather(reload_task, return_exceptions=True)
                terminal.close()
                await app_task
                renderer.terminal = None
    ''')


def test_cancelled_threaded_staging_drains_before_namespace_cleanup(run_copy):
    run_copy('''
        import threading
        import sys
        started = threading.Event()
        release = threading.Event()
        prefixes = []
        stage = frame._stage
        def paused_stage(sources, prefix):
            prefixes.append(prefix)
            modules = stage(sources, prefix)
            started.set()
            assert release.wait(3)
            return modules
        frame._stage = paused_stage
        edit('cli.py', '"  current model: ', '"  cancelled model: ')
        task = asyncio.create_task(frame.checkpoint())
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            await asyncio.sleep(.02)
            assert not task.done()
            release.set()
            result = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError)
            assert frame.generation == 0 and not frame.reloading
            assert all(not name.startswith(prefixes[0]) for name in sys.modules)
            assert all(getattr(loader, 'prefix', None) != prefixes[0] for loader in sys.meta_path)
            await frame.checkpoint()
            assert frame.generation == 1, sink.getvalue()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    ''')


def test_missing_component_entry_rejects_candidate_before_commit(run_copy):
    run_copy('''
        edit('agent.py', 'async def _step(', 'async def _renamed_step(')
        await frame.checkpoint()
        assert frame.generation == 0
        assert 'Missing component API in agent.Agent' in sink.getvalue()
        responses.append({'role': 'assistant', 'content': 'still working'})
        assert await agent.run('question') == 'still working'
    ''')


def test_renderer_failure_does_not_stop_frame_reporting_or_watching(run_copy):
    run_copy('''
        original = (root / 'cli.py').read_text()
        watcher = asyncio.create_task(frame.watch())
        try:
            edit('cli.py', '        if separate and strip_sequences(text).strip():', '        raise ValueError("broken renderer")\\n        if separate and strip_sequences(text).strip():')
            async with asyncio.timeout(4):
                while frame.generation < 1:
                    await asyncio.sleep(.05)
            assert not watcher.done()
            frame.report_error('frame remains alive')
            assert 'frame remains alive' in sink.getvalue()
            (root / 'cli.py').write_text(original)
            async with asyncio.timeout(4):
                while frame.generation < 2:
                    await asyncio.sleep(.05)
            await cli._handle_command(session, '/model')
            assert 'current model: test/model' in sink.getvalue()
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
    ''')


@pytest.mark.parametrize("filename", ["extra.py", "cli.py"])
def test_new_component_classes_keep_state_and_receive_future_edits(run_copy, filename):
    run_copy(f'''
        import textwrap
        import sys
        path = root / {filename!r}
        original = path.read_text() if path.exists() else ''
        path.write_text(original + textwrap.dedent("""

        class Counter:
            def __init__(self):
                self.value = 0
            def next(self):
                self.value += 1
                return self.value
        """))
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        module = sys.modules[frame.namespace + '.' + path.stem]
        counter = module.Counter()
        session.extensions['counter'] = counter
        assert counter.next() == 1
        edit({filename!r}, 'self.value += 1', 'self.value += 2')
        path.write_text(path.read_text() + '\\nclass Child(Counter):\\n    pass\\n')
        await frame.checkpoint()
        assert frame.generation == 2, sink.getvalue()
        assert session.extensions['counter'] is counter
        assert counter.next() == 3
        module = sys.modules[frame.namespace + '.' + path.stem]
        child = module.Child()
        assert child.next() == 2
        edit({filename!r}, 'self.value += 2', 'self.value += 3')
        await frame.checkpoint()
        assert frame.generation == 3, sink.getvalue()
        assert counter.next() == 6 and child.next() == 5
    ''')


def test_task_environment_and_command_logs_survive_reload_and_rejection(run_copy):
    run_copy('''
        import shlex
        import sys
        environment = registry.services['project_environment']
        environment.python_override = sys.executable
        archive = registry.services['command_archive']
        command = shlex.join([sys.executable, '-c', "print('retained output')"])
        result = await registry.invoke('run_command', {'command': command})
        assert not result.is_error, result.content
        log = next(iter(archive.logs.values()))
        responses.append({'role': 'assistant', 'content': 'task progress'})
        await agent.run('Keep the database intact')
        task = agent.history.task
        sources = dict(task.sources)
        original = (root / 'agent.py').read_text()
        (root / 'agent.py').write_text(original + '\\n# compatible edit\\n')
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        assert registry.services['project_environment'] is environment
        assert environment.python_override == sys.executable
        assert registry.services['command_archive'] is archive
        assert agent.history.task is task and task.sources == sources
        page = await registry.invoke('read_command_output', {'log_id': log.id})
        assert page.content['content'] == 'retained output\\n'
        (root / 'agent.py').write_text('invalid Python !!!')
        await frame.checkpoint()
        assert frame.generation == 1
        page = await registry.invoke('read_command_output', {'log_id': log.id})
        assert not page.is_error and 'retained output' in page.content['content']
        agent.reset()
        assert not archive.logs and not agent.history.task.sources
    ''')
