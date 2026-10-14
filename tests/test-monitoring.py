"""Command subscriptions deliver bounded evidence in real request prompts."""

import asyncio
import json
import shlex
import sys
from importlib import import_module

import pytest

from slipagent.agent import Agent
from slipagent.artifacts import ArtifactStore, ReadArtifactTool
from slipagent.jobs import CommandJobs, CommandJobsTool
from slipagent.monitoring import MonitorCommandTool, REPLAY_BYTES
from slipagent.tools import build_default_registry
from slipagent.tools.base import ToolRegistry, ToolResult, current_invocation
from slipagent.tools.discovery import tool_catalog
from slipagent.tools.output import CommandArchive, ReadCommandOutputTool
from slipagent.tools.shell import RunCommandTool
from slipagent.types import ToolCall, content_text
from test_agent import StubClient, completion

run_copy = import_module('test-runtime').run_copy


@pytest.fixture
async def monitor_service():
    archive, jobs = CommandArchive(1_000_000), CommandJobs()
    archive.artifacts = ArtifactStore(1_000_000)
    gate = asyncio.Event()
    async def run():
        await gate.wait()
        return ToolResult.ok('done')
    log = archive.start('disposable test job', step_id=3, call_id='launch')
    job = jobs.start(log, run)
    try:
        yield archive, jobs, job, gate, MonitorCommandTool(jobs)
    finally:
        await jobs.aclose()
        await archive.artifacts.aclose()
        await archive.aclose()


async def start(tool, job, **kwargs):
    token = current_invocation.set((7, 'watch'))
    try:
        response = await tool.run('start', job_id=job.log.id, **kwargs)
    finally:
        current_invocation.reset(token)
    assert not response.is_error, response.content
    return response.content['monitor_id']


async def test_split_match_delivers_result_and_origin_without_stopping_job(monitor_service):
    archive, jobs, job, gate, tool = monitor_service
    key = await start(tool, job, pattern='ready')
    job.log.write('stdout', 'server rea')
    assert jobs.monitors.monitors[key].state == 'active'
    job.log.write('stdout', 'dy on port 1234\n')
    record = jobs.monitors.monitors[key].snapshot()
    assert record['initialized_turn'] == 7 and record['initializing_call_id'] == 'watch'
    assert record['state'] == 'ready' and record['result']['reason'] == 'matched'
    assert record['result']['last_match']['text'].endswith('ready on port 1234\n')
    assert record['result']['output']['stdout']['text'] == 'server ready on port 1234\n'
    assert job.state == 'running' and not job.task.done()
    saved = await ReadArtifactTool(archive.artifacts).run(record_id=record['artifact']['record_id'])
    assert 'server ready on port 1234' in content_text(saved.content)


async def test_repeated_matches_coalesce_notices_but_retain_raw_output(monitor_service):
    archive, jobs, job, gate, tool = monitor_service
    notices = []
    jobs.on_completion = lambda: notices.extend(jobs.monitors.notices())
    key = await start(tool, job, pattern='ERR', repeat=True)
    job.log.write('stderr', 'ERR first\n')
    job.log.write('stderr', 'ERR second\nERR third\n')
    assert len(notices) == 1
    gate.set()
    await job.task
    await asyncio.sleep(0)
    assert len(notices) == 2
    result = jobs.monitors.monitors[key].snapshot()['result']
    assert result['matches'] == 3 and result['suppressed_notifications'] == 2
    assert result['reason'] == 'job_finished'
    output = await ReadCommandOutputTool(archive).run(log_id=job.log.id, stream='stderr')
    assert output.content['content'] == 'ERR first\nERR second\nERR third\n'


async def test_match_count_is_independent_of_chunk_boundaries(monitor_service):
    _, jobs, job, _, tool = monitor_service
    key = await start(tool, job, pattern='aa', repeat=True)
    for chunk in ['aa', 'a', 'aaa']:
        job.log.write('stdout', chunk)
    assert jobs.monitors.monitors[key].matches == 3


async def test_stream_filter_does_not_join_stdout_and_stderr(monitor_service):
    _, jobs, job, _, tool = monitor_service
    key = await start(tool, job, pattern='ready', stream='stderr')
    job.log.write('stdout', 'ready')
    job.log.write('stderr', 'rea')
    job.log.write('stdout', 'dy')
    assert jobs.monitors.monitors[key].matches == 0
    job.log.write('stderr', 'dy')
    assert jobs.monitors.monitors[key].matches == 1


async def test_fast_finished_job_replays_output_and_exit_status(monitor_service):
    _, jobs, job, gate, tool = monitor_service
    job.log.write('stdout', 'passed\n')
    job.log.returncode = 0
    gate.set()
    await job.task
    key = await start(tool, job)
    result = jobs.monitors.monitors[key].snapshot()['result']
    assert result['reason'] == 'job_finished' and result['job_state'] == 'completed'
    assert result['returncode'] == 0 and result['output']['stdout']['text'] == 'passed\n'
    assert result['job_result'] == {'is_error': False, 'text': 'done', 'omitted_characters': 0}


async def test_failed_job_delivers_error_even_without_process_output(monitor_service):
    archive, jobs, _, _, tool = monitor_service
    async def fail():
        return ToolResult.error('Could not start command: executable unavailable')
    job = jobs.start(archive.start('missing test command'), fail)
    key = await start(tool, job)
    await job.task
    await asyncio.sleep(0)
    result = jobs.monitors.monitors[key].snapshot()['result']
    assert result['job_state'] == 'failed'
    assert result['job_result']['is_error']
    assert result['job_result']['text'] == 'Could not start command: executable unavailable'


async def test_replay_and_prompt_truncation_are_explicit(monitor_service):
    _, jobs, job, _, tool = monitor_service
    job.log.write('stdout', 'x' * (REPLAY_BYTES + 100) + 'ready')
    key = await start(tool, job, pattern='ready')
    record = jobs.monitors.monitors[key].snapshot()
    assert record['replay_omitted_bytes']['stdout'] == 105
    output = record['result']['output']['stdout']
    assert len(output['text']) == 1000 and output['text'].endswith('ready')
    assert output['omitted_characters'] == REPLAY_BYTES - 1000


async def test_quota_loss_is_reported_without_suppressing_live_match(monitor_service):
    archive, jobs, job, _, tool = monitor_service
    archive.quota_bytes = 2
    key = await start(tool, job, pattern='ready')
    job.log.write('stdout', 'ready')
    result = jobs.monitors.monitors[key].snapshot()['result']
    assert result['last_match']['text'] == 'ready'
    assert result['retention']['stdout']['lost_bytes'] == 3
    assert result['retention']['stdout']['retention_error']


async def test_wait_cancel_acknowledge_and_timeout_do_not_stop_process(monitor_service):
    _, jobs, job, _, tool = monitor_service
    key = await start(tool, job, pattern='later', timeout=.1)
    waiter = asyncio.create_task(tool.run('wait', monitor_id=key))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert jobs.monitors.monitors[key].state == 'active'
    assert (await tool.run('acknowledge', monitor_id=key)).is_error
    result = await tool.run('wait', monitor_id=key, timeout=1)
    assert result.content['result']['reason'] == 'timed_out'
    assert job.state == 'running' and not job.task.done()
    assert not (await tool.run('acknowledge', monitor_id=key)).is_error
    assert not jobs.monitors.snapshots()
    key = await start(tool, job)
    assert (await tool.run('cancel', monitor_id=key)).content['result']['reason'] == 'cancelled'
    assert job.state == 'running'


async def test_capacity_rejects_without_dropping_results(monitor_service):
    _, jobs, job, _, tool = monitor_service
    for _ in range(8):
        key = await start(tool, job)
        await tool.run('cancel', monitor_id=key)
    response = await tool.run('start', job_id=job.log.id)
    assert response.is_error and len(jobs.monitors.snapshots()) == 8
    await tool.run('acknowledge', monitor_id=key)
    await start(tool, job)


@pytest.mark.parametrize('arguments', [{'pattern': ''}, {'pattern': 'x' * 257}, {'repeat': True}, {'job_id': 'missing'}])
async def test_invalid_initialization_returns_error(monitor_service, arguments):
    _, _, job, _, tool = monitor_service
    result = await tool.run('start', **{'job_id': job.log.id, **arguments})
    assert result.is_error


async def test_prompt_lifecycle_delivers_result_and_hides_when_unavailable(monitor_service):
    _, jobs, job, _, tool = monitor_service
    registry = ToolRegistry([tool], services={'command_jobs': jobs})
    agent = Agent(StubClient([completion('done')]), registry, 'test', system_prompt='Test')
    async def system(specs=None):
        context = await agent._context_view(registry.specs() if specs is None else specs, 1)
        return content_text(context[0].content)
    assert 'Command Monitoring Setup' in await system()
    key = await start(tool, job, pattern='ready')
    active = await system()
    assert 'Active Command Monitors' in active and 'Command Monitoring Setup' not in active
    job.log.write('stdout', 'ready with actual result')
    ready = await system()
    assert 'Command Monitor Results Ready' in ready
    assert 'ready with actual result' in ready and 'initialized_turn' in ready and '7' in ready
    assert 'Active Command Monitors' not in ready
    assert 'Command Monitor' not in await system([])
    await tool.run('acknowledge', monitor_id=key)
    assert 'Command Monitoring Setup' in await system()
    await agent._compactor().aclose()


async def test_real_process_result_enters_next_request_without_extra_read(workspace):
    registry = build_default_registry(workspace)
    client = StubClient([completion('done')])
    agent = Agent(client, registry, 'test', system_prompt='Test')
    try:
        result = await registry.get('run_command').run(shlex.join([sys.executable, '-c', "print('tests passed')"]), background=True)
        tool = registry.get('monitor_command')
        token = current_invocation.set((4, 'subscribe'))
        try:
            monitor = await tool.run('start', job_id=result.content['job_id'])
        finally:
            current_invocation.reset(token)
        await CommandJobsTool(registry.resources.jobs).run('wait', result.content['job_id'])
        await asyncio.sleep(0)
        await agent.run('Report the result')
        text = content_text(client.calls[-1]['messages'][0].content)
        assert 'tests passed' in text and 'Command Monitor Results Ready' in text
        assert monitor.content['monitor_id'] in text and 'initialized_turn' in text
        assert len(client.calls) == 1
    finally:
        await registry.aclose()


async def test_reset_cancels_monitor_timers(monitor_service):
    _, jobs, job, _, tool = monitor_service
    key = await start(tool, job, pattern='x', repeat=True)
    job.log.write('stdout', 'x')
    job.log.write('stdout', 'x')
    monitor = jobs.monitors.monitors[key]
    timer, flush = monitor.timer, monitor.flush_timer
    await jobs.stop_all()
    jobs.clear()
    assert timer.cancelled() and flush.cancelled()
    assert not jobs.monitors.snapshots() and 'monitor_output' not in job.log.__dict__


async def test_rate_limited_notice_flushes_without_further_output(monitor_service, monkeypatch):
    import slipagent.monitoring as monitoring
    monkeypatch.setattr(monitoring, 'NOTICE_SECONDS', .01)
    _, jobs, job, _, tool = monitor_service
    delivered = []
    flushed = asyncio.Event()
    def notify():
        delivered.extend(jobs.monitors.notices())
        if len(delivered) == 2:
            flushed.set()
    jobs.on_completion = notify
    await start(tool, job, pattern='x', repeat=True)
    job.log.write('stdout', 'x')
    job.log.write('stdout', 'x')
    await asyncio.wait_for(flushed.wait(), 1)
    assert delivered[-1]['matches'] == 2
    assert delivered[-1]['suppressed_notifications'] == 1
    assert job.state == 'running'


async def test_result_storage_failure_still_delivers_inline_evidence(monitor_service):
    archive, jobs, job, _, tool = monitor_service
    archive.artifacts.quota_bytes = 1
    key = await start(tool, job, pattern='ready')
    job.log.write('stdout', 'ready and available')
    result = jobs.monitors.monitors[key].snapshot()
    assert result['storage_error'] and 'artifact' not in result
    assert result['result']['output']['stdout']['text'] == 'ready and available'


async def test_deferred_setup_obeys_allowlist_and_preview_does_not_consume(workspace):
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([]), registry, 'test', system_prompt='Test')
    try:
        context = await agent._context_view(registry.specs(), 1)
        assert 'Command Monitoring Setup' in content_text(context[0].content)
        catalog = tool_catalog(registry)
        assert 'monitor_command' not in catalog.loaded
        agent.exposed_tools = ('run_command',)
        context = await agent._context_view(registry.specs(), 1)
        assert 'Command Monitoring Setup' not in content_text(context[0].content)
        assert 'monitor_command' not in content_text(context[0].content)
    finally:
        await registry.aclose()


async def test_actual_tool_dispatch_records_initialization_turn_and_delivers_result(monitor_service):
    _, jobs, job, gate, tool = monitor_service
    job.log.write('stdout', 'completed result delivered inline')
    gate.set()
    await job.task
    client = StubClient([completion(tool_calls=[ToolCall('watch-call', 'monitor_command',
                            {'action': 'start', 'job_id': job.log.id})]), completion('done')])
    registry = ToolRegistry([tool], services={'command_jobs': jobs})
    agent = Agent(client, registry, 'test', system_prompt='Test')
    try:
        await agent.run('Watch this job')
        monitor = next(iter(jobs.monitors.monitors.values()))
        assert monitor.step_id == 1 and monitor.call_id == 'watch-call'
        request = content_text(client.calls[1]['messages'][0].content)
        assert 'Command Monitor Results Ready' in request and 'completed result delivered inline' in request
        before = monitor.snapshot()
        await agent._context_view(registry.specs(), 3, preview=True)
        assert monitor.snapshot() == before
        assert len(client.calls) == 2
    finally:
        await agent._compactor().aclose()


async def test_saved_result_survives_store_reopen_without_restarting_watch(monitor_service, tmp_path):
    archive, jobs, job, gate, tool = monitor_service
    archive.artifacts.use_directory(tmp_path / 'saved-results')
    key = await start(tool, job, pattern='ready')
    job.log.write('stdout', 'ready: persistent result')
    record_id = jobs.monitors.monitors[key].record['record_id']
    await tool.run('acknowledge', monitor_id=key)
    restored = ArtifactStore(1_000_000)
    restored.use_directory(tmp_path / 'saved-results')
    try:
        page = await ReadArtifactTool(restored).run(record_id=record_id)
        assert 'ready: persistent result' in content_text(page.content)
        assert 'initialized_turn' in content_text(page.content)
        assert not jobs.monitors.snapshots()
    finally:
        await restored.aclose()


def test_active_subscription_survives_accepted_and_rejected_reload(run_copy):
    run_copy('''
        from slipagent.tools.base import current_invocation
        jobs = registry.resources.jobs
        log = registry.resources.archive.start('temporary job')
        release = asyncio.Event()
        async def run():
            await release.wait()
            return ToolResult.ok('done')
        job = jobs.start(log, run)
        token = current_invocation.set((9, 'monitor-call'))
        try:
            started = await registry.get('monitor_command').run('start', job_id=job.log.id, pattern='ready')
        finally:
            current_invocation.reset(token)
        key = started.content['monitor_id']
        log.write('stdout', 'rea')
        frame.request()
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        edit('monitoring.py', 'TAIL_CHARS = 1000', 'TAIL_CHARS = (')
        await frame.checkpoint()
        assert frame.generation == 1
        log.write('stdout', 'dy: the result')
        result = jobs.monitors.monitors[key].snapshot()
        assert result['state'] == 'ready' and result['initialized_turn'] == 9
        assert result['result']['output']['stdout']['text'] == 'ready: the result'
        assert session.registry.services['command_jobs'] is jobs
        release.set()
        await job.task
    ''')
