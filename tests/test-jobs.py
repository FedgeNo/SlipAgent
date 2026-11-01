
from slipagent.types import content_text

from slipagent.types import decode_json_content
import asyncio
import json
import shlex
import sys

import pytest

from slipagent.jobs import CommandJobs, CommandJobsTool, CommandInputTool
from slipagent.tools.shell import RunCommandTool
from slipagent.tools.output import CommandArchive, ReadCommandOutputTool


async def test_interactive_job_accepts_literal_input_and_eof_across_tool_replacement(workspace):
    archive, jobs = CommandArchive(100000), CommandJobs()
    tool = RunCommandTool(workspace, archive, jobs)
    command = shlex.join([sys.executable, '-u', '-c',
                          "import sys; print('ready'); print('received:' + sys.stdin.read())"])
    try:
        started = await tool.run(command, background=True, interactive=True)
        key = started.content['job_id']
        first = await CommandInputTool(jobs).run(key, text='héllo')
        assert not first.is_error and first.content['bytes_sent'] == 6
        # Rebuilt tools borrow the same job-owned pipe.
        second = await CommandInputTool(jobs).run(key, text=' world\n', eof=True)
        assert not second.is_error
        result = await CommandJobsTool(jobs).run('wait', key)
        assert result.content['state'] == 'completed'
        assert result.content['stdin'] == 'closed'
        output = content_text((await ReadCommandOutputTool(archive).run(log_id=key)).content)
        assert 'received:héllo world' in output
        assert (await CommandInputTool(jobs).run(key, text='again')).is_error
    finally:
        await jobs.aclose()
        await archive.aclose()


async def test_interactive_input_validation_and_stop(workspace):
    archive, jobs = CommandArchive(100000), CommandJobs()
    tool, inputs = RunCommandTool(workspace, archive, jobs), CommandInputTool(jobs)
    command = shlex.join([sys.executable, '-c', 'import sys; sys.stdin.read()'])
    try:
        assert (await tool.run(command, interactive=True)).is_error
        assert not archive.logs
        key = (await tool.run(command, background=True, interactive=True)).content['job_id']
        assert (await inputs.run(key)).is_error
        assert (await inputs.run(key, text='é' * 32769)).is_error
        assert (await inputs.run('missing', text='test')).is_error
        assert not (await inputs.run(key, text='accepted')).is_error
        await CommandJobsTool(jobs).run('stop', key)
        assert jobs.jobs[key].input.state == 'closed'
        assert (await inputs.run(key, eof=True)).is_error
        plain = (await tool.run('echo plain', background=True)).content['job_id']
        assert (await inputs.run(plain, text='test')).is_error
    finally:
        await jobs.aclose()
        await archive.aclose()


async def test_cancelled_input_closes_pipe_without_stopping_owned_job(workspace, monkeypatch):
    archive, jobs = CommandArchive(100000), CommandJobs()
    command = shlex.join([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        key = (await RunCommandTool(workspace, archive, jobs).run(command, background=True, interactive=True)).content['job_id']
        channel = jobs.jobs[key].input
        await asyncio.wait_for(channel.ready.wait(), 5)
        entered = asyncio.Event()

        async def blocked():
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(channel.process.stdin, 'drain', blocked)
        sending = asyncio.create_task(CommandInputTool(jobs).run(key, text='hello'))
        await asyncio.wait_for(entered.wait(), 5)
        sending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sending
        assert channel.state == 'closed'
        assert jobs.active
    finally:
        await jobs.aclose()
        await archive.aclose()


@pytest.mark.parametrize('failure', ['broken_pipe', 'timeout'])
async def test_failed_input_closes_pipe_and_reports_partial_delivery(workspace, monkeypatch, failure):
    from slipagent import jobs as jobs_module

    archive, jobs = CommandArchive(100000), CommandJobs()
    command = shlex.join([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        key = (await RunCommandTool(workspace, archive, jobs).run(command, background=True, interactive=True)).content['job_id']
        channel = jobs.jobs[key].input
        await asyncio.wait_for(channel.ready.wait(), 5)

        async def fail():
            if failure == 'broken_pipe':
                raise BrokenPipeError('closed by child')
            await asyncio.Event().wait()

        monkeypatch.setattr(channel.process.stdin, 'drain', fail)
        monkeypatch.setattr(jobs_module, 'INPUT_TIMEOUT', .01)
        result = await CommandInputTool(jobs).run(key, text='hello')
        assert result.is_error
        assert 'partially delivered' in result.content
        assert channel.state == 'closed'
        assert (await CommandInputTool(jobs).run(key, text='retry')).is_error
    finally:
        await jobs.aclose()
        await archive.aclose()


async def test_background_output_wait_timeout_and_stop(workspace):
    archive, jobs = CommandArchive(100000), CommandJobs()
    tool, control = RunCommandTool(workspace, archive, jobs), CommandJobsTool(jobs)
    command = shlex.join([sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"])
    try:
        response = await tool.run(command, background=True)
        metadata = response.content
        key = metadata["job_id"]
        for _ in range(200):
            output = await ReadCommandOutputTool(archive).run(log_id=key)
            if "ready" in content_text(output.content):
                break
            await asyncio.sleep(.01)
        assert "ready" in content_text(output.content)
        waited = decode_json_content((await control.run("wait", key, timeout=.01)).content)
        assert waited["state"] == "running"
        with pytest.raises(ValueError, match="Stop background"):
            jobs.clear()
        stopped = decode_json_content((await control.run("stop", key)).content)
        assert stopped["state"] == "stopped" and stopped["finished"]
        assert not jobs.active
        assert "ready" in content_text((await ReadCommandOutputTool(archive).run(log_id=key)).content)
    finally:
        await jobs.aclose()
        await archive.aclose()


@pytest.mark.parametrize('interactive', [False, True])
async def test_shutdown_during_process_start_reaps_job(workspace, interactive):
    archive, jobs = CommandArchive(10000), CommandJobs()
    tool = RunCommandTool(workspace, archive, jobs)
    try:
        await tool.run(shlex.join([sys.executable, "-c", "import time; time.sleep(60)"]),
                       background=True, interactive=interactive)
        await jobs.aclose()
        assert not jobs.active
        assert all(log.finished for log in archive.logs.values())
        if interactive:
            assert all(job.input.state == 'closed' for job in jobs.jobs.values())
    finally:
        await jobs.aclose()
        await archive.aclose()


async def test_completion_is_announced_once_without_starting_a_model_turn(workspace):
    from slipagent.agent import Agent
    from slipagent.tools.base import ToolRegistry
    archive, jobs = CommandArchive(10000), CommandJobs()
    events = []
    agent = Agent(None, ToolRegistry(services={"command_jobs": jobs}), "test", on_event=events.append)
    try:
        result = await RunCommandTool(workspace, archive, jobs).run(shlex.join([sys.executable, "-c", "print('done')"]), background=True)
        key = result.content["job_id"]
        await CommandJobsTool(jobs).run("wait", key)
        assert len(events) == 1 and events[0].kind == "notice"
        agent._notify_jobs()
        agent._notify_jobs()
        assert len(events) == 1 and events[0].kind == "notice"
        assert key in {job["job_id"] for job in agent.registry.context_notes["background_commands"]["completed"]}
        assert not agent.running
    finally:
        await agent.registry.aclose()
        await archive.aclose()


async def test_cleanup_keeps_unreported_completions(workspace):
    from slipagent.jobs import CommandJob
    from slipagent.tools.base import ToolResult
    from slipagent.tools.output import CommandLog

    jobs = CommandJobs()
    archive = CommandArchive(1_000_000)
    for index in range(128):
        job = CommandJob(CommandLog(archive, f"echo {index}", None, None))
        job.state, job.notified = "completed", index != 0
        jobs.jobs[job.log.id] = job
    unreported, oldest_reported = list(jobs.jobs)[:2]

    async def finished():
        return ToolResult.ok("done")

    jobs.start(CommandLog(archive, "true", None, None), finished)
    assert unreported in jobs.jobs and oldest_reported not in jobs.jobs and len(jobs.jobs) == 128
    for job in jobs.jobs.values():
        if job.task is not None:
            await job.task
        job.notified = False
    with pytest.raises(ValueError, match="not been reported yet"):
        jobs.check_capacity()
    await jobs.aclose()
