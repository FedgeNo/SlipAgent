
from slipagent.types import content_text

from slipagent.types import decode_json_content
import asyncio
import json
import shlex
import sys

import pytest

from slipagent.jobs import CommandJobs, CommandJobsTool
from slipagent.tools.shell import RunCommandTool
from slipagent.tools.output import CommandArchive, ReadCommandOutputTool


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


async def test_shutdown_during_process_start_reaps_job(workspace):
    archive, jobs = CommandArchive(10000), CommandJobs()
    tool = RunCommandTool(workspace, archive, jobs)
    try:
        await tool.run(shlex.join([sys.executable, "-c", "import time; time.sleep(60)"]), background=True)
        await jobs.aclose()
        assert not jobs.active
        assert all(log.finished for log in archive.logs.values())
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
