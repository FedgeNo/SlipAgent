"""Undelivered command results and accounting completeness survive session resume."""

import asyncio
import json

from slipagent.agent import Agent
from slipagent.jobs import CommandJobs
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry, ToolResult
from slipagent.tools.output import CommandArchive
from slipagent.types import Usage
from test_agent import StubClient, completion


async def test_separate_idle_completions_survive_preview_delivery_history_and_resume(workspace):
    jobs, archive = CommandJobs(), CommandArchive(10000)
    journal = SessionJournal(str(workspace.root))
    registry = ToolRegistry(services={"command_jobs": jobs, "session_journal": journal})
    client = StubClient([completion("First"), completion("Second")])
    agent = Agent(client, registry, "test")
    async def done():
        return ToolResult.ok("done")
    try:
        await agent.run("Start")
        logs = [archive.start("first"), archive.start("second")]
        for log in logs:
            job = jobs.start(log, done)
            await job.task
            await asyncio.sleep(0)
        expected = {log.id for log in logs}
        assert {item["job_id"] for item in registry.context_notes["background_commands"]["completed"]} == expected
        pending = journal.load(journal.session_id)
        assert {item["job_id"] for item in pending["state"]["background_commands"]["completed"]} == expected
        assert not any(job.notified for job in jobs.jobs.values())
        await agent._context_view([], 0, preview=True)
        assert not any(job.notified for job in jobs.jobs.values())
        await agent.run("Continue")
        delivered = json.dumps([message.to_api() for message in client.calls[-1]["messages"]])
        assert all(key in delivered for key in expected)
        assert all(job.notified for job in jobs.jobs.values())
        assert "background_commands" not in registry.context_notes
        saved = journal.load(journal.session_id)
        assert {item["job_id"] for step in saved["history"].steps
                for item in getattr(step, "background_commands", [])} == expected
        journal.restore(agent, saved)
        view = await agent._context_view([], 0, preview=True)
        assert all(key in json.dumps([message.to_api() for message in view]) for key in expected)
    finally:
        await registry.aclose()
        await archive.aclose()


async def test_cancelled_usage_completeness_survives_resume(workspace):
    journal = SessionJournal(str(workspace.root))
    registry = ToolRegistry(services={"session_journal": journal})
    agent = Agent(StubClient([completion("Done")]), registry, "test")
    try:
        await agent.run("Start")
        agent.usage = Usage(prompt_tokens=10, total_tokens=10, cost=0.1) + Usage(complete=False)
        agent._persist()
        saved = journal.load(journal.session_id)
        journal.restore(agent, saved)
        assert not agent.usage.complete
        assert agent.total_cost is None
        assert agent.usage.prompt_tokens == 10
    finally:
        await registry.aclose()
