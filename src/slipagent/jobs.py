"""Session-owned command jobs; completion never starts another model turn."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from .lifecycle import Lifetime, finish_cleanup
from .tools.base import Tool, ToolResult
from .tools.output import CommandLog


class CommandJob:
    def __init__(self, log: CommandLog) -> None:
        self.log = log
        self.state = "running"
        self.result: ToolResult | None = None
        self.task: asyncio.Task[None] | None = None
        self.notified = False

    def metadata(self) -> dict[str, Any]:
        return {"job_id": self.log.id, "state": self.state, **self.log.metadata()}


class CommandJobs:
    """Keep jobs independent of rebuilt tools, foreground waits and /stop."""
    def __init__(self) -> None:
        self.jobs: dict[str, CommandJob] = {}
        self.lifetime = Lifetime("background commands")
        self.on_completion: Callable[[], None] | None = None

    @property
    def active(self) -> bool:
        return any(job.task is not None and not job.task.done() for job in self.jobs.values())

    def check_capacity(self) -> None:
        if not self.lifetime.active:
            raise ValueError("Background commands are shutting down")
        if sum(job.state == "running" for job in self.jobs.values()) >= 4:
            raise ValueError("Four background commands are already running; wait for or stop one before starting another")

    def start(self, log: CommandLog, run: Callable[[], Awaitable[ToolResult]]) -> CommandJob:
        self.check_capacity()
        while len(self.jobs) >= 128:
            oldest = next((key for key, job in self.jobs.items() if job.state != "running"), None)
            if oldest is None:
                break
            del self.jobs[oldest]
        job = CommandJob(log)
        self.jobs[log.id] = job

        async def execute() -> None:
            try:
                job.result = await run()
                job.state = "failed" if job.result.is_error else "completed"
            except asyncio.CancelledError:
                job.state = "stopped"
                raise
            except Exception as exc:
                job.state = "failed"
                job.result = ToolResult.error(f"Background command failed: {exc}; inspect its log before retrying")
            finally:
                log.finish(log.returncode, log.timed_out)

        job.task = self.lifetime.spawn(execute())
        def settled(task: asyncio.Task[None]) -> None:
            # Cancellation can win before execute() reaches its first line.
            if task.cancelled() and job.state == "running":
                job.state = "stopped"
                log.finish(log.returncode, log.timed_out)
            if self.on_completion is not None:
                self.on_completion()
        job.task.add_done_callback(settled)
        return job

    async def stop_all(self) -> None:
        """Drain the current jobs without closing the reusable session owner."""
        async def drain() -> None:
            tasks = [job.task for job in self.jobs.values() if job.task is not None and not job.task.done()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await finish_cleanup(asyncio.create_task(drain()))

    def clear(self) -> None:
        if self.active:
            raise ValueError("Stop background commands before clearing the session")
        self.jobs.clear()

    def completions(self) -> list[dict[str, Any]]:
        completed = []
        for job in self.jobs.values():
            if job.state != "running" and not job.notified:
                completed.append(job.metadata())
                job.notified = True
        return completed

    async def aclose(self) -> None:
        self.on_completion = None
        await self.lifetime.aclose()


class CommandJobsTool(Tool):
    name = "command_jobs"
    progress_exempt = True
    description = (
        "Manage commands started with run_command background=true.\n\n"
        "List/status returns job_id, state, exit status and log_id; read_command_output retrieves "
        "live pages or tails of either stream.\n\n"
        "Wait waits at most 30 seconds without cancelling the job.\n\n"
        "Stop kills its process group and drains cleanup.\n\n"
        "Completion does not wake the model; request status/wait when you need the outcome. /stop "
        "leaves jobs running."
    )
    parameters = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["list", "status", "wait", "stop"]},
        "job_id": {"type": "string"},
        "timeout": {"type": "number", "minimum": 0.1, "maximum": 30},
        "offset": {"type": "integer", "minimum": 0},
    }, "required": ["action"]}

    def __init__(self, jobs: CommandJobs) -> None:
        self.jobs = jobs

    async def run(self, action: str, job_id: str | None = None, timeout: float = 10, offset: int = 0) -> ToolResult:
        if action == "list":
            jobs = list(self.jobs.jobs.values())
            return ToolResult.ok(json.dumps({"jobs": [job.metadata() for job in jobs[offset:offset + 50]],
                                            "next_offset": offset + 50 if offset + 50 < len(jobs) else None}))
        job = self.jobs.jobs.get(job_id or "")
        if job is None:
            return ToolResult.error("Unknown job_id; use command_jobs action=list. Jobs from an earlier process cannot be controlled; retained output remains accessible by log_id.")
        assert job.task is not None
        task = job.task
        if action == "wait":
            # Waiting cancellation must not cancel the independent command.
            await asyncio.wait({job.task}, timeout=timeout)
        elif action == "stop":
            job.task.cancel()
            async def reap() -> None:
                await asyncio.gather(task, return_exceptions=True)
            await finish_cleanup(asyncio.create_task(reap()))
        result = job.metadata()
        if job.result is not None:
            result["result"] = job.result.content
            result["is_error"] = job.result.is_error
        return ToolResult.ok(json.dumps(result, ensure_ascii=False))
