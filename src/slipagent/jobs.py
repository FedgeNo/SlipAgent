"""Session-owned command jobs; completion never starts another model step."""

from __future__ import annotations

from .prompts import load_prompt

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from .lifecycle import Lifetime, finish_cleanup
from .tools.base import Tool, ToolResult
from .tools.output import CommandLog
from .monitoring import CommandMonitors

MAX_INPUT_BYTES = 65536
INPUT_TIMEOUT = 5.0


class CommandInput:
    """A pipe owned by its background job, never by a replacement tool."""

    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self.process: asyncio.subprocess.Process | None = None
        self.state = "starting"

    def attach(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.state = "open"
        self.ready.set()

    def close(self) -> None:
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()
        self.process = None
        self.state = "closed"
        self.ready.set()


class CommandJob:
    def __init__(self, log: CommandLog, *, input_channel: CommandInput | None = None) -> None:
        self.log = log
        self.state = "running"
        self.result: ToolResult | None = None
        self.task: asyncio.Task[None] | None = None
        self.notified = False
        self._input = input_channel

    @property
    def input(self) -> CommandInput | None:
        # Jobs retained from before stdin support were started without a pipe.
        return self.__dict__.get('_input')

    def metadata(self) -> dict[str, Any]:
        return {"job_id": self.log.id, "state": self.state,
                "stdin": self.input.state if self.input is not None else "disabled", **self.log.metadata()}


class CommandJobs:
    """Keep jobs independent of rebuilt tools, foreground waits and /stop."""
    def __init__(self) -> None:
        self.jobs: dict[str, CommandJob] = {}
        self.lifetime = Lifetime("background commands")
        self.on_completion: Callable[[], None] | None = None

    @property
    def monitors(self) -> CommandMonitors:
        # Explicit lazy migration for session owners retained across reloads.
        if "_monitors" not in self.__dict__:
            self._monitors = CommandMonitors(self)
        return self._monitors

    @property
    def active(self) -> bool:
        return any(job.task is not None and not job.task.done() for job in self.jobs.values())

    def check_capacity(self) -> None:
        if not self.lifetime.active:
            raise ValueError("Background commands are shutting down")
        if sum(job.state == "running" for job in self.jobs.values()) >= 4:
            raise ValueError("Four background commands are already running; wait for or stop one before starting another")
        if len(self.jobs) >= 128 and self._removable() is None and any(
                job.state != "running" for job in self.jobs.values()):
            raise ValueError("Earlier background command results have not been reported yet; "
                             "start another after the next step")

    def _removable(self) -> str | None:
        # Only reported completions may leave the retained list.
        return next((key for key, job in self.jobs.items() if job.state != "running" and job.notified), None)

    def start(self, log: CommandLog, run: Callable[[], Awaitable[ToolResult]], *,
              input_channel: CommandInput | None = None) -> CommandJob:
        self.check_capacity()
        while len(self.jobs) >= 128:
            oldest = self._removable()
            if oldest is None:
                break
            del self.jobs[oldest]
        job = CommandJob(log, input_channel=input_channel)
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
                if job.input is not None:
                    job.input.close()
                log.finish(log.returncode, log.timed_out)

        job.task = self.lifetime.spawn(execute())
        def settled(task: asyncio.Task[None]) -> None:
            # Cancellation can win before execute() reaches its first line.
            if task.cancelled() and job.state == "running":
                job.state = "stopped"
                if job.input is not None:
                    job.input.close()
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
        self.monitors.clear()
        self.jobs.clear()

    def completions(self) -> list[dict[str, Any]]:
        completed = []
        for job in self.jobs.values():
            if job.state != "running" and not job.notified:
                completed.append(job.metadata())
        return completed

    async def aclose(self) -> None:
        self.on_completion = None
        try:
            await self.lifetime.aclose()
        finally:
            self.monitors.clear()


class CommandJobsTool(Tool):
    name = "command_jobs"
    parameter_prompts = "tools/command-jobs-parameters.json"
    progress_exempt = True
    description_prompt = 'tools/command-jobs.md'
    description = load_prompt(description_prompt)
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
            return ToolResult.ok({"jobs": [job.metadata() for job in jobs[offset:]],
                                            "next_offset": None})
        job = self.jobs.jobs.get(job_id or "")
        if job is None:
            return ToolResult.error("Unknown job_id; use command_jobs action=list. Jobs from an earlier process cannot be controlled; retained output remains accessible by log_id.")
        if job.task is None:
            return ToolResult.error("Background job has not started yet; retry command_jobs shortly.")
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
        return ToolResult.ok(result)


class CommandInputTool(Tool):
    name = "command_input"
    description = ""
    description_prompt = 'tools/command-input.md'
    parameter_prompts = 'tools/command-input-parameters.json'
    parameters = {"type": "object", "properties": {
        "job_id": {"type": "string"},
        "text": {"type": "string"},
        "eof": {"type": "boolean"},
    }, "required": ["job_id"]}

    def __init__(self, jobs: CommandJobs) -> None:
        self.jobs = jobs

    async def run(self, job_id: str, text: str = '', eof: bool = False) -> ToolResult:
        try:
            data = text.encode('utf-8')
        except UnicodeEncodeError:
            return ToolResult.error("text must be valid UTF-8.")
        if len(data) > MAX_INPUT_BYTES or (not data and not eof):
            return ToolResult.error("Supply text and/or eof=true; text must be at most 65536 UTF-8 bytes. No input was sent.")
        job = self.jobs.jobs.get(job_id)
        if job is None:
            return ToolResult.error("Unknown job_id; use command_jobs action=list. Jobs from an earlier process cannot receive input.")
        channel = job.input
        if channel is None:
            return ToolResult.error("This job has no input pipe; start run_command with background=true and interactive=true.")
        try:
            await asyncio.wait_for(channel.ready.wait(), INPUT_TIMEOUT)
        except TimeoutError:
            return ToolResult.error("Job input is not ready; no input was sent. Inspect command_jobs before retrying.")
        process = channel.process
        if (job.state != 'running' or process is None or process.returncode is not None
                or process.stdin is None or process.stdin.is_closing()):
            return ToolResult.error("Job input is closed or the process has exited; no input was sent.")
        try:
            process.stdin.write(data)
            await asyncio.wait_for(process.stdin.drain(), INPUT_TIMEOUT)
            if eof:
                channel.close()
        except asyncio.CancelledError:
            channel.close()
            raise
        except (OSError, RuntimeError, TimeoutError) as exc:
            channel.close()
            return ToolResult.error(f"Input delivery failed: {exc}. Input may have been partially delivered; stdin was closed. Inspect output before retrying.")
        return ToolResult.ok({"job_id": job_id, "bytes_sent": len(data), "eof": eof,
                              "notice": "Bytes submitted to stdin; consumption and command success are not confirmed. Inspect command output and status."})
