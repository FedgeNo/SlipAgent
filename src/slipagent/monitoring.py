"""Event-driven subscriptions to session-owned command output and completion."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from .tools.base import Tool, ToolResult, current_invocation
from .types import content_text

if TYPE_CHECKING:
    from .jobs import CommandJob, CommandJobs

TAIL_CHARS = 1000
REPLAY_BYTES = 65536
NOTICE_SECONDS = 5.0
MAX_MONITORS = 8


class CommandMonitor:
    def __init__(self, job: CommandJob, pattern: str | None, stream: str, repeat: bool,
                 timeout: float) -> None:
        self.id = uuid.uuid4().hex
        self.job = job
        invocation = current_invocation.get()
        self.step_id: int | None = invocation[0] if invocation else None
        self.call_id: str | None = invocation[1] if invocation else None
        self.pattern, self.stream, self.repeat = pattern, stream, repeat
        self.state = "active"
        self.reason = ""
        self.deadline = time.monotonic() + timeout
        self.tails = {name: "" for name in job.log.streams}
        self.characters = {name: 0 for name in job.log.streams}
        self.carry = {name: "" for name in job.log.streams}
        self.replay_omitted = {name: 0 for name in job.log.streams}
        self.matches = 0
        self.suppressed = 0
        self.last_match: dict[str, Any] | None = None
        self.notice_pending = False
        self.last_notice = float("-inf")
        self.timer: asyncio.TimerHandle | None = None
        self.flush_timer: asyncio.TimerHandle | None = None
        self.ready = asyncio.Event()
        self.result: dict[str, Any] | None = None
        self.record: dict[str, Any] | None = None
        self.storage_error = ""

    def snapshot(self) -> dict[str, Any]:
        data = {"monitor_id": self.id, "job_id": self.job.log.id,
                "initialized_turn": self.step_id, "initializing_call_id": self.call_id,
                "state": self.state, "condition": "output_match" if self.pattern else "completion",
                "pattern": self.pattern, "stream": self.stream, "repeat": self.repeat,
                "matches": self.matches, "suppressed_notifications": self.suppressed,
                "seconds_remaining": max(0, round(self.deadline - time.monotonic(), 1)) if self.state == "active" else 0,
                "replay_omitted_bytes": dict(self.replay_omitted)}
        if self.result is not None:
            data["result"] = self.result
        elif self.last_match is not None:
            data["latest_match"] = dict(self.last_match)
        if self.record is not None:
            data["artifact"] = self.record
        if self.storage_error:
            data["storage_error"] = self.storage_error
        return data


class CommandMonitors:
    """Borrow job/log ownership; keep bounded summaries without replacing raw logs."""

    def __init__(self, jobs: CommandJobs) -> None:
        self.jobs = jobs
        self.monitors: dict[str, CommandMonitor] = {}

    def start(self, job_id: str, pattern: str | None, stream: str, repeat: bool,
              timeout: float) -> CommandMonitor:
        if not self.jobs.lifetime.active:
            raise ValueError("Background command service is shutting down")
        if len(self.monitors) >= MAX_MONITORS:
            raise ValueError("Eight monitors are retained; acknowledge ready results or cancel and acknowledge an active monitor first")
        job = self.jobs.jobs.get(job_id)
        if job is None:
            raise ValueError("Unknown job_id; use command_jobs action=list. Monitors cannot attach to a previous process's jobs")
        monitor = CommandMonitor(job, pattern, stream, repeat, timeout)
        self.monitors[monitor.id] = monitor
        # Read a bounded retained tail before subscribing, without yielding: no
        # output can slip between replay and attaching the live callback.
        for name, output in job.log.streams.items():
            if output.path is None:
                continue
            try:
                with output.path.open("rb") as source:
                    start = max(0, output.retained_bytes - REPLAY_BYTES)
                    source.seek(start)
                    raw = source.read(REPLAY_BYTES)
                while raw and raw[0] & 0xC0 == 0x80:
                    raw, start = raw[1:], start + 1
                monitor.replay_omitted[name] = start
                self._feed(monitor, name, raw.decode("utf-8"), notify=False)
            except (OSError, UnicodeError) as exc:
                self.finish(monitor, "failed", error=f"Cannot read initial command output: {exc}")
                return monitor
        if "monitor_output" not in job.log.__dict__:
            job.log.__dict__["monitor_output"] = lambda name, text: self.feed(job_id, name, text)
            if job.task is not None:
                # Subscribe once per job, including jobs predating this feature.
                job.task.add_done_callback(lambda task: self.job_finished(job_id))
        if monitor.matches and not repeat:
            self.finish(monitor, "matched")
        elif job.state != "running":
            self.finish(monitor, "job_finished")
        else:
            monitor.timer = asyncio.get_running_loop().call_later(timeout, lambda: self.finish(monitor, "timed_out"))
            if monitor.matches:
                self._notice(monitor)
        return monitor

    def feed(self, job_id: str, stream: str, text: str) -> None:
        for monitor in tuple(self.monitors.values()):
            if monitor.job.log.id == job_id and monitor.state == "active":
                self._feed(monitor, stream, text)

    def _feed(self, monitor: CommandMonitor, stream: str, text: str, *, notify: bool = True) -> None:
        monitor.tails[stream] = (monitor.tails[stream] + text)[-TAIL_CHARS:]
        monitor.characters[stream] += len(text)
        pattern = monitor.pattern
        if not pattern or monitor.stream not in ("both", stream):
            return
        combined = monitor.carry[stream] + text
        count, end, index = 0, 0, -1
        while (found := combined.find(pattern, end)) >= 0:
            count += 1
            index = found
            end = found + len(pattern)
        monitor.carry[stream] = combined[max(end, len(combined) - len(pattern) + 1):]
        if not count:
            return
        # Only a bounded sample enters prompts; original streams stay in the
        # command archive, including matches whose notifications are coalesced.
        excerpt = combined[max(0, index - 128):index + len(pattern) + 128]
        monitor.last_match = {"stream": stream, "text": excerpt}
        monitor.matches += count
        monitor.suppressed += max(0, count - 1)
        if notify:
            if not monitor.repeat:
                self.finish(monitor, "matched")
            else:
                self._notice(monitor)

    def _notice(self, monitor: CommandMonitor) -> None:
        now = time.monotonic()
        if now < monitor.last_notice + NOTICE_SECONDS:
            monitor.suppressed += 1
            if monitor.flush_timer is None:
                monitor.flush_timer = asyncio.get_running_loop().call_later(
                    monitor.last_notice + NOTICE_SECONDS - now, lambda: self._flush(monitor))
            return
        self._flush(monitor)

    def _flush(self, monitor: CommandMonitor) -> None:
        if monitor.flush_timer is not None:
            monitor.flush_timer.cancel()
            monitor.flush_timer = None
        monitor.last_notice = time.monotonic()
        monitor.notice_pending = True
        if self.jobs.on_completion is not None:
            self.jobs.on_completion()

    def finish(self, monitor: CommandMonitor, reason: str, *, error: str = "") -> None:
        if monitor.state != "active":
            return
        monitor.state = "ready"
        monitor.reason = reason
        for timer in (monitor.timer, monitor.flush_timer):
            if timer is not None:
                timer.cancel()
        monitor.timer = monitor.flush_timer = None
        log = monitor.job.log
        monitor.result = {"reason": reason, "job_state": monitor.job.state,
                          "returncode": log.returncode, "timed_out": log.timed_out,
                          "matches": monitor.matches, "last_match": monitor.last_match,
                          "suppressed_notifications": monitor.suppressed,
                          "output": {name: {"text": text, "omitted_characters": max(0, monitor.characters[name] - len(text)),
                                             "earlier_replay_omitted_bytes": monitor.replay_omitted[name]}
                                     for name, text in monitor.tails.items()},
                          "log_id": log.id, "retention": log.metadata()["streams"]}
        if error:
            monitor.result["error"] = error
        if reason == "job_finished" and monitor.job.result is not None:
            outcome = content_text(monitor.job.result.content)
            monitor.result["job_result"] = {"is_error": monitor.job.result.is_error,
                                            "text": outcome[:TAIL_CHARS],
                                            "omitted_characters": max(0, len(outcome) - TAIL_CHARS)}
        store = log.archive.artifacts
        if store is not None:
            try:
                monitor.record = store.save("Command monitor result", json.dumps(monitor.snapshot(), ensure_ascii=False),
                                            kind="monitor_result", step_id=monitor.step_id, call_id=monitor.call_id)
            except (OSError, ValueError, UnicodeError) as exc:
                monitor.storage_error = str(exc)
        monitor.ready.set()
        self._flush(monitor)

    def job_finished(self, job_id: str) -> None:
        for monitor in tuple(self.monitors.values()):
            if monitor.job.log.id == job_id:
                self.finish(monitor, "job_finished")

    def snapshots(self) -> list[dict[str, Any]]:
        return [monitor.snapshot() for monitor in self.monitors.values()]

    def notices(self) -> list[dict[str, Any]]:
        notices = []
        for monitor in self.monitors.values():
            if monitor.notice_pending:
                notices.append(monitor.snapshot())
                monitor.notice_pending = False
        return notices

    def clear(self) -> None:
        for monitor in self.monitors.values():
            for timer in (monitor.timer, monitor.flush_timer):
                if timer is not None:
                    timer.cancel()
            monitor.job.log.__dict__.pop("monitor_output", None)
        self.monitors.clear()


class MonitorCommandTool(Tool):
    name = "monitor_command"
    description_prompt = "tools/monitor-command.md"
    parameter_prompts = "tools/monitor-command-parameters.json"
    description = ""
    progress_exempt = True
    parameters = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["start", "list", "wait", "cancel", "acknowledge"]},
        "job_id": {"type": "string"}, "monitor_id": {"type": "string"},
        "pattern": {"type": "string"},
        "stream": {"type": "string", "enum": ["stdout", "stderr", "both"]},
        "repeat": {"type": "boolean"},
        "timeout": {"type": "number", "minimum": 0.1, "maximum": 600},
    }, "required": ["action"]}

    def __init__(self, jobs: CommandJobs) -> None:
        self.jobs = jobs

    async def run(self, action: str, job_id: str | None = None, monitor_id: str | None = None,
                  pattern: str | None = None, stream: str = "both", repeat: bool = False,
                  timeout: float = 30) -> ToolResult:
        monitors = self.jobs.monitors
        if action == "start":
            if pattern is not None and not 1 <= len(pattern) <= 256:
                return ToolResult.error("pattern must contain 1 to 256 characters; omit it to watch completion")
            if repeat and pattern is None:
                return ToolResult.error("repeat requires a literal output pattern")
            try:
                return ToolResult.ok(monitors.start(job_id or "", pattern, stream, repeat, timeout).snapshot())
            except ValueError as exc:
                return ToolResult.error(str(exc))
        if action == "list":
            return ToolResult.ok({"monitors": monitors.snapshots()})
        monitor = monitors.monitors.get(monitor_id or "")
        if monitor is None:
            return ToolResult.error("Unknown monitor_id; use monitor_command action=list. Monitors belong to this process and are not resumed automatically")
        if action == "wait":
            try:
                await asyncio.wait_for(monitor.ready.wait(), min(timeout, 30))
            except TimeoutError:
                pass
        elif action == "cancel":
            monitors.finish(monitor, "cancelled")
        elif action == "acknowledge":
            if monitor.state == "active":
                return ToolResult.error("Monitor is still active; cancel it before acknowledging")
            del monitors.monitors[monitor.id]
        return ToolResult.ok(monitor.snapshot())
