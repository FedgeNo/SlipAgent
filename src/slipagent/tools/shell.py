"""Shell execution tool.

Runs commands with the workspace as the working directory and captures output.
Blocking work is handed to a subprocess via asyncio so the event loop stays
responsive. Observations use bounded previews; a default-registry command also
writes its full decoded streams to the shared quota-limited session archive.
"""

from __future__ import annotations

from ..prompts import load_prompt

import asyncio
import codecs
import os
import signal
import json
from collections import deque
from typing import TYPE_CHECKING

from ..workspace import Workspace, WorkspaceError
from ..activity import OutputProgress, command_output
from .base import Tool, ToolResult
from .output import CommandArchive, CommandLog, MAX_OUTPUT_CHARS as MAX_OUTPUT_CHARS
from ..lifecycle import finish_cleanup

if TYPE_CHECKING:
    from ..jobs import CommandJobs

DEFAULT_TIMEOUT = 120.0
MAX_TIMEOUT = 600.0


class RunCommandTool(Tool):
    parameter_prompts = 'tools/run-command-parameters.json'
    name = "run_command"
    description_prompt = 'tools/run-command.md'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": ""},
            "timeout": {
                "type": "number",
                "description": (
                    ""
                ),
                "minimum": 0.1,
            },
            "poll": {"type": "boolean", "description": ""},
            "background": {"type": "boolean", "description": ""},
        },
        "required": ["command"],
    }

    def __init__(self, workspace: Workspace, archive: CommandArchive | None = None, jobs: CommandJobs | None = None) -> None:
        self.workspace = workspace
        self.archive = archive
        self.jobs = jobs

    async def run(self, command: str, timeout: float = DEFAULT_TIMEOUT, poll: bool = False, background: bool = False) -> ToolResult:
        if not command.strip():
            return ToolResult.error("command must not be empty.")

        try:
            self.workspace.resolve(".")
        except WorkspaceError as exc:
            return ToolResult.error(str(exc))

        limit = min(float(timeout), MAX_TIMEOUT)
        if background:
            if self.jobs is None or self.archive is None:
                return ToolResult.error("Background command service is unavailable")
            self.jobs.check_capacity()
        log = self.archive.start(command) if self.archive is not None else None
        if background:
            assert self.jobs is not None and log is not None
            job = self.jobs.start(log, lambda: self._execute(command, limit, log, background=True))
            # Ensure subprocess ownership is established before a following stop.
            await asyncio.sleep(0)
            return ToolResult.ok({**job.metadata(), "notice": log.notice()})
        return await self._execute(command, limit, log)

    async def _execute(self, command: str, limit: float, log: CommandLog | None, *, background: bool = False) -> ToolResult:
        spawn = asyncio.create_task(asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace.root), env=_subprocess_env(), start_new_session=os.name == "posix",
        ))
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            async def reap_spawn() -> None:
                process = await spawn
                await kill_and_drain(process, log=log)
                if log is not None:
                    log.finish(process.returncode, False)
            await finish_cleanup(asyncio.create_task(reap_spawn()))
            raise
        except OSError as exc:
            if log is not None:
                log.finish(None, False)
            return ToolResult.error(f"Could not start command: {exc}")

        publish = None if background else command_output.get()
        progress = OutputProgress(publish) if publish is not None else None
        try:
            stdout, stderr, timed_out = await capture_process(process, limit, log=log, progress=progress)
        finally:
            if progress is not None:
                progress.close()
        result = _render(command, process.returncode or 0, stdout, stderr, archived=log is not None)
        if log is not None:
            result.content += "\n" + log.notice()
        if timed_out:
            return ToolResult.error(
                f"Command timed out after {limit:g}s and was killed. Effects may be partial; "
                "inspect before retrying. Captured output follows.\n" + result.content
            )
        return result


async def capture_process(process: asyncio.subprocess.Process, timeout: float, *,
                          log: CommandLog | None = None,
                          progress: OutputProgress | None = None) -> tuple[str, str, bool]:
    """Drain bounded output and retain it through timeout cleanup.

    Shield readers while the deadline runs, kill the process group, then drain
    EOF so timeout output remains available. The optional log belongs to the
    caller's archive and is finalized on success, timeout, or cancellation.
    """
    stdout = asyncio.create_task(_capture(process.stdout, log=log, channel="stdout", progress=progress))
    stderr = asyncio.create_task(_capture(process.stderr, log=log, channel="stderr", progress=progress))
    completed = asyncio.gather(stdout, stderr, process.wait())
    timed_out = False
    try:
        try:
            await asyncio.wait_for(asyncio.shield(completed), timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await _kill(process)
            await completed
        return stdout.result(), stderr.result(), timed_out
    except BaseException:
        async def cleanup() -> None:
            await _kill(process)
            # Drain the readers after killing descendants so retained logs also
            # include output written just before cancellation.
            await asyncio.gather(completed, return_exceptions=True)
        await finish_cleanup(asyncio.create_task(cleanup()))
        raise
    finally:
        if log is not None:
            log.finish(process.returncode, timed_out)


async def _capture(stream: asyncio.StreamReader | None, *, log: CommandLog | None = None,
                   channel: str = "stdout", progress: OutputProgress | None = None) -> str:
    if stream is None:
        return ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    prefix: list[str] = []
    suffix: deque[str] = deque()
    prefix_size = suffix_size = 0
    total = 0
    head_limit = MAX_OUTPUT_CHARS // 2
    tail_limit = MAX_OUTPUT_CHARS + 1 - head_limit
    while True:
        raw = await stream.read(65536)
        text = decoder.decode(raw, final=not raw)
        if log is not None:
            log.write(channel, text)
        if progress is not None:
            progress.feed(text)
        total += len(text)
        head = text[:max(0, head_limit - prefix_size)]
        if head:
            prefix.append(head)
            prefix_size += len(head)
        if text:
            suffix.append(text)
            suffix_size += len(text)
        # Keep chunks so large outputs do not repeatedly copy the entire tail.
        while suffix_size > tail_limit:
            excess = suffix_size - tail_limit
            first = suffix.popleft()
            if len(first) > excess:
                suffix.appendleft(first[excess:])
                suffix_size -= excess
            else:
                suffix_size -= len(first)
        if not raw:
            remaining = min(max(0, total - prefix_size), tail_limit)
            return "".join(prefix) + ("".join(suffix)[-remaining:] if remaining else "")


async def _kill(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


async def kill_and_drain(process: asyncio.subprocess.Process, *, log: CommandLog | None = None) -> None:
    """Reap a process whose normal pipe readers have not started or have stopped.

    Process.wait can wait for pipe EOF after the child exits. Keep bounded readers
    running while killing it so a full asyncio pipe buffer cannot deadlock close.
    Callers must first settle any previous readers to avoid competing reads.
    """
    readers = [asyncio.create_task(_capture(process.stdout, log=log, channel="stdout")),
               asyncio.create_task(_capture(process.stderr, log=log, channel="stderr"))]
    try:
        await _kill(process)
    finally:
        await asyncio.gather(*readers)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    # Flush Python child output promptly for capture and progress updates.
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS, *, notice: str | None = None,
              archived: bool = False) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    notice = notice or f"… [output truncated at {limit} chars]"
    head = limit // 2
    retention = "use the command log for retained middle output" if archived else "middle not archived"
    return (text[:head] + f"\n{notice} (beginning and end; {retention})\n"
            + text[-(limit - head):]), True


def _render(command: str, returncode: int, stdout: str, stderr: str, *, archived: bool = False) -> ToolResult:
    stdout, stdout_cut = _truncate(stdout, archived=archived)
    stderr, stderr_cut = _truncate(stderr, archived=archived)

    parts = [f"$ {command}", f"exit code: {returncode}"]
    if stdout.strip():
        parts.append("--- stdout ---\n" + stdout.rstrip())
    if stderr.strip():
        parts.append("--- stderr ---\n" + stderr.rstrip())
    if stdout_cut or stderr_cut:
        parts.append("(output was truncated)")

    content = "\n".join(parts)
    if returncode != 0:
        return ToolResult.error(content)
    return ToolResult.ok(content)


def shell_tools(workspace: Workspace, archive: CommandArchive | None = None, jobs: CommandJobs | None = None) -> list[Tool]:
    return [RunCommandTool(workspace, archive, jobs)]
