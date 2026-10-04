"""Shell execution tool.

Runs commands with the workspace as the working directory and captures output.
Blocking work is handed to a subprocess via asyncio so the event loop stays
responsive. Observations use bounded previews; a default-registry command also
writes its full decoded streams to the shared quota-limited session archive.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import signal

from ..workspace import Workspace, WorkspaceError
from ..activity import OutputProgress, command_output
from .base import Tool, ToolResult
from .output import CommandArchive, CommandLog

DEFAULT_TIMEOUT = 120.0
MAX_TIMEOUT = 600.0
MAX_OUTPUT_CHARS = 30_000


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the workspace root and return its output. Use it "
        "to run tests, linters, type checkers, and builds, and to inspect files "
        "that dedicated tools cannot read. The command runs under a non-login "
        "shell, so prefer explicit paths. Long-running commands are killed at "
        "the timeout. The observation shows at most 30000 characters per stream, "
        "retaining the beginning and end. Session command logs retain full decoded "
        "output within the configured disk quota; read_command_output retrieves "
        "pages or tails using the returned log ID. Lost bytes are reported explicitly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout": {
                "type": "number",
                "description": (
                    f"Timeout in seconds (default {DEFAULT_TIMEOUT:g}, max {MAX_TIMEOUT:g})."
                ),
                "minimum": 0.1,
            },
            "poll": {"type": "boolean", "description": "True only when deliberately polling for an external change; exempts unchanged results from loop detection. Default false."},
        },
        "required": ["command"],
    }

    def __init__(self, workspace: Workspace, archive: CommandArchive | None = None) -> None:
        self.workspace = workspace
        self.archive = archive

    async def run(self, command: str, timeout: float = DEFAULT_TIMEOUT, poll: bool = False) -> ToolResult:
        if not command.strip():
            return ToolResult.error("command must not be empty.")

        try:
            self.workspace.resolve(".")
        except WorkspaceError as exc:
            return ToolResult.error(str(exc))

        limit = min(float(timeout), MAX_TIMEOUT)
        log = self.archive.start(command) if self.archive is not None else None
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace.root),
                env=_subprocess_env(),
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            if log is not None:
                log.finish(None, False)
            return ToolResult.error(f"Could not start command: {exc}")

        publish = command_output.get()
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
        await _kill(process)
        completed.cancel()
        await asyncio.gather(completed, return_exceptions=True)
        raise
    finally:
        if log is not None:
            log.finish(process.returncode, timed_out)


async def _capture(stream: asyncio.StreamReader | None, *, log: CommandLog | None = None,
                   channel: str = "stdout", progress: OutputProgress | None = None) -> str:
    if stream is None:
        return ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    prefix = ""
    suffix = ""
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
        prefix += text[:max(0, head_limit - len(prefix))]
        suffix = (suffix + text)[-tail_limit:]
        if not raw:
            remaining = min(max(0, total - len(prefix)), tail_limit)
            return prefix + (suffix[-remaining:] if remaining else "")


async def _kill(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    # Keep tool output and agent output from interleaving confusingly.
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


def shell_tools(workspace: Workspace, archive: CommandArchive | None = None) -> list[Tool]:
    return [RunCommandTool(workspace, archive)]
