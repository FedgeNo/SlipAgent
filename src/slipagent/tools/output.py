"""Session-owned command logs with a shared disk quota and bounded text pages.

Commands keep draining after quota or disk failure so logging cannot deadlock a
child. Earlier bytes are never evicted to make room for newer commands. Stream
offsets count UTF-8 bytes; page limits count decoded characters. Invalid process
bytes have already been replaced by the capture decoder before arriving here.
"""

from __future__ import annotations

from ..prompts import load_prompt

import json
import os
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from .base import Tool, ToolResult, current_invocation


class OutputStream:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.file: BinaryIO | None = None
        self.retained_bytes = 0
        self.lost_bytes = 0
        self.error = ""


class CommandLog:
    def __init__(self, archive: CommandArchive, command: str, step_id: int | None, call_id: str | None) -> None:
        self.archive = archive
        self.id = uuid.uuid4().hex
        self.command = command
        self.step_id = step_id
        self.call_id = call_id
        self.streams = {name: OutputStream() for name in ("stdout", "stderr")}
        self.returncode: int | None = None
        self.timed_out = False
        self.finished = False

    def write(self, name: str, text: str) -> None:
        stream = self.streams[name]
        raw = text.encode("utf-8")
        if not raw:
            return
        if stream.error:
            stream.lost_bytes += len(raw)
            return
        remaining = max(0, self.archive.quota_bytes - self.archive.used_bytes)
        # Never leave half a character at the retention boundary. Once bytes are
        # lost this stream stays stopped, so a file always represents one prefix.
        kept = raw[:remaining].decode("utf-8", errors="ignore").encode("utf-8")
        written = 0
        try:
            if kept:
                if stream.file is None:
                    stream.path = self.archive.directory / (self.id + "-" + name)
                    fd = os.open(stream.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    stream.file = os.fdopen(fd, "wb", buffering=0)
                while written < len(kept):
                    count = stream.file.write(kept[written:])
                    if not count:
                        raise OSError("log write made no progress")
                    written += count
            if len(kept) < len(raw):
                stream.error = f"Session log quota of {self.archive.quota_bytes} bytes reached; later bytes from this stream are not archived."
        except OSError as exc:
            stream.error = f"Log storage failed: {exc}; later bytes from this stream are not archived."
        valid_written = len(kept[:written].decode("utf-8", errors="ignore").encode("utf-8"))
        stream.retained_bytes += valid_written
        self.archive.used_bytes += written
        stream.lost_bytes += len(raw) - valid_written

    def finish(self, returncode: int | None, timed_out: bool) -> None:
        if self.finished and self.returncode == returncode and self.timed_out == timed_out:
            return
        self.returncode = returncode
        self.timed_out = timed_out
        self.finished = True
        for stream in self.streams.values():
            if stream.file is not None:
                if getattr(self.archive, "_persistent", None) is not None:
                    try:
                        os.fsync(stream.file.fileno())
                    except OSError as exc:
                        stream.error = f"{stream.error} Output durability could not be verified: {exc}".strip()
                stream.file.close()
                stream.file = None
        callback = getattr(self.archive, "on_log", None)
        if callback is not None:
            callback(self)

    def metadata(self) -> dict[str, Any]:
        return {"log_id": self.id, "step_id": self.step_id, "call_id": self.call_id,
                "command": self.command[:512] + ("…" if len(self.command) > 512 else ""),
                "command_characters": len(self.command), "returncode": self.returncode,
                "timed_out": self.timed_out, "finished": self.finished,
                "streams": {name: {"retained_bytes": stream.retained_bytes,
                                    "lost_bytes": stream.lost_bytes, "retention_error": stream.error}
                            for name, stream in self.streams.items()}}

    def notice(self) -> str:
        errors = " ".join(f"{name}: {stream.error} Lost {stream.lost_bytes} UTF-8 bytes."
                          for name, stream in self.streams.items() if stream.error)
        arguments = json.dumps({"log_id": self.id, "stream": "stdout", "offset": 0, "limit": 8000})
        return load_prompt("command-output-recovery.txt", log_id=self.id, arguments=arguments,
            retention=load_prompt("command-output-saved.txt" if getattr(self.archive, "_persistent", None) else "command-output-temporary.txt"),
            errors=errors).rstrip()


class CommandArchive:
    """Own files once per registry/session; reload candidates borrow this owner."""

    def __init__(self, quota_bytes: int, *, directory: Path | None = None) -> None:
        if type(quota_bytes) is not int or quota_bytes < 1:
            raise ValueError("Command archive quota must be a positive integer")
        self.quota_bytes = quota_bytes
        self.used_bytes = 0
        self.logs: dict[str, CommandLog] = {}
        self._parent = directory
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    @property
    def directory(self) -> Path:
        persistent: Path | None = getattr(self, "_persistent", None)
        if persistent is not None:
            return persistent
        if self._temporary is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="slipagent-logs-", dir=self._parent)
        return Path(self._temporary.name)

    def use_directory(self, directory: Path, on_log: Callable[[CommandLog], None]) -> None:
        """Attach private durable storage; clear detaches it without deleting it."""
        self._persistent: Path | None = directory
        self.on_log: Callable[[CommandLog], None] | None = on_log

    def start(self, command: str, *, step_id: int | None = None, call_id: str | None = None) -> CommandLog:
        invocation = current_invocation.get()
        if invocation is not None and step_id is None and call_id is None:
            step_id, call_id = invocation
        log = CommandLog(self, command, step_id, call_id)
        self.logs[log.id] = log
        callback = getattr(self, "on_log", None)
        if callback is not None:
            callback(log)
        return log

    def clear(self) -> None:
        for log in self.logs.values():
            log.finish(log.returncode, log.timed_out)
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self.logs.clear()
        self.used_bytes = 0
        self._persistent = None
        self.on_log = None

    async def aclose(self) -> None:
        self.clear()


class ReadCommandOutputTool(Tool):
    parameter_prompts = 'tools/read-command-output-parameters.json'
    name = "read_command_output"
    progress_exempt = True  # Re-reading a log tail is deliberate polling.
    description_prompt = 'tools/read-command-output.txt'
    description = load_prompt(description_prompt)
    parameters = {"type": "object", "properties": {
        "log_id": {"type": "string"}, "step_id": {"type": "integer", "minimum": 1},
        "call_id": {"type": "string"}, "stream": {"type": "string", "enum": ["stdout", "stderr"]},
        "offset": {"type": "integer", "minimum": 0,
                   "description": ""},
        "limit": {"type": "integer", "minimum": 1, "maximum": 16000,
                  "description": ""},
        "tail": {"type": "boolean", "description": ""},
    }}

    def __init__(self, archive: CommandArchive) -> None:
        self.archive = archive

    async def run(self, *, log_id: str | None = None, step_id: int | None = None,
                  call_id: str | None = None, stream: str = "stdout", offset: int = 0,
                  limit: int = 8000, tail: bool = False) -> ToolResult:
        if log_id is None:
            records = [log.metadata() for log in self.archive.logs.values()
                       if (step_id is None or log.step_id == step_id) and (call_id is None or log.call_id == call_id)]
            # Listing is paged too: long sessions must not create an oversized
            # observation just by asking which logs are available.
            page = records[offset:offset + min(limit, 50)]
            end = offset + len(page)
            return ToolResult.ok({"logs": page, "total_logs": len(records),
                                            "offset_unit": "records", "limit_unit": "records",
                                            "next_offset": end if end < len(records) else None})
        log = self.archive.logs.get(log_id)
        if log is None:
            return ToolResult.error("Unknown command log in this session. Omit log_id to list available logs.")
        target = log.streams[stream]
        content = ""
        start = offset
        try:
            if target.path is not None:
                with target.path.open("rb") as file:
                    if tail:
                        start = max(0, target.retained_bytes - limit * 4)
                    file.seek(start)
                    raw = file.read(max(0, min(limit * 4, target.retained_bytes - start)))
                    if tail:
                        while raw and raw[0] & 0xC0 == 0x80:
                            raw, start = raw[1:], start + 1
                        text = raw.decode("utf-8")
                        content = text[-limit:]
                        start += len(text[:-limit].encode("utf-8"))
                    else:
                        # A maximum-size byte read may stop inside its final
                        # code point. Keep that code point for the next page.
                        import codecs
                        content = codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)[:limit]
        except (OSError, UnicodeError) as exc:
            return ToolResult.error(f"Cannot read log at byte offset {offset}: {exc}. Use offsets returned by previous pages.")
        end = start + len(content.encode("utf-8"))
        return ToolResult.ok({"log_id": log.id, "step_id": log.step_id, "call_id": log.call_id,
            "stream": stream, "offset": start, "next_offset": end if end < target.retained_bytes else None,
            "offset_unit": "UTF-8 bytes", "limit_unit": "characters",
            "retained_bytes": target.retained_bytes, "lost_bytes": target.lost_bytes,
            "retention_error": target.error, "finished": log.finished, "returncode": log.returncode,
            "timed_out": log.timed_out, "content": content})
