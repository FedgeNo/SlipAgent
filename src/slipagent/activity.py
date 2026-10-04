"""Per-invocation progress delivery without adding state to stable tool contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import ContextVar

command_output: ContextVar[Callable[[str], None] | None] = ContextVar("command_output", default=None)


class OutputProgress:
    """Throttle display updates; complete output remains in the command archive."""

    def __init__(self, publish: Callable[[str], None]) -> None:
        self.publish = publish
        self.pending = ""
        self.truncated = False
        self.timer: asyncio.TimerHandle | None = None
        self.closed = False

    def feed(self, text: str) -> None:
        if self.closed or not text:
            return
        self.pending += text
        if len(self.pending) > 32768:
            self.pending = self.pending[-32768:]
            self.truncated = True
        if self.timer is None:
            self.timer = asyncio.get_running_loop().call_later(0.1, self.flush)

    def flush(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        text, self.pending = self.pending, ""
        if self.truncated:
            text = "\n… Live output abbreviated; complete retained output is available through read_command_output.\n" + text
            self.truncated = False
        if text:
            self.publish(text)

    def close(self) -> None:
        self.flush()
        self.closed = True
