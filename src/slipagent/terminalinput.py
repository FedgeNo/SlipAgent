"""Keep the active UI's POSIX input mode intact across long SSH sessions."""

from __future__ import annotations

import os
from typing import Any

from prompt_toolkit.application import Application


class TerminalInputMode:
    def __init__(self, app: Application[None]) -> None:
        self.app = app
        self.fd: int | None = None
        self.modes: list[Any] | None = None
        if os.name == 'posix':
            try:
                fd = app.input.fileno()
            except NotImplementedError:
                return
            if os.isatty(fd):
                self.fd = fd

    def capture(self, app: Application[None]) -> None:
        # before_render runs after prompt_toolkit enters its owned raw mode.
        if self.fd is not None and self.modes is None:
            import termios
            self.modes = termios.tcgetattr(self.fd)

    def restore(self) -> bool:
        if (self.fd is None or self.modes is None or not self.app.is_running
                or self.app._running_in_terminal or self.app.input.closed
                or not os.isatty(self.fd)):
            return False
        import termios
        current = termios.tcgetattr(self.fd)
        cooked = termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG
        translated = termios.IXON | termios.IXOFF | termios.ICRNL | termios.INLCR | termios.IGNCR
        if not ((current[3] ^ self.modes[3]) & cooked or (current[0] ^ self.modes[0]) & translated):
            return False
        # TCSANOW preserves pending mouse packets and typed input. The native
        # application still owns restoring the original shell modes on exit.
        termios.tcsetattr(self.fd, termios.TCSANOW, self.modes)
        return True
