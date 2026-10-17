"""Negotiate SGR pixel mouse input while retaining cell-based UI hit testing."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import Input
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding import KeyBindings, KeyPress, KeyPressEvent
from prompt_toolkit.key_binding.bindings.mouse import load_mouse_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.output.vt100 import Vt100_Output

if TYPE_CHECKING:
    from .terminal import TerminalUI


_MODE_PREFIX = "\x1b[?1016;"
_SIZE_PREFIX = "\x1b[6;"
_MODE_REPORT = re.compile(r"\x1b\[\?1016;([0-4])\$y\Z")
_SIZE_REPORT = re.compile(r"\x1b\[6;([0-9]{1,5});([0-9]{1,5})t\Z")
_SIZE_PARTIAL = re.compile(r"[0-9]{0,5}(?:;[0-9]{0,5})?\Z")
_MOUSE_REPORT = re.compile(r"\x1b\[<([0-9]+);([0-9]+);([0-9]+)([mM])\Z")


class PixelMouseParser(Vt100Parser):
    """Consume capability replies before the ordinary keyboard/paste parser."""

    def __init__(self, callback: Callable[[KeyPress], None], mouse: PixelMouse) -> None:
        self.callback = callback
        self.mouse = mouse
        self.pending = ""
        super().__init__(self._key)

    def _key(self, key: KeyPress) -> None:
        match = _MOUSE_REPORT.fullmatch(key.data) if key.key == Keys.Vt100MouseEvent else None
        if match is not None:
            if self.mouse.enabling:
                # The confirmation separates buffered cell reports from pixels.
                return
            if self.mouse.enabled:
                button, x, y, suffix = match.groups()
                width, height = self.mouse.cell_size
                column, row = (int(x) - 1) / width, (int(y) - 1) / height
                if column < 0 or row < 0:
                    return
                key = KeyPress(Keys.Vt100MouseEvent,
                               f"\x1b[<{button};{int(column) + 1};{int(row) + 1}{suffix}")
                key.__dict__["pixel_position"] = (column, row)
        self.callback(key)

    def feed(self, data: str) -> None:
        for index, char in enumerate(data):
            if self._in_bracketed_paste:
                super().feed(data[index:])
                return
            if not self.pending and char != "\x1b":
                super().feed(char)
                continue
            self.pending += char
            mode = _MODE_REPORT.fullmatch(self.pending)
            size = _SIZE_REPORT.fullmatch(self.pending)
            if mode is not None or size is not None:
                self.pending = ""
                if mode is not None:
                    self.mouse.mode_report(int(mode[1]))
                elif size is not None:
                    self.mouse.size_report(int(size[2]), int(size[1]))
                continue
            if (_MODE_PREFIX.startswith(self.pending) or _SIZE_PREFIX.startswith(self.pending)
                    or self.pending.startswith(_MODE_PREFIX)
                    and any(f"{value}$y".startswith(self.pending[len(_MODE_PREFIX):]) for value in range(5))
                    or self.pending.startswith(_SIZE_PREFIX)
                    and _SIZE_PARTIAL.fullmatch(self.pending[len(_SIZE_PREFIX):])):
                continue
            ordinary, self.pending = self.pending, ""
            if ordinary.endswith("\x1b"):
                ordinary, self.pending = ordinary[:-1], "\x1b"
            super().feed(ordinary)

    def flush(self) -> None:
        # Keep recognized, fragmented replies out of the prompt. Bare Escape
        # and ordinary key prefixes retain prompt_toolkit's normal timeout.
        if not (self.pending.startswith(_MODE_PREFIX) or self.pending.startswith(_SIZE_PREFIX)):
            ordinary, self.pending = self.pending, ""
            super().feed(ordinary)
        super().flush()


class PixelMouse:
    """Own mode negotiation, cell measurements, and input-parser restoration."""

    def __init__(self, input: Input, output: Vt100_Output, parser: Vt100Parser) -> None:
        self.input = input
        self.output = output
        self.original_parser = parser
        self.parser = PixelMouseParser(parser.feed_key_callback, self)
        self.app: Application[Any] | None = None
        self.cell_size = (0, 0)
        self.screen_size: Size | None = output.get_size()
        self.supported: bool | None = None
        self.enabled = False
        self.enabling = False
        self.deadline = 0.0
        self.closed = False
        setattr(input, "vt100_parser", self.parser)
        self._write("\x1b[?1016l\x1b[?1016$p\x1b[16t")

    @classmethod
    def attach(cls, app: Application[Any]) -> PixelMouse | None:
        parser = getattr(app.input, "vt100_parser", None)
        if isinstance(parser, Vt100Parser) and isinstance(app.output, Vt100_Output):
            mouse = cls(app.input, app.output, parser)
            mouse.app = app
            app.before_render += mouse.before_render
            return mouse
        return None

    def _write(self, text: str) -> None:
        self.output.write_raw(text)
        self.output.flush()

    def _enable(self) -> None:
        if (self.supported and all(self.cell_size) and not self.enabled
                and not self.enabling and not self.closed):
            self.enabling = True
            self.deadline = time.monotonic() + 2
            self._write("\x1b[?1016h\x1b[?1016$p")

    def mode_report(self, value: int) -> None:
        if self.closed:
            return
        if self.enabling:
            self.enabling = False
            self.enabled = value in (1, 3)
            if not self.enabled:
                self.supported = False
                self._write("\x1b[?1016l\x1b[?1006h")
        elif self.supported is None:
            self.supported = value in (1, 2, 3)
            self._enable()

    def size_report(self, width: int, height: int) -> None:
        if width > 0 and height > 0 and not self.closed:
            self.cell_size = (width, height)
            self._enable()

    def before_render(self, app: Application[Any]) -> None:
        if self.supported and not app.is_done and not app.renderer._mouse_support_enabled:
            # Resizing/resuming resets the renderer's mouse encoding. Measure
            # again even if a font change leaves the same character-grid size.
            self.enabled = self.enabling = False
            self.cell_size = (0, 0)
            self.screen_size = None

    def refresh(self) -> None:
        if self.closed:
            return
        if self.enabling and time.monotonic() >= self.deadline:
            self.enabling = self.enabled = False
            self.supported = False
            self._write("\x1b[?1016l\x1b[?1006h")
        size = self.output.get_size()
        if size != self.screen_size:
            self.screen_size = size
            if self.supported:
                self._write("\x1b[16t")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.enabled = self.enabling = False
        self._write("\x1b[?1016l")
        if self.app is not None:
            self.app.before_render -= self.before_render
            self.app = None
        if getattr(self.input, "vt100_parser", None) is self.parser:
            setattr(self.input, "vt100_parser", self.original_parser)


def bind_pixel_mouse(bindings: KeyBindings, ui: TerminalUI) -> None:
    native = load_mouse_bindings().get_bindings_for_keys((Keys.Vt100MouseEvent,))[0].handler

    @bindings.add(Keys.Vt100MouseEvent)
    def mouse(event: KeyPressEvent) -> Any:
        ui.__dict__["_pixel_position"] = event.key_sequence[-1].__dict__.get("pixel_position")
        try:
            return native(event)
        finally:
            ui.__dict__.pop("_pixel_position", None)
