"""Read-only panel ownership over disk-backed snapshots and controls."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from prompt_toolkit.layout import HSplit
from prompt_toolkit.formatted_text import StyleAndTextTuples

from .inspection import IndexedText
from .uiwindows import TranscriptWindow
from .viewers import DiffViewControl, TextViewControl
from .links import ActionControl
from .focus import FocusCoordinator

FileSignature = tuple[int, int, int, int, int]


def file_signature(metadata: os.stat_result) -> FileSignature:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)


class TextViewerPanel:
    """Own one immutable snapshot and its renderer, without agent/edit state."""

    control_type = TextViewControl

    def __init__(self, title: str, text: str | IndexedText) -> None:
        self.title = " ".join(title.split())
        self.control = self.control_type(text)
        self.changed = False
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.control.close()


class DiffViewerPanel(TextViewerPanel):
    control_type = DiffViewControl


class FileViewerPanel(TextViewerPanel):
    """Track the source independently of the immutable displayed snapshot."""

    def __init__(self, title: str, text: IndexedText, source: Path, loaded: FileSignature) -> None:
        super().__init__(title, text)
        self.source = source
        self.loaded = loaded
        self.check_pending = False
        self.check_at = 0.0

    async def check(self) -> None:
        try:
            try:
                current = file_signature(await asyncio.to_thread(self.source.stat))
            except OSError:
                current = None
            if not self.closed:
                self.changed = current != self.loaded
        finally:
            self.check_pending = False


class ViewerDeck:
    """Own the active panel, viewport, and stale asynchronous-open tokens."""

    def __init__(self, header: Callable[[], StyleAndTextTuples]) -> None:
        self.panel = TextViewerPanel("", "")
        self.active = False
        self.request = 0
        self.window = TranscriptWindow(self.panel.control)
        self.window._following = False
        self.container = HSplit([
            FocusCoordinator.header(ActionControl(header), lambda: self.window, style="class:status"), self.window,
        ])

    def next_request(self) -> int:
        self.request += 1
        return self.request

    def show(self, panel: TextViewerPanel, line: int = 1) -> None:
        self.next_request()
        self.panel.close()
        self.panel = panel
        self.window.content = panel.control
        lines = panel.control.lines
        self.window.vertical_scroll = lines.line_index(line) if isinstance(lines, IndexedText) else max(0, line - 1)
        self.window.vertical_scroll_2 = self.window._pending_scroll = 0
        self.window._following = False
        self.active = True

    def close(self) -> None:
        self.next_request()
        self.active = False
        self.panel.close()

    def tick(self, spawn: Callable[[Coroutine[Any, Any, None]], object], invalidate: Callable[[], None]) -> None:
        panel = self.panel
        now = time.monotonic()
        if not self.active or not isinstance(panel, FileViewerPanel) or panel.check_pending or now < panel.check_at:
            return
        panel.check_pending = True
        panel.check_at = now + 1
        async def check() -> None:
            await panel.check()
            if panel is self.panel and self.active:
                invalidate()
        spawn(check())
