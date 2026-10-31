"""Interactive, disk-backed transcript search without pausing agent work."""

from __future__ import annotations

import asyncio
import pickle
import re
from bisect import bisect_right
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent
from prompt_toolkit.widgets import TextArea
from wcwidth import iter_graphemes, width as display_width

from .focus import FocusCoordinator
from .links import CallbackLink
from .transcript import TranscriptFile
from .uiwindows import TranscriptWindow


def excerpt(text: str, start: int, end: int, columns: int) -> StyleAndTextTuples:
    """Center a match in a single grapheme-safe row, marking clipped context."""
    text = text.replace('\t', ' ').replace('\r', ' ')
    columns = max(1, columns)
    match = text[start:end]
    left_budget = max(0, (columns - display_width(match)) // 2 - 1)
    before = ''
    for cluster in reversed(list(iter_graphemes(text[:start]))):
        if display_width(cluster + before) > left_budget:
            break
        before = cluster + before
    prefix = ('…' if len(before) < start else '') + before
    pieces = [("", prefix), ("reverse bold", match), ("", text[end:])]
    truncated = sum(display_width(part) for _, part in pieces) > columns
    available = columns - int(truncated)
    result: StyleAndTextTuples = []
    for style, part in pieces:
        shown = ''
        for cluster in iter_graphemes(part):
            cells = max(0, display_width(cluster))
            if cells > available:
                break
            shown += cluster
            available -= cells
        result.append((style, shown))
        if len(shown) < len(part):
            break
    if truncated:
        result.append(("", '…'))
    return result


class FeedSearch:
    def __init__(self, terminal: Any) -> None:
        from .terminal import TranscriptControl

        self.terminal = terminal
        self.active = False
        self.show_results = True
        self.previous: Any = None
        self.task: asyncio.Task[None] | None = None
        self.results = TranscriptFile()
        self.query = ''
        self.columns = 0
        self.flow: Any = None
        self.status = 'Type a term to search the feed.'
        self.field = TextArea(multiline=False, height=5, prompt='Search: ', accept_handler=self.accept,
                              style='class:user')
        self.field.buffer.on_text_changed += self.changed
        self.window = TranscriptWindow(TranscriptControl(self.content, self.mouse))
        self.container = FocusCoordinator.panel(FormattedTextControl(lambda: self.status), self.window,
                                                lambda: self.window)

    def open(self) -> None:
        if not self.active:
            self.previous = self.terminal.app.layout.current_control
            self.active = True
        self.terminal._focus.focus(self.field)
        self.changed(self.field.buffer)

    def cancel(self) -> None:
        self.active = False
        if self.task is not None:
            self.task.cancel()
        self.terminal._focus.restore(self.previous)
        self.terminal.app.invalidate()

    def changed(self, buffer: Any) -> None:
        if not self.active:
            return
        if self.task is not None:
            self.task.cancel()
        self.show_results = True
        self.task = self.terminal.app.create_background_task(self.search(buffer.text))
        self.terminal.app.invalidate()

    def accept(self, buffer: Any) -> bool:
        self.changed(buffer)
        return False

    async def search(self, query: str) -> None:
        self.results.close()
        self.results = TranscriptFile()
        self.query = query
        self.window.seek()
        self.status = 'Searching…' if query else 'Type a term to search the feed.'
        self.terminal.app.invalidate()
        if not query:
            return
        ui = self.terminal
        columns = ui.main_area.width
        content = ui._transcript_content(columns)
        # Restored transcripts fill upward independently of the working turn.
        layouts = ui._transcript_layouts
        state = layouts.states[columns]
        while state.start:
            task = layouts.tasks.get(columns)
            if task is None:
                content = ui._transcript_content(columns)
                state = layouts.states[columns]
            else:
                await asyncio.shield(task)
                content = ui._transcript_content(columns)
                state = layouts.states[columns]
        flow = ui._flow
        self.columns, self.flow = columns, flow
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        margin = columns + len(query)
        buffer = ''
        base = processed = 0
        starts: list[int] = []
        rows: list[int] = []

        def collect(final: bool = False) -> None:
            nonlocal buffer, base, processed, starts, rows
            stop = len(buffer) + 1 if final else max(0, len(buffer) - margin)
            for match in pattern.finditer(buffer, max(0, processed - base)):
                if match.start() >= stop:
                    break
                row = rows[max(0, bisect_right(starts, base + match.start()) - 1)]
                left = max(buffer.rfind('\n', 0, match.start()) + 1, match.start() - columns)
                right = buffer.find('\n', match.end())
                right = min(len(buffer) if right < 0 else right, match.end() + columns)
                record = (row, buffer[left:right], match.start() - left, match.end() - left)
                self.results.append_bytes([pickle.dumps(record, pickle.HIGHEST_PROTOCOL)])
                processed = base + match.end()
            processed = max(processed, base + stop)
            discard = max(0, processed - base - margin)
            buffer = buffer[discard:]
            base += discard
            keep = max(0, bisect_right(starts, base) - 1)
            starts, rows = starts[keep:], rows[keep:]

        for row in range(max(0, content.line_count - 1)):
            if flow is not ui._flow or columns != ui.main_area.width:
                self.changed(self.field.buffer)
                return
            starts.append(base + len(buffer))
            rows.append(row)
            buffer += flow.copy_row(row).copy_text
            collect()
            if row % 128 == 0:
                self.terminal.app.invalidate()
                await asyncio.sleep(0)
        collect(final=True)
        self.status = f'{len(self.results)} results — click to go to the feed; Esc closes search.'
        ui.app.invalidate()

    def content(self, width: int) -> UIContent:
        results = self.results

        def get_line(row: int) -> StyleAndTextTuples:
            if results is not self.results or row % 2 or not 0 <= row // 2 < len(results):
                return []
            index = row // 2
            _, text, start, end = pickle.loads(results.read_bytes(index))
            action = CallbackLink('', ('feed-search', index), self.terminal._link_hover(),
                                  lambda: self.terminal.app.create_background_task(self.activate(index)))
            return [(fragment[0] + ' class:clickable', fragment[1], action.mouse)
                    for fragment in excerpt(text, start, end, width)]

        content = UIContent(get_line=get_line, line_count=max(1, len(results) * 2 - 1), show_cursor=False)
        content.__dict__['focus_rows'] = range(0, len(results) * 2, 2)
        content.__dict__['is_current'] = lambda: results is self.results
        return content

    def mouse(self, event: MouseEvent) -> object:
        content = self.content(self.terminal.main_area.width)
        column = 0
        for fragment in content.get_line(event.position.y):
            end = column + len(fragment[1])
            if column <= event.position.x < end and len(fragment) == 3:
                return fragment[2](event)
            column = end
        return NotImplemented

    async def activate(self, index: int) -> None:
        ui = self.terminal
        ui._transcript_content(ui.main_area.width)
        if self.flow is not ui._flow or self.columns != ui.main_area.width:
            if self.task is not None and not self.task.done():
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            await self.search(self.query)
        if index >= len(self.results):
            return
        row, *_ = pickle.loads(self.results.read_bytes(index))
        self.show_results = False
        ui.transcript.seek(max(0, row - 1))
        ui._focus.focus(self.field)
        ui.app.invalidate()

    def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
        self.results.close()
