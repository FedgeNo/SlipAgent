"""Show neighboring editor sections as a continuous, bounded viewport."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import BufferControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType


class EditorControl(BufferControl):
    def __init__(self, panel: Any, previous: BufferControl) -> None:
        super().__init__(buffer=previous.buffer, input_processors=previous.input_processors,
                         lexer=previous.lexer, focusable=previous.focusable,
                         focus_on_click=previous.focus_on_click)
        self.panel = panel
        self.prefix = 0
        self.active_rows = 0
        self.neighbor_click = False

    def neighbor_lines(self, page: int) -> list[str]:
        file = self.panel.file
        if file is None or not 0 <= page < file.page_count:
            return []
        text = file.page_text(page)
        lines = Document(text).lines
        if text.endswith("\n") and page < file.page_count - 1:
            lines = lines[:-1]
        return lines

    def create_content(self, width: int, height: int, preview_search: bool = False) -> UIContent:
        active = super().create_content(width, height, preview_search)
        before = self.neighbor_lines(self.panel.page - 1)
        after = self.neighbor_lines(self.panel.page + 1)
        self.prefix = len(before)
        active_rows = active.line_count - int(bool(after) and self.buffer.text.endswith("\n"))
        self.active_rows = active_rows

        def get_line(row: int) -> StyleAndTextTuples:
            if row < len(before):
                return [("", before[row] + " ")]
            row -= len(before)
            if row < active_rows:
                return active.get_line(row)
            return [("", after[row - active_rows] + " ")]

        return UIContent(get_line=get_line,
                         line_count=len(before) + active_rows + len(after),
                         cursor_position=Point(active.cursor_position.x, active.cursor_position.y + len(before)))

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        if self.neighbor_click and mouse_event.event_type == MouseEventType.MOUSE_UP:
            self.neighbor_click = False
            return None
        row = mouse_event.position.y - self.prefix
        if 0 <= row < self.active_rows:
            return super().mouse_handler(MouseEvent(Point(mouse_event.position.x, row),
                                                    mouse_event.event_type, mouse_event.button, mouse_event.modifiers))
        if mouse_event.event_type == MouseEventType.MOUSE_DOWN:
            target = self.panel.page - 1 if row < 0 else self.panel.page + 1
            target_row = mouse_event.position.y if row < 0 else row - self.active_rows
            if self.panel.change_page(target - self.panel.page):
                self.neighbor_click = True
                self.buffer.cursor_position = self.buffer.document.translate_row_col_to_index(target_row, mouse_event.position.x)
                self.panel.terminal.app.invalidate()
        return None
