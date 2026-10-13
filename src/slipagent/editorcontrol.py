"""Show neighboring editor sections as a continuous, bounded viewport."""

from __future__ import annotations

from typing import Any
from bisect import bisect_right

from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import BufferControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from wcwidth import iter_graphemes, width as display_width

from .editorbuffer import grapheme_edge


def visual_offsets(content: UIContent, width: int, wrapped: bool) -> list[int]:
    offsets = [0]
    for row in range(content.line_count):
        offsets.append(offsets[-1] + (content.get_height_for_line(row, width, None) if wrapped else 1))
    return offsets


class EditorWindow(Window):
    """Scroll by displayed rows without snapping whole wrapped lines into view."""

    def _scroll_when_linewrapping(self, ui_content: UIContent, width: int, height: int) -> None:
        width = max(1, width)
        offsets = visual_offsets(ui_content, width, True)
        cursor = ui_content.cursor_position
        line = fragment_list_to_text(ui_content.get_line(cursor.y))
        cursor_row = offsets[cursor.y] + display_width(line[:cursor.x]) // width
        top = offsets[min(self.vertical_scroll, ui_content.line_count - 1)] + self.vertical_scroll_2
        if cursor_row < top:
            top = cursor_row
        elif cursor_row >= top + height:
            top = cursor_row - height + 1
        self._set_visual_scroll(offsets, top, height)
        self.horizontal_scroll = 0

    def _set_visual_scroll(self, offsets: list[int], top: int, height: int) -> None:
        top = max(0, min(top, max(0, offsets[-1] - height)))
        self.vertical_scroll = max(0, bisect_right(offsets, top) - 1)
        self.vertical_scroll_2 = top - offsets[self.vertical_scroll]

    def scroll_rows(self, rows: int) -> None:
        info = self.render_info
        if info is None:
            return
        content = self.content.create_content(info.window_width, info.window_height)
        offsets = visual_offsets(content, max(1, info.window_width), bool(self.wrap_lines()))
        top = offsets[min(self.vertical_scroll, content.line_count - 1)] + self.vertical_scroll_2
        self._set_visual_scroll(offsets, top + rows, info.window_height)


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

    def move_visual(self, rows: int) -> None:
        window = self.panel.input.window
        info = window.render_info
        width = max(1, info.window_width if info is not None else self.panel.terminal.output.get_size().columns)
        height = info.window_height if info is not None else 1
        content = self.create_content(width, height)
        wrapped = bool(window.wrap_lines())
        offsets = visual_offsets(content, width, wrapped)
        cursor = content.cursor_position
        line = fragment_list_to_text(content.get_line(cursor.y))
        display_column = display_width(line[:cursor.x])
        column = self.buffer.preferred_column
        if column is None:
            column = display_column % width if wrapped else display_column
        current = offsets[cursor.y] + (display_column // width if wrapped else 0)
        target = max(0, min(offsets[-1] - 1, current + rows))
        row = max(0, bisect_right(offsets, target) - 1)
        desired_column = (target - offsets[row]) * width + column if wrapped else column
        text = fragment_list_to_text(content.get_line(row))[:-1]
        index = 0
        used = 0
        for cluster in iter_graphemes(text):
            size = display_width(cluster)
            if used + size > desired_column:
                break
            used += size
            index += len(cluster)
        local_row = row - self.prefix
        if local_row < 0:
            changed = self.panel.change_page(-1)
            local_row = row
        elif local_row >= self.active_rows:
            changed = self.panel.change_page(1)
            local_row -= self.active_rows
        else:
            changed = True
        if changed:
            self.buffer.cursor_position = self.buffer.document.translate_row_col_to_index(local_row, index)
            self.buffer.preferred_column = column

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        if self.neighbor_click and mouse_event.event_type == MouseEventType.MOUSE_UP:
            self.neighbor_click = False
            return None
        row = mouse_event.position.y - self.prefix
        if 0 <= row < self.active_rows:
            result = super().mouse_handler(MouseEvent(Point(mouse_event.position.x, row),
                                                      mouse_event.event_type, mouse_event.button, mouse_event.modifiers))
            position = self.buffer.cursor_position
            edge = grapheme_edge(self.buffer.text, position, -1)
            if position not in (0, grapheme_edge(self.buffer.text, edge, 1)):
                self.buffer.cursor_position = edge
            return result
        if mouse_event.event_type == MouseEventType.MOUSE_DOWN:
            target = self.panel.page - 1 if row < 0 else self.panel.page + 1
            target_row = mouse_event.position.y if row < 0 else row - self.active_rows
            if self.panel.change_page(target - self.panel.page):
                self.neighbor_click = True
                self.buffer.cursor_position = self.buffer.document.translate_row_col_to_index(target_row, mouse_event.position.x)
                self.panel.terminal.app.invalidate()
        return None
