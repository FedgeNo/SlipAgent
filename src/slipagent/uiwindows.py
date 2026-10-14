"""Wrapped scrolling windows shared by transcript and inspection panels."""

from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import UIContent, UIControl


class TranscriptWindow(Window):
    """Scroll wrapped output by screen rows without moving input focus."""

    def __init__(self, content: UIControl) -> None:
        super().__init__(content, wrap_lines=True)
        self._following = True
        self._pending_scroll = 0

    def scroll(self, rows: int) -> None:
        self._pending_scroll += rows

    def reveal_row(self, row: int) -> None:
        self.__dict__["_reveal_row"] = row
        self._following = False
        self._pending_scroll = 0

    def _scroll(self, ui_content: UIContent, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return

        def line_height(line: int) -> int:
            return ui_content.get_height_for_line(line, width, self.get_line_prefix)

        # The tail can start inside a wrapped line, including lines taller
        # than the entire transcript viewport.
        remaining = height
        bottom = (0, 0)
        for line in range(ui_content.line_count - 1, -1, -1):
            size = line_height(line)
            if size > remaining:
                bottom = (line, size - remaining)
                break
            remaining -= size
            if remaining == 0:
                bottom = (line, 0)
                break

        line, offset = min(
            bottom if self._following else (self.vertical_scroll, self.vertical_scroll_2), bottom,
        )
        offset = min(offset, line_height(line) - 1)
        rows = self._pending_scroll
        self._pending_scroll = 0
        while rows < 0 and (line, offset) > (0, 0):
            if offset:
                offset -= 1
            else:
                line -= 1
                offset = line_height(line) - 1
            rows += 1
        while rows > 0 and (line, offset) < bottom:
            if offset + 1 < line_height(line):
                offset += 1
            else:
                line += 1
                offset = 0
            rows -= 1
        self.vertical_scroll, self.vertical_scroll_2 = min((line, offset), bottom)
        selected = self.__dict__.pop("_reveal_row", None)
        if selected is not None:
            selected = min(max(0, selected), ui_content.line_count - 1)
            if selected < self.vertical_scroll or selected == self.vertical_scroll and self.vertical_scroll_2:
                self.vertical_scroll, self.vertical_scroll_2 = max(0, selected - (height > 1)), 0
            else:
                before = sum(line_height(row) for row in range(self.vertical_scroll, selected)) - self.vertical_scroll_2
                overflow = before + min(height, line_height(selected)) - height
                if overflow > 0:
                    self.vertical_scroll, self.vertical_scroll_2 = max(0, selected - (height > 1)), 0
        self.vertical_scroll, self.vertical_scroll_2 = min((self.vertical_scroll, self.vertical_scroll_2), bottom)
        self._following = (self.vertical_scroll, self.vertical_scroll_2) == bottom
        self.horizontal_scroll = 0


class ListWindow(TranscriptWindow):
    """Reveal selected items by the minimum number of wrapped screen rows."""

    def __init__(self, content: UIControl) -> None:
        super().__init__(content)
        self._reveal_selection = True
        self._size: tuple[int, int] | None = None

    def reveal(self) -> None:
        self._reveal_selection = True
        self._following = False
        self._pending_scroll = 0

    def _scroll(self, ui_content: UIContent, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        reveal = self._reveal_selection or self._size != (width, height)
        self._reveal_selection = False
        self._size = (width, height)
        if reveal:
            self._following = False
        super()._scroll(ui_content, width, height)
        if not reveal:
            return

        index = ui_content.cursor_position.y
        if index < self.vertical_scroll or index == self.vertical_scroll and self.vertical_scroll_2:
            self.vertical_scroll = index
            self.vertical_scroll_2 = 0
        else:
            def line_height(line: int) -> int:
                return ui_content.get_height_for_line(line, width, self.get_line_prefix)

            before = sum(line_height(line) for line in range(self.vertical_scroll, index)) - self.vertical_scroll_2
            # Items taller than the viewport reveal their beginning; all
            # shorter items include their final wrapped row.
            rows = before + min(line_height(index), height) - height
            if rows > 0:
                self._following = False
                self._pending_scroll = rows
                super()._scroll(ui_content, width, height)
        self._following = False
