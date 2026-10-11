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
        self._following = (self.vertical_scroll, self.vertical_scroll_2) == bottom
        self.horizontal_scroll = 0


