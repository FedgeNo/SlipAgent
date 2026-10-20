"""Wrapped scrolling windows shared by transcript and inspection panels."""

from collections.abc import Callable

from prompt_toolkit.application.current import get_app
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType


class TranscriptWindow(Window):
    """Scroll wrapped output by screen rows without moving input focus."""

    def __init__(self, content: UIControl) -> None:
        super().__init__(content, wrap_lines=True)
        self._following = True
        self._pending_scroll = 0

    def scroll(self, rows: int) -> None:
        self._pending_scroll += rows

    def seek(self, row: int = 0, *, following: bool = False) -> None:
        """Set a new viewport origin and discard pending scroll gestures."""
        self.vertical_scroll = max(0, row)
        self.vertical_scroll_2 = self._pending_scroll = 0
        self._following = following
        self.__dict__.pop("_reveal_row", None)

    def reveal_row(self, row: int) -> None:
        self.__dict__["_reveal_row"] = row
        self._following = False
        self._pending_scroll = 0

    def bottom_origin(self, ui_content: UIContent, width: int, height: int) -> tuple[int, int]:
        """The line and row offset that fill the viewport with the end of the content."""
        # The tail can start inside a wrapped line, including lines taller
        # than the entire transcript viewport.
        remaining = height
        for line in range(ui_content.line_count - 1, -1, -1):
            size = ui_content.get_height_for_line(line, width, self.get_line_prefix)
            if size > remaining:
                return line, size - remaining
            remaining -= size
            if remaining == 0:
                return line, 0
        return 0, 0

    def _scroll(self, ui_content: UIContent, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return

        def line_height(line: int) -> int:
            return ui_content.get_height_for_line(line, width, self.get_line_prefix)

        bottom = self.bottom_origin(ui_content, width, height)
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


class TranscriptScrollbar(UIControl):
    """A clickable, draggable track for the visible transcript window."""

    def __init__(self, target: Callable[[], TranscriptWindow | None]) -> None:
        self.target = target
        self.height = 0
        self.drag_offset: float | None = None

    def geometry(self) -> tuple[int, int, tuple[int, int]]:
        """The thumb's top row and height, and the viewport origin at the end of the content."""
        window = self.target()
        info = window.render_info if window is not None else None
        if window is None or info is None or self.height <= 0:
            return 0, self.height, (0, 0)
        # The track ends where wheel scrolling stops: at the origin that shows
        # the last row, which a wrapped tail puts past the last line index that
        # fits a viewport of unwrapped lines.
        bottom = window.bottom_origin(info.ui_content, info.window_width, info.window_height)
        visible = max(1, len(info.displayed_lines))
        thumb = max(1, min(self.height, self.height * visible // max(1, info.content_height)))
        top = round((self.height - thumb) * min(info.vertical_scroll, bottom[0]) / bottom[0]) if bottom[0] else 0
        return top, thumb, bottom

    def create_content(self, width: int, height: int) -> UIContent:
        self.height = height
        top, thumb, bottom = self.geometry()
        return UIContent(
            get_line=lambda row: [("class:scroll-track", "┃" if bottom[0] and top <= row < top + thumb else " ")],
            line_count=height, show_cursor=False,
        )

    def mouse_handler(self, mouse_event: MouseEvent) -> object:
        return self.pointer(mouse_event, mouse_event.position.y)

    def pointer(self, mouse_event: MouseEvent, row: float) -> object:
        window = self.target()
        if window is None:
            return None
        top, thumb, bottom = self.geometry()
        maximum = bottom[0]
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            self.drag_offset = None
            return None
        if mouse_event.event_type == MouseEventType.MOUSE_DOWN and mouse_event.button == MouseButton.LEFT:
            if top <= row < top + thumb:
                # The painted thumb is cell-rounded; anchor dragging to the
                # actual position so grabbing it does not quantize the viewport.
                origin = (self.height - thumb) * min(window.vertical_scroll, maximum) / maximum if maximum else 0
                self.drag_offset = row - origin
                return None
            self.drag_offset = thumb // 2
        elif mouse_event.event_type != MouseEventType.MOUSE_MOVE or self.drag_offset is None:
            return NotImplemented
        travel = self.height - thumb
        # The anchor keeps a cell-rounded thumb from reaching the end of the
        # track, so the track's end cells themselves mean the ends of the
        # content, as does a thumb painted in them: holding it there keeps the
        # feed following each new row rather than the row it reached first.
        position = min(travel, max(0, row - self.drag_offset))
        if row >= self.height - 1 or round(position) >= travel:
            position = travel
        elif row < 1 or round(position) <= 0:
            position = 0
        line = round(maximum * position / travel) if travel else 0
        window.vertical_scroll, window.vertical_scroll_2 = bottom if line >= maximum else (line, 0)
        window._pending_scroll = 0
        window._following = line >= maximum
        get_app().invalidate()
        return None


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
