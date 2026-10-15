"""Mouse routing between the feed, input, editor, menus, and inspector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

if TYPE_CHECKING:
    from .terminal import TerminalUI


def route_mouse(ui: TerminalUI, app: Application[None]) -> None:
    from .terminal import FEED_LINK_PREFIXES, InputWindow

    original = app.renderer.mouse_handlers

    def mouse_handler(event: MouseEvent) -> object:
        if event.event_type == MouseEventType.MOUSE_UP:
            ui._scrollbar.drag_offset = None
        previous_hover = ui.__dict__.get("_hovered_control")
        if event.event_type == MouseEventType.MOUSE_MOVE:
            ui.__dict__["_hovered_control"] = None
        if event.event_type == MouseEventType.MOUSE_DOWN:
            ui._selection_dragging = False
            if event.button == MouseButton.LEFT:
                ui._focus.click(event.position)
        if event.event_type in (MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN):
            size = ui.output.get_size()
            if ui._menu_future is not None:
                ui._move_menu(-1 if event.event_type == MouseEventType.SCROLL_UP else 1)
                return None
            if ui._editor_active and event.position.y < size.rows - ui._footer_rows() - ui._queued_prompt_height() and ui._menu_future is None:
                if not ui.main_area.in_inspector(event.position.x):
                    ui._editor_panel.move_cursor(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
                    app.invalidate()
                    return None
            if ui._menu_future is None and size.rows - ui._footer_rows() + 1 <= event.position.y < size.rows - 1:
                if ui._editor_active:
                    return None
                window = ui.input.window
                if isinstance(window, InputWindow):
                    window.scroll(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
                app.invalidate()
                return None
            if ui.main_area.in_inspector(event.position.x) and event.position.y < size.rows - ui._footer_rows() - ui._queued_prompt_height() and ui._menu_future is None:
                screen = app.renderer._last_screen
                if screen is not None:
                    for pane in (ui._files_window, ui._calls_window):
                        position = screen.visible_windows_to_write_positions.get(pane)
                        if position is not None and position.ypos <= event.position.y < position.ypos + position.height:
                            pane.scroll(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
                            break
                app.invalidate()
                return None
            ui._scroll_output(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
            app.invalidate()
            return None
        try:
            screen = app.renderer._last_screen
            if screen is not None:
                position = screen.visible_windows_to_write_positions.get(ui.transcript)
                info = ui.transcript.render_info
                if position is not None and info is not None and ui.transcript in app.layout.walk_through_modal_area() and (
                    position.xpos <= event.position.x < position.xpos + position.width
                    and position.ypos <= event.position.y < position.ypos + position.height
                ):
                    source = info.visible_line_to_row_col.get(event.position.y - position.ypos)
                    if source is not None and any(token.startswith(FEED_LINK_PREFIXES)
                                                  for style, _, *_ in info.ui_content.get_line(source[0])
                                                  for token in style.split()):
                        # Window clamps blank columns to the last character, which would activate the link.
                        return ui._select_transcript(MouseEvent(
                            Point(x=source[1] + event.position.x - position.xpos, y=source[0]),
                            event.event_type, event.button, event.modifiers,
                        ))
                for hit_window in (ui._menu_window, ui._files_window, ui._calls_window):
                    position = screen.visible_windows_to_write_positions.get(hit_window)
                    if position is None or not (position.xpos <= event.position.x < position.xpos + position.width
                                               and position.ypos <= event.position.y < position.ypos + position.height):
                        continue
                    info = hit_window.render_info
                    row = event.position.y - position.ypos
                    source = info.visible_line_to_row_col.get(row) if info is not None else None
                    # Window otherwise clamps blank rows onto the last item.
                    if source is None or info is None or not fragment_list_to_text(info.ui_content.get_line(source[0])):
                        return None
            return original.mouse_handlers[event.position.y][event.position.x](event)
        finally:
            # Releases outside the transcript must also end its drag.
            if event.event_type == MouseEventType.MOUSE_UP:
                ui._selection_dragging = False
            if previous_hover != ui.__dict__.get("_hovered_control"):
                app.invalidate()

    # Wheels preserve drafts; clicks retain native handling after focus.
    handlers = MouseHandlers()
    size = ui.output.get_size()
    handlers.set_mouse_handler_for_range(0, size.columns, 0, size.rows, mouse_handler)
    app.renderer.mouse_handlers = handlers
