"""Keyboard commands and focus filters for the terminal interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys

if TYPE_CHECKING:
    from .terminal import TerminalUI


def build_bindings(ui: TerminalUI) -> KeyBindings:
    bindings = KeyBindings()
    in_menu = Condition(lambda: ui._menu_future is not None and ui.app.layout.current_window is ui._menu_window)
    in_setting = Condition(lambda: ui.__dict__.get("_options_field_active", False) and ui.app.layout.current_control is ui._options_panel.field.control)
    in_inspection = Condition(ui._inspection_focused) & ~in_menu & ~in_setting
    in_editor = Condition(lambda: ui.__dict__.get("_editor_active", False) and ui.app.layout.current_control is ui._editor_input.control) & ~in_menu & ~in_setting
    in_viewer = Condition(lambda: ui._viewer.active and ui.app.layout.current_window is ui._viewer_window) & ~in_menu & ~in_editor & ~in_setting & ~in_inspection

    in_link = Condition(lambda: ui.app.layout.current_control.__dict__.get("_keyboard_key") is not None)
    in_feed = Condition(lambda: ui.app.layout.current_window is ui.transcript)
    at_feed_prompt = Condition(lambda: ui.app.layout.current_control is ui.input.control and
                              not (ui._editor_active or ui._viewer_active or ui.context_visible))

    @bindings.add("tab", filter=~in_editor)
    def cycle_panel(event: KeyPressEvent) -> None:
        ui._cycle_panel()

    @bindings.add("s-tab")
    def cycle_component(event: KeyPressEvent) -> None:
        ui._cycle_panel(components=True)

    @bindings.add("enter", filter=in_link, eager=True)
    def activate_link(event: KeyPressEvent) -> None:
        ui._focus.activate()

    @bindings.add("escape", filter=in_link | in_feed, eager=True)
    def leave_feed(event: KeyPressEvent) -> None:
        if ui._menu_future is not None:
            ui._finish_menu(None)
        elif ui._options_field_active:
            ui._options_panel.cancel()
        else:
            ui._focus.restore_prompt()

    @bindings.add(Keys.Any, filter=in_link | in_feed)
    def ignore_feed_text(event: KeyPressEvent) -> None:
        pass

    @bindings.add("up", filter=in_feed)
    @bindings.add("down", filter=in_feed)
    @bindings.add("pageup", filter=in_feed | at_feed_prompt)
    @bindings.add("pagedown", filter=in_feed | at_feed_prompt)
    def scroll_feed(event: KeyPressEvent) -> None:
        key = event.key_sequence[-1].key
        rows = max(1, ui.output.get_size().rows - ui._footer_rows() - ui._queued_prompt_height() - 1) if key in {"pageup", "pagedown"} else 1
        ui.transcript.content.__dict__.pop("_keyboard_key", None)
        ui.transcript.scroll(-rows if key in {"up", "pageup"} else rows)
        ui.app.invalidate()

    @bindings.add("home", filter=in_feed)
    @bindings.add("end", filter=in_feed)
    def edge_feed(event: KeyPressEvent) -> None:
        ui.transcript.content.__dict__.pop("_keyboard_key", None)
        ui.transcript.seek(following=event.key_sequence[-1].key == Keys.End)
        ui.app.invalidate()

    @bindings.add("up", filter=in_inspection)
    @bindings.add("down", filter=in_inspection)
    def move_inspection(event: KeyPressEvent) -> None:
        ui._move_inspection(-1 if event.key_sequence[-1].key == Keys.Up else 1)

    @bindings.add("enter", filter=in_inspection)
    def activate_inspection(event: KeyPressEvent) -> None:
        ui._activate_inspection()

    @bindings.add("escape", filter=in_inspection, eager=True)
    def leave_inspection(event: KeyPressEvent) -> None:
        ui._focus.restore_primary()

    @bindings.add(Keys.Any, filter=in_inspection)
    def ignore_inspection_text(event: KeyPressEvent) -> None:
        pass

    @bindings.add("f4", filter=~in_menu & ~in_setting)
    def options_screen(event: KeyPressEvent) -> None:
        ui.app.create_background_task(ui._options_panel.show())

    @bindings.add("escape", filter=in_setting, eager=True)
    def cancel_setting(event: KeyPressEvent) -> None:
        ui._options_panel.cancel()

    @bindings.add("tab", filter=in_editor)
    def editor_tab(event: KeyPressEvent) -> None:
        if not ui._editor_panel.busy and not ui._editor_panel.read_only:
            ui._editor_input.buffer.insert_text(" " * ui._options_panel.values.get("tab_width", 4))

    @bindings.add("c-z", filter=in_editor)
    @bindings.add("c-y", filter=in_editor)
    def editor_undo(event: KeyPressEvent) -> None:
        if not ui._editor_panel.busy and not ui._editor_panel.read_only:
            if event.key_sequence[-1].key == Keys.ControlZ:
                ui._editor_input.buffer.undo()
            else:
                ui._editor_input.buffer.redo()

    @bindings.add("c-s", filter=in_editor)
    def save_editor(event: KeyPressEvent) -> None:
        ui.app.create_background_task(ui._save_editor())

    @bindings.add("backspace", filter=in_editor)
    @bindings.add("delete", filter=in_editor)
    def delete_editor(event: KeyPressEvent) -> None:
        direction = -1 if event.key_sequence[-1].key == Keys.Backspace else 1
        for _ in range(event.arg):
            ui._editor_panel.delete_grapheme(direction)

    @bindings.add("escape", filter=in_editor, eager=True)
    def close_editor(event: KeyPressEvent) -> None:
        ui.app.create_background_task(ui._close_editor())

    @bindings.add("c-r", filter=in_editor)
    def reload_editor(event: KeyPressEvent) -> None:
        ui.app.create_background_task(ui._reload_editor())

    @bindings.add("pageup", filter=in_editor)
    @bindings.add("pagedown", filter=in_editor)
    def page_editor(event: KeyPressEvent) -> None:
        ui._editor_panel.move_screen(-1 if event.key_sequence[-1].key == Keys.PageUp else 1)

    @bindings.add("up", filter=in_editor)
    @bindings.add("down", filter=in_editor)
    def move_editor_vertical(event: KeyPressEvent) -> None:
        ui._editor_panel.move_cursor(-event.arg if event.key_sequence[-1].key == Keys.Up else event.arg)

    @bindings.add("left", filter=in_editor)
    @bindings.add("right", filter=in_editor)
    def move_editor_horizontal(event: KeyPressEvent) -> None:
        direction = -1 if event.key_sequence[-1].key == Keys.Left else 1
        for _ in range(event.arg):
            ui._editor_panel.move_horizontal(direction)

    @bindings.add("c-home", filter=in_editor)
    @bindings.add("c-end", filter=in_editor)
    def move_editor_end(event: KeyPressEvent) -> None:
        ui._editor_panel.move_to_end(end=event.key_sequence[-1].key == Keys.ControlEnd)

    @bindings.add("c-o")
    def inspect_panel(event: KeyPressEvent) -> None:
        ui.toggle_inspection()

    @bindings.add("escape", filter=in_viewer, eager=True)
    def close_view(event: KeyPressEvent) -> None:
        ui.close_viewer()

    @bindings.add("c-r", filter=in_viewer)
    def reload_view(event: KeyPressEvent) -> None:
        ui.app.create_background_task(ui._reload_viewer())

    @bindings.add("up", filter=in_viewer)
    @bindings.add("down", filter=in_viewer)
    @bindings.add("pageup", filter=in_viewer)
    @bindings.add("pagedown", filter=in_viewer)
    def scroll_view(event: KeyPressEvent) -> None:
        key = event.key_sequence[-1].key
        rows = max(1, ui.output.get_size().rows - ui._footer_rows() - ui._queued_prompt_height() - 1) if key in {"pageup", "pagedown"} else 1
        ui._viewer_window.scroll(-rows if key in {"up", "pageup"} else rows)
        ui.app.invalidate()

    @bindings.add("home", filter=in_viewer)
    @bindings.add("end", filter=in_viewer)
    def edge_view(event: KeyPressEvent) -> None:
        ui._viewer_window.seek(following=event.key_sequence[-1].key == Keys.End)
        ui.app.invalidate()

    @bindings.add("f2", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting)
    def toggle_markdown(event: KeyPressEvent) -> None:
        ui.configure_markdown(source=not ui.__dict__.get("_markdown_source", False))

    @bindings.add("f3", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting)
    def copy_response(event: KeyPressEvent) -> None:
        if ui.last_markdown():
            ui.copy_response()

    @bindings.add("up", filter=in_menu)
    @bindings.add("down", filter=in_menu)
    def move_menu(event: KeyPressEvent) -> None:
        ui._move_menu(-1 if event.key_sequence[-1].key == Keys.Up else 1)

    @bindings.add("enter", filter=in_menu, eager=True)
    def select_menu(event: KeyPressEvent) -> None:
        ui._finish_menu(ui._menu_options[ui._menu_index][0])

    @bindings.add("escape", filter=in_menu, eager=True)
    @bindings.add("c-d", filter=in_menu)
    def cancel_menu(event: KeyPressEvent) -> None:
        ui._finish_menu(None)

    @bindings.add("escape", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting & ~in_inspection & ~in_link & ~in_feed, eager=True)
    def stop_now(event: KeyPressEvent) -> None:
        callback = getattr(ui, "_interrupt_callback", None)
        if callback is not None:
            callback()

    @bindings.add(Keys.Any, filter=in_menu)
    def ignore_menu_text(event: KeyPressEvent) -> None:
        pass

    @bindings.add("c-\\", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting)
    def toggle_context(event: KeyPressEvent) -> None:
        ui.context_visible = not ui.context_visible
        ui._focus.focus(ui.context_window if ui.context_visible else ui.input)
        ui.app.invalidate()

    in_context = Condition(lambda: ui.context_visible and ui.app.layout.current_window is ui.context_window) & ~in_menu & ~in_viewer & ~in_editor & ~in_setting & ~in_inspection

    @bindings.add("up", filter=in_context)
    @bindings.add("down", filter=in_context)
    @bindings.add("pageup", filter=in_context)
    @bindings.add("pagedown", filter=in_context)
    def scroll_context(event: KeyPressEvent) -> None:
        key = event.key_sequence[-1].key
        rows = max(1, ui.output.get_size().rows - ui._footer_rows() - ui._queued_prompt_height() - 1) if key in {"pageup", "pagedown"} else 1
        ui.context_window.scroll(-rows if key in {"up", "pageup"} else rows)
        ui.app.invalidate()

    @bindings.add("home", filter=in_context)
    def context_top(event: KeyPressEvent) -> None:
        ui.context_window.seek()
        ui.app.invalidate()

    @bindings.add("end", filter=in_context)
    def context_bottom(event: KeyPressEvent) -> None:
        ui.context_window._following = True
        ui.app.invalidate()

    @bindings.add("<scroll-up>")
    def scroll_up(event: KeyPressEvent) -> None:
        ui._scroll_output(-3)

    @bindings.add("<scroll-down>")
    def scroll_down(event: KeyPressEvent) -> None:
        ui._scroll_output(3)

    @bindings.add("c-d", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting & ~in_inspection)
    def eof(event: KeyPressEvent) -> None:
        if not event.current_buffer.text:
            ui._end_input()
        else:
            event.current_buffer.delete()

    return bindings
