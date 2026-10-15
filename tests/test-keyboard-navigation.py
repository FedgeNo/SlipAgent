"""Actual keyboard traversal of visible regions and shared mouse actions."""

import asyncio
import io
import json
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from slipagent.links import ActionControl
from slipagent.terminal import TerminalUI, SuggestedPrompt
from slipagent.types import ToolCall


@pytest.fixture
def ui(tmp_path):
    sink = io.StringIO()
    sizes = [Size(rows=24, columns=100)]
    output = Vt100_Output(sink, lambda: sizes[0], term="xterm-256color", enable_cpr=False)
    with create_pipe_input() as pipe:
        terminal = TerminalUI(lambda width: "workspace  │  Model: stub", sink,
                              input=pipe, output=output, project_directory=tmp_path)
        terminal._folder_opener = ["unused"]
        yield terminal, sizes
        terminal.close()


def render(ui):
    with set_app(ui.app):
        ui.app.before_render.fire()
        ui.app.renderer.render(ui.app, ui.app.layout)


def press(ui, key):
    with set_app(ui.app):
        ui.app.key_processor.feed(KeyPress(key))
        ui.app.key_processor.process_keys()
        render(ui)


def selected(ui):
    return ui.app.layout.current_control, ui.app.layout.current_control.__dict__.get("_keyboard_key")


def click(ui, window, x=0, y=0):
    render(ui)
    position = ui.app.renderer._last_screen.visible_windows_to_write_positions[window]
    point = Point(position.xpos + x, position.ypos + y)
    ui._route_mouse_wheel(ui.app)
    with set_app(ui.app):
        for kind in (MouseEventType.MOUSE_DOWN, MouseEventType.MOUSE_UP):
            ui.app.renderer.mouse_handlers.mouse_handlers[point.y][point.x](
                MouseEvent(point, kind, MouseButton.LEFT, frozenset()))
    render(ui)


def reach(ui, predicate, key=Keys.Tab):
    for _ in range(60):
        if predicate():
            return
        press(ui, key)
    pytest.fail("Keyboard route did not reach the requested visible target")


@pytest.mark.parametrize("key", [Keys.Tab, Keys.BackTab])
async def test_route_reaches_feed_links_lists_and_footer_actions(ui, tmp_path, key):
    terminal, _ = ui
    path = tmp_path / "example.py"
    path.write_text("print('example')\n")
    terminal.track_call(ToolCall("read", "read_file", {"path": "example.py"}), 1)
    terminal.write_file_reference("example.py")
    terminal.write_directory_reference(".")
    terminal.write_artifact_reference({"record_id": 7, "kind": "output"})
    terminal.write_suggestion("Explain this code")
    terminal.toggle_inspection()
    render(terminal)
    origin = selected(terminal)
    visited = []
    for _ in range(40):
        press(terminal, key)
        visited.append(selected(terminal))
        if selected(terminal) == origin:
            break
    else:
        pytest.fail("Traversal did not wrap to its starting point")
    controls = [control for control, _ in visited]
    assert terminal.transcript.content in controls
    assert terminal._files_window.content in controls
    assert terminal._calls_window.content in controls
    keys = [value for _, value in visited]
    for kind in ("file", "directory", "artifact", "suggestion"):
        assert any(isinstance(value, tuple) and value[0] == "feed" and value[1].startswith(f"class:{kind}-")
                   for value in keys) is (key == Keys.Tab)
    assert "inspect" in keys and "options" in keys
    assert "commands" in keys and terminal._folder_mouse in keys
    assert terminal.input.text == ""


async def test_shift_tab_advances_focus_and_enter_submits_suggestion(ui):
    terminal, _ = ui
    terminal.input.text = "keep this draft"
    terminal.write_suggestion("Explain these changes")
    render(terminal)
    before = selected(terminal)
    press(terminal, Keys.BackTab)
    assert selected(terminal) != before
    assert terminal.app.layout.current_window is terminal.transcript
    reach(terminal, lambda: selected(terminal)[1] == ("feed", "class:suggestion-0"))
    content = terminal.transcript.content.create_content(100, 20)
    with set_app(terminal.app):
        content = terminal.transcript.content.create_content(100, 20)
    assert any("class:keyboard-focus" in part[0] for row in range(content.line_count) for part in content.get_line(row))
    press(terminal, Keys.Enter)
    message = await asyncio.wait_for(terminal.read_line(), 1)
    assert isinstance(message, SuggestedPrompt) and message == "Explain these changes"
    assert terminal.input.text == "keep this draft"


async def test_click_prompt_focuses_and_places_cursor_on_the_first_click(ui):
    terminal, _ = ui
    terminal.input.text = "abcdefgh"
    terminal.toggle_inspection()
    terminal._focus.focus(terminal._files_window)
    click(terminal, terminal.input.window, x=5)
    assert terminal.app.layout.current_control is terminal.input.control
    assert terminal.input.buffer.cursor_position == 3
    press(terminal, "X")
    assert terminal.input.text == "abcXdefgh"


@pytest.mark.parametrize("files", [True, False])
async def test_click_list_title_focuses_that_list_for_arrows(ui, files):
    terminal, _ = ui
    for number in range(3):
        terminal.track_call(ToolCall(str(number), "read_file", {"path": f"file-{number}"}), 1)
    terminal.toggle_inspection()
    window = terminal._files_window if files else terminal._calls_window
    group = terminal._inspection_panel.children[0 if files else 2]
    before = terminal._inspection_selection(files)
    click(terminal, group.children[0], x=5)
    assert terminal.app.layout.current_window is window
    press(terminal, Keys.Down)
    assert terminal._inspection_selection(files) != before
    assert not terminal._editor_active and not terminal._viewer_active


@pytest.mark.parametrize("view", ["feed", "viewer", "context", "editor"])
async def test_click_main_area_focuses_it(ui, tmp_path, view):
    terminal, _ = ui
    if view == "feed":
        terminal.write("Feed text")
        window = terminal.transcript
    elif view == "viewer":
        terminal.open_viewer("Output", "Text")
        window = terminal._viewer_window
    elif view == "context":
        terminal.set_context(json.dumps({"messages": [{"role": "user", "content": "Text"}]}))
        terminal.context_visible = True
        window = terminal.context_window
    else:
        path = tmp_path / "example.py"
        path.write_text("Text")
        await terminal._open_editor_now(path)
        window = terminal._editor_input.window
    terminal.toggle_inspection()
    terminal._focus.focus(terminal._files_window)
    click(terminal, window, x=1)
    assert terminal.app.layout.current_window is window


@pytest.mark.parametrize("kind", ["file", "directory", "artifact"])
async def test_feed_links_activate_their_existing_operations(ui, tmp_path, monkeypatch, kind):
    terminal, _ = ui
    path = tmp_path / "example.py"
    path.write_text("example\n")
    action = AsyncMock()
    if kind == "file":
        monkeypatch.setattr(terminal, "_inspect_file", action)
        terminal.write_file_reference("example.py", line=4)
        arguments, keywords = (str(path),), {"line": 4}
    elif kind == "directory":
        monkeypatch.setattr(terminal, "_explore_directory", action)
        terminal.write_directory_reference(".")
        arguments, keywords = (tmp_path,), {}
    else:
        monkeypatch.setattr(terminal, "open_artifact", action)
        terminal.write_artifact_reference({"record_id": 7, "kind": "output"})
        arguments, keywords = (7,), {}
    render(terminal)
    reach(terminal, lambda: selected(terminal)[1] == ("feed", f"class:{kind}-0"))
    press(terminal, Keys.Enter)
    await asyncio.sleep(0)
    action.assert_awaited_once_with(*arguments, **keywords)


async def test_offscreen_feed_actions_join_the_route_but_disabled_actions_do_not(ui):
    terminal, _ = ui
    terminal.write_suggestion("Old suggestion")
    for number in range(100):
        terminal.write(f"row {number}")
    terminal.write_suggestion("New suggestion")
    render(terminal)
    keys = [key for _, key, _, _ in terminal._focus.stops()]
    assert ("feed", "class:suggestion-0") in keys
    assert ("feed", "class:suggestion-101") in keys
    terminal.set_working(True)
    render(terminal)
    assert all(not isinstance(key, tuple) or key[0] != "feed" for _, key, _, _ in terminal._focus.stops())


async def test_tab_reveals_each_selected_feed_link_and_shift_tab_leaves_feed(ui):
    terminal, _ = ui
    terminal.write_suggestion('First item')
    for index in range(60):
        terminal.write(f'line {index}')
    terminal.write_suggestion('Second item')
    for index in range(60):
        terminal.write(f'later line {index}')
    terminal.write_suggestion('Third item')
    terminal.toggle_inspection()
    render(terminal)
    terminal._focus.focus(terminal.input)
    press(terminal, Keys.Tab)
    assert terminal.app.layout.current_window is terminal.transcript
    for expected in [0, 61, 122]:
        press(terminal, Keys.Tab)
        assert selected(terminal)[1] == ('feed', f'class:suggestion-{expected}')
        point = terminal.transcript.content.__dict__['_keyboard_point']
        assert point.y in terminal.transcript.render_info.displayed_lines
        scroll = terminal.transcript.vertical_scroll
        render(terminal)
        assert terminal.transcript.vertical_scroll == scroll
    press(terminal, Keys.BackTab)
    assert terminal.app.layout.current_window is terminal._files_window


async def test_shift_tab_skips_feed_links_and_still_focuses_feed_component(ui):
    terminal, _ = ui
    terminal.write_suggestion('Skip this item')
    terminal.toggle_inspection()
    render(terminal)
    press(terminal, Keys.BackTab)
    assert selected(terminal) == (terminal.transcript.content, None)
    press(terminal, Keys.BackTab)
    assert terminal.app.layout.current_window is terminal._files_window


@pytest.mark.parametrize('key', [Keys.Tab, Keys.BackTab])
async def test_tabbing_shows_component_tip_without_flashing_or_losing_notification(ui, monkeypatch, key):
    terminal, _ = ui
    clock = [100.0]
    monkeypatch.setattr('slipagent.terminal.time.monotonic', lambda: clock[0])
    terminal.notify('Saved notification')
    render(terminal)
    press(terminal, key)
    assert terminal._notification_text(100) == 'Shift+Tab to select UI component.'
    assert not any('status-flash' in style for style, *_ in terminal._activity_notice([]))
    assert terminal._notification == 'Saved notification'
    clock[0] += 16
    assert terminal._notification_text(100) == 'Saved notification'


async def test_inspect_uses_prewarmed_width_and_keeps_new_output(ui, monkeypatch):
    terminal, _ = ui
    for index in range(60):
        terminal.write(f'line {index}: ' + 'text ' * 20)
    render(terminal)
    layouts = terminal._transcript_layouts
    await asyncio.gather(*list(layouts.tasks.values()))
    formatted = []
    append = terminal._append_transcript_record
    monkeypatch.setattr(terminal, '_append_transcript_record', lambda flow, index: (formatted.append(index), append(flow, index)))
    for _ in range(4):
        terminal.toggle_inspection()
        render(terminal)
    assert formatted == []
    assert len(layouts.states) == 2
    terminal.write('New output after toggles')
    render(terminal)
    await asyncio.gather(*list(layouts.tasks.values()))
    assert formatted == [60, 60]
    for state in layouts.states.values():
        assert state.count == 61
        assert 'New output after toggles' in '\n'.join(
            ''.join(part[1] for part in row) for row in [
                *[state.flow.get_row(index) for index in range(len(state.flow.rows))],
                *state.flow.content_rows(),
            ])


async def test_wrapping_cache_is_bounded_and_reset_on_theme_or_clear(ui):
    terminal, sizes = ui
    terminal.write_markdown('**Original** text')
    render(terminal)
    layouts = terminal._transcript_layouts
    await asyncio.gather(*list(layouts.tasks.values()))
    old = list(layouts.states.values())
    terminal.configure_markdown(theme='light')
    render(terminal)
    assert all(state.flow.rows._file.closed for state in old)
    sizes[0] = Size(rows=24, columns=120)
    render(terminal)
    assert len(layouts.states) == 2
    terminal.clear_transcript()
    terminal.write('Replacement')
    render(terminal)
    content = terminal._transcript_content(120)
    text = '\n'.join(''.join(part[1] for part in content.get_line(row)) for row in range(content.line_count))
    assert 'Replacement' in text and 'Original' not in text


@pytest.mark.parametrize('close', [False, True])
async def test_pending_prewarm_is_cancelled_before_storage_closes(ui, close):
    terminal, _ = ui
    terminal.write('Old data')
    render(terminal)
    pending = list(terminal._transcript_layouts.tasks.values())
    assert pending
    if close:
        terminal.close()
    else:
        terminal.clear_transcript()
        terminal.write('New data')
        render(terminal)
    await asyncio.gather(*pending, return_exceptions=True)
    assert all(task.cancelled() for task in pending)


async def test_feed_scroll_and_escape_do_not_change_draft_or_interrupt(ui):
    terminal, _ = ui
    interrupted = []
    terminal.set_interrupt_handler(lambda: interrupted.append(True))
    terminal.input.text = "draft"
    for number in range(100):
        terminal.write(f"row {number}")
    render(terminal)
    bottom = terminal.transcript.vertical_scroll
    press(terminal, Keys.PageUp)
    assert terminal.transcript.vertical_scroll < bottom
    reach(terminal, lambda: terminal.app.layout.current_window is terminal.transcript)
    press(terminal, Keys.Home)
    assert terminal.transcript.vertical_scroll == 0
    press(terminal, Keys.Down)
    assert terminal.transcript.vertical_scroll == 1
    press(terminal, Keys.End)
    assert terminal.transcript._following
    press(terminal, Keys.Escape)
    assert terminal.app.layout.current_control is terminal.input.control
    assert terminal.input.text == "draft" and interrupted == []


async def test_editor_tab_indents_but_other_traversal_keys_leave_editor(ui, tmp_path):
    terminal, _ = ui
    path = tmp_path / "code.py"
    path.write_text("original\n")
    await terminal._open_editor_now(path)
    render(terminal)
    press(terminal, Keys.Tab)
    assert terminal._editor_input.text.startswith("    original")
    text = terminal._editor_input.text
    press(terminal, Keys.BackTab)
    assert terminal.app.layout.current_control is not terminal._editor_input.control
    reach(terminal, lambda: terminal.app.layout.current_control is terminal._editor_input.control,
          Keys.BackTab)
    press(terminal, Keys.BackTab)
    assert terminal.app.layout.current_control is not terminal._editor_input.control
    reach(terminal, lambda: selected(terminal)[1] == ("editor-action", "Save File [Ctrl+S]"), Keys.BackTab)
    press(terminal, Keys.Enter)
    for _ in range(100):
        if path.read_text() == text:
            break
        await asyncio.sleep(.01)
    assert path.read_text() == text
    reach(terminal, lambda: terminal.app.layout.current_control is terminal.input.control)
    assert terminal.input.text == ""


async def test_menu_list_and_back_link_are_separate_keyboard_stops(ui):
    terminal, _ = ui
    task = asyncio.create_task(terminal.choose("Pick", [("a", "First"), ("b", "Second")]))
    await asyncio.sleep(0)
    render(terminal)
    press(terminal, Keys.Down)
    assert terminal._menu_index == 1
    reach(terminal, lambda: selected(terminal)[1] == "menu-back")
    press(terminal, Keys.Enter)
    assert await asyncio.wait_for(task, 1) is None


async def test_reloaded_focus_owner_skips_draft_hidden_by_editor_menu(ui, tmp_path):
    terminal, _ = ui
    path = tmp_path / "code.py"
    path.write_text("original\n")
    await terminal._open_editor_now(path)
    owner = terminal._focus
    owner.editor = lambda: terminal._editor_input
    task = asyncio.create_task(terminal.choose("Pick", [("done", "Done")]))
    await asyncio.sleep(0)
    try:
        terminal.refresh()
        render(terminal)
        assert terminal._focus is owner
        assert all(window.content is not terminal.input.control for window, _, _, _ in owner.stops())
        reach(terminal, lambda: selected(terminal)[1] == "menu-back")
        press(terminal, Keys.Escape)
        assert await asyncio.wait_for(task, 1) is None
    finally:
        terminal._finish_menu(None)
        await task


@pytest.mark.parametrize("view", ["feed", "viewer", "context"])
async def test_scrolling_main_view_while_menu_open_keeps_menu_selection(ui, view):
    terminal, _ = ui
    text = "\n".join(f"row {number}" for number in range(100))
    if view == "viewer":
        terminal.open_viewer("Output", text)
        window = terminal._viewer_window
    elif view == "context":
        terminal.set_context(json.dumps({"messages": [{"role": "user", "content": text}]}))
        terminal.context_visible = True
        window = terminal.context_window
    else:
        terminal.write(text)
        window = terminal.transcript
    task = asyncio.create_task(terminal.choose("Pick", [("a", "First"), ("b", "Second")]))
    await asyncio.sleep(0)
    try:
        render(terminal)
        reach(terminal, lambda: terminal.app.layout.current_window is window)
        before = window.vertical_scroll
        press(terminal, Keys.PageUp if view == "feed" else Keys.PageDown)
        assert window.vertical_scroll != before
        assert terminal._menu_index == 0
    finally:
        terminal._finish_menu(None)
        await task


async def test_resize_and_disappearing_links_restore_visible_focus(ui):
    terminal, size = ui
    terminal.toggle_inspection()
    render(terminal)
    reach(terminal, lambda: terminal._inspection_focused())
    size[0] = Size(rows=24, columns=40)
    render(terminal)
    assert not terminal._inspection_focused()
    for window, _, _, _ in terminal._focus.stops():
        assert window not in (terminal._files_window, terminal._calls_window)
    reach(terminal, lambda: selected(terminal)[1] == "inspect")
    terminal._editor_active = True
    terminal._focus.prompt_active = False
    render(terminal)
    # A rebuilt footer must not leave focus attached to an invisible old control.
    terminal.refresh()
    render(terminal)
    assert any(window.content is terminal.app.layout.current_control
               for window, _, _, _ in terminal._focus.stops())


@pytest.mark.parametrize("columns,inspection,expected_width", [(100, False, 100), (100, True, 75), (59, True, 59)])
async def test_queued_preview_stops_at_visible_inspection_boundary(ui, columns, inspection, expected_width):
    terminal, sizes = ui
    sizes[0] = Size(rows=24, columns=columns)
    if inspection:
        terminal.toggle_inspection()
    terminal.queue_prompt("queued message " * 30)
    render(terminal)
    screen = terminal.app.renderer._last_screen
    row = sizes[0].rows - terminal._footer_rows() - 1
    text = "".join(screen.data_buffer[row][column].char for column in range(columns))
    assert text[:expected_width].rstrip().endswith("…")
    assert not text[expected_width:].strip()
