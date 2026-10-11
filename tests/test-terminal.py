"""Render the footer through a VT terminal emulator while editing live input."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pyte
import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.application.current import set_app
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output.vt100 import Vt100_Output

from slipagent.terminal import TerminalUI
from slipagent.agent import AgentEvent
from slipagent.cli import BANNER, HELP, Renderer, Style, _status_bar, _danger_suffix
from slipagent.tools.base import ToolResult
from slipagent.types import Message, ToolCall, Usage


async def wait_until(predicate) -> None:
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.02)


async def test_footer_background_separates_controls_from_feed(display):
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        ui.input.text = "\n".join(f"input {n}" for n in range(5))
        ui.notify("Notice stays at bottom.")
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Notice stays at bottom." in snapshot()[17])
            assert "Ready" in snapshot()[17]
            assert "input 0" in snapshot()[18]
            assert "input 4" in snapshot()[22]
            assert not snapshot()[16].strip()
            assert not snapshot()[15].strip()
            assert "readout" in snapshot()[23]
            assert screen.buffer[17][99].bg == "000040"
            assert screen.buffer[23][99].bg == "000040"
            assert screen.buffer[18][99].bg == "004000"
            assert screen.buffer[16][99].bg == "default"
        finally:
            ui.close()
            await task


async def test_menu_back_link_cancels_only_when_its_word_is_clicked(display):
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=140)
    screen.resize(lines=24, columns=140)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        ui.input.text = "keep draft"
        task = asyncio.create_task(ui.run())
        choice = asyncio.create_task(ui.choose("Choices", [("one", "One")]))
        try:
            await wait_until(lambda: "Esc = back" in snapshot()[23])
            start = snapshot()[23].index("back")
            assert screen.buffer[23][start].underscore
            assert not screen.buffer[23][start - 2].underscore
            pipe.send_text(f"\x1b[<0;{start + 8};24M\x1b[<0;{start + 8};24m")
            await asyncio.sleep(.05)
            assert not choice.done()
            pipe.send_text(f"\x1b[<35;{start + 1};24M")
            await wait_until(lambda: snapshot() and screen.buffer[23][start].reverse)
            pipe.send_text(f"\x1b[<0;{start + 1};24M\x1b[<0;{start + 1};24m")
            assert await asyncio.wait_for(choice, 3) is None
            assert ui.input.text == "keep draft"
        finally:
            ui.close()
            await choice
            await task


@pytest.mark.parametrize("action", ["save", "discard", "cancel", "save_failure"])
async def test_editor_unsaved_choices_preserve_or_save_as_selected(display, tmp_path, monkeypatch, action):
    from unittest.mock import AsyncMock
    path = tmp_path / "file.txt"
    path.write_text("original\n")
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        try:
            await ui._open_editor_now(path)
            ui._editor_input.text = "edited\n"
            choose = AsyncMock(return_value="save" if action == "save_failure" else action)
            monkeypatch.setattr(ui, "choose", choose)
            if action == "save_failure":
                monkeypatch.setattr(ui._editor_panel, "save", AsyncMock(return_value=False))
            await ui._close_editor()
            choose.assert_awaited_once_with("Unsaved Changes", [("save", "Save"), ("discard", "Don't Save"), ("cancel", "Cancel")])
            assert ui._editor_active is (action in {"cancel", "save_failure"})
            assert path.read_text() == ("edited\n" if action == "save" else "original\n")
            if ui._editor_active:
                assert ui._editor_input.text == "edited\n"
        finally:
            ui._close_transcript()


@pytest.mark.parametrize("action", ["clean", "save", "discard", "cancel", "save_failure"])
async def test_opening_another_touched_file_keeps_agent_held(display, tmp_path, monkeypatch, action):
    from unittest.mock import AsyncMock, Mock
    from slipagent.tools.base import ToolRegistry
    from slipagent.useredits import UserEdits
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("original\n")
    second.write_text("second\n")
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output, project_directory=tmp_path)
        agent = SimpleNamespace(running=True, history=SimpleNamespace(steps=[]), _persist=Mock())
        session = SimpleNamespace(agent=agent, registry=ToolRegistry(), reloader=SimpleNamespace(busy=0))
        edits = UserEdits(session, ui)
        ui.__dict__["_user_edits"] = edits
        await edits.request(first)
        boundary = asyncio.create_task(edits.boundary())
        switch = None
        try:
            await wait_until(lambda: ui._editor_active)
            choose = AsyncMock(return_value="save" if action == "save_failure" else action)
            monkeypatch.setattr(ui, "choose", choose)
            if action != "clean":
                ui._editor_input.text = "edited\n"
            if action == "save_failure":
                monkeypatch.setattr(ui._editor_panel, "save", AsyncMock(return_value=False))
            switch = asyncio.create_task(ui._inspect_file("second.txt"))
            if action in {"cancel", "save_failure"}:
                await switch
                assert ui._editor_panel.file.path == first
                assert ui._editor_input.text == "edited\n"
            else:
                await wait_until(lambda: ui._editor_panel.file is not None and ui._editor_panel.file.path == second)
                assert ui._editor_input.text == "second\n"
            assert first.read_text() == ("edited\n" if action == "save" else "original\n")
            assert not edits.released.is_set()
            assert not boundary.done()
            assert session.reloader.busy == 1
            if action == "clean":
                choose.assert_not_awaited()
            if action == "save":
                assert "+edited" in next(iter(session.registry.context_notes["user_edits"].values()))["diff"]
        finally:
            ui._close_transcript()
            await boundary
            if switch is not None:
                await switch
            assert session.reloader.busy == 0


async def test_scroll_bottom_link_is_centered_and_returns_feed_to_tail(display):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui.input.text = "keep this draft"
        ui.write("\n".join(f"Output {line}" for line in range(80)))
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Output 79" in "\n".join(snapshot()[:15]))
            assert "Scroll to Bottom" not in snapshot()[17]
            ui.transcript.scroll(-20)
            ui.app.invalidate()
            await wait_until(lambda: not ui.transcript._following)
            ui.app.invalidate()
            await wait_until(lambda: "Scroll to Bottom" in snapshot()[17])
            assert snapshot()[17].index("Scroll to Bottom") == 42
            pipe.send_text("\x1b[<0;50;18M\x1b[<0;50;18m")
            await wait_until(lambda: ui.transcript._following)
            await wait_until(lambda: "Scroll to Bottom" not in snapshot()[17])
            assert "Output 79" in "\n".join(snapshot()[:15])
            assert ui.input.text == "keep this draft"
        finally:
            ui.close()
            await task


async def test_cancelled_save_finishes_publication_and_records_diff_before_cleanup(display, tmp_path, monkeypatch):
    import threading
    from unittest.mock import Mock
    from slipagent.editor import FileEditor
    path = tmp_path / "file.txt"
    path.write_text("original\n")
    started, release = threading.Event(), threading.Event()
    original_save = FileEditor.save
    def slow_save(file, **kwargs):
        started.set()
        assert release.wait(3)
        original_save(file, **kwargs)
    monkeypatch.setattr(FileEditor, "save", slow_save)
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui.__dict__["_user_edits"] = SimpleNamespace(record=Mock(), close=Mock())
        save = None
        try:
            await ui._open_editor_now(path)
            ui._editor_input.text = "user\n"
            save = asyncio.create_task(ui._editor_panel.save())
            assert await asyncio.to_thread(started.wait, 2)
            save.cancel()
            await asyncio.sleep(.02)
            assert not save.done()
            release.set()
            result = await asyncio.gather(save, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError)
            assert path.read_text() == "user\n"
            ui.__dict__["_user_edits"].record.assert_called_once()
        finally:
            release.set()
            if save is not None:
                await asyncio.gather(save, return_exceptions=True)
            ui._close_transcript()


async def test_options_field_isolated_from_prompt_queue_and_draft(display, monkeypatch):
    from slipagent import options
    from unittest.mock import Mock
    monkeypatch.setattr(options, "configured_folder_opener", lambda command: ("/test/opener",) if command else None)
    saved = Mock()
    monkeypatch.setattr(options, "save_ui_preference", saved)
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui.input.text = "keep this draft"
        ui.write("Feed stays in place.")
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Options" in snapshot()[17])
            pipe.send_text("\x1bOS")  # F4.
            await wait_until(lambda: ui._menu_title == "Inspection Options")
            ui._finish_menu("folder_opener")
            await wait_until(lambda: ui._options_field_active)
            await wait_until(lambda: snapshot()[17].startswith("Folder Opener"))
            assert snapshot()[0].startswith("Feed stays in place.")
            assert "Folder Opener" not in "\n".join(snapshot()[:15])
            pipe.send_text("opener --flag {path}\r")
            await wait_until(lambda: saved.called)
            saved.assert_called_once_with("folder_opener", "opener --flag {path}")
            await wait_until(lambda: ui._menu_future is not None)
            ui._finish_menu("done")
            await wait_until(lambda: not ui._options_panel.active)
            assert ui.input.text == "keep this draft"
            assert ui._lines.empty()
            assert ui._folder_opener == ("/test/opener",)
        finally:
            ui.close()
            await task


async def test_file_line_link_opens_at_target_and_selection_offers_path(display, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseEvent, MouseEventType, MouseButton
    path = tmp_path / "file.txt"
    path.write_text("one\ntwo\nthree\n")
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        request = AsyncMock()
        ui.__dict__["_user_edits"] = SimpleNamespace(request=request, close=lambda: None)
        task = asyncio.create_task(ui.run())
        try:
            ui.write_file_reference("file.txt", line=3)
            await wait_until(lambda: "file.txt:3" in snapshot()[0])
            pipe.send_text("\x1b[<0;7;1M\x1b[<0;7;1m")
            await wait_until(lambda: request.await_count == 1)
            request.assert_awaited_once_with(path, 3)
            ui._selection_start = Point(x=4, y=0)
            ui._selection_end = Point(x=12, y=0)
            menu = AsyncMock()
            monkeypatch.setattr(ui, "_selection_menu", menu)
            ui._select_transcript(MouseEvent(Point(x=6, y=0), MouseEventType.MOUSE_UP, MouseButton.RIGHT, frozenset()))
            await wait_until(lambda: menu.await_count == 1)
            assert menu.call_args.args[1] == path
        finally:
            ui.close()
            await task


async def test_editor_mouse_places_cursor_on_text_and_at_line_end_in_blank_space(display, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("first line\nsecond line")
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ui.app.is_running)
            await ui._open_editor_now(path)
            await wait_until(lambda: snapshot()[1].startswith("first line"))
            pipe.send_text("\x1b[<0;4;2M\x1b[<0;4;2m")
            await wait_until(lambda: ui._editor_input.buffer.cursor_position == 3)
            pipe.send_text("\x1b[<0;40;2M\x1b[<0;40;2m")
            await wait_until(lambda: ui._editor_input.buffer.cursor_position == len("first line"))
            pipe.send_text("\x1b[<0;40;10M\x1b[<0;40;10m")
            await wait_until(lambda: ui._editor_input.buffer.cursor_position == len(path.read_text()))
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("long_line", [False, True])
async def test_editor_reload_preserves_logical_cursor_line(display, tmp_path, long_line):
    path = tmp_path / "file.txt"
    prefix = "x" * (4096 * 40) + "\n" if long_line else "first\n"
    text = prefix + "".join(f"line-{n}\n" for n in range(1, 101))
    path.write_text(text)
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        try:
            await ui._open_editor_now(path, line=35)
            assert ui._editor_input.buffer.document.current_line == "line-34"
            path.write_text(text.replace("line-", "fresh-"))
            await ui._reload_editor()
            assert ui._editor_input.buffer.document.current_line == "fresh-34"
            file = ui._editor_panel.file
            file.set_page(0, "inserted\n" + file.page_text(0))
            assert await ui._editor_panel.save()
            await ui._reload_editor()
            assert ui._editor_input.buffer.document.current_line == "fresh-34"
            path.write_text("first\nlast\n")
            await ui._reload_editor()
            assert ui._editor_input.buffer.document.current_line == "last"
        finally:
            ui._close_transcript()


async def test_editor_standard_page_keys_overlap_and_retain_edits(display, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("".join(f"line-{n:03}\n" for n in range(100)))
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        ui.input.text = "keep draft"
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ui.app.is_running)
            await ui._open_editor_now(path)
            panel = ui._editor_panel
            buffer = panel.input.buffer
            buffer.insert_text("user edit\n")
            await wait_until(lambda: "line-012" in "\n".join(snapshot()[:15]))
            # Start with the cursor at the bottom of the viewport, as an editor
            # normally does after consecutive downward movement.
            panel.move_cursor(12)
            await wait_until(lambda: buffer.document.cursor_position_row == 13)
            await asyncio.sleep(.05)
            before = snapshot()[1:15]
            height = panel.input.window.render_info.window_height
            pipe.send_text("\x1b[6~")
            await wait_until(lambda: buffer.document.cursor_position_row == 13 + height - 1)
            await wait_until(lambda: snapshot()[1:15] != before)
            assert set(row.strip() for row in before) & set(row.strip() for row in snapshot()[1:15])
            # Cross a storage boundary and retain the same one-line overlap.
            before = snapshot()[1:15]
            pipe.send_text("\x1b[6~")
            await wait_until(lambda: panel.page == 1)
            await wait_until(lambda: snapshot()[1:15] != before)
            assert set(row.strip() for row in before) & set(row.strip() for row in snapshot()[1:15])
            before = snapshot()[1:15]
            pipe.send_text("\x1b[5~")
            await wait_until(lambda: panel.page == 0)
            await wait_until(lambda: snapshot()[1:15] != before)
            assert set(row.strip() for row in before) & set(row.strip() for row in snapshot()[1:15])
            assert buffer.text.startswith("user edit\nline-000\n")
            assert ui.input.text == "keep draft"
        finally:
            ui.close()
            await task


async def test_editor_saves_and_requires_choice_before_discarding(display, tmp_path):
    from unittest.mock import Mock
    path = tmp_path / "file.txt"
    path.write_text("original\n")
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        ui.__dict__["_user_edits"] = SimpleNamespace(record=Mock(), close=Mock())
        ui.input.text = "prompt draft"
        ui.input.buffer.cursor_position = 3
        prompt_buffer = ui.input.buffer
        prompt_document = prompt_buffer.document
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ui.app.is_running)
            await ui._open_editor_now(path)
            await wait_until(lambda: "exit editor" in snapshot()[17])
            assert "Editor Actions" in snapshot()[18]
            assert "prompt draft" not in "\n".join(snapshot()[18:23])
            pipe.send_text("user ")
            await wait_until(lambda: ui._editor_input.text.startswith("user original"))
            pipe.send_text("\x13")
            await wait_until(lambda: path.read_text().startswith("user original"))
            await wait_until(lambda: not ui._editor_panel.busy)
            assert ui.__dict__["_user_edits"].record.call_count == 1
            assert "+user original" in ui.__dict__["_user_edits"].record.call_args.args[1]
            pipe.send_text("unsaved ")
            await wait_until(lambda: "unsaved" in ui._editor_input.text)
            pipe.send_text("\x1b")
            await wait_until(lambda: ui._menu_title == "Unsaved Changes")
            assert ui._editor_active
            await wait_until(lambda: "Don't Save" in "\n".join(snapshot()[18:23]))
            assert "Save" in snapshot()[18]
            assert "Cancel" in snapshot()[20]
            assert not snapshot()[16].strip()
            assert not snapshot()[15].strip()
            pipe.send_text("\x1b")
            await wait_until(lambda: ui._menu_future is None)
            assert ui._editor_active
            close = asyncio.create_task(ui._close_editor())
            await wait_until(lambda: ui._menu_future is not None)
            ui._finish_menu("discard")
            await close
            assert not ui._editor_active
            assert path.read_text() == "user original\n"
            assert ui.input.text == "prompt draft"
            assert ui.input.buffer is prompt_buffer
            assert ui.input.buffer.document == prompt_document
            ui.__dict__["_user_edits"].close.assert_called_once()
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("action", ["copy", "prompt", "absolute", "relative", None])
async def test_selection_actions_preserve_draft_and_use_reference_paths(display, tmp_path, monkeypatch, action):
    from unittest.mock import AsyncMock, Mock
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        ui.input.text = "existing draft"
        choose = AsyncMock(return_value=action)
        copy = Mock()
        monkeypatch.setattr(ui, "choose", choose)
        monkeypatch.setattr(ui, "_copy_text", copy)
        try:
            await ui._selection_menu("selected\ntext", tmp_path / "some folder")
            assert len(choose.call_args.args[1]) == 4
            if action == "prompt":
                assert ui.input.text == "existing draft\nselected\ntext"
                copy.assert_not_called()
            else:
                assert ui.input.text == "existing draft"
                if action is None:
                    copy.assert_not_called()
                else:
                    expected = {"copy": "selected\ntext", "absolute": str(tmp_path / "some folder"), "relative": "some folder"}
                    copy.assert_called_once_with(expected[action])
            choose.reset_mock()
            await ui._selection_menu("ordinary prose")
            assert len(choose.call_args.args[1]) == 2
        finally:
            ui._close_transcript()


async def test_right_click_keeps_selection_and_no_selection_shows_notice(display, monkeypatch):
    from unittest.mock import AsyncMock
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseEvent, MouseEventType, MouseButton
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        menu = AsyncMock()
        monkeypatch.setattr(ui, "_selection_menu", menu)
        task = asyncio.create_task(ui.run())
        try:
            ui.write("select this text")
            await wait_until(lambda: ui.app.is_running)
            ui._transcript_content(100)
            for kind, x in [(MouseEventType.MOUSE_DOWN, 0), (MouseEventType.MOUSE_UP, 6)]:
                ui._select_transcript(MouseEvent(Point(x=x, y=0), kind, MouseButton.LEFT, frozenset()))
            selected = ui.selected_text()
            for kind in (MouseEventType.MOUSE_DOWN, MouseEventType.MOUSE_UP):
                ui._select_transcript(MouseEvent(Point(x=3, y=0), kind, MouseButton.RIGHT, frozenset()))
            await wait_until(lambda: menu.await_count == 1)
            assert ui.selected_text() == selected
            menu.assert_awaited_once_with(selected, None)
            ui._selection_start = ui._selection_end = None
            ui._select_transcript(MouseEvent(Point(x=3, y=0), MouseEventType.MOUSE_UP, MouseButton.RIGHT, frozenset()))
            assert "Select text" in str(ui._notification_text())
        finally:
            ui.close()
            await task


async def test_pending_editor_status_keeps_spinner_and_read_only_calls_open_immediately(display, tmp_path):
    from slipagent.useredits import UserEdits
    from slipagent.terminal import PULSE_FRAMES
    from slipagent.tools.base import ToolRegistry
    from unittest.mock import AsyncMock
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        session = SimpleNamespace(agent=SimpleNamespace(running=True), registry=ToolRegistry(), reloader=None)
        edits = UserEdits(session, ui)
        ui.__dict__["_user_edits"] = edits
        opened = AsyncMock()
        ui._open_editor_now = opened
        try:
            ui.set_working(True)
            await ui._inspect_call({"tool": "list_dir", "call_id": "listing", "result": "entries"})
            assert ui._viewer_active and ui.working
            assert edits.pending is None and edits.released.is_set()
            ui.close_viewer()
            await edits.request(tmp_path / "file.txt")
            opened.assert_not_awaited()
            for frame in PULSE_FRAMES:
                ui._activity_frame = frame
                assert ui._activity_status() == [("class:pulse", f"{frame} Opening editor after this turn")]
            edits.close()
            assert "Working" in ui._activity_status()[0][1]
        finally:
            ui._close_transcript()


async def test_editor_wheel_moves_cursor_three_lines_and_viewport_follows(display, tmp_path):
    path = tmp_path / "lines.txt"
    original = "\n".join(f"line {n:03}" for n in range(32))
    path.write_text(original)
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ui.app.is_running)
            await ui._open_editor_now(path)
            buffer = ui._editor_input.buffer
            buffer.cursor_position = 4
            pipe.send_text("\x1b[<65;5;5M")
            await wait_until(lambda: buffer.document.cursor_position_row == 3)
            assert buffer.document.cursor_position_col == 4
            pipe.send_text("\x1b[<65;5;5M" * 5)
            await wait_until(lambda: buffer.document.cursor_position_row == 18)
            await wait_until(lambda: "line 018" in "\n".join(snapshot()[:15]))
            assert ui._editor_input.window.vertical_scroll > 0
            pipe.send_text("\x1b[<64;5;5M")
            await wait_until(lambda: buffer.document.cursor_position_row == 15)
            pipe.send_text("\x1b[<64;5;5M" * 10)
            await wait_until(lambda: buffer.document.cursor_position_row == 0)
            await wait_until(lambda: "line 000" in snapshot()[1])
            assert buffer.text == original
            assert not ui._editor_panel.dirty()
        finally:
            ui.close()
            await task


async def test_editor_wheel_crosses_pages_and_retains_edits(display, tmp_path):
    path = tmp_path / "long.txt"
    original = "\n".join(f"line {n:03}" for n in range(80))
    path.write_text(original)
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ui.app.is_running)
            await ui._open_editor_now(path, line=31)
            panel = ui._editor_panel
            buffer = panel.input.buffer
            buffer.insert_text("edited ")
            pipe.send_text("\x1b[<65;5;5M")
            await wait_until(lambda: panel.page == 1 and buffer.document.cursor_position_row == 1)
            await wait_until(lambda: "line 033" in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[<64;5;5M")
            await wait_until(lambda: panel.page == 0 and buffer.document.cursor_position_row == 30)
            assert "edited line 030" in buffer.text
            pipe.send_text("\x1b[<65;5;5M" * 30)
            await wait_until(lambda: panel.page == 2 and buffer.document.cursor_position_row == 15)
            await wait_until(lambda: "line 079" in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[<64;5;5M" * 30)
            await wait_until(lambda: panel.page == 0 and buffer.document.cursor_position_row == 0)
            await wait_until(lambda: "line 000" in snapshot()[1])
            panel.flush()
            assert panel.file is not None
            assert "".join(panel.file.chunks()) == original.replace("line 030", "edited line 030")
            assert path.read_text() == original
        finally:
            ui.close()
            await task


async def test_editor_arrows_cross_sections_and_document_endpoints(display, tmp_path):
    path = tmp_path / "arrows.txt"
    original = "\n".join(f"line {n:03}" for n in range(80))
    path.write_text(original)
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ui.app.is_running)
            await ui._open_editor_now(path, line=32)
            panel = ui._editor_panel
            buffer = panel.input.buffer
            buffer.cursor_position += 4
            await wait_until(lambda: "line 031" in "\n".join(snapshot()[:15]))
            before = snapshot()[1:15]
            pipe.send_text("\x1b[B")
            await wait_until(lambda: panel.page == 1 and buffer.document.cursor_position_row == 0)
            assert buffer.document.cursor_position_col == 4
            await wait_until(lambda: "line 032" in "\n".join(snapshot()[:15]))
            assert set(before) & set(snapshot()[1:15])
            pipe.send_text("\x1b[A")
            await wait_until(lambda: panel.page == 0 and buffer.document.cursor_position_row == 31)
            assert buffer.document.cursor_position_col == 4
            # Right/left traverse the newline between stored sections.
            buffer.cursor_position = len(buffer.text) - 1
            pipe.send_text("\x1b[C")
            await wait_until(lambda: panel.page == 1 and buffer.cursor_position == 0)
            pipe.send_text("\x1b[D")
            await wait_until(lambda: panel.page == 0 and buffer.cursor_position == len(buffer.text) - 1)
            pipe.send_text("\x1b[1;5F")
            await wait_until(lambda: panel.page == 2 and buffer.cursor_position == len(buffer.text))
            await wait_until(lambda: "line 079" in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[1;5H")
            await wait_until(lambda: panel.page == 0 and buffer.cursor_position == 0)
            await wait_until(lambda: "line 000" in snapshot()[1])
            assert path.read_text() == original
            assert not panel.dirty()
        finally:
            ui.close()
            await task


async def test_editor_click_opens_visible_neighbor_section_at_clicked_position(display, tmp_path):
    path = tmp_path / "click.txt"
    path.write_text("\n".join(f"line {n:03}" for n in range(80)))
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ui.app.is_running)
            await ui._open_editor_now(path, line=32)
            panel = ui._editor_panel
            panel.input.window.vertical_scroll = 25
            ui.app.invalidate()
            await wait_until(lambda: "line 032" in "\n".join(snapshot()[:15]))
            row = next(i for i, text in enumerate(snapshot()[:15]) if text.startswith("line 032"))
            pipe.send_text(f"\x1b[<0;5;{row + 1}M\x1b[<0;5;{row + 1}m")
            await wait_until(lambda: panel.page == 1)
            await asyncio.sleep(.05)
            assert panel.input.buffer.document.cursor_position_row == 0
            assert panel.input.buffer.document.cursor_position_col == 4
            assert not panel.dirty()
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("step", [1, 3])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_editor_vertical_navigation_keeps_cursor_at_edges(display, tmp_path, step, wrapped):
    path = tmp_path / "scroll.txt"
    lines = [f"line {n:03}" + ("x" * 250 if wrapped and n % 3 else "") for n in range(160)]
    path.write_text("\n".join(lines))
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ui.app.is_running)
            await ui._open_editor_now(path)
            panel = ui._editor_panel
            await wait_until(lambda: "line 000" in snapshot()[1])
            for direction in [1, -1]:
                for index in range(1, 45):
                    previous = (panel.page, panel.input.buffer.cursor_position)
                    key = ("\x1b[B" if direction > 0 else "\x1b[A") if step == 1 else ("\x1b[<65;5;5M" if direction > 0 else "\x1b[<64;5;5M")
                    pipe.send_text(key)
                    await wait_until(lambda: (panel.page, panel.input.buffer.cursor_position) != previous)
                    await asyncio.sleep(.035)
                    if index * step >= 14:
                        assert panel.input.window.render_info.cursor_position.y == (13 if direction > 0 else 0), (index, direction, panel.page, panel.input.window.vertical_scroll, snapshot()[:15])
            assert panel.page == 0 and panel.input.buffer.cursor_position == 0
        finally:
            ui.close()
            await task


async def test_read_only_viewer_scrolls_targets_lines_keeps_work_running_and_esc_only_closes(display):
    from unittest.mock import Mock
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        interrupted = Mock()
        ui.set_interrupt_handler(interrupted)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Ready" in snapshot()[17])
            ui.input.text = "unfinished draft"
            ui.set_working(True)
            ui.notify("Still working.")
            ui.open_viewer("Text Record", "\n".join(f"line {n:03}" for n in range(100)), line=80)
            await wait_until(lambda: "line 079" in snapshot()[1])
            assert "Still working." in snapshot()[17]
            ui.write("background result arrived")
            ui.refresh()
            pipe.send_text("no editing\x13")
            await asyncio.sleep(.05)
            assert ui.input.text == "unfinished draft"
            assert ui.working
            pipe.send_text("\x1b[F")
            await wait_until(lambda: "line 099" in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[H")
            await wait_until(lambda: "line 000" in snapshot()[1])
            pipe.send_text("\x1b")
            await wait_until(lambda: not ui._viewer_active)
            await wait_until(lambda: "background result arrived" in "\n".join(snapshot()))
            assert ui.working
            interrupted.assert_not_called()
            assert ui.input.text == "unfinished draft"
        finally:
            ui.close()
            await task


async def test_artifact_link_opens_colored_diff_without_desktop_and_preserves_notice(display, tmp_path):
    from slipagent.artifacts import ArtifactStore
    sink, _, output, screen, snapshot = display
    store = ArtifactStore(1_000_000)
    store.use_directory(tmp_path)
    record = store.save("edit_file 'demo.txt'", "--- demo.txt\n+++ demo.txt\n@@ -1 +1 @@\n-old\n+new", kind="diff", call_id="c1", step_id=1)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui.__dict__["_artifact_store"] = store
        task = asyncio.create_task(ui.run())
        try:
            ui.write_artifact_reference(record)
            ui.notify("Saved diff.")
            await wait_until(lambda: "View Diff" in snapshot()[0])
            assert screen.buffer[0][5].underscore
            pipe.send_text("\x1b[<0;8;1M\x1b[<0;8;1m")
            await wait_until(lambda: "Read Only" in snapshot()[0])
            assert snapshot()[1].startswith("edit_file 'demo.txt'")
            assert snapshot()[2].strip() == ""
            assert "1        │ -old" in snapshot()[6]
            assert "1 │ +new" in snapshot()[7]
            assert screen.buffer[6][16].fg == "ff6666"
            assert screen.buffer[7][16].fg == "66dd88"
            assert "Saved diff." in snapshot()[17]
            pipe.send_text("\x1b")
            await wait_until(lambda: "View Diff" in snapshot()[0])
        finally:
            ui.close()
            await task


async def test_inspection_panel_toggles_by_combo_and_click_and_opens_files_and_calls(display, tmp_path):
    sink, _, output, _, snapshot = display
    (tmp_path / "demo.txt").write_text("file contents\n")
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        task = asyncio.create_task(ui.run())
        try:
            call = ToolCall("call-1", "read_file", {"path": "demo.txt"})
            ui.track_call(call, 1)
            ui.finish_call(call, 1, 'file contents\n"quoted" and literal \\n', False)
            await wait_until(lambda: "Inspect [Ctrl+O]" in snapshot()[17])
            pipe.send_text("\x0f")
            await wait_until(lambda: "Touched" in snapshot()[0])
            assert snapshot()[0].index("Touched") == 75
            assert "Calls" in snapshot()[8]
            assert "demo.txt [read]" in snapshot()[1]
            assert "read_file [completed]" in snapshot()[9]
            pipe.send_text("\x1b[<0;78;2M\x1b[<0;78;2m")
            await wait_until(lambda: "file contents" in snapshot()[1])
            pipe.send_text("\x1b")
            await wait_until(lambda: not ui._viewer_active)
            pipe.send_text("\x1b[<0;78;10M\x1b[<0;78;10m")
            await wait_until(lambda: "call-1" in snapshot()[0])
            assert ui._viewer_active
            text = "\n".join(ui._viewer_window.content.lines)
            assert '  file contents\n  "quoted" and literal \\n' in text
            assert 'result:\n' in text
            pipe.send_text("\x1b")
            await wait_until(lambda: not ui._viewer_active)
            pipe.send_text("\x1b[<0;90;18M\x1b[<0;90;18m")
            await wait_until(lambda: not ui._inspection_visible)
            await wait_until(lambda: "Touched" not in snapshot()[0])
        finally:
            ui.close()
            await task


async def test_input_wheel_scrolls_wrapped_draft_without_moving_cursor_or_feed(display):
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=30)
    screen.resize(lines=24, columns=30)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            ui.write("feed stays here")
            text = " ".join(f"word{n:02d}" for n in range(80))
            pipe.send_text(text)
            await wait_until(lambda: "word79" in "\n".join(snapshot()[18:23]))
            before = snapshot()[18:23]
            cursor = ui.input.buffer.cursor_position
            history = ui.input.buffer.history.get_strings()
            pipe.send_text("\x1b[<64;5;19M" * 30)
            await wait_until(lambda: "word00" in snapshot()[18])
            assert snapshot()[18:23] != before
            assert ui.input.text == text and ui.input.buffer.cursor_position == cursor
            assert ui.input.buffer.history.get_strings() == history
            assert "feed stays here" in snapshot()[0]
            pipe.send_text("\x1b[<65;5;19M" * 30)
            await wait_until(lambda: "word79" in "\n".join(snapshot()[18:23]))
            pipe.send_text("\x1b[<64;5;19M" * 30)
            await wait_until(lambda: "word00" in snapshot()[18])
            pipe.send_text("X")
            await wait_until(lambda: "word79X" in "\n".join(snapshot()[18:23]))
            assert ui.input.text == text + "X"
        finally:
            ui.close()
            await task


async def test_file_viewer_detects_external_changes_and_reloads_without_whole_file_reads(display, tmp_path, monkeypatch):
    from pathlib import Path
    sink, _, output, _, snapshot = display
    path = tmp_path / "live.txt"
    path.write_text("original\n")
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Ready" in snapshot()[17])
            def no_full_read(*args, **kwargs):
                raise AssertionError("File viewer must not call read_text")
            monkeypatch.setattr(Path, "read_text", no_full_read)
            await ui._inspect_file("live.txt")
            await wait_until(lambda: "original" in snapshot()[1])
            original = ui._viewer_window.content.lines
            path.write_text("external change\n")
            await wait_until(lambda: "File changed on disk" in snapshot()[0])
            assert "original" in snapshot()[1]
            pipe.send_text("\x12")
            await wait_until(lambda: "external change" in snapshot()[1])
            assert original.text.closed
            assert "File changed on disk" not in snapshot()[0]
            path.unlink()
            await wait_until(lambda: "File changed on disk" in snapshot()[0])
            assert "external change" in snapshot()[1]
        finally:
            ui.close()
            await task


async def test_sidebar_wheel_scrolls_each_pane_and_uses_menu_colors(display):
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        for n in range(30):
            ui.track_call(ToolCall(f"call{n}", "read_file" if n % 2 else "write_file", {"path": f"file{n}"}), n)
        task = asyncio.create_task(ui.run())
        try:
            ui.toggle_inspection()
            await wait_until(lambda: "file29" in snapshot()[1])
            assert snapshot()[7][75:].strip() == ""
            assert screen.buffer[1][75].fg == "999999"
            assert screen.buffer[0][75].fg != "999999"
            calls = [row[75:] for row in snapshot()[8:15]]
            pipe.send_text("\x1b[<65;80;2M" * 3)
            await wait_until(lambda: "file29" not in snapshot()[1])
            assert [row[75:] for row in snapshot()[8:15]] == calls
            files = [row[75:] for row in snapshot()[:7]]
            pipe.send_text("\x1b[<65;80;10M" * 3)
            await wait_until(lambda: [row[75:] for row in snapshot()[8:15]] != calls)
            assert [row[75:] for row in snapshot()[:7]] == files
        finally:
            ui.close()
            await task


def test_inspection_panel_tracks_shell_references_without_claiming_changes(display, tmp_path):
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui.__dict__["_project_directory"] = tmp_path
        call = ToolCall("shell-1", "run_command", {"command": 'rg -e "not a path" source > output; mystery hidden'})
        ui.track_call(call, 1)
        assert ui._touched_items == {"output": "shell write (inferred)", "source": "shell read (inferred)"}
        tracking = ui._inspection_calls[(1, "shell-1")]["shell_tracking"]
        assert tracking["incomplete"] and tracking["limitations"]
        ui.finish_call(call, 1, "exit code: 1", True)
        assert ui._touched_items["output"] == "shell write (inferred)"
        read = ToolCall("read-1", "read_file", {"path": "source"})
        ui.track_call(read, 2)
        ui.finish_call(read, 2, "contents", False)
        assert ui._touched_items["source"] == "read"
        ui.track_call(ToolCall("shell-2", "run_command", {"command": "cat source"}), 3)
        assert ui._touched_items["source"] == "read"
        ui.track_call(ToolCall("shell-3", "run_command", {"command": "cat output > output"}), 4)
        assert ui._touched_items["output"] == "shell read/write (inferred)"


@pytest.mark.parametrize("available", [False, True])
async def test_inspection_directories_use_explorer_without_editor_hold(display, tmp_path, monkeypatch, available):
    from unittest.mock import AsyncMock, Mock
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseEvent, MouseEventType, MouseButton
    sink, _, output, _, _ = display
    (tmp_path / "directory").mkdir()
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        ui._folder_opener = ("test-opener",) if available else None
        explore = AsyncMock()
        coordinator = SimpleNamespace(request=AsyncMock(), close=Mock())
        monkeypatch.setattr(ui, "_explore_directory", explore)
        ui.__dict__["_user_edits"] = coordinator
        try:
            for name, arguments in [("list_dir", {}), ("grep", {"path": "directory"}), ("glob", {"path": "directory"})]:
                call = ToolCall(name, name, arguments)
                ui.track_call(call, 1)
                ui.finish_call(call, 1, {}, False)
            assert ui._touched_items == {".": "read", "directory": "read"}
            fragments = ui._inspection_entries(files=True)
            entry = next(fragment for fragment in fragments if fragment[1].startswith("directory"))
            assert ("class:clickable" in entry[0]) is available
            if available:
                entry[2](MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
                await asyncio.sleep(0)
                explore.assert_awaited_once_with(tmp_path / "directory")
            else:
                assert len(entry) == 2
                explore.assert_not_awaited()
            coordinator.request.assert_not_awaited()
        finally:
            ui._close_transcript()


def test_inspection_panel_records_partial_path_operations(display):
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        call = ToolCall("copy-1", "copy_path", {"source": "original", "destination": "copy"})
        ui.track_call(call, 1)
        ui.finish_call(call, 1, {"changes": [{"path": "copy/child", "action": "created", "kind": "file"}],
                                 "error": "disk full"}, True)
        assert ui._touched_items["copy/child"] == "created"
        assert ui._inspection_calls[(1, "copy-1")]["status"] == "error"
        assert ui._touched_items["original"] == "requested"


async def test_notifications_remain_visible_in_menu_context_and_after_resize(display):
    sink, size, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        choice = None
        try:
            ui.notify("Saved changes.")
            await wait_until(lambda: "Saved changes." in snapshot()[17])
            assert snapshot()[17].index("Saved changes.") > snapshot()[17].index("Ready")
            assert "Saved changes." not in snapshot()[23]
            ui.write("feed contents\n" * 50)
            choice = asyncio.create_task(ui.choose("Commands", [("one", "One")]))
            await wait_until(lambda: snapshot()[17].startswith("Commands"))
            assert "Saved changes." in snapshot()[17]
            pipe.send_text("\x1b")
            await choice
            ui.context_visible = True
            ui.app.invalidate()
            await wait_until(lambda: "Context" in snapshot()[0])
            assert "Saved changes." in snapshot()[17]
            size[0] = Size(rows=24, columns=12)
            screen.resize(lines=24, columns=12)
            output.get_size = lambda: size[0]
            ui.app._on_resize()
            await wait_until(lambda: ui._notification_text() in snapshot()[17])
            ui.notify("")
            assert ui._notification_text().endswith("…")
            assert "Saved" not in ui._notification_text()
        finally:
            if choice is not None and not choice.done():
                choice.cancel()
            ui.close()
            await task


@pytest.mark.parametrize("working", [False, True])
async def test_ctrl_c_has_no_application_binding_or_exit_action(display, working):
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            ui.working = working
            ui.input.text = "draft"
            ui.input.buffer.cursor_position = len(ui.input.text)
            for key in (Keys.ControlC, Keys.SIGINT):
                bindings = [binding for binding in ui._bindings().get_bindings_for_keys((key,))
                            if binding.keys == (key,)]
                assert not bindings
            await wait_until(lambda: ui.app.is_running)
            pipe.send_text('\x03x')
            await wait_until(lambda: ui.input.text == 'draftx')
            assert not task.done()
            assert ui._lines.empty()
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("columns", [18, 100])
@pytest.mark.parametrize("paint_first", [False, True])
def test_queued_notice_disappears_without_reformatting_history(display, monkeypatch, columns, paint_first):
    from slipagent.transcript import WrappedTranscript
    sink, _, output, _, _ = display
    ui = TerminalUI(lambda width: "status", sink, output=output)
    renderer = Renderer(Style(True), sink, False)
    renderer.terminal = ui
    renderer.emit("Old transcript " * 2000)
    literal = "Keep (queued — the agent is still working) in this prompt"
    renderer.user_prompt(literal, queued=True)
    renderer.user_prompt("same", queued=True)
    renderer.user_prompt("same", queued=True)
    renderer.emit("Working output " * 100)
    ui.input.text = "unfinished draft"
    if paint_first:
        before = ui._transcript_content(columns)
        ui.transcript._following = False
        ui.transcript.vertical_scroll = before.line_count - 3
        old_row = before.get_line(ui.transcript.vertical_scroll)
    examined = []
    append = WrappedTranscript.append
    def measure(self, text, **kwargs):
        examined.append(text)
        return append(self, text, **kwargs)
    monkeypatch.setattr(WrappedTranscript, "append", measure)
    renderer.handle(AgentEvent(kind="user_message_sent", text=literal))
    renderer.handle(AgentEvent(kind="user_message_sent", text="same"))
    content = ui._transcript_content(columns)
    if paint_first:
        assert not examined
        assert content.get_line(ui.transcript.vertical_scroll) == old_row
        assert not ui.transcript._following
    text = "\n".join("".join(part[1] for part in content.get_line(i)) for i in range(content.line_count))
    assert text.count("queued") == 2  # User's literal text and the remaining identical prompt's notice.
    assert text.count("> same") == 2
    assert ui.input.text == "unfinished draft"
    # Resize/reload rebuild from the updated originals; notices cannot return.
    ui.refresh()
    content = ui._transcript_content(100)
    text = "\n".join("".join(part[1] for part in content.get_line(i)) for i in range(content.line_count))
    assert text.count("(queued — the agent is still working)") == 2
    assert literal in text
    renderer.handle(AgentEvent(kind="user_message_sent", text="same"))
    content = ui._transcript_content(100)
    text = "\n".join("".join(part[1] for part in content.get_line(i)) for i in range(content.line_count))
    assert text.count("(queued — the agent is still working)") == 1
    assert ui._flow.prompt_at(content.line_count - 1) is not None
    renderer.emit("The new answer")
    content = ui._transcript_content(100)
    text = "\n".join("".join(part[1] for part in content.get_line(i)) for i in range(content.line_count))
    assert "The new answer" in text and literal in text
    ui._close_transcript()


def test_long_thought_updates_only_format_a_bounded_tail(display, monkeypatch):
    import slipagent.transcript as terminal
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui.write_chunk("Thinking through the next step. " * 2000, first=True)
        with set_app(ui.app):
            ui.transcript.content.create_content(100, 17)
            examined = []
            original = terminal.wrap
            def measure(text, *args, **kwargs):
                examined.append(len(text))
                return original(text, *args, **kwargs)
            monkeypatch.setattr(terminal, "wrap", measure)
            for text in ("Now ", "check ", "the result."):
                ui.write_chunk(text)
                ui.app.render_counter += 1
                ui.transcript.content.create_content(100, 17)
            # Counting processed input avoids machine-dependent timing tests.
            assert sum(examined) < 3000


def test_unchanged_repaint_does_not_scan_scrollback(display, monkeypatch):
    import prompt_toolkit.layout.controls as controls
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui.write("Old output\n" * 2000)
        with set_app(ui.app):
            ui.transcript.content.create_content(100, 17)
            examined = []
            original = controls.split_lines
            def measure(fragments):
                examined.append(sum(len(fragment[1]) for fragment in fragments))
                return original(fragments)
            monkeypatch.setattr(controls, "split_lines", measure)
            ui.app.render_counter += 1
            content = ui.transcript.content.create_content(100, 17)
            assert content.line_count >= 2000
            assert sum(examined) < 1000


def test_streaming_rewraps_only_the_active_output_block(display, monkeypatch):
    import slipagent.transcript as terminal
    sink, _, output, _, _ = display
    ui = TerminalUI(lambda width: "status", sink, output=output)
    for index in range(500):
        ui.write(f"completed block {index} " * 10)
    ui.write_chunk("Active reply " * 10, first=True)
    ui.transcript.content.create_content(100, 17)
    wrapped = []
    original = terminal.wrap
    def record_wrap(text, *args, **kwargs):
        wrapped.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(terminal, "wrap", record_wrap)
    ui.write_chunk("continues here")
    ui.transcript.content.create_content(100, 17)
    assert wrapped and all("completed block" not in text for text in wrapped)


def test_refresh_migrates_legacy_transcript_and_keeps_live_state(display):
    from prompt_toolkit.layout.controls import FormattedTextControl
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui._raw_output.close()
        # State left by the previous implementation before its methods reload.
        ui._raw_output = ["\x1b[1mfirst\x1b[0m", "growing reply"]
        del ui._continuation_indents
        del ui._pinned_prompt
        ui.transcript.content = FormattedTextControl("old display")
        ui._transcript = [("", "old cached text")]
        ui._wrapped_spans = [(0, 0)]
        app = ui.app
        window = ui.transcript
        ui.input.text = "draft"
        ui._lines.put_nowait("queued")
        ui.refresh()
        ui.write_chunk(" continues")
        content = ui.transcript.content.create_content(100, 17)
        assert content.get_line(0) == [("bold", "first")]
        assert content.get_line(1) == [("", "growing reply continues")]
        assert ui.app is app and ui.transcript is window
        assert ui.input.text == "draft" and ui._lines.get_nowait() == "queued"
        ui.close()


def test_failed_refresh_leaves_legacy_transcript_available_for_rollback(display, monkeypatch):
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        archive = ui._raw_output
        legacy = ["retained output"]
        ui._raw_output = legacy
        def fail():
            raise ValueError("rejected bindings")
        monkeypatch.setattr(ui, "_bindings", fail)
        try:
            with pytest.raises(ValueError, match="rejected bindings"):
                ui.refresh()
            assert ui._raw_output is legacy
        finally:
            if ui._raw_output is not legacy:
                ui._raw_output.close()
            ui._raw_output = archive
            ui.close()


def test_scrollback_reads_only_requested_rows_and_preserves_originals(display, monkeypatch):
    from slipagent.transcript import TranscriptFile
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        text = "\n".join(f"row {index}" for index in range(1000))
        ui.write(text)
        try:
            content = ui.transcript.content.create_content(100, 17)
            reads = []
            original = TranscriptFile.__getitem__
            def measure(archive, index):
                reads.append(index)
                return original(archive, index)
            monkeypatch.setattr(TranscriptFile, "__getitem__", measure)
            for index in [0, 500, 998, 0]:
                assert content.get_line(index) == [("", f"row {index}")]
            assert len(reads) == 3
            assert ui._raw_output[0] == text
        finally:
            ui.close()


async def test_long_reasoning_stream_keeps_input_and_stop_responsive(display):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ui.set_working(True)
        task = asyncio.create_task(ui.run())
        try:
            ui.write_chunk("Considering the next step. " * 4000, first=True)
            await wait_until(lambda: "Considering" in "\n".join(snapshot()[:15]))
            for word in ["Now ", "check ", "the ", "result."]:
                ui.write_chunk("\x1b[2m" + word + "\x1b[0m")
                pipe.send_text(word)
                await asyncio.sleep(.04)
            await wait_until(lambda: "Now check the result." in snapshot()[18])
            pipe.send_text("\x15/stop\r")
            assert await asyncio.wait_for(ui.read_line(), 1) == "/stop"
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("cancel", [False, True])
async def test_secret_entry_has_no_prompt_history_and_restores_draft(display, cancel):
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output)
        ordinary = ui.input.buffer
        ordinary.history.append_string("previous task")
        ordinary.text = "unfinished draft"
        ordinary.cursor_position = 4
        task = asyncio.create_task(ui.ask("Key: ", password=True))
        await asyncio.sleep(0)
        secret = ui.input.buffer
        secret.text = "FAKE-SECRET-AUDIT"
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            secret.validate_and_handle()
            assert await task == "FAKE-SECRET-AUDIT"
        assert ui.input.buffer is ordinary
        assert ordinary.text == "unfinished draft" and ordinary.cursor_position == 4
        assert ordinary.history.get_strings() == ["previous task"]
        assert secret.text == ""
        assert secret.history.get_strings() == []


@pytest.fixture
def display():
    sink = io.StringIO()
    size = [Size(rows=24, columns=100)]
    output = Vt100_Output(sink, lambda: size[0], term="xterm-256color", enable_cpr=False)
    screen = pyte.Screen(100, 24)
    parser = pyte.Stream(screen)
    offset = [0]
    def snapshot():
        data = sink.getvalue()
        parser.feed(data[offset[0]:])
        offset[0] = len(data)
        return screen.display
    return sink, size, output, screen, snapshot


async def test_menu_uses_footer_and_scrolls_selection_through_refresh_and_resize(display):
    sink, size, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        choice = None
        try:
            ui.write("Transcript stays visible")
            await wait_until(lambda: "Ready" in snapshot()[17])
            before = snapshot()[:15]
            choice = asyncio.create_task(ui.choose("Commands", [
                (str(i), f"Option {i} — A saved conversation title with additional details")
                for i in range(12)
            ]))
            await wait_until(lambda: snapshot()[17].strip() == "Commands")
            assert snapshot()[:15] == before
            assert snapshot()[23].startswith("↑/↓ or wheel = move | Enter or click = select | Esc = back")
            assert "› Option 0" in snapshot()[18]
            assert "Option 4" in snapshot()[22]
            assert screen.buffer[19][0].fg == "999999"
            assert screen.buffer[19][99].bg == "400040"
            assert screen.buffer[17][0].fg != "999999"
            pipe.send_text("\x1b[A" + "\x1b[B" * 8)
            await wait_until(lambda: "› Option 8" in "\n".join(snapshot()[18:23]))
            assert "Option 0" not in "\n".join(snapshot()[18:23])
            pipe.send_text("\x1b[B" * 8)
            await wait_until(lambda: "› Option 11" in "\n".join(snapshot()[18:23]))
            pipe.send_text("\x1b[A" * 8)
            await wait_until(lambda: "› Option 3" in "\n".join(snapshot()[18:23]))
            assert "Option 11" not in "\n".join(snapshot()[18:23])
            ui.refresh()
            size[0] = Size(rows=24, columns=40)
            screen.resize(lines=24, columns=40)
            ui.app._on_resize()
            await wait_until(lambda: "› Option 3" in "\n".join(snapshot()[18:23]))
            assert snapshot()[23].startswith("↑↓/wheel move | Enter/click select | Esc")
            pipe.send_text("\x1b[A" * 3)
            await wait_until(lambda: "› Option 0" in "\n".join(snapshot()[18:23]))
            pipe.send_text("\x1b[B" * 3)
            await wait_until(lambda: "› Option 3" in "\n".join(snapshot()[18:23]))
            pipe.send_text("\r")
            assert await asyncio.wait_for(choice, 3) == "3"
            await wait_until(lambda: "Ready" in snapshot()[17])
            assert ui._lines.empty()
        finally:
            ui.close()
            if choice is not None:
                await choice
            await task


@pytest.mark.parametrize("columns", [40, 100])
async def test_menu_supports_wheel_navigation_and_clicking_wrapped_items(display, columns):
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=columns)
    screen.resize(lines=24, columns=columns)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        choice = asyncio.create_task(ui.choose("Commands", [
            (str(i), f"Option {i} with enough description to wrap on a narrow screen")
            for i in range(12)
        ]))
        try:
            await wait_until(lambda: snapshot()[17].strip() == "Commands")
            pipe.send_text("\x1b[<65;5;20M")
            await wait_until(lambda: ui._menu_index == 1)
            pipe.send_text("\x1b[<64;5;20M")
            await wait_until(lambda: ui._menu_index == 0)
            pipe.send_text("\x1b[B")
            await wait_until(lambda: "› Option 1" in "\n".join(snapshot()[18:23]))
            rows = snapshot()
            row = next(i for i in range(18, 23) if "Option " in rows[i] and "›" not in rows[i])
            selected = rows[row].split("Option ")[1].split()[0]
            # On a narrow screen, click the continuation of an unselected item.
            if columns == 40 and row + 1 < 23:
                row += 1
            pipe.send_text(f"\x1b[<0;5;{row + 1}M\x1b[<0;5;{row + 1}m")
            assert await asyncio.wait_for(choice, 3) == selected
            assert ui._lines.empty()
        finally:
            ui.close()
            await choice
            await task


async def test_ready_commands_link_preserves_draft_and_inserts_command(display):
    from slipagent.cli import command_menu_options
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Ready  Commands" in snapshot()[17])
            assert screen.buffer[17][7].underscore
            pipe.send_text("\x1b[<35;60;18M")
            await asyncio.sleep(0.1)
            assert ui.__dict__.get("_hovered_control") != "commands"
            pipe.send_text("\x1b[<0;60;18M\x1b[<0;60;18m")
            await asyncio.sleep(0.1)
            assert ui._menu_future is None
            pipe.send_text("unfinished draft")
            await wait_until(lambda: "unfinished draft" in snapshot()[18])
            pipe.send_text("\x1b[<35;8;18M")
            await wait_until(lambda: snapshot() and screen.buffer[17][7].reverse)
            pipe.send_text("\x1b[<35;1;1M")
            await wait_until(lambda: snapshot() and not screen.buffer[17][7].reverse)
            pipe.send_text("\x1b[<0;8;18M\x1b[<0;8;18m")
            await wait_until(lambda: ui._menu_future is not None)
            options = command_menu_options()
            assert ui._menu_options == options
            index = next(i for i, (value, _) in enumerate(options) if value == "/rename ")
            assert "<name>" in options[index][1]
            ui._move_menu(index)
            await wait_until(lambda: "› /rename <name>" in "\n".join(snapshot()[18:23]))
            pipe.send_text("\r")
            await wait_until(lambda: ui.input.text == "/rename ")
            assert "unfinished draft" in ui.input.buffer.history.get_strings()
            assert ui.input.buffer.cursor_position == len("/rename ")
            assert ui._lines.empty()
            ui.set_working(True)
            await wait_until(lambda: "Commands" not in snapshot()[17])
            ui.set_working(False)
            await wait_until(lambda: "Ready  Commands" in snapshot()[17])
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("working", [False, True])
@pytest.mark.parametrize("columns", [40, 100])
async def test_viewer_exit_link_replaces_commands_and_closes_only_on_its_words(display, working, columns):
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=columns)
    screen.resize(lines=24, columns=columns)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Commands" in snapshot()[17])
            ui.input.text = "saved draft"
            ui.set_working(working)
            ui.open_viewer("Contents", "sample text", diff=working)
            await wait_until(lambda: "ESC to exit viewer" in snapshot()[17])
            row = snapshot()[17]
            assert "Commands" not in row
            assert "esc to interrupt" not in row
            start = row.index("exit viewer")
            assert screen.buffer[17][start].underscore
            assert not screen.buffer[17][start - 2].underscore
            blank = 21 if columns == 40 else 70
            pipe.send_text(f"\x1b[<35;{blank};18M\x1b[<0;{blank};18M\x1b[<0;{blank};18m")
            await asyncio.sleep(0.1)
            assert ui._viewer_active
            assert ui.__dict__.get("_hovered_control") != "exit-viewer"
            pipe.send_text(f"\x1b[<35;{start + 1};18M")
            await wait_until(lambda: snapshot() and screen.buffer[17][start].reverse)
            pipe.send_text(f"\x1b[<0;{start + 1};18M\x1b[<0;{start + 1};18m")
            await wait_until(lambda: not ui._viewer_active)
            assert ui.working == working
            assert ui.input.text == "saved draft"
            ui.set_working(False)
            await wait_until(lambda: "Ready  Commands" in snapshot()[17])
        finally:
            ui.close()
            await task


async def test_menu_blank_rows_do_not_hover_or_select_the_last_item(display):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        choice = asyncio.create_task(ui.choose("Menu", [("one", "One"), ("two", "Two")]))
        try:
            await wait_until(lambda: "› One" in snapshot()[18])
            pipe.send_text("\x1b[<35;70;22M")
            await asyncio.sleep(0.1)
            assert ui.__dict__.get("_hovered_control") is None
            pipe.send_text("\x1b[<0;70;22M\x1b[<0;70;22m")
            await asyncio.sleep(0.1)
            assert not choice.done()
            # A real menu row remains clickable across the available width.
            pipe.send_text("\x1b[<0;70;20M\x1b[<0;70;20m")
            assert await asyncio.wait_for(choice, 3) == "two"
        finally:
            ui.close()
            await choice
            await task


@pytest.mark.parametrize("available", [True, False])
async def test_project_directory_link_requires_opener_and_disables_after_failure(display, tmp_path, monkeypatch, available):
    from slipagent import terminal
    from unittest.mock import Mock
    sink, _, output, screen, snapshot = display
    monkeypatch.setattr(terminal, "folder_opener", lambda: ("test-opener",) if available else None)
    launch = Mock(return_value=False)
    monkeypatch.setattr(terminal, "open_folder", launch)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "project  │  model: test  │  free: 42", sink,
                        input=pipe, output=output, project_directory=tmp_path)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "project" in snapshot()[23])
            assert screen.buffer[23][0].underscore is available
            assert not screen.buffer[23][11].underscore
            if available:
                pipe.send_text("\x1b[<35;2;24M")
                await wait_until(lambda: snapshot() and screen.buffer[23][0].reverse)
            pipe.send_text("\x1b[<0;2;24M\x1b[<0;2;24m")
            if available:
                await wait_until(lambda: "Could not open" in "\n".join(snapshot()))
                launch.assert_called_once_with(tmp_path, ("test-opener",))
                assert not screen.buffer[23][0].underscore
            else:
                await asyncio.sleep(.05)
                launch.assert_not_called()
            assert ui._lines.empty()
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("available", [True, False])
async def test_tool_directory_links_wrap_hover_click_and_preserve_selection(display, tmp_path, monkeypatch, available):
    from unittest.mock import Mock
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseEvent, MouseEventType, MouseButton
    from slipagent import terminal
    sink, _, output, _, _ = display
    directory = tmp_path / "directory with spaces"
    directory.mkdir()
    monkeypatch.setattr(terminal, "folder_opener", lambda: ("test-opener",) if available else None)
    launch = Mock(return_value=True)
    monkeypatch.setattr(terminal, "open_folder", launch)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", sink, input=pipe, output=output, project_directory=tmp_path)
        renderer = Renderer(Style(False), sink, False)
        renderer.terminal = ui
        renderer.handle(AgentEvent(kind="tool_start", tool_call=ToolCall(id="directory", name="list_dir", arguments={"path": directory.name})))
        for width in (30, 100):
            content = ui._transcript_content(width)
            links = [(y, sum(len(t) for _, t, *_ in content.get_line(y)[:i]))
                     for y in range(content.line_count)
                     for i, (style, _, *_) in enumerate(content.get_line(y)) if "class:clickable" in style]
            assert bool(links) is available
        if available:
            y, x = links[0]
            point = Point(x=x, y=y)
            def event(kind, position=point):
                return MouseEvent(position, kind, MouseButton.LEFT, frozenset())
            ui._select_transcript(event(MouseEventType.MOUSE_MOVE))
            assert any("class:hover" in style for style, _, *_ in ui._transcript_content(100).get_line(y))
            copy = Mock()
            monkeypatch.setattr(ui, "_copy_text", copy)
            with set_app(ui.app):
                ui._select_transcript(event(MouseEventType.MOUSE_DOWN))
                ui._select_transcript(event(MouseEventType.MOUSE_UP))
                await asyncio.sleep(.05)
                launch.assert_called_once_with(directory, ("test-opener",))
                ui._select_transcript(event(MouseEventType.MOUSE_DOWN))
                ui._select_transcript(event(MouseEventType.MOUSE_UP, Point(x=x + 5, y=y)))
                copy.assert_not_called()
                assert ui.selected_text()
                assert launch.call_count == 1
            ui.clear_transcript()
            assert not ui.__dict__["_directory_links"]
        ui._close_transcript()


async def test_danger_status_suffix_is_red_last_and_survives_resize(display, tmp_path):
    from types import SimpleNamespace
    from slipagent.workspace import Workspace
    sink, size, output, screen, snapshot = display
    session = SimpleNamespace(workspace=Workspace(tmp_path), agent=SimpleNamespace(model="stub/model"), free_calls=42)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda columns: _status_bar(session, Style(True), columns=columns), sink,
                        input=pipe, output=output, status_suffix=lambda: _danger_suffix(session, Style(True)))
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Free: 42" in snapshot()[23])
            assert "Danger Mode" not in snapshot()[23]
            session.workspace.access.danger = True
            ui.app.invalidate()
            await wait_until(lambda: " | Danger Mode" in snapshot()[23])
            assert snapshot()[23].index("Free: 42") < snapshot()[23].index("Danger Mode")
            start = snapshot()[23].index("Danger Mode")
            assert all(screen.buffer[23][column].fg == "ff6666" for column in range(start, start + len("Danger Mode")))
            choice = asyncio.create_task(ui.choose("Commands", [("one", "One")]))
            await wait_until(lambda: snapshot()[17].strip() == "Commands")
            assert " | Danger Mode" in snapshot()[23]
            assert "Enter or click = select" in snapshot()[23]
            pipe.send_text("\x1b")
            assert await asyncio.wait_for(choice, 3) is None
            await wait_until(lambda: "Ready" in snapshot()[17])
            size[0] = Size(rows=24, columns=35)
            screen.resize(lines=24, columns=35)
            ui.app._on_resize()
            await wait_until(lambda: " | Danger Mode" in snapshot()[23] and "Free: 42" in snapshot()[23])
            session.workspace.access.danger = False
            ui.app.invalidate()
            await wait_until(lambda: "Danger Mode" not in snapshot()[23])
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("cancel_key", ["\x1b", "\x04"])
async def test_menu_cancellation_preserves_input_history_scroll_and_running_turn(display, cancel_key):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        interrupts = []
        ui.set_interrupt_handler(lambda: interrupts.append(True))
        task = asyncio.create_task(ui.run())
        choice = None
        try:
            await wait_until(lambda: "Ready" in snapshot()[17])
            pipe.send_text("previous prompt\r")
            assert await asyncio.wait_for(ui.read_line(), 3) == "previous prompt"
            pipe.send_text("draft in progress\x1b[D")
            await wait_until(lambda: ui.input.buffer.text == "draft in progress" and
                             ui.input.buffer.cursor_position == len("draft in progres"))
            history = ui.input.buffer.history.get_strings()
            cursor = ui.input.buffer.cursor_position
            ui.write("\n".join(f"line {i}" for i in range(70)))
            ui.set_working(True)
            await wait_until(lambda: "line 69" in "\n".join(snapshot()[:15]))
            ui._scroll_output(-10)
            await wait_until(lambda: "line 69" not in "\n".join(snapshot()[:15]))
            before = snapshot()[:15]
            choice = asyncio.create_task(ui.choose("Commands", [("one", "One"), ("two", "Two")]))
            await wait_until(lambda: "Commands" in snapshot()[17])
            pipe.send_text("ignored\x7f\x1b[A\x1c" + cancel_key)
            assert await asyncio.wait_for(choice, 3) is None
            await wait_until(lambda: "Working (esc to interrupt)" in snapshot()[17])
            assert snapshot()[:15] == before
            assert ui.input.buffer.text == "draft in progress"
            assert ui.input.buffer.cursor_position == cursor
            assert ui.input.buffer.history.get_strings() == history
            assert ui._lines.empty()  # Neither /stop nor EOF was submitted.
            assert not interrupts
            assert not ui.context_visible
            pipe.send_text("\r")
            assert await asyncio.wait_for(ui.read_line(), 3) == "draft in progress"
        finally:
            ui.close()
            if choice is not None:
                await choice
            await task


async def test_escape_interrupts_directly_and_preserves_draft(display):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        interrupted = asyncio.Event()
        ui.set_interrupt_handler(interrupted.set)
        ui.set_working(True)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Working" in snapshot()[17])
            pipe.send_text("keep this draft")
            await wait_until(lambda: ui.input.buffer.text == "keep this draft")
            pipe.send_text("\x1b")
            await asyncio.wait_for(interrupted.wait(), 2)
            assert ui.input.buffer.text == "keep this draft"
            assert ui._lines.empty()
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("cancel_task", [False, True])
async def test_menu_releases_waiter_on_close_or_cancellation(display, cancel_task):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Ready" in snapshot()[17])
            # Refresh migrates terminals created before menus were available.
            del ui._menu_future
            ui.refresh()
            assert await ui.choose("Empty", []) is None
            choice = asyncio.create_task(ui.choose("Commands", [("one", "One")]))
            await wait_until(lambda: "Commands" in snapshot()[17])
            if cancel_task:
                choice.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await choice
                await wait_until(lambda: "Ready" in snapshot()[17])
                pipe.send_text("still usable\r")
                assert await asyncio.wait_for(ui.read_line(), 3) == "still usable"
            else:
                ui.close()
                assert await asyncio.wait_for(choice, 3) is None
        finally:
            ui.close()
            await task


async def test_footer_stays_fixed_while_transcript_scrolls_and_input_survives(display) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "CWD │ model: stub/model │ free: 42", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            pipe.send_text("partially typed")
            await wait_until(lambda: "partially typed" in snapshot()[18])
            for number in range(50):
                ui.write(f"output {number}")
            ui.set_working(True)
            await wait_until(lambda: "output 49" in "\n".join(snapshot()[:15]))
            rows = snapshot()
            assert rows[15].strip() == ""
            assert "Working" in rows[17]
            assert "partially typed" in rows[18]
            assert ui._notification_text() in rows[17]
            assert "model: stub/model" in rows[23]
            assert "free: 42" in rows[23]
            assert "output 0" not in "\n".join(rows[:17])
            pipe.send_text("\r")
            assert await ui.read_line() == "partially typed"
        finally:
            ui.close()
            await task


async def test_latest_user_prompt_stays_pinned_until_next_prompt_reaches_top(display):
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(True), sink, False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            ui.write("Startup information\n" * 4)
            renderer.user_prompt("Keep the original task visible")
            await wait_until(lambda: "> Keep the original task visible" in "\n".join(snapshot()[:15]))
            assert snapshot()[0].strip() == "Startup information"
            pipe.send_text("unfinished draft")
            await wait_until(lambda: "unfinished draft" in snapshot()[18])
            ui.write("\n".join(f"result {i}" for i in range(50)))
            await wait_until(lambda: snapshot()[0].rstrip() == "> Keep the original task visible")
            assert "result 49" in "\n".join(snapshot()[1:15])
            assert screen.buffer[0][0].fg == "ffffff"
            assert screen.buffer[0][0].bg == "004000"
            assert "unfinished draft" in snapshot()[18]
            assert "readout" in snapshot()[23]

            before = snapshot()[1:15]
            # Wheel events over the pinned prompt must still scroll history.
            pipe.send_text("\x1b[<64;1;1M" * 3)
            await wait_until(lambda: snapshot()[1:15] != before)
            assert snapshot()[0].rstrip() == "> Keep the original task visible"
            pipe.send_text("\x1b[<65;1;1M" * 30)
            await wait_until(lambda: "result 49" in "\n".join(snapshot()[1:15]))

            renderer.user_prompt("The next task", queued=True)
            await wait_until(lambda: "> The next task" in "\n".join(snapshot()[1:15]))
            assert snapshot()[0].rstrip() == "> Keep the original task visible"
            assert "> The next task" in "\n".join(snapshot()[:15])
            ui.write("\n".join(f"next result {i}" for i in range(50)))
            await wait_until(lambda: snapshot()[0].rstrip() == "> The next task")
            assert "next result 49" in "\n".join(snapshot()[1:15])
            assert "queued" not in snapshot()[0]
            assert "unfinished draft" in snapshot()[18]
        finally:
            ui.close()
            await task


async def test_pinned_prompt_follows_scrollback_in_both_directions(display):
    sink, size, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(True), sink, False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            ui.write("Startup information\n" * 4)
            for prompt in ("First task\nFirst details", "Second task", "Third task\nThird details"):
                renderer.user_prompt(prompt)
                ui.write("\n".join(f"{prompt.splitlines()[0]} output {i}" for i in range(40)))
            await wait_until(lambda: snapshot()[0].rstrip() == "> Third task")
            pipe.send_text("unfinished draft")
            await wait_until(lambda: "unfinished draft" in snapshot()[18])

            async def scroll_to(text, offset=0):
                content = ui.transcript.content.create_content(size[0].columns, 17)
                row = next(i for i in range(content.line_count)
                           if "".join(part[1] for part in content.get_line(i)) == text) + offset
                ui._scroll_output(row - ui.transcript.vertical_scroll)
                await wait_until(lambda: ui.transcript.vertical_scroll == row)

            # Crossing each prompt boundary changes the header in either direction.
            for text, offset, pinned in (
                ("> Third task", -1, "> Second task"),
                ("> Second task", -1, "> First task"),
                ("Startup information", 0, "Startup information"),
                ("> First task", 0, "> First task"),
                ("> Second task", 0, "> Second task"),
                ("> Third task", 0, "> Third task"),
            ):
                await scroll_to(text, offset)
                await wait_until(lambda: snapshot()[0].rstrip() == pinned)
                assert "unfinished draft" in snapshot()[18]
                assert "readout" in snapshot()[23]

            # A wheel event on the header crosses back into the previous task.
            pipe.send_text("\x1b[<64;1;1M")
            await wait_until(lambda: snapshot()[0].rstrip() == "> Second task")
            assert screen.buffer[0][0].fg == "ffffff"
            pipe.send_text("\x1b[<65;1;1M")
            await wait_until(lambda: snapshot()[0].rstrip() == "> Third task")

            # Narrowing doubles many output rows; lookup must use the new file offsets.
            size[0] = Size(rows=24, columns=15)
            screen.resize(lines=24, columns=15)
            ui.app.invalidate()
            await wait_until(lambda: ui._wrapped_columns == 15)
            await scroll_to("> Second task", -1)
            await wait_until(lambda: snapshot()[0].rstrip() == "> First task")
            ui.refresh()
            await scroll_to("> Third task")
            await wait_until(lambda: snapshot()[0].rstrip() == "> Third task")
        finally:
            ui.close()
            await task


async def test_pinned_prompt_truncates_resizes_and_survives_context_view_and_refresh(display):
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=20)
    screen.resize(lines=24, columns=20)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(True), sink, False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            renderer.user_prompt("alpha beta gamma delta epsilon\nKeep tests passing")
            ui.write("\n".join(f"output {i}" for i in range(50)))
            await wait_until(lambda: snapshot()[0].rstrip() == "> alpha beta gamma…")
            assert snapshot()[1].startswith("output ")
            assert screen.buffer[0][0].fg == "ffffff"
            size[0] = Size(rows=24, columns=40)
            screen.resize(lines=24, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: snapshot()[0].rstrip() == "> alpha beta gamma delta epsilon")
            assert "output 49" in "\n".join(snapshot()[1:15])

            ui.set_context(json.dumps({"messages": [{"role": "system", "content": "System context"}]}))
            pipe.send_text("\x1c")
            await wait_until(lambda: "System context" in "\n".join(snapshot()[:15]))
            assert "> alpha" not in "\n".join(snapshot()[:15])
            pipe.send_text("\x1c")
            await wait_until(lambda: snapshot()[0].rstrip() == "> alpha beta gamma delta epsilon")
            ui.refresh()
            ui.write("after refresh")
            await wait_until(lambda: "after refresh" in "\n".join(snapshot()[2:15]))
            assert snapshot()[0].rstrip() == "> alpha beta gamma delta epsilon"
        finally:
            ui.close()
            await task


async def test_oversized_pinned_prompt_keeps_output_and_footer_visible(display):
    sink, size, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(True), sink, False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            prompt = " ".join(f"requirement {i}" for i in range(100))
            renderer.user_prompt(prompt)
            ui.write("\n".join(f"output {i}" for i in range(50)))
            await wait_until(lambda: snapshot()[0].startswith("> requirement 0 "))
            assert snapshot()[0].rstrip().endswith("…")
            assert snapshot()[1].startswith("output ")
            assert "output 49" in "\n".join(snapshot()[1:15])
            assert "readout" in snapshot()[23]
            # The complete prompt is still part of ordinary scrollback.
            content = ui.transcript.content.create_content(100, 17)
            assert any("requirement 99" in "".join(part[1] for part in content.get_line(i))
                       for i in range(content.line_count))
            size[0] = Size(rows=10, columns=100)
            screen.resize(lines=10, columns=100)
            ui.app._on_resize()
            await wait_until(lambda: "readout" in snapshot()[9])
            assert "output 49" in "\n".join(snapshot()[:3])
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("prompt, width, expected", [
    ("alpha beta later", 12, "> alpha…"),
    ("short task", 12, "> short task"),
    ("abcdefghijklmnopqr", 20, "> abcdefghijklmnopqr"),
    ("abcdefghijklmnopqrs", 20, "> abcdefghijklmnopq…"),
    ("x" * 40, 12, "> xxxxxxxxx…"),
    ("猫" * 10, 10, "> 猫猫猫…"),
    ("e\u0301" * 12, 10, "> " + "e\u0301" * 7 + "…"),
    ("👩‍💻" * 8, 10, "> " + "👩‍💻" * 3 + "…"),
    ("long prompt", 1, "…"),
])
def test_pinned_prompt_ellipsis_preserves_words_and_graphemes(display, prompt, width, expected):
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda columns: "readout", sink, input=pipe, output=output)
        try:
            renderer = Renderer(Style(True), sink, False)
            renderer.terminal = ui
            renderer.user_prompt(prompt)
            ui.write("output\n" * 50)
            ui.transcript.vertical_scroll = 40
            content = ui._pinned_prompt_content(width)
            assert content.line_count == 1
            assert "".join(fragment[1] for fragment in content.get_line(0)) == expected
            assert content.get_line(1) == []
        finally:
            ui.close()


@pytest.mark.parametrize("ansi_color", [
    pytest.param("38;2;0;255;0", id="saved-green"),
    pytest.param("38;2;102;255;102", id="light-green"),
])
async def test_pinned_line_requires_green_prompt_prefix_and_survives_reload(display, ansi_color):
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            # No renderer metadata: selection comes from the styled transcript.
            ui.write(f"\x1b[{ansi_color}m> Saved task\x1b[0m\n")
            ui.write(f"> Ordinary quote\n\x1b[{ansi_color}mGreen without marker\x1b[0m\n")
            ui.write("\n".join(f"output {i}" for i in range(40)))
            await wait_until(lambda: snapshot()[0].rstrip() == "> Saved task")
            assert screen.buffer[0][0].fg == "ffffff"
            # Rebuild from the file, as for a session without prompt metadata.
            del ui._pinned_prompt
            ui.refresh()
            ui.write("after refresh")
            await wait_until(lambda: "after refresh" in "\n".join(snapshot()[1:15]))
            assert snapshot()[0].rstrip() == "> Saved task"
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("failed", [False, True])
async def test_command_streams_stay_in_viewer_for_live_and_restored_feed(display, failed):
    sink, _, output, _, _ = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(False), sink, True)
        renderer.terminal = ui
        call = ToolCall("command", "run_command", {"command": "test-command"})
        body = "$ test-command\nexit code: 1\n--- stdout ---\nPRIVATE STDOUT\n--- stderr ---\nPRIVATE STDERR"
        result = ToolResult(body, failed)
        renderer.handle(AgentEvent("tool_start", tool_call=call, step=1))
        renderer.handle(AgentEvent("tool_output", tool_call=call, text="PRIVATE LIVE OUTPUT"))
        renderer.handle(AgentEvent("tool_end", tool_call=call, result=result, step=1))
        recorded = "\n".join(ui._raw_output)
        assert "exit code: 1" in recorded
        assert "PRIVATE" not in recorded
        assert ui._inspection_calls[(1, "command")]["result"] == body
        messages = [Message.assistant("", [call]), Message.tool_result("command", {"status": "error" if failed else "success", "content": body})]
        await renderer.restore_transcript(messages, [])
        recorded = "\n".join(ui._raw_output)
        assert "exit code: 1" in recorded
        assert "PRIVATE" not in recorded
        assert messages[-1].content["content"] == body


async def test_restored_transcript_replaces_output_and_scrolls_to_end(display):
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(True), sink, False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        first_reply = Message.assistant("Saved first answer")
        first_reply.reasoning = "Saved readable thoughts"
        first_reply.reasoning_details = [{"signature": "opaque signature"}]
        messages = [
            Message.system("System instructions stay in the context view"),
            Message.user("Saved first task"), first_reply,
            Message.user("Saved next task"),
            Message.assistant("Reading the file", [
                ToolCall("read", "read_file", {"path": "app.py"}),
                ToolCall("legacy", "custom_tool", {}),
            ]),
            Message.tool_result("read", json.dumps({"status": "error", "content": "Saved tool failure"})),
            Message.tool_result("legacy", '{"status": [], "data": "Legacy tool output"}'),
            Message.assistant("\n".join(f"Saved output {i}" for i in range(40)) + "\nSaved final answer"),
        ]
        original = [message.to_api() for message in messages]
        try:
            ui.write("Unrelated old transcript\n" * 50)
            await wait_until(lambda: "Unrelated old transcript" in snapshot()[0])
            ui._scroll_output(-1000)
            ui.input.text = "unfinished draft"
            ui.context_visible = True
            await renderer.restore_transcript(messages, [])
            await wait_until(lambda: "Saved final answer" in "\n".join(snapshot()[:15]))
            assert snapshot()[0].rstrip() == "> Saved next task"
            assert "unfinished draft" in snapshot()[18]
            assert "readout" in snapshot()[23]
            recorded = "\n".join(ui._raw_output)
            assert "Unrelated old transcript" not in recorded
            assert "System instructions" not in recorded and "opaque signature" not in recorded
            assert "Saved readable thoughts" in recorded and "Saved first answer" in recorded
            assert "read_file" in recorded and "Saved tool failure" in recorded
            assert 'status: list (0 items):' in recorded and 'Legacy tool output' in recorded
            assert "\x1b[38;2;255;0;255m" in recorded
            assert "\x1b[38;2;255;102;102m" in recorded
            assert [message.to_api() for message in messages] == original
            ui._scroll_output(-1000)
            await wait_until(lambda: "Saved first answer" in "\n".join(snapshot()[:15]))
            assert snapshot()[0].rstrip() == "> Saved first task"
            assert screen.buffer[0][0].fg == "ffffff"
            await renderer.restore_transcript(messages, ["Queued correction"])
            await wait_until(lambda: "> Queued correction" in "\n".join(snapshot()[:15]))
            assert "queued" in "\n".join(snapshot()[:15])
        finally:
            ui.close()
            await task


async def test_working_indicator_pulses_and_becomes_idle(display) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output, color=False)
        task = asyncio.create_task(ui.run())
        try:
            ui.set_working(True)
            await wait_until(lambda: "Working" in snapshot()[17])
            first = snapshot()[17]
            await wait_until(lambda: snapshot()[17] != first)
            ui.set_working(True, stopping=True)
            await wait_until(lambda: "Stopping After This Step" in snapshot()[17])
            ui.set_working(False)
            await wait_until(lambda: "Ready" in snapshot()[17])
            ui.set_command_status("Applying /model (Esc to interrupt)")
            await wait_until(lambda: "Applying /model" in snapshot()[17])
            assert not ui.working
            ui.set_command_status("")
            ui.set_working(True)
            await wait_until(lambda: "Working" in snapshot()[17])
        finally:
            ui.close()
            await task


def test_status_updates_flash_three_times_then_remain_steady(display, monkeypatch):
    sink, _, output, _, _ = display
    clock = [0.0]
    monkeypatch.setattr("slipagent.terminal.time.monotonic", lambda: clock[0])
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        try:
            ui.notify("Saved changes.")
            ui._activity()
            for start in [0.0, .3, .6]:
                clock[0] = start + .01
                assert any("status-flash" in fragment[0] for fragment in ui._activity())
                clock[0] = start + .16
                assert all("status-flash" not in fragment[0] for fragment in ui._activity())
            clock[0] = 1.2
            assert all("status-flash" not in fragment[0] for fragment in ui._activity())
            assert "Saved changes." in "".join(fragment[1] for fragment in ui._activity())
            ui.set_working(True)
            assert any("status-flash" in fragment[0] for fragment in ui._activity())
            clock[0] = 2.5
            assert all("status-flash" not in fragment[0] for fragment in ui._activity())
            clock[0] = 2.75
            assert all("status-flash" not in fragment[0] for fragment in ui._activity())
        finally:
            ui.close()


def test_title_animation_matches_working_and_stops_when_ready(display, monkeypatch):
    sink, _, output, _, _ = display
    titles = []
    clock = [0.0]
    monkeypatch.setattr(output, "set_title", titles.append)
    monkeypatch.setattr("slipagent.terminal.time.monotonic", lambda: clock[0])
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        try:
            ui.set_title("Task | SlipAgent")
            assert titles[-1] == "Task | SlipAgent"
            assert ui._activity()[0][1] == "Ready"
            ui.set_working(True)
            assert titles[-1] == "▌ Task | SlipAgent"
            clock[0] = .25
            ui.app.before_render.fire()
            assert titles[-1] == "▛ Task | SlipAgent"
            assert ui._activity()[0][1].startswith(titles[-1][0])
            ui.set_title("Renamed | SlipAgent")
            assert titles[-1] == "▛ Renamed | SlipAgent"
            ui._editor_active = True
            ui.app.before_render.fire()
            assert titles[-1] == "Renamed | SlipAgent"
            held_count = len(titles)
            clock[0] = .5
            ui.app.before_render.fire()
            assert len(titles) == held_count
            ui._editor_active = False
            ui.app.before_render.fire()
            assert titles[-1].endswith(" Renamed | SlipAgent")
            ui.set_working(False)
            assert titles[-1] == "Renamed | SlipAgent"
            assert ui._activity()[0][1] == "Ready"
            count = len(titles)
            clock[0] = .5
            ui.app.before_render.fire()
            assert len(titles) == count
            assert titles[-1] == "Renamed | SlipAgent"
            assert ui._activity()[0][1] == "Ready"
            from slipagent.terminal import PULSE_FRAMES
            from wcwidth import width
            assert all(width(frame) == 1 for frame in PULSE_FRAMES)
            ready = ui.app.style.get_attrs_for_style_str("class:idle")
            working = ui.app.style.get_attrs_for_style_str("class:pulse")
            assert ready.color == working.color
        finally:
            ui.close()


async def test_footer_resizes_and_wraps_input_above_notices_and_readout(display) -> None:
    sink, size, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: f"({width}) free: 42", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            size[0] = Size(rows=16, columns=40)
            screen.resize(lines=16, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: f"({ui._readout_width()})" in snapshot()[15])
            pipe.send_text("long input " * 8)
            await wait_until(lambda: "long input" in snapshot()[10])
            rows = snapshot()
            assert rows[8].strip() == ""
            assert "Ready" in rows[9]
            assert "long input" in rows[10]
            assert ui._notification_text() in rows[9]
            assert "free: 42" in rows[15]
        finally:
            ui.close()
            await task


async def test_input_word_wrap_preserves_editing_submission_and_resize(display) -> None:
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=20)
    screen.resize(lines=24, columns=20)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        text = "alpha beta gamma delta epsilon"
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            pipe.send_text(text)
            await wait_until(lambda: "epsilon" in snapshot()[19])
            assert snapshot()[18].rstrip() == "> alpha beta gamma"
            assert snapshot()[19].rstrip() == "delta epsilon"
            assert ui.input.buffer.text == text
            pipe.send_text("\x1b[D" * len("delta epsilon") + "X")
            await wait_until(lambda: "Xdelta epsilon" in snapshot()[19])
            assert ui.input.buffer.text == "alpha beta gamma Xdelta epsilon"
            size[0] = Size(rows=24, columns=40)
            screen.resize(lines=24, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: "gamma Xdelta epsilon" in snapshot()[18])
            assert snapshot()[19].strip() == ""
            pipe.send_text("\r")
            assert await ui.read_line() == "alpha beta gamma Xdelta epsilon"
            assert ui.input.buffer.history.get_strings()[-1] == "alpha beta gamma Xdelta epsilon"
        finally:
            ui.close()
            await task


def test_startup_and_help_list_all_commands_with_requested_spacing():
    banner = BANNER.format(model="test/model", workspace="project", tools="read_file", mcp="")
    for command in [
        "/help", "/menu", "/danger", "/tools", "/model", "/models", "/key", "/cost", "/mcp",
        "/rename", "/reset", "/reload", "/generations", "/init", "/stop", "/exit", "/quit",
    ]:
        assert command in banner
        assert command in HELP
    assert banner.splitlines()[1] == ""
    assert "\n\nType a task and press Enter." in banner


def test_startup_tool_list_keeps_hanging_indent_after_resize(display, tmp_path):
    from wcwidth import width

    sink, _, output, _, _ = display
    tools = ["read_file", "write_file", "list_dir"] + [f"mcp__server__tool_{i}" for i in range(40)]
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda columns: "status", sink, input=pipe, output=output)
        renderer = Renderer(Style(False), sink, verbose=False)
        renderer.terminal = ui
        try:
            renderer.show_banner(model="test/model", workspace=tmp_path, tools=tools)
            renderer.emit("  ordinary output wraps with its original indentation")
            for columns in [80, 40, 120, 14, 13, 80]:
                content = ui.transcript.content.create_content(columns, 17)
                rows = ["".join(fragment[1] for fragment in content.get_line(i))
                        for i in range(content.line_count)]
                start = next(i for i, row in enumerate(rows) if row.startswith("  Tools:"))
                end = next(i for i in range(start + 1, len(rows)) if not rows[i].strip())
                tool_rows = rows[start:end]
                assert len(tool_rows) > 1
                assert all(width(row) <= columns for row in tool_rows)
                if columns > 13:
                    assert all(row.startswith(" " * 13) for row in tool_rows[1:])
                assert "".join(row.strip().replace(" ", "") for row in tool_rows) == "Tools:" + ",".join(tools)
                ordinary = next(i for i, row in enumerate(rows) if row.startswith("  ordinary"))
                assert all(row.startswith("  ") for row in rows[ordinary:] if row.strip())
        finally:
            ui.close()


@pytest.mark.parametrize("text, first, second", [
    ("alpha 中文字符 beta gamma", "> alpha 中文字符", "beta gamma"),
    ("abcdefghijklmnopqrstuvwx tail", "> abcdefghijklmnopqr", "stuvwx tail"),
])
async def test_input_wrap_handles_wide_words_and_long_identifiers(display, text, first, second):
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=20)
    screen.resize(lines=24, columns=20)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            pipe.send_text(text)
            await wait_until(lambda: second in snapshot()[19])
            assert snapshot()[18].rstrip() == first
            pipe.send_text("\r")
            assert await ui.read_line() == text
        finally:
            ui.close()
            await task


async def test_turn_token_counts_follow_output_without_blank_line(monkeypatch):
    from types import SimpleNamespace
    import slipagent.cli as cli
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, verbose=False)
    async def run(prompt):
        renderer.handle(AgentEvent(kind="assistant_text", text="Finished."))
        return "Finished."
    async def read(*args):
        await asyncio.Event().wait()
    session = SimpleNamespace(
        renderer=renderer,
        extensions={},
        agent=SimpleNamespace(run=run, pending=[], stopped=False,
                              usage=Usage(prompt_tokens=10, completion_tokens=5, cost=0.0)),
    )
    monkeypatch.setattr(cli, "_read_line", read)
    assert await cli._run_request(session, "task", renderer.style) is False
    assert "Finished.\n    (in=10 out=5 cost=$0.000000)" in sink.getvalue()


@pytest.mark.parametrize("color", [True, False])
async def test_context_toggle_shows_ordered_system_prompts_and_preserves_draft(display, color):
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=26, columns=100)
    screen.resize(lines=26, columns=100)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output, color=color)
        renderer = Renderer(Style(color), sink, False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            renderer.emit("Ordinary transcript.")
            await wait_until(lambda: "Ordinary transcript." in "\n".join(snapshot()[:18]))
            pipe.send_text("unfinished draft")
            await wait_until(lambda: "unfinished draft" in snapshot()[20])
            before = snapshot()[:18]
            request = json.dumps({"model": "test/model", "messages": [
                {"role": "system", "content": "SYSTEM ONE"},
                {"role": "system", "content": "SYSTEM TWO"},
                {"role": "user", "content": "USER REQUEST"},
                {"role": "assistant", "content": "PREVIOUS ANSWER"},
            ]})
            renderer.handle(AgentEvent(kind="context", text=request))
            assert snapshot()[:18] == before
            pipe.send_text("\x1c")
            await wait_until(lambda: "SYSTEM TWO" in "\n".join(snapshot()[:18]))
            visible = "\n".join(snapshot()[:18])
            assert visible.index("SYSTEM ONE") < visible.index("SYSTEM TWO") < visible.index("USER REQUEST") < visible.index("PREVIOUS ANSWER")
            for text in ["SYSTEM ONE", "SYSTEM TWO"]:
                rows = snapshot()
                row = next(index for index, value in enumerate(rows) if text in value)
                assert screen.buffer[row][rows[row].index(text)].fg == ("ffff00" if color else "default")
            assert "unfinished draft" in snapshot()[20]
            assert "readout" in snapshot()[25]
            ui.refresh()
            assert "SYSTEM ONE" in "\n".join(snapshot()[:18])
            pipe.send_text("\x1c")
            await wait_until(lambda: snapshot()[:18] == before)
            assert ui.input.buffer.text == "unfinished draft"
            assert ui.input.buffer.cursor_position == len("unfinished draft")
            assert ui._lines.empty()
        finally:
            ui.close()
            await task


async def test_context_scrolls_and_updates_without_changing_transcript(display):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            for number in range(50):
                ui.write(f"transcript {number}")
            await wait_until(lambda: "transcript 49" in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[<64;1;5M" * 3)
            await wait_until(lambda: "transcript 49" not in "\n".join(snapshot()[:15]))
            before = snapshot()[:15]
            ui.set_context(json.dumps({"messages": [{"role": "system", "content": "\n".join(f"instruction {n}" for n in range(100))}]}))
            pipe.send_text("\x1c")
            await wait_until(lambda: "instruction 0" in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[<65;1;5M" * 3)
            await wait_until(lambda: "instruction 0" not in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[1~")
            await wait_until(lambda: "instruction 0" in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[6~")
            await wait_until(lambda: "instruction 0" not in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1b[4~")
            await wait_until(lambda: "instruction 99" in "\n".join(snapshot()[:15]))
            ui.set_context(json.dumps({"messages": [{"role": "system", "content": "UPDATED REQUEST"}], "tools": [{"name": "example"}], "response_format": {"type": "json_object"}}))
            await wait_until(lambda: "UPDATED REQUEST" in "\n".join(snapshot()[:15]))
            assert "example" in "\n".join(snapshot()[:15])
            pipe.send_text("\x1b[4~")
            await wait_until(lambda: "json_object" in "\n".join(snapshot()[:15]))
            pipe.send_text("\x1c")
            await wait_until(lambda: snapshot()[:15] == before)
        finally:
            ui.close()
            await task


async def test_password_prompt_masks_value_and_restores_normal_input(display) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        ask = asyncio.create_task(ui.ask("key: ", password=True))
        try:
            await wait_until(lambda: "key:" in snapshot()[18])
            pipe.send_text("secret-value")
            await wait_until(lambda: "************" in snapshot()[18])
            assert "secret-value" not in "\n".join(snapshot())
            pipe.send_text("\r")
            assert await ask == "secret-value"
            await wait_until(lambda: ">" in snapshot()[18])
        finally:
            ask.cancel()
            ui.close()
            await task


async def test_eof_stays_closed_after_current_turn(display) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            pipe.send_text("\x04")
            assert await asyncio.wait_for(ui.read_line(), 2) is None
            assert await asyncio.wait_for(ui.read_line(), 2) is None
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("queued", [False, True])
def test_user_prompts_have_blank_lines_on_both_sides(queued) -> None:
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, verbose=False)
    renderer.emit("previous output")
    if queued:
        renderer.handle(AgentEvent(kind="user_message", text="my request"))
    else:
        renderer.user_prompt("my request")
    renderer.emit("next output")
    lines = sink.getvalue().splitlines()
    assert lines[:4] == ["", "previous output", "", "> my request"]
    assert lines[-2:] == ["", "next output"]


@pytest.mark.parametrize("mouse_row", [5, 16, 17, 24])
async def test_mouse_wheel_scrolls_transcript_without_changing_input(display, mouse_row) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            pipe.send_text("old prompt\r")
            assert await ui.read_line() == "old prompt"
            history = ui.input.buffer.history.get_strings()
            pipe.send_text("draft")
            await wait_until(lambda: "draft" in snapshot()[18])
            for number in range(50):
                ui.write(f"output {number}")
            await wait_until(lambda: "output 49" in "\n".join(snapshot()[:15]))
            before = snapshot()[:15]
            footer = snapshot()[18:]
            pipe.send_text(f"\x1b[<64;1;{mouse_row}M" * 3)
            await wait_until(lambda: snapshot()[:15] != before)
            scrolled = snapshot()[:15]
            assert "output 49" not in "\n".join(scrolled)
            assert snapshot()[18:] == footer
            ui.write("new output")
            await asyncio.sleep(.2)
            assert snapshot()[:15] == scrolled
            pipe.send_text(f"\x1b[<65;1;{mouse_row}M" * 30)
            await wait_until(lambda: "new output" in "\n".join(snapshot()[:15]))
            ui.write("following output")
            await wait_until(lambda: "following output" in "\n".join(snapshot()[:15]))
            assert "draft" in snapshot()[18]
            assert ui.input.buffer.history.get_strings() == history
            pipe.send_text("\r")
            assert await ui.read_line() == "draft"
        finally:
            ui.close()
            await task


async def test_wheel_scrolls_inside_wrapped_lines_and_survives_resize(display) -> None:
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=16, columns=40)
    screen.resize(lines=16, columns=40)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            ui.write("".join(f"{number:02d}" + "x" * 38 for number in range(30)))
            await wait_until(lambda: any(row.startswith("29xxx") for row in snapshot()[:9]))
            pipe.send_text("\x1b[<64;1;5M" * 3)
            await wait_until(lambda: any(row.startswith("20xxx") for row in snapshot()[:9]))
            assert not any(row.startswith("29xxx") for row in snapshot()[:9])
            size[0] = Size(rows=16, columns=80)
            screen.resize(lines=16, columns=80)
            ui.app.invalidate()
            await asyncio.sleep(.2)
            pipe.send_text("\x1b[<64;1;5M" * 30)
            await wait_until(lambda: snapshot()[0].startswith("00xxx"))
            pipe.send_text("\x1b[<65;1;5M" * 30)
            await wait_until(lambda: "29xxx" in "\n".join(snapshot()[:9]))
            assert ">" in snapshot()[10]
        finally:
            ui.close()
            await task


async def test_wheel_events_without_coordinates_preserve_prompt_history(display) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            pipe.send_text("previous prompt\r")
            assert await ui.read_line() == "previous prompt"
            for number in range(50):
                ui.write(f"output {number}")
            await wait_until(lambda: "output 49" in "\n".join(snapshot()[:15]))
            before = snapshot()[:15]
            ui.app.key_processor.feed(KeyPress(Keys.ScrollUp))
            ui.app.key_processor.process_keys()
            await wait_until(lambda: snapshot()[:15] != before)
            assert ui.input.text == ""
            ui.app.key_processor.feed(KeyPress(Keys.ScrollDown))
            ui.app.key_processor.process_keys()
            await wait_until(lambda: snapshot()[:15] == before)
            assert ui.input.text == ""
        finally:
            ui.close()
            await task


async def test_output_wraps_at_words_preserves_spacing_and_reflows_on_resize(display) -> None:
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=20)
    screen.resize(lines=24, columns=20)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            ui.write("\x1b[1malpha beta gamma delta epsilon\x1b[0m\n\nName    Count\nalpha   12\nbeta    3")
            ui.write("    one two three four five")
            await wait_until(lambda: "Count" in "\n".join(snapshot()[:15]))
            assert [line.rstrip() for line in snapshot()[:6]] == [
                "alpha beta gamma", "delta epsilon", "", "Name    Count", "alpha   12", "beta    3",
            ]
            assert screen.buffer[0][0].bold
            assert screen.buffer[1][0].bold
            assert not screen.buffer[3][0].bold
            assert [line.rstrip() for line in snapshot()[6:8]] == [
                "    one two three", "    four five",
            ]
            size[0] = Size(rows=24, columns=40)
            screen.resize(lines=24, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: snapshot()[0].rstrip() == "alpha beta gamma delta epsilon")
            assert [line.rstrip() for line in snapshot()[1:5]] == [
                "", "Name    Count", "alpha   12", "beta    3",
            ]
            assert snapshot()[5].rstrip() == "    one two three four five"
            assert ">" in snapshot()[18]
            assert ui._notification_text() in snapshot()[17]
            assert "readout" in snapshot()[23]
        finally:
            ui.close()
            await task


async def test_output_wrap_uses_display_width_for_wide_characters(display) -> None:
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=14)
    screen.resize(lines=24, columns=14)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            ui.write("界界界 test 界界界")
            await wait_until(lambda: "test" in snapshot()[0])
            assert snapshot()[0].rstrip() == "界界界 test"
            assert snapshot()[1].rstrip() == "界界界"
        finally:
            ui.close()
            await task


async def test_streaming_chunks_wrap_reflow_and_keep_the_footer_and_draft(display) -> None:
    sink, size, output, screen, snapshot = display
    size[0] = Size(rows=24, columns=20)
    screen.resize(lines=24, columns=20)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(True), sink, verbose=False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[18])
            pipe.send_text("draft")
            await wait_until(lambda: "draft" in snapshot()[18])
            renderer.handle(AgentEvent(kind="assistant_delta", text="alpha be"))
            await wait_until(lambda: "alpha be" in snapshot()[1])
            renderer.handle(AgentEvent(kind="assistant_delta", text="ta gamma delta epsilon"))
            await wait_until(lambda: "delta epsilon" in snapshot()[2])
            assert snapshot()[1].rstrip() == "alpha beta gamma"
            assert not screen.buffer[1][0].bold
            size[0] = Size(rows=24, columns=40)
            screen.resize(lines=24, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: snapshot()[1].rstrip() == "alpha beta gamma delta epsilon")
            renderer.handle(AgentEvent(kind="stream_end"))
            renderer.handle(AgentEvent(kind="retry", text="Response rejected. Retrying from the start."))
            renderer.handle(AgentEvent(kind="reasoning_delta", text="Starting again."))
            await wait_until(lambda: "Thinking: Starting again." in "\n".join(snapshot()[:15]))
            rows = snapshot()
            row = next(index for index, value in enumerate(rows) if "Response rejected" in value)
            assert screen.buffer[row][2].fg == "ff6666"
            assert "draft" in rows[18]
            assert rows[15].strip() == ""
            assert ui._notification_text() in rows[17]
            assert "readout" in rows[23]
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("color", [True, False])
async def test_requested_colors_apply_to_user_input_tool_calls_and_errors(display, color) -> None:
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output, color=color)
        renderer = Renderer(Style(color), sink, verbose=False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            renderer.user_prompt("request")
            renderer.handle(AgentEvent(kind="tool_start", tool_call=ToolCall("call", "list_dir")))
            renderer.handle(AgentEvent(kind="tool_end", result=ToolResult.error("tool failed")))
            renderer.handle(AgentEvent(kind="warning", text="request failed"))
            renderer.emit(renderer.style.dim("muted detail"))
            renderer.emit("white detail")
            await wait_until(lambda: "request failed" in "\n".join(snapshot()[:15]))
            pipe.send_text("draft")
            await wait_until(lambda: "draft" in snapshot()[18])
            for text, expected in [
                ("> request", "ffffff"), ("list_dir", "ff00ff"),
                ("tool failed", "ff6666"), ("request failed", "ff6666"), ("draft", "ffffff"),
                ("muted detail", "aaaaaa"), ("readout", "aaaaaa"), ("white detail", "default"),
            ]:
                rows = snapshot()
                row = next(index for index, value in enumerate(rows) if text in value)
                column = rows[row].index(text)
                assert screen.buffer[row][column].fg == (expected if color else "default")
        finally:
            ui.close()
            await task


def test_independent_output_blocks_start_with_a_blank_line() -> None:
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, verbose=False)
    renderer.emit("first block\nsecond line")
    renderer.emit("next block")
    assert sink.getvalue().splitlines() == ["", "first block", "second line", "", "next block"]


def test_each_model_reply_starts_a_block_with_its_following_tool_calls() -> None:
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, verbose=True)
    renderer.handle(AgentEvent(kind="step_start", step=1))
    renderer.handle(AgentEvent(kind="assistant_text", step=1, text="Checking files."))
    for name in ["list_dir", "read_file"]:
        renderer.handle(AgentEvent(kind="tool_start", step=1, tool_call=ToolCall(name, name)))
        renderer.handle(AgentEvent(kind="tool_end", step=1, result=ToolResult.ok("result\nsecond line")))
    renderer.handle(AgentEvent(kind="step_end", step=1))
    renderer.handle(AgentEvent(kind="step_start", step=2))
    renderer.handle(AgentEvent(kind="tool_start", step=2, tool_call=ToolCall("grep", "grep")))
    renderer.handle(AgentEvent(kind="step_start", step=3))
    renderer.handle(AgentEvent(kind="assistant_text", step=3, text="Finished."))
    renderer.handle(AgentEvent(kind="step_start", step=1))
    renderer.handle(AgentEvent(kind="assistant_text", step=1, text="Next task."))
    assert sink.getvalue().splitlines() == [
        "", "Checking files.", "  ⚙ list_dir({})", "    result", "    second line",
        "  ⚙ read_file({})", "    result", "    second line", "  ⚙ grep({})", "", "Finished.", "", "Next task.",
    ]


def test_consecutive_model_replies_have_gaps_without_splitting_their_lines() -> None:
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, verbose=False)
    renderer.handle(AgentEvent(kind="assistant_text", text="First reply.\nSecond line."))
    renderer.handle(AgentEvent(kind="assistant_text", text="Next reply."))
    assert sink.getvalue().splitlines() == ["", "First reply.", "Second line.", "", "Next reply."]


async def test_output_unit_gaps_render_without_separating_model_text_and_tools(display) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(True), sink, verbose=True)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            renderer.emit("command response\nsecond line")
            renderer.handle(AgentEvent(kind="step_start", step=1))
            renderer.handle(AgentEvent(kind="assistant_text", step=1, text="Checking files."))
            renderer.handle(AgentEvent(kind="tool_start", step=1, tool_call=ToolCall("call", "list_dir")))
            renderer.handle(AgentEvent(kind="tool_end", step=1, result=ToolResult.ok("result")))
            renderer.handle(AgentEvent(kind="step_start", step=2))
            renderer.handle(AgentEvent(kind="assistant_text", step=2, text="Finished."))
            renderer.emit("next command response")
            await wait_until(lambda: "next command response" in "\n".join(snapshot()[:15]))
            assert [line.rstrip() for line in snapshot()[:11]] == [
                "", "command response", "second line", "", "Checking files.",
                "  ⚙ list_dir({})", "    result", "", "Finished.", "", "next command response",
            ]
            assert ">" in snapshot()[18]
        finally:
            ui.close()
            await task
