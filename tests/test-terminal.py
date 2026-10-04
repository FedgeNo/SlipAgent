"""Render the footer through a VT terminal emulator while editing live input."""

from __future__ import annotations

import asyncio
import io
import json

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
from slipagent.cli import BANNER, HELP, Renderer, Style
from slipagent.tools.base import ToolResult
from slipagent.types import Message, ToolCall, Usage


async def wait_until(predicate) -> None:
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.02)


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
        for name in ("_pinned_prompt", "_prompt_pending"):
            del ui.__dict__[name]
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
            await wait_until(lambda: "Considering" in "\n".join(snapshot()[:17]))
            for word in ["Now ", "check ", "the ", "result."]:
                ui.write_chunk("\x1b[2m" + word + "\x1b[0m")
                pipe.send_text(word)
                await asyncio.sleep(.04)
            await wait_until(lambda: "Now check the result." in snapshot()[19])
            pipe.send_text("\x03")
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


async def test_footer_stays_fixed_while_transcript_scrolls_and_input_survives(display) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "CWD │ model: stub/model │ free: 42", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[19])
            pipe.send_text("partially typed")
            await wait_until(lambda: "partially typed" in snapshot()[19])
            for number in range(50):
                ui.write(f"output {number}")
            ui.set_working(True)
            await wait_until(lambda: "output 49" in "\n".join(snapshot()[:17]))
            rows = snapshot()
            assert rows[17].strip() == ""
            assert "Working" in rows[18]
            assert "partially typed" in rows[19]
            assert rows[22].strip() == ""
            assert "model: stub/model" in rows[23]
            assert "free: 42" in rows[23]
            assert "output 0" not in "\n".join(rows[:17])
            pipe.send_text("\r")
            assert await ui.read_line() == "partially typed"
        finally:
            ui.close()
            await task


async def test_latest_user_prompt_pins_at_top_and_releases_on_next_prompt(display):
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(True), sink, False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            ui.write("Startup information\n" * 4)
            renderer.user_prompt("Keep the original task visible")
            await wait_until(lambda: "> Keep the original task visible" in "\n".join(snapshot()[:17]))
            assert snapshot()[0].strip() == "Startup information"
            pipe.send_text("unfinished draft")
            await wait_until(lambda: "unfinished draft" in snapshot()[19])
            ui.write("\n".join(f"result {i}" for i in range(50)))
            await wait_until(lambda: snapshot()[0].rstrip() == "> Keep the original task visible")
            assert "result 49" in "\n".join(snapshot()[1:17])
            assert screen.buffer[0][0].fg == "00ff00"
            assert "unfinished draft" in snapshot()[19]
            assert "readout" in snapshot()[23]

            before = snapshot()[1:17]
            # Wheel events over the pinned prompt must still scroll history.
            pipe.send_text("\x1b[<64;1;1M" * 3)
            await wait_until(lambda: snapshot()[1:17] != before)
            assert snapshot()[0].rstrip() == "> Keep the original task visible"
            pipe.send_text("\x1b[<65;1;1M" * 30)
            await wait_until(lambda: "result 49" in "\n".join(snapshot()[1:17]))

            renderer.user_prompt("The next task", queued=True)
            await wait_until(lambda: snapshot()[0].rstrip() != "> Keep the original task visible")
            assert "> The next task" in "\n".join(snapshot()[:17])
            ui.write("\n".join(f"next result {i}" for i in range(50)))
            await wait_until(lambda: snapshot()[0].rstrip() == "> The next task")
            assert "next result 49" in "\n".join(snapshot()[1:17])
            assert "queued" not in snapshot()[0]
            assert "unfinished draft" in snapshot()[19]
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
            await wait_until(lambda: "unfinished draft" in snapshot()[19])

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
                assert "unfinished draft" in snapshot()[19]
                assert "readout" in snapshot()[23]

            # A wheel event on the header crosses back into the previous task.
            pipe.send_text("\x1b[<64;1;1M")
            await wait_until(lambda: snapshot()[0].rstrip() == "> Second task")
            assert screen.buffer[0][0].fg == "00ff00"
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


async def test_pinned_prompt_wraps_resizes_and_survives_context_view_and_refresh(display):
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
            await wait_until(lambda: snapshot()[0].rstrip() == "> alpha beta gamma")
            assert [row.rstrip() for row in snapshot()[:2]] == [
                "> alpha beta gamma", "delta epsilon",
            ]
            assert all(screen.buffer[row][0].fg == "00ff00" for row in range(2))
            size[0] = Size(rows=24, columns=40)
            screen.resize(lines=24, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: snapshot()[0].rstrip() == "> alpha beta gamma delta epsilon")
            assert "output 49" in "\n".join(snapshot()[1:17])

            ui.set_context(json.dumps({"messages": [{"role": "system", "content": "System context"}]}))
            pipe.send_text("\x1c")
            await wait_until(lambda: "System context" in "\n".join(snapshot()[:17]))
            assert "> alpha" not in "\n".join(snapshot()[:17])
            pipe.send_text("\x1c")
            await wait_until(lambda: snapshot()[0].rstrip() == "> alpha beta gamma delta epsilon")
            ui.refresh()
            ui.write("after refresh")
            await wait_until(lambda: "after refresh" in "\n".join(snapshot()[2:17]))
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
            assert snapshot()[13].strip() == "…"
            assert "output 49" in "\n".join(snapshot()[14:17])
            assert "readout" in snapshot()[23]
            # The complete prompt is still part of ordinary scrollback.
            content = ui.transcript.content.create_content(100, 17)
            assert any("requirement 99" in "".join(part[1] for part in content.get_line(i))
                       for i in range(content.line_count))
            size[0] = Size(rows=10, columns=100)
            screen.resize(lines=10, columns=100)
            ui.app.invalidate()
            await wait_until(lambda: "readout" in snapshot()[9])
            assert "output 49" in "\n".join(snapshot()[:3])
        finally:
            ui.close()
            await task


async def test_pinned_line_requires_green_prompt_prefix_and_survives_reload(display):
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            # No renderer metadata: selection comes from the styled transcript.
            ui.write("\x1b[38;2;0;255;0m> Saved task\x1b[0m\n")
            ui.write("> Ordinary quote\n\x1b[38;2;0;255;0mGreen without marker\x1b[0m\n")
            ui.write("\n".join(f"output {i}" for i in range(40)))
            await wait_until(lambda: snapshot()[0].rstrip() == "> Saved task")
            assert screen.buffer[0][0].fg == "00ff00"
            # Rebuild from the file, as for a session without prompt metadata.
            for name in ("_pinned_prompt", "_prompt_pending"):
                del ui.__dict__[name]
            ui.refresh()
            ui.write("after refresh")
            await wait_until(lambda: "after refresh" in "\n".join(snapshot()[1:17]))
            assert snapshot()[0].rstrip() == "> Saved task"
        finally:
            ui.close()
            await task


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
            await wait_until(lambda: "Saved final answer" in "\n".join(snapshot()[:17]))
            assert snapshot()[0].rstrip() == "> Saved next task"
            assert "unfinished draft" in snapshot()[19]
            assert "readout" in snapshot()[23]
            recorded = "\n".join(ui._raw_output)
            assert "Unrelated old transcript" not in recorded
            assert "System instructions" not in recorded and "opaque signature" not in recorded
            assert "Saved readable thoughts" in recorded and "Saved first answer" in recorded
            assert "read_file" in recorded and "Saved tool failure" in recorded
            assert '{"status": [], "data": "Legacy tool output"}' in recorded
            assert "\x1b[38;2;255;0;255m" in recorded
            assert "\x1b[38;2;255;0;0m" in recorded
            assert [message.to_api() for message in messages] == original
            ui._scroll_output(-1000)
            await wait_until(lambda: "Saved first answer" in "\n".join(snapshot()[:17]))
            assert snapshot()[0].rstrip() == "> Saved first task"
            assert screen.buffer[0][0].fg == "00ff00"
            await renderer.restore_transcript(messages, ["Queued correction"])
            await wait_until(lambda: "> Queued correction" in "\n".join(snapshot()[:17]))
            assert "queued" in "\n".join(snapshot()[:17])
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
            await wait_until(lambda: "Working" in snapshot()[18])
            first = snapshot()[18]
            await wait_until(lambda: snapshot()[18] != first)
            ui.set_working(True, stopping=True)
            await wait_until(lambda: "Stopping After This Turn" in snapshot()[18])
            ui.set_working(False)
            await wait_until(lambda: "Ready" in snapshot()[18])
        finally:
            ui.close()
            await task


async def test_footer_resizes_and_wraps_input_above_blank_and_readout(display) -> None:
    sink, size, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: f"model: stub │ free: 42 ({width})", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[19])
            size[0] = Size(rows=16, columns=40)
            screen.resize(lines=16, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: "(40)" in snapshot()[15])
            pipe.send_text("long input " * 8)
            await wait_until(lambda: "long input" in snapshot()[11])
            rows = snapshot()
            assert rows[9].strip() == ""
            assert "Ready" in rows[10]
            assert "long input" in rows[12]
            assert rows[14].strip() == ""
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
            await wait_until(lambda: ">" in snapshot()[19])
            pipe.send_text(text)
            await wait_until(lambda: "epsilon" in snapshot()[20])
            assert snapshot()[19].rstrip() == "> alpha beta gamma"
            assert snapshot()[20].rstrip() == "delta epsilon"
            assert ui.input.buffer.text == text
            pipe.send_text("\x1b[D" * len("delta epsilon") + "X")
            await wait_until(lambda: "Xdelta epsilon" in snapshot()[20])
            assert ui.input.buffer.text == "alpha beta gamma Xdelta epsilon"
            size[0] = Size(rows=24, columns=40)
            screen.resize(lines=24, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: "gamma Xdelta epsilon" in snapshot()[19])
            assert snapshot()[20].strip() == ""
            pipe.send_text("\r")
            assert await ui.read_line() == "alpha beta gamma Xdelta epsilon"
            assert ui.input.buffer.history.get_strings()[-1] == "alpha beta gamma Xdelta epsilon"
        finally:
            ui.close()
            await task


def test_startup_and_help_list_all_commands_with_requested_spacing():
    banner = BANNER.format(model="test/model", workspace="project", tools="read_file", mcp="")
    for command in [
        "/help", "/tools", "/model", "/models", "/key", "/cost", "/mcp",
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
                start = next(i for i, row in enumerate(rows) if row.startswith("  tools:"))
                end = next(i for i in range(start + 1, len(rows)) if not rows[i].strip())
                tool_rows = rows[start:end]
                assert len(tool_rows) > 1
                assert all(width(row) <= columns for row in tool_rows)
                if columns > 13:
                    assert all(row.startswith(" " * 13) for row in tool_rows[1:])
                assert "".join(row.strip().replace(" ", "") for row in tool_rows) == "tools:" + ",".join(tools)
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
            await wait_until(lambda: ">" in snapshot()[19])
            pipe.send_text(text)
            await wait_until(lambda: second in snapshot()[20])
            assert snapshot()[19].rstrip() == first
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
        agent=SimpleNamespace(run=run, pending=[], stopped=False,
                              usage=Usage(prompt_tokens=10, completion_tokens=5, cost=0.0)),
    )
    monkeypatch.setattr(cli, "_read_line", read)
    assert await cli._run_turn(session, "task", renderer.style) is False
    assert "Finished.\n    (in=10 out=5 cost=$0.000000)" in sink.getvalue()


@pytest.mark.parametrize("color", [True, False])
async def test_context_toggle_shows_ordered_system_prompts_and_preserves_draft(display, color):
    sink, _, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output, color=color)
        renderer = Renderer(Style(color), sink, False)
        renderer.terminal = ui
        task = asyncio.create_task(ui.run())
        try:
            renderer.emit("Ordinary transcript.")
            await wait_until(lambda: "Ordinary transcript." in "\n".join(snapshot()[:17]))
            pipe.send_text("unfinished draft")
            await wait_until(lambda: "unfinished draft" in snapshot()[19])
            before = snapshot()[:17]
            request = json.dumps({"model": "test/model", "messages": [
                {"role": "system", "content": "SYSTEM ONE"},
                {"role": "system", "content": "SYSTEM TWO"},
                {"role": "user", "content": "USER REQUEST"},
                {"role": "assistant", "content": "PREVIOUS ANSWER"},
            ]})
            renderer.handle(AgentEvent(kind="context", text=request))
            assert snapshot()[:17] == before
            pipe.send_text("\x1c")
            await wait_until(lambda: "SYSTEM TWO" in "\n".join(snapshot()[:17]))
            visible = "\n".join(snapshot()[:17])
            assert visible.index("SYSTEM ONE") < visible.index("SYSTEM TWO") < visible.index("USER REQUEST") < visible.index("PREVIOUS ANSWER")
            for text in ["SYSTEM ONE", "SYSTEM TWO"]:
                rows = snapshot()
                row = next(index for index, value in enumerate(rows) if text in value)
                assert screen.buffer[row][rows[row].index(text)].fg == ("ffff00" if color else "default")
            assert "unfinished draft" in snapshot()[19]
            assert "readout" in snapshot()[23]
            ui.refresh()
            assert "SYSTEM ONE" in "\n".join(snapshot()[:17])
            pipe.send_text("\x1c")
            await wait_until(lambda: snapshot()[:17] == before)
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
            await wait_until(lambda: "transcript 49" in "\n".join(snapshot()[:17]))
            pipe.send_text("\x1b[<64;1;5M" * 3)
            await wait_until(lambda: "transcript 49" not in "\n".join(snapshot()[:17]))
            before = snapshot()[:17]
            ui.set_context(json.dumps({"messages": [{"role": "system", "content": "\n".join(f"instruction {n}" for n in range(100))}]}))
            pipe.send_text("\x1c")
            await wait_until(lambda: "instruction 0" in "\n".join(snapshot()[:17]))
            pipe.send_text("\x1b[<65;1;5M" * 3)
            await wait_until(lambda: "instruction 0" not in "\n".join(snapshot()[:17]))
            pipe.send_text("\x1b[1~")
            await wait_until(lambda: "instruction 0" in "\n".join(snapshot()[:17]))
            pipe.send_text("\x1b[6~")
            await wait_until(lambda: "instruction 0" not in "\n".join(snapshot()[:17]))
            pipe.send_text("\x1b[4~")
            await wait_until(lambda: "instruction 99" in "\n".join(snapshot()[:17]))
            ui.set_context(json.dumps({"messages": [{"role": "system", "content": "UPDATED REQUEST"}], "tools": [{"name": "example"}], "response_format": {"type": "json_object"}}))
            await wait_until(lambda: "UPDATED REQUEST" in "\n".join(snapshot()[:17]))
            assert "example" in "\n".join(snapshot()[:17])
            assert "json_object" in "\n".join(snapshot()[:17])
            pipe.send_text("\x1c")
            await wait_until(lambda: snapshot()[:17] == before)
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
            await wait_until(lambda: "key:" in snapshot()[19])
            pipe.send_text("secret-value")
            await wait_until(lambda: "************" in snapshot()[19])
            assert "secret-value" not in "\n".join(snapshot())
            pipe.send_text("\r")
            assert await ask == "secret-value"
            await wait_until(lambda: ">" in snapshot()[19])
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
            await wait_until(lambda: ">" in snapshot()[19])
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


@pytest.mark.parametrize("mouse_row", [5, 18, 20, 24])
async def test_mouse_wheel_scrolls_transcript_without_changing_input(display, mouse_row) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[19])
            pipe.send_text("old prompt\r")
            assert await ui.read_line() == "old prompt"
            history = ui.input.buffer.history.get_strings()
            pipe.send_text("draft")
            await wait_until(lambda: "draft" in snapshot()[19])
            for number in range(50):
                ui.write(f"output {number}")
            await wait_until(lambda: "output 49" in "\n".join(snapshot()[:17]))
            before = snapshot()[:17]
            footer = snapshot()[17:]
            pipe.send_text(f"\x1b[<64;1;{mouse_row}M" * 3)
            await wait_until(lambda: snapshot()[:17] != before)
            scrolled = snapshot()[:17]
            assert "output 49" not in "\n".join(scrolled)
            assert snapshot()[17:] == footer
            ui.write("new output")
            await asyncio.sleep(.2)
            assert snapshot()[:17] == scrolled
            pipe.send_text(f"\x1b[<65;1;{mouse_row}M" * 30)
            await wait_until(lambda: "new output" in "\n".join(snapshot()[:17]))
            ui.write("following output")
            await wait_until(lambda: "following output" in "\n".join(snapshot()[:17]))
            assert "draft" in snapshot()[19]
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
            assert ">" in snapshot()[11]
        finally:
            ui.close()
            await task


async def test_wheel_events_without_coordinates_preserve_prompt_history(display) -> None:
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: ">" in snapshot()[19])
            pipe.send_text("previous prompt\r")
            assert await ui.read_line() == "previous prompt"
            for number in range(50):
                ui.write(f"output {number}")
            await wait_until(lambda: "output 49" in "\n".join(snapshot()[:17]))
            before = snapshot()[:17]
            ui.app.key_processor.feed(KeyPress(Keys.ScrollUp))
            ui.app.key_processor.process_keys()
            await wait_until(lambda: snapshot()[:17] != before)
            assert ui.input.text == ""
            ui.app.key_processor.feed(KeyPress(Keys.ScrollDown))
            ui.app.key_processor.process_keys()
            await wait_until(lambda: snapshot()[:17] == before)
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
            await wait_until(lambda: "Count" in "\n".join(snapshot()[:17]))
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
            assert ">" in snapshot()[19]
            assert snapshot()[22].strip() == ""
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
            await wait_until(lambda: ">" in snapshot()[19])
            pipe.send_text("draft")
            await wait_until(lambda: "draft" in snapshot()[19])
            renderer.handle(AgentEvent(kind="assistant_delta", text="alpha be"))
            await wait_until(lambda: "alpha be" in snapshot()[1])
            renderer.handle(AgentEvent(kind="assistant_delta", text="ta gamma delta epsilon"))
            await wait_until(lambda: "delta epsilon" in snapshot()[2])
            assert snapshot()[1].rstrip() == "alpha beta gamma"
            assert screen.buffer[1][0].bold
            size[0] = Size(rows=24, columns=40)
            screen.resize(lines=24, columns=40)
            ui.app.invalidate()
            await wait_until(lambda: snapshot()[1].rstrip() == "alpha beta gamma delta epsilon")
            renderer.handle(AgentEvent(kind="stream_end"))
            renderer.handle(AgentEvent(kind="retry", text="Response rejected. Retrying from the start."))
            renderer.handle(AgentEvent(kind="reasoning_delta", text="Starting again."))
            await wait_until(lambda: "Thinking: Starting again." in "\n".join(snapshot()[:17]))
            rows = snapshot()
            row = next(index for index, value in enumerate(rows) if "Response rejected" in value)
            assert screen.buffer[row][2].fg == "ff0000"
            assert "draft" in rows[19]
            assert rows[17].strip() == ""
            assert rows[22].strip() == ""
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
            await wait_until(lambda: "request failed" in "\n".join(snapshot()[:17]))
            pipe.send_text("draft")
            await wait_until(lambda: "draft" in snapshot()[19])
            for text, expected in [
                ("> request", "00ff00"), ("list_dir", "ff00ff"),
                ("tool failed", "ff0000"), ("request failed", "ff0000"), ("draft", "00ff00"),
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
            await wait_until(lambda: "next command response" in "\n".join(snapshot()[:17]))
            assert [line.rstrip() for line in snapshot()[:11]] == [
                "", "command response", "second line", "", "Checking files.",
                "  ⚙ list_dir({})", "    result", "", "Finished.", "", "next command response",
            ]
            assert ">" in snapshot()[19]
        finally:
            ui.close()
            await task
