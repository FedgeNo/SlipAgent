"""Waiting prompts stay above the controls until their request is submitted."""

import asyncio
from importlib import import_module

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.data_structures import Size
from wcwidth import width as display_width

from slipagent.cli import Renderer, Style
from slipagent.events import AgentEvent
from slipagent.terminal import TerminalUI
from slipagent.types import Message

display = import_module("test-terminal").display
wait_until = import_module("test-terminal").wait_until


def feed(ui, columns=100):
    content = ui._transcript_content(columns)
    return "\n".join("".join(part[1] for part in content.get_line(i)) for i in range(content.line_count))


async def test_waiting_prompts_stay_above_footer_then_enter_feed_at_submission(display):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(False), sink, False)
        renderer.terminal = ui
        renderer.user_prompt("Original task")
        renderer.handle(AgentEvent(kind="assistant_delta", text="Working"))
        first = "First correction " + "long text " * 20 + "\nFull second line"
        renderer.user_prompt(first, queued=True)
        renderer.user_prompt("Second correction", queued=True)
        renderer.handle(AgentEvent(kind="assistant_delta", text=" continuously."))
        ui.input.text = "unfinished draft"
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Second correction" in snapshot()[16])
            assert snapshot()[15].rstrip().endswith("…")
            assert "First correction" in snapshot()[15]
            assert "Ready" in snapshot()[17]
            assert "queued" not in "\n".join(snapshot())
            assert "First correction" not in feed(ui)
            assert "Second correction" not in feed(ui)
            assert "Working continuously." in feed(ui)
            renderer.handle(AgentEvent(kind="user_message_sent", text=first))
            renderer.handle(AgentEvent(kind="assistant_text", text="Processing the correction."))
            await wait_until(lambda: "First correction" not in snapshot()[15])
            text = feed(ui)
            assert text.index("Working continuously.") < text.index("First correction")
            assert text.index("Full second line") < text.index("Processing the correction.")
            assert "Second correction" not in text
            assert "Second correction" in snapshot()[16]
            assert ui.input.text == "unfinished draft"
        finally:
            ui.close()
            await task


@pytest.mark.parametrize("columns", [1, 2, 8, 18, 100])
def test_pending_preview_is_one_line_with_grapheme_safe_ellipsis(display, columns):
    sink, _, output, _, _ = display
    ui = TerminalUI(lambda width: "readout", sink, output=output)
    renderer = Renderer(Style(False), sink, False)
    renderer.terminal = ui
    prompt = "A 👩‍💻 é 界 " * 20 + "\nnext line\x1b[2J"
    try:
        renderer.user_prompt(prompt, queued=True)
        content = ui._queued_prompt_content(columns)
        assert content.line_count == 1
        text = "".join(part[1] for part in content.get_line(0))
        assert display_width(text) <= columns
        assert text.endswith("…")
        assert "\n" not in text and "\x1b" not in text
        assert prompt not in feed(ui)
        renderer.handle(AgentEvent(kind="user_message_sent", text=prompt))
        assert "next line" in feed(ui)
    finally:
        ui.close()


async def test_restore_and_reset_keep_unsent_duplicates_out_of_feed(display):
    sink, _, output, _, _ = display
    ui = TerminalUI(lambda width: "readout", sink, output=output)
    renderer = Renderer(Style(False), sink, False)
    renderer.terminal = ui
    try:
        await renderer.restore_transcript([Message.user("Original"), Message.assistant("Previous work"),
                                           Message.user("same")], ["same", "later"], queued_messages={2: "same"})
        assert "same" not in feed(ui) and "later" not in feed(ui)
        renderer.handle(AgentEvent(kind="user_message_sent", text="same"))
        assert feed(ui).count("> same") == 1
        renderer.handle(AgentEvent(kind="user_message_sent", text="same"))
        assert feed(ui).count("> same") == 2
        renderer.handle(AgentEvent(kind="user_queue_reset"))
        assert ui._queued_prompt_content(100).line_count == 0
        renderer.handle(AgentEvent(kind="user_message_sent", text="later"))
        assert "later" not in feed(ui)
    finally:
        ui.close()


def test_reload_moves_legacy_prompt_and_notice_into_pending_area(display):
    sink, _, output, _, _ = display
    ui = TerminalUI(lambda width: "readout", sink, output=output)
    try:
        ui.write("Previous work\n")
        ui.write("> Legacy correction\n", user_prompt=True)
        index = len(ui._raw_output)
        ui.write_chunk("  (queued — the agent is still working)\n")
        ui.write("More work\n")
        ui.__dict__.pop("_queued_prompts")
        ui.__dict__["_queued_notices"] = {index: "Legacy correction"}
        ui.input.text = "draft"
        ui.refresh()
        text = feed(ui)
        assert "Previous work" in text and "More work" in text
        assert "Legacy correction" not in text and "queued" not in text
        content = ui._queued_prompt_content(100)
        assert "Legacy correction" in "".join(part[1] for part in content.get_line(0))
        assert ui.input.text == "draft"
    finally:
        ui.close()


def test_failed_refresh_preserves_legacy_queue_and_transcript(display, monkeypatch):
    sink, _, output, _, _ = display
    ui = TerminalUI(lambda width: "readout", sink, output=output)
    try:
        ui.write("> Legacy correction\n", user_prompt=True)
        index = len(ui._raw_output)
        ui.write_chunk("  (queued — the agent is still working)\n")
        before = list(ui._raw_output)
        ui.__dict__.pop("_queued_prompts")
        ui.__dict__["_queued_notices"] = {index: "Legacy correction"}
        def fail():
            raise ValueError("Invalid presentation")
        with monkeypatch.context() as patch:
            patch.setattr(ui, "_style", fail)
            with pytest.raises(ValueError, match="Invalid presentation"):
                ui.refresh()
        assert list(ui._raw_output) == before
        assert ui.__dict__["_queued_notices"] == {index: "Legacy correction"}
        ui.refresh()
        assert "Legacy correction" not in feed(ui)
    finally:
        ui.close()


async def test_many_pending_prompts_leave_feed_and_controls_visible_after_resize(display):
    sink, size, output, screen, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        renderer = Renderer(Style(False), sink, False)
        renderer.terminal = ui
        renderer.emit("Previous output")
        for index in range(20):
            renderer.user_prompt(f"Correction {index:02}", queued=True)
        size[0] = Size(rows=8, columns=40)
        screen.resize(8, 40)
        task = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: "Correction 19" in snapshot()[2])
            assert "Correction 18" in snapshot()[1]
            assert "Ready" in snapshot()[3]
            assert "readout" in snapshot()[7]
            assert ui.transcript.render_info.window_height == 1
            renderer.handle(AgentEvent(kind="user_message_sent", text="Correction 00"))
            assert "> Correction 00" in feed(ui)
            assert "Correction 01" not in feed(ui)
            size[0] = Size(rows=24, columns=100)
            screen.resize(24, 100)
            ui.refresh()
            await wait_until(lambda: "Correction 19" in snapshot()[16])
            assert "Ready" in snapshot()[17]
        finally:
            ui.close()
            await task
