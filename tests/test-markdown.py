"""Markdown presentation preserves source and copy text independently."""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from slipagent.markdown import MarkdownRenderer, THEMES, code_blocks
from slipagent.terminal import TerminalUI
from slipagent.transcript import WrappedTranscript, compact_fragments
from slipagent import cli


def visible(rows):
    return ["".join(fragment[1] for fragment in row.fragments) for row in rows]


def test_reasoning_markdown_stream_is_grey_and_inline_code_has_neutral_background():
    from slipagent.agent import AgentEvent
    from slipagent.palette import THOUGHT_COLOR, THOUGHT_CODE_BACKGROUND_COLOR

    with create_pipe_input() as terminal_input:
        ui = TerminalUI(lambda width: "", io.StringIO(), input=terminal_input, output=DummyOutput())
        renderer = cli.Renderer(cli.Style(True), io.StringIO(), False)
        renderer.terminal = ui
        try:
            for chunk in ["# Plan\n\nUse **bold** and `val", "ues`.\n\n```python\nreturn 42\n```"]:
                renderer.handle(AgentEvent(kind="reasoning_delta", text=chunk))
                ui._transcript_content(80)
            renderer._finish_stream()
            for width in (80, 30):
                ui._transcript_content(width)
                rows = [ui._flow.copy_row(i) for i in range(len(ui._flow.rows))]
                text = "\n".join(visible(rows))
                assert "Plan" in text and "# Plan" not in text
                assert "**bold**" not in text and "`values`" not in text
                fragments = [part for row in rows for part in row.fragments if part[1].strip()]
                assert all(THOUGHT_COLOR in style for style, *_ in fragments)
                assert any("bold" in style for style, *_ in fragments)
                assert any("bg:" + THOUGHT_CODE_BACKGROUND_COLOR in style for style, *_ in fragments)
            renderer.handle(AgentEvent(kind="assistant_delta", text="Answer"))
            renderer._finish_stream()
            ui._transcript_content(80)
            assert THEMES["dark"]["body"] in ui._flow.copy_row(len(ui._flow.rows) - 1).fragments[0][0]
        finally:
            ui._close_transcript()


def test_headings_share_style_and_body_bold_share_color_without_added_spacing():
    renderer = MarkdownRenderer()
    rows = renderer.render("# First\n## Second\nplain **bold**", 80)
    assert visible(rows) == ["First", "Second", "plain bold"]
    assert rows[0].fragments[0][0] == rows[1].fragments[0][0] == "bold #ffffff"
    assert rows[2].fragments[0][0] == "#ffffff"
    assert rows[2].fragments[-1][0] == "bold #ffffff"


@pytest.mark.parametrize("theme", THEMES)
def test_syntax_roles_and_background_follow_theme(theme):
    renderer = MarkdownRenderer(theme)
    source = '```python\nreturn "hi" + 42\n```'
    rows = renderer.render(source, 80)
    assert visible(rows)[0].strip() == 'return "hi" + 42'
    assert "python" not in "".join(visible(rows))
    if theme == "monochrome":
        assert all("#" not in style and "bg:" not in style for style, _ in rows[0].fragments)
    else:
        assert any(THEMES[theme]["keyword"] in style for style, _ in rows[0].fragments)
        assert all("bg:" + THEMES[theme]["code-bg"] in style for style, _ in rows[0].fragments)


def test_links_tables_and_literal_controls():
    renderer = MarkdownRenderer(color=False)
    assert renderer.plain("[site](https://example.com)") == "site (https://example.com)"
    table = "| Name | Value |\n| --- | --- |\n| alpha | beta |"
    assert visible(renderer.render(table, 10)) == ["Name: ", "alpha", "Value: ", "beta"]
    assert "\\x1b" in renderer.plain("\x1b[31mexample")


def test_wrapped_code_copy_excludes_markers_and_padding_preserves_tabs():
    source = "```unknown-language\n\talpha = '界界界'\n```"
    renderer = MarkdownRenderer()
    rows = renderer.render(source, 12)
    assert any("↪ " in line for line in visible(rows))
    assert "".join(row.selected(0) for row in rows) == "\talpha = '界界界'\n"
    assert code_blocks(source) == ["\talpha = '界界界'\n"]


@pytest.mark.parametrize("text, columns, expected", [
    ("12345 next", 5, ["12345", "next"]),
    ("12345 1234 next", 10, ["12345 1234", "next"]),
    ("界界界 next", 6, ["界界界", "next"]),
])
def test_full_width_words_hide_wrap_spaces_but_copy_original_prose(text, columns, expected):
    rows = MarkdownRenderer().rows([("", text)], columns)
    assert visible(rows) == expected
    assert "".join(row.selected(0) for row in rows) == text
    assert rows[-1].selected(0) == "next"
    assert rows[0].selected(0, columns) == expected[0]


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_streamed_final_rows_match_complete_with_reference_links(chunk_size):
    source = '# Title\n\nA **bold** [reference][r].\n\n```python\nx = "界"\n```\n\n[r]: https://example.com\n'
    flow = WrappedTranscript(20)
    try:
        for start in range(0, len(source), chunk_size):
            flow.append_markdown(source[start:start + chunk_size], first=start == 0)
            flow.content_rows()
        flow.finish_markdown()
        expected = MarkdownRenderer().render(source, 20)
        assert [flow.get_row(i) for i in range(len(flow.rows))] == [compact_fragments(row.fragments) for row in expected]
        assert "".join(flow.copy_row(i).selected(0) for i in range(len(flow.rows))) == "".join(row.copy_text for row in expected)
    finally:
        flow.close()


def test_terminal_source_toggle_and_copy_modes(monkeypatch):
    with create_pipe_input() as terminal_input:
        ui = TerminalUI(lambda width: "", io.StringIO(), input=terminal_input, output=DummyOutput())
        copied = []
        monkeypatch.setattr(ui, "_copy_text", copied.append)
        try:
            source = "Explanation\n\n```python\nx = 1\n```"
            ui.write_markdown(source)
            ui._transcript_content(20)
            assert ui.copy_response() == "x = 1\n"
            assert ui.copy_response("text") == "Explanation\n\nx = 1"
            ui.configure_markdown(source=True, theme="ironbow")
            ui._transcript_content(20)
            assert ui.copy_response() == source
            assert copied[-1] == source
        finally:
            ui._close_transcript()


@pytest.mark.parametrize("source_output", [False, True])
async def test_redirected_answer_plain_by_default_or_markdown_on_request(monkeypatch, source_output):
    source = "**Result**\n\n```python\nx = 1\n```"
    renderer = cli.Renderer(cli.Style(False), io.StringIO(), False)
    renderer.output_markdown = source_output
    sink = io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", sink)
    monkeypatch.setattr(cli, "_shutdown", AsyncMock())
    agent = SimpleNamespace(run=AsyncMock(return_value=source), wait_for_compaction=AsyncMock(), stopped=False)
    session = SimpleNamespace(agent=agent, renderer=renderer)
    assert await cli.run_one_shot(session, "show code") == 0
    assert sink.getvalue() == (source + "\n" if source_output else "Result\n\nx = 1\n")


@pytest.mark.parametrize("release_button", [MouseButton.LEFT, MouseButton.UNKNOWN])
def test_mouse_release_stops_dragging_and_retains_selection_without_copying(monkeypatch, release_button):
    with create_pipe_input() as terminal_input:
        ui = TerminalUI(lambda width: "", io.StringIO(), input=terminal_input, output=DummyOutput())
        copied = []
        monkeypatch.setattr(ui, "_copy_text", copied.append)
        def mouse(kind, button, x):
            return ui._select_transcript(MouseEvent(Point(x=x, y=0), kind, button, frozenset()))
        try:
            ui.write_markdown("abcdefgh")
            ui._transcript_content(20)
            mouse(MouseEventType.MOUSE_DOWN, MouseButton.LEFT, 1)
            mouse(MouseEventType.MOUSE_MOVE, MouseButton.LEFT, 3)
            mouse(MouseEventType.MOUSE_UP, release_button, 4)
            assert copied == []
            mouse(MouseEventType.MOUSE_MOVE, MouseButton.NONE, 7)
            mouse(MouseEventType.MOUSE_MOVE, MouseButton.LEFT, 7)
            mouse(MouseEventType.MOUSE_UP, release_button, 7)
            assert copied == []
            assert ui.selected_text() == "bcde"
            mouse(MouseEventType.MOUSE_DOWN, MouseButton.RIGHT, 6)
            assert ui.selected_text() == "bcde"
            mouse(MouseEventType.MOUSE_MOVE, MouseButton.LEFT, 7)
            assert ui.selected_text() == "bcde"
        finally:
            ui._close_transcript()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("source, columns, first, last, expected", [
    ("first\nsecond\n", 20, Point(0, 0), Point(5, 1), "first\nsecond"),
    ("first\nsecond\n", 20, Point(1, 0), Point(3, 1), "irst\nseco"),
    ("abcdef\n", 20, Point(1, 0), Point(4, 0), "bcde"),
    ("abcdefghij\n", 5, Point(0, 0), Point(4, 1), "abcdefghij"),
    ("first\nA🌕\n", 20, Point(0, 0), Point(1, 1), "first\nA🌕"),
    ("first\nAe\u0301\n", 20, Point(0, 0), Point(1, 1), "first\nAe\u0301"),
])
def test_drag_selection_includes_last_cell_in_copied_and_highlighted_text(source, columns, first, last, expected, reverse):
    with create_pipe_input() as terminal_input:
        ui = TerminalUI(lambda width: "", io.StringIO(), input=terminal_input, output=DummyOutput())
        try:
            ui.write(source)
            ui._transcript_content(columns)
            start, end = (last, first) if reverse else (first, last)
            for kind, point in [(MouseEventType.MOUSE_DOWN, start),
                                (MouseEventType.MOUSE_MOVE, end), (MouseEventType.MOUSE_UP, end)]:
                ui._select_transcript(MouseEvent(point, kind, MouseButton.LEFT, frozenset()))
            assert ui.selected_text() == expected
            content = ui._transcript_content(columns)
            highlighted = "".join(text for row in range(content.line_count)
                                  for style, text, *_ in content.get_line(row) if "reverse" in style.split())
            assert highlighted == expected.replace("\n", "")
        finally:
            ui._close_transcript()


def test_plain_text_click_does_not_select_a_character():
    with create_pipe_input() as terminal_input:
        ui = TerminalUI(lambda width: "", io.StringIO(), input=terminal_input, output=DummyOutput())
        try:
            ui.write("abcdef\n")
            ui._transcript_content(20)
            for kind in (MouseEventType.MOUSE_DOWN, MouseEventType.MOUSE_UP):
                ui._select_transcript(MouseEvent(Point(2, 0), kind, MouseButton.LEFT, frozenset()))
            assert ui.selected_text() == ""
            assert not any("reverse" in style.split() for style, *_ in ui._transcript_content(20).get_line(0))
        finally:
            ui._close_transcript()


@pytest.mark.parametrize("end", ["outside", "no-button", "right-button"])
def test_mouse_drag_cancels_when_release_is_outside_or_button_is_lost(monkeypatch, end):
    with create_pipe_input() as terminal_input:
        ui = TerminalUI(lambda width: "", io.StringIO(), input=terminal_input, output=DummyOutput())
        copied = []
        monkeypatch.setattr(ui, "_copy_text", copied.append)
        def event(kind, button, x=1):
            return MouseEvent(Point(x=x, y=0), kind, button, frozenset())
        try:
            ui.write_markdown("abcdefgh")
            ui._transcript_content(20)
            ui._select_transcript(event(MouseEventType.MOUSE_DOWN, MouseButton.LEFT))
            if end == "outside":
                ui._route_mouse_wheel(ui.app)
                ui.app.renderer.mouse_handlers.mouse_handlers[0][0](event(MouseEventType.MOUSE_UP, MouseButton.LEFT))
            elif end == "no-button":
                ui._select_transcript(event(MouseEventType.MOUSE_MOVE, MouseButton.NONE, 3))
            else:
                ui._select_transcript(event(MouseEventType.MOUSE_UP, MouseButton.RIGHT, 3))
            assert not ui._selection_dragging
            ui._select_transcript(event(MouseEventType.MOUSE_MOVE, MouseButton.LEFT, 6))
            ui._select_transcript(event(MouseEventType.MOUSE_UP, MouseButton.LEFT, 6))
            assert copied == []
        finally:
            ui._close_transcript()
