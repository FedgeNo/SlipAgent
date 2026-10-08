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


def test_headings_share_style_and_body_bold_share_color_without_added_spacing():
    renderer = MarkdownRenderer()
    rows = renderer.render("# First\n## Second\nplain **bold**", 80)
    assert visible(rows) == ["First", "Second", "plain bold"]
    assert rows[0].fragments[0][0] == rows[1].fragments[0][0] == "bold #ffffff"
    assert rows[2].fragments[0][0] == "#dddddd"
    assert rows[2].fragments[-1][0] == "bold #dddddd"


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
    agent = SimpleNamespace(run=AsyncMock(return_value=source), wait_for_compaction=AsyncMock())
    session = SimpleNamespace(agent=agent, renderer=renderer)
    assert await cli.run_one_shot(session, "show code") == 0
    assert sink.getvalue() == (source + "\n" if source_output else "Result\n\nx = 1\n")


@pytest.mark.parametrize("release_button", [MouseButton.LEFT, MouseButton.UNKNOWN])
def test_mouse_release_stops_selection_and_copies_once(monkeypatch, release_button):
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
            assert copied == ["bcd"]
            mouse(MouseEventType.MOUSE_MOVE, MouseButton.NONE, 7)
            mouse(MouseEventType.MOUSE_MOVE, MouseButton.LEFT, 7)
            mouse(MouseEventType.MOUSE_UP, release_button, 7)
            assert copied == ["bcd"]
            assert ui.selected_text() == "bcd"
            mouse(MouseEventType.MOUSE_DOWN, MouseButton.RIGHT, 6)
            assert ui.selected_text() == ""
            mouse(MouseEventType.MOUSE_MOVE, MouseButton.LEFT, 7)
            assert ui.selected_text() == ""
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
