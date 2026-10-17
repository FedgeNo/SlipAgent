"""Pixel mouse negotiation and parsing stay isolated from keyboard input."""

import io
from types import SimpleNamespace

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.keys import Keys
from prompt_toolkit.output.vt100 import Vt100_Output

from slipagent.pixelmouse import PixelMouse


@pytest.fixture
def pixel_mouse():
    keys = []
    sink = io.StringIO()
    size = [Size(rows=24, columns=100)]
    output = Vt100_Output(sink, lambda: size[0], term="xterm-256color", enable_cpr=False)
    input = SimpleNamespace(vt100_parser=Vt100Parser(keys.append))
    mouse = PixelMouse(input, output, input.vt100_parser)
    try:
        yield mouse, keys, sink, size
    finally:
        mouse.close()


@pytest.mark.parametrize("status", [0, 4])
def test_unsupported_pixel_mode_retains_normal_mouse_input(pixel_mouse, status):
    mouse, keys, sink, _ = pixel_mouse
    mouse.parser.feed(f"\x1b[?1016;{status}$y\x1b[6;20;10t\x1b[<0;75;9M")
    assert "\x1b[?1016h" not in sink.getvalue()
    assert [(key.key, key.data) for key in keys] == [(Keys.Vt100MouseEvent, "\x1b[<0;75;9M")]
    assert "pixel_position" not in keys[0].__dict__


@pytest.mark.parametrize("reply", ["", "\x1b[?1016;2$y", "\x1b[6;20;10t", "\x1b[?1016;2$y\x1b[6;0;0t"])
def test_pixel_mode_requires_support_and_valid_cell_measurements(pixel_mouse, reply):
    mouse, keys, sink, _ = pixel_mouse
    mouse.parser.feed(reply + "draft")
    assert "\x1b[?1016h" not in sink.getvalue()
    assert "".join(key.data for key in keys) == "draft"


def test_fragmented_reports_preserve_keyboard_mouse_and_paste(pixel_mouse):
    mouse, keys, sink, _ = pixel_mouse
    for char in "\x1b[6;20;10t\x1b[?1016;2$y":
        mouse.parser.feed(char)
    assert "\x1b[?1016h\x1b[?1016$p" in sink.getvalue()
    assert keys == []
    # Ignore coordinates until confirmation establishes their units.
    mouse.parser.feed("\x1b[<0;4;5M")
    assert keys == []
    mouse.parser.feed("\x1b[?1016;1$y\x1b[<32;746;12M")
    assert keys[-1].data == "\x1b[<32;75;1M"
    assert keys[-1].__dict__["pixel_position"] == (74.5, .55)
    keys.clear()
    pasted = "literal \x1b[?1016;4$y and \x1b[6;40;20t"
    for char in "draft\x1b[A\x1b[200~" + pasted + "\x1b[201~":
        mouse.parser.feed(char)
    mouse.parser.feed("\x1b")
    mouse.parser.flush()
    assert "".join(key.data for key in keys[:5]) == "draft"
    assert keys[5].key == Keys.Up
    assert keys[6].key == Keys.BracketedPaste and keys[6].data == pasted
    assert keys[7].key == Keys.Escape
    mouse.parser.feed("\x1b[<0;746;12M")
    assert keys[-1].__dict__["pixel_position"] == (74.5, .55)


def test_resize_updates_pixel_conversion_without_changing_queued_events(pixel_mouse):
    mouse, keys, sink, size = pixel_mouse
    mouse.parser.feed("\x1b[?1016;2$y\x1b[6;20;10t\x1b[?1016;1$y\x1b[<0;746;12M")
    queries = sink.getvalue().count("\x1b[16t")
    size[0] = Size(rows=30, columns=120)
    mouse.refresh()
    assert sink.getvalue().count("\x1b[16t") == queries + 1
    mouse.refresh()
    assert sink.getvalue().count("\x1b[16t") == queries + 1
    mouse.parser.feed("\x1b[6;10;5t\x1b[<0;746;12M")
    assert keys[0].__dict__["pixel_position"] == (74.5, .55)
    assert keys[1].__dict__["pixel_position"] == (149.0, 1.1)
    assert keys[1].data == "\x1b[<0;150;2M"


def test_missing_mode_confirmation_restores_cell_input(pixel_mouse, monkeypatch):
    mouse, keys, sink, _ = pixel_mouse
    monkeypatch.setattr("slipagent.pixelmouse.time.monotonic", lambda: 100.0)
    mouse.parser.feed("\x1b[?1016;2$y\x1b[6;20;10t")
    monkeypatch.setattr("slipagent.pixelmouse.time.monotonic", lambda: 103.0)
    mouse.refresh()
    assert sink.getvalue().endswith("\x1b[?1016l\x1b[?1006h")
    # A late activation reply must not restart a failed negotiation.
    mouse.parser.feed("\x1b[?1016;1$y")
    assert sink.getvalue().endswith("\x1b[?1016l\x1b[?1006h")
    mouse.parser.feed("draft\x1b[<0;75;9M")
    assert keys[-1].data == "\x1b[<0;75;9M"
    assert "pixel_position" not in keys[-1].__dict__


def test_close_restores_input_parser_and_disables_pixel_mode(pixel_mouse):
    mouse, keys, sink, _ = pixel_mouse
    mouse.parser.feed("\x1b[?1016;2$y\x1b[6;20;10t\x1b[?1016;1$y")
    mouse.close()
    assert mouse.input.vt100_parser is mouse.original_parser
    assert sink.getvalue().endswith("\x1b[?1016l")
    closed_output = sink.getvalue()
    mouse.parser.feed("\x1b[?1016;2$y\x1b[6;20;10t")
    mouse.close()
    mouse.refresh()
    assert sink.getvalue() == closed_output
    assert keys == []
