"""Command definitions preserve argument-dependent routing and input syntax."""

import pytest

from slipagent.cli import _background_command, _changes_session, _command_help, command_menu_options


@pytest.mark.parametrize("command,argument,exclusive,background", [
    ("key", "status", False, True),
    ("key", "show", False, True),
    ("key", "replacement-key", True, False),
    ("key", "", True, False),
    ("temperature", "", False, True),
    ("temperature", "0.5", True, False),
    ("mcp", "", False, False),
    ("mcp", "remove test", True, False),
    ("danger", "status", False, False),
    ("danger", "off", True, False),
    ("task", "", False, False),
    ("task", "new", True, False),
    ("planning", "on", True, False),
    ("planning", "off", True, False),
    ("model", "", True, False),
    ("exit", "", False, False),
])
def test_command_routing_keeps_reads_available_and_changes_exclusive(command, argument, exclusive, background):
    assert _changes_session(command, argument) is exclusive
    assert _background_command(f"/{command} {argument}".rstrip()) is background


def test_ordinary_text_is_never_a_background_command():
    assert not _background_command("key status")
    assert not _background_command("temperature")


def test_menu_inserts_only_the_command_prefix_and_help_keeps_all_variants():
    choices = dict(command_menu_options())
    assert "/rename " in choices
    assert "<name>" in choices["/rename "]
    assert "/mcp add " in choices
    assert "<command>" in choices["/mcp add "]
    assert "/danger off" in choices
    assert _command_help("mcp") == _command_help("/mcp")
    assert all(part in _command_help("mcp") for part in ("/mcp add", "/mcp save", "/mcp remove"))
    assert "/quit" in _command_help("quit")
    assert "/exit" in _command_help("exit")
