"""Command definitions preserve argument-dependent routing and input syntax."""

import pytest

from slipagent.cli import CommandRunner, command_menu_options
from slipagent.commands import command_help


@pytest.mark.parametrize("command,argument,exclusive,background", [
    ("key", "status", False, True),
    ("key", "show", False, True),
    ("key", "replacement-key", True, False),
    ("key", "", True, False),
    ("temperature", "", False, True),
    ("temperature", "0.5", True, False),
    ("context", "", False, True),
    ("context", "300k", True, False),
    ("context", "reset", True, False),
    ("context", "default", True, False),
    ("mcp", "", False, False),
    ("mcp", "remove test", True, False),
    ("danger", "status", False, False),
    ("danger", "off", True, False),
    ("overthinking", "", True, False),
    ("overthinking", "on", True, False),
    ("overthinking", "off", True, False),
    ("task", "", False, False),
    ("task", "new", True, False),
    ("planning", "on", True, False),
    ("planning", "off", True, False),
    ("model", "", True, False),
    ("exit", "", False, False),
    ("help", "", False, True),
    ("help", "theme", False, True),
    ("theme", "light", False, True),
    ("markdown", "source", False, True),
    ("copy", "", False, True),
])
def test_command_routing_keeps_reads_available_and_changes_exclusive(command, argument, exclusive, background):
    assert CommandRunner(None).changes_session(command, argument) is exclusive
    assert CommandRunner(None).runs_in_background(f"/{command} {argument}".rstrip()) is background


def test_ordinary_text_is_never_a_background_command():
    assert not CommandRunner(None).runs_in_background("key status")
    assert not CommandRunner(None).runs_in_background("temperature")


def test_menu_inserts_only_the_command_prefix_and_help_keeps_all_variants():
    choices = dict(command_menu_options())
    assert "/rename " in choices
    assert "<name>" in choices["/rename "]
    assert "/mcp add " in choices
    assert "<command>" in choices["/mcp add "]
    assert "/danger off" in choices
    assert "/overthinking [on|off]" in choices["/overthinking "]
    assert "/overthinking [on|off]" in command_help("overthinking")
    assert command_help("mcp") == command_help("/mcp")
    assert all(part in command_help("mcp") for part in ("/mcp add", "/mcp save", "/mcp remove"))
    assert "/quit" in command_help("quit")
    assert "/exit" in command_help("exit")
