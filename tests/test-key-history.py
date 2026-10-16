"""Inline API keys reach command dispatch without entering input history."""

import io

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from slipagent.terminal import TerminalUI


@pytest.fixture
def terminal():
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", io.StringIO(), input=pipe, output=DummyOutput())
        try:
            yield ui
        finally:
            ui.close()


@pytest.mark.parametrize("command, retained", [
    ("/key FAKE-SECRET", False),
    ("/key google FAKE-SECRET", False),
    ("/key kilo FAKE-SECRET", False),
    ("  /KEY\tFAKE-SECRET  ", False),
    ("/key show extra", False),
    ("/key", True),
    ("/key show", True),
    ("/key STATUS  ", True),
    ("/help", True),
    ("Explain the /key command", True),
])
async def test_submitted_key_skips_history_but_reaches_dispatch(terminal, command, retained):
    buffer = terminal.input.buffer
    buffer.history.append_string("previous task")
    buffer.text = command
    buffer.validate_and_handle()
    assert await terminal.read_line() == command
    assert buffer.text == ""
    assert buffer.history.get_strings() == ["previous task"] + ([command] if retained else [])
    buffer.text = "next task"
    buffer.validate_and_handle()
    assert await terminal.read_line() == "next task"
    assert buffer.history.get_strings()[-1] == "next task"


@pytest.mark.parametrize("draft, retained", [("/key FAKE-SECRET", False), ("unfinished draft", True)])
def test_command_menu_does_not_save_a_key_draft(terminal, draft, retained):
    buffer = terminal.input.buffer
    buffer.history.append_string("previous task")
    buffer.text = draft
    terminal.insert_command("/help")
    assert buffer.text == "/help"
    assert buffer.history.get_strings() == ["previous task"] + ([draft] if retained else [])
    assert terminal._lines.empty()
