"""An interactive launch without any API key asks for one instead of exiting."""

import os

import pytest

from slipagent import cli
from slipagent.config import dotenv_path


@pytest.fixture
def no_keys(monkeypatch):
    for name in tuple(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name)


def answers(monkeypatch, secret, *replies):
    remaining = list(replies)
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: secret)
    monkeypatch.setattr("builtins.input", lambda prompt: remaining.pop(0))


@pytest.mark.parametrize("save", ["", "n"])
def test_detected_key_is_installed_and_optionally_saved(no_keys, monkeypatch, save):
    answers(monkeypatch, "nvapi-test-key", save)
    assert cli._first_run_key()
    assert os.environ["NVIDIA_API_KEY"] == "nvapi-test-key"
    saved = dotenv_path().read_text() if dotenv_path().exists() else ""
    assert ("NVIDIA_API_KEY=nvapi-test-key" in saved) is (save == "")


def test_unrecognized_key_asks_for_its_provider(no_keys, monkeypatch, capsys):
    answers(monkeypatch, "plain-key", "unknown", "kilo", "n")
    assert cli._first_run_key()
    assert os.environ["KILO_API_KEY"] == "plain-key"
    assert "https://openrouter.ai/keys" in capsys.readouterr().err


def test_blank_key_exits(no_keys, monkeypatch):
    answers(monkeypatch, "")
    assert not cli._first_run_key()
    assert "NVIDIA_API_KEY" not in os.environ


async def test_interactive_dispatch_without_keys_runs_setup_first(no_keys, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "_first_run_key", lambda: calls.append(True) or False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("TERM", "xterm")
    args = cli.build_parser().parse_args(["--workspace", str(tmp_path), "--no-mcp", "--no-session"])
    assert await cli._dispatch(args, None) == 2
    assert calls == [True]


def test_tests_never_resolve_an_installation_dotenv():
    from slipagent.config import data_directory

    # Saving a key in a test must not reach a real checkout or installation file.
    assert dotenv_path().parent == data_directory()
