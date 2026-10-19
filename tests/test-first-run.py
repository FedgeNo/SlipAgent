"""An interactive launch without any API key asks for one instead of exiting."""

import argparse
import os

import pytest

from slipagent import cli
from slipagent.config import command_environment, dotenv_path


@pytest.fixture
def no_keys(monkeypatch):
    from slipagent import config
    for name in tuple(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "_SESSION_NAMES", set())


def answers(monkeypatch, secret, *replies):
    remaining = list(replies)
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: secret)
    monkeypatch.setattr("builtins.input", lambda prompt: remaining.pop(0))


@pytest.mark.parametrize("save", ["", "n"])
def test_detected_key_is_installed_and_optionally_saved(no_keys, monkeypatch, save):
    answers(monkeypatch, "nvapi-test-key", save)
    assert cli.Launcher(argparse.Namespace()).first_run_key()
    assert os.environ["NVIDIA_API_KEY"] == "nvapi-test-key"
    # A pasted key is not part of the user's shell environment.
    assert "NVIDIA_API_KEY" not in command_environment()
    saved = dotenv_path().read_text() if dotenv_path().exists() else ""
    assert ("NVIDIA_API_KEY=nvapi-test-key" in saved) is (save == "")


def test_unrecognized_key_asks_for_its_provider(no_keys, monkeypatch, capsys):
    answers(monkeypatch, "plain-key", "unknown", "kilo", "n")
    assert cli.Launcher(argparse.Namespace()).first_run_key()
    assert os.environ["KILO_API_KEY"] == "plain-key"
    assert "https://openrouter.ai/keys" in capsys.readouterr().err


def test_blank_key_exits(no_keys, monkeypatch):
    answers(monkeypatch, "")
    assert not cli.Launcher(argparse.Namespace()).first_run_key()
    assert "NVIDIA_API_KEY" not in os.environ


async def test_interactive_dispatch_without_keys_runs_setup_first(no_keys, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli.Launcher, "first_run_key", lambda self: calls.append(True) or False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("TERM", "xterm")
    args = cli.build_parser().parse_args(["--workspace", str(tmp_path), "--no-mcp", "--no-session"])
    assert await cli.Launcher(args).dispatch(None) == 2
    assert calls == [True]


def test_tests_never_resolve_an_installation_dotenv():
    from slipagent.config import data_directory

    # Saving a key in a test must not reach a real checkout or installation file.
    assert dotenv_path().parent == data_directory()


async def test_interactive_dispatch_reads_saved_keys_before_asking(no_keys, monkeypatch, tmp_path):
    from slipagent.config import ConfigError, data_directory

    dotenv = data_directory() / ".env"
    dotenv.parent.mkdir(parents=True, exist_ok=True)
    dotenv.write_text("KILO_API_KEY=saved-key\n")
    monkeypatch.delenv("SLIPAGENT_NO_DOTENV")
    # Registered with monkeypatch so the loaded value is removed afterwards.
    monkeypatch.setenv("KILO_API_KEY", "")
    monkeypatch.delenv("KILO_API_KEY")
    calls = []
    monkeypatch.setattr(cli.Launcher, "first_run_key", lambda self: calls.append(True) or False)

    async def stop(*args, **kwargs):
        raise ConfigError("stop after the key check")
    monkeypatch.setattr(cli.Launcher, "build_session", stop)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("TERM", "xterm")
    args = cli.build_parser().parse_args(["--workspace", str(tmp_path), "--no-mcp", "--no-session"])
    assert await cli.Launcher(args).dispatch(None) == 2
    assert calls == []
    assert os.environ["KILO_API_KEY"] == "saved-key"
