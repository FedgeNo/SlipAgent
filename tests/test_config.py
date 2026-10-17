"""Config resolution: `.env` loading, precedence, and persistence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from slipagent.config import (
    DEFAULT_MODEL,
    Config,
    ConfigError,
    data_directory,
    dotenv_path,
    find_dotenv,
    ignored_project_settings,
    load_dotenv,
    save_dotenv_value,
)


def own_dotenv(text: str) -> Path:
    path = data_directory() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_dotenv_loads_simple_pairs() -> None:
    path = own_dotenv(
        "# a comment\n"
        "OPENROUTER_API_KEY=sk-or-v1-abc\n"
        "\n"
        "export OPENROUTER_MODEL=some/model\n"
        'OPENROUTER_TITLE="quoted title"\n',
    )
    env: dict[str, str] = {}

    loaded = load_dotenv(env)

    assert loaded == path
    assert env["OPENROUTER_API_KEY"] == "sk-or-v1-abc"
    assert env["OPENROUTER_MODEL"] == "some/model"
    assert env["OPENROUTER_TITLE"] == "quoted title"


def test_dotenv_does_not_override_existing_values() -> None:
    own_dotenv("OPENROUTER_MODEL=from-file\n")
    env = {"OPENROUTER_MODEL": "from-shell"}

    load_dotenv(env)

    assert env["OPENROUTER_MODEL"] == "from-shell"


def test_dotenv_can_be_disabled() -> None:
    own_dotenv("OPENROUTER_API_KEY=sk-or-v1-abc\n")
    env = {"SLIPAGENT_NO_DOTENV": "1"}

    assert load_dotenv(env) is None
    assert "OPENROUTER_API_KEY" not in env


def test_project_dotenv_is_never_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    own = own_dotenv("OPENROUTER_MODEL=own\n")
    project = tmp_path / "project" / "nested"
    project.mkdir(parents=True)
    (tmp_path / "project" / ".env").write_text(
        "OPENROUTER_API_KEY=project-key\nOPENROUTER_BASE_URL=https://elsewhere.invalid\n"
        "SLIPAGENT_STATE_DIR=/elsewhere\nDATABASE_URL=postgres://app\n", encoding="utf-8")
    monkeypatch.chdir(project)

    env: dict[str, str] = {}
    assert load_dotenv(env) == own
    assert env == {"OPENROUTER_MODEL": "own"}
    assert ignored_project_settings(project) == (
        tmp_path / "project" / ".env", ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "SLIPAGENT_STATE_DIR"])


def test_project_dotenv_without_slipagent_settings_is_not_reported(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("DATABASE_URL=postgres://app\n", encoding="utf-8")
    assert ignored_project_settings(tmp_path) is None


def test_dotenv_falls_back_to_the_install_directory(tmp_path: Path, monkeypatch) -> None:
    """A `.env` beside an older install is used when the data directory has none."""
    from slipagent import config as config_module
    install = tmp_path / "install" / "slipagent"
    install.mkdir(parents=True)
    monkeypatch.setattr(config_module, "__file__", str(install / "config.py"))
    # The fallback is only consulted while dotenv files are enabled.
    monkeypatch.delenv("SLIPAGENT_NO_DOTENV")
    env_file = install.parent / ".env"
    env_file.write_text("OPENROUTER_API_KEY=install-test-key\n")
    environ: dict[str, str] = {}
    assert load_dotenv(environ) == env_file
    assert environ["OPENROUTER_API_KEY"] == "install-test-key"
    # The loaded installation file is SlipAgent's own, not an ignored project file.
    assert ignored_project_settings(install.parent) is None


def test_config_requires_a_key() -> None:
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Config.from_env(environ={})


def test_config_precedence_cli_over_env() -> None:
    config = Config.from_env(
        api_key="sk-or-v1-cli",
        model="cli/model",
        environ={"OPENROUTER_API_KEY": "sk-or-v1-env",
                 "OPENROUTER_MODEL": "env/model"},
    )

    assert config.model == "cli/model"
    assert config.api_key == "sk-or-v1-cli"


def test_config_falls_back_to_env_then_default() -> None:
    assert Config.from_env(environ={"OPENROUTER_API_KEY": "k"}).model == DEFAULT_MODEL
    assert Config.from_env(
        environ={"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "env/model"}
    ).model == "env/model"


def test_default_model_is_the_configured_one() -> None:
    assert DEFAULT_MODEL == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_saved_model_is_reused_and_explicit_choices_take_precedence():
    from slipagent.config import save_model_choice
    save_model_choice("chosen/model")
    assert Config.from_env(environ={"OPENROUTER_API_KEY": "k"}).model == "chosen/model"
    assert Config.from_env(model="flag/model", environ={"OPENROUTER_API_KEY": "k"}).model == "flag/model"
    assert Config.from_env(environ={"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "env/model"}).model == "env/model"


def test_saved_model_precedes_dotenv_default(monkeypatch):
    from slipagent.config import save_model_choice
    save_model_choice("chosen/model")
    monkeypatch.setattr("slipagent.config.load_dotenv", lambda: os.environ.setdefault("OPENROUTER_MODEL", "dotenv/model"))
    assert Config.from_env().model == "chosen/model"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_creates_file_when_missing(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    save_dotenv_value(target, "OPENROUTER_API_KEY", "sk-or-v1-new")

    assert target.read_text().strip() == "OPENROUTER_API_KEY=sk-or-v1-new"


def test_save_replaces_in_place_preserving_comments(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text(
        "# keep me\nOPENROUTER_API_KEY=old\nOTHER=1\n", encoding="utf-8"
    )

    save_dotenv_value(target, "OPENROUTER_API_KEY", "new")

    lines = target.read_text().splitlines()
    assert lines == ["# keep me", "OPENROUTER_API_KEY=new", "OTHER=1"]


def test_save_appends_when_absent(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("# header\n", encoding="utf-8")

    save_dotenv_value(target, "OPENROUTER_API_KEY", "new")

    assert target.read_text().splitlines()[-1] == "OPENROUTER_API_KEY=new"


def test_save_sets_owner_only_permissions(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    save_dotenv_value(target, "OPENROUTER_API_KEY", "secret")

    assert oct(target.stat().st_mode)[-3:] == "600"


def test_find_dotenv_does_not_mutate_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    path = own_dotenv("OPENROUTER_API_KEY=sk-or-v1-abc\n")
    # Start from a known-empty environment so the assertion holds whether or not
    # the surrounding shell exports these variables.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert find_dotenv() == path
    assert os.environ.get("OPENROUTER_API_KEY") != "secret-project-key"


def test_dotenv_path_reports_the_file_that_would_be_used(tmp_path: Path, monkeypatch) -> None:
    from slipagent import config as config_module
    # Isolate from any `.env` above this checkout's real install location.
    monkeypatch.setattr(config_module, "__file__", str(tmp_path / "install" / "slipagent" / "config.py"))
    assert dotenv_path() == data_directory() / ".env"
    target = own_dotenv("OPENROUTER_API_KEY=sk-or-v1-abc\n")
    assert dotenv_path() == target


@pytest.mark.parametrize("changes", [{"max_steps": 0}, {"max_steps": -1},
                                     {"max_tokens": 0}, {"temperature": float("nan")}])
def test_config_rejects_invalid_limits(changes: dict) -> None:
    with pytest.raises(ConfigError):
        Config.from_env(environ={"OPENROUTER_API_KEY": "test"}, **changes)


def test_dotenv_rejects_injected_lines_without_modifying_file(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("OTHER=keep\n")
    with pytest.raises(ConfigError):
        save_dotenv_value(target, "OPENROUTER_API_KEY", "key\nOTHER=changed")
    assert target.read_text() == "OTHER=keep\n"


def test_dotenv_save_failure_preserves_existing_credentials(tmp_path, monkeypatch) -> None:
    target = tmp_path / ".env"
    target.write_text("OPENROUTER_API_KEY=old\n")
    def fail_replace(*args):
        raise OSError("failed commit")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="failed commit"):
        save_dotenv_value(target, "OPENROUTER_API_KEY", "new")
    assert target.read_text() == "OPENROUTER_API_KEY=old\n"
    assert list(tmp_path.iterdir()) == [target]


def test_startup_notice_names_ignored_settings_without_values(tmp_path: Path, monkeypatch, capsys) -> None:
    import argparse
    from slipagent import cli

    monkeypatch.delenv("SLIPAGENT_NO_DOTENV", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=secret-project-key\nAPP_PORT=8000\n", encoding="utf-8")
    cli._notice_ignored_project_settings(argparse.Namespace(workspace=str(tmp_path)))
    error = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in error and str(tmp_path / ".env") in error
    assert "secret-project-key" not in error and "APP_PORT" not in error
    assert os.environ.get("OPENROUTER_API_KEY") != "secret-project-key"


def test_startup_notice_respects_disabled_dotenv(tmp_path: Path, capsys) -> None:
    import argparse
    from slipagent import cli

    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=secret-project-key\n", encoding="utf-8")
    cli._notice_ignored_project_settings(argparse.Namespace(workspace=str(tmp_path)))
    assert capsys.readouterr().err == ""


async def test_commands_do_not_receive_dotenv_values_but_keep_exported_ones(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    from slipagent import config as config_module
    from slipagent.tools.shell import RunCommandTool
    from slipagent.workspace import Workspace

    monkeypatch.delenv("SLIPAGENT_NO_DOTENV", raising=False)
    monkeypatch.setattr(config_module, "_DOTENV_NAMES", set())
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "exported-by-user")
    own_dotenv("MISTRAL_API_KEY=from-slipagent-dotenv\nOPENROUTER_API_KEY=shadowed\n")
    # Undo the loaded value after the test; monkeypatch restores only what it set.
    monkeypatch.setenv("MISTRAL_API_KEY", "placeholder")
    monkeypatch.delenv("MISTRAL_API_KEY")
    load_dotenv()
    assert os.environ["MISTRAL_API_KEY"] == "from-slipagent-dotenv"
    child = config_module.command_environment()
    assert "MISTRAL_API_KEY" not in child and child["OPENROUTER_API_KEY"] == "exported-by-user"
    result = await RunCommandTool(Workspace(tmp_path)).run(
        'printf "%s|%s" "${MISTRAL_API_KEY-unset}" "$OPENROUTER_API_KEY"')
    assert "unset|exported-by-user" in result.content
    assert "from-slipagent-dotenv" not in result.content


def test_mcp_servers_receive_a_minimal_environment(monkeypatch) -> None:
    from slipagent.mcp import server_environment

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("TERM", "() { :; }; echo injected")
    env = server_environment({"EXA_API_KEY": "configured", "PATH": "/opt/server/bin"})
    assert env["PATH"] == "/opt/server/bin" and env["EXA_API_KEY"] == "configured"
    assert env["LC_ALL"] == "C.UTF-8" and env["HTTPS_PROXY"] == "http://proxy.invalid:8080"
    assert "OPENROUTER_API_KEY" not in env and "TERM" not in env
