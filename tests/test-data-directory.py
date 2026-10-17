"""Persistent data uses one platform root independently of the project."""

from pathlib import Path

import pytest

from slipagent.config import data_directory, _preferences_path, find_dotenv, dotenv_path
from slipagent.mcp import config_path, load_servers, save_servers, ServerSpec
from slipagent.sessions import SessionJournal


@pytest.mark.parametrize("platform,variable,relative", [
    ("linux", None, ".local/share/slipagent"),
    ("linux", "relative", ".local/share/slipagent"),
    ("darwin", None, "Library/Application Support/SlipAgent/data"),
    ("win32", None, "AppData/Local/SlipAgent"),
])
def test_platform_defaults(tmp_path, monkeypatch, platform, variable, relative):
    monkeypatch.delenv("SLIPAGENT_STATE_DIR")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr("slipagent.config.sys.platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    if variable:
        monkeypatch.setenv("XDG_DATA_HOME", variable)
    assert data_directory() == tmp_path / relative


def test_xdg_root_is_shared_across_projects_and_data_types(tmp_path, monkeypatch):
    monkeypatch.delenv("SLIPAGENT_STATE_DIR")
    monkeypatch.setattr("slipagent.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    root = tmp_path / "data" / "slipagent"
    assert data_directory() == root
    assert _preferences_path() == root / "preferences.json"
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_servers(first, {"example": ServerSpec("example", "dummy-command")})
    assert config_path(first) == config_path(second) == root / "mcp.json"
    assert load_servers(second)["example"].command == "dummy-command"
    assert SessionJournal(str(first)).directory.parent == root
    assert SessionJournal(str(second)).directory.parent == root
    (second).mkdir()
    (second / ".mcp.json").write_text('{"mcpServers":{"untrusted":{"command":"do-not-run"}}}')
    assert "untrusted" not in load_servers(second)


def test_explicit_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "other"))
    root = tmp_path / "chosen"
    monkeypatch.setenv("SLIPAGENT_STATE_DIR", str(root))
    assert data_directory() == root
    assert config_path() == root / "mcp.json"
    assert _preferences_path().parent == root
    assert SessionJournal(str(tmp_path / "project")).directory.parent == root


def test_user_data_dotenv_is_used_and_project_dotenv_is_not(tmp_path, monkeypatch):
    root = data_directory()
    root.mkdir(parents=True, exist_ok=True)
    global_env = root / ".env"
    global_env.write_text("DUMMY_SETTING=global\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("DUMMY_SETTING=project\n")
    monkeypatch.chdir(project)
    assert find_dotenv() == global_env
    assert dotenv_path() == global_env
