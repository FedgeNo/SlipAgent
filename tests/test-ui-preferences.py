"""Inspection options persist alongside model preferences and use literal paths."""

import json
from unittest.mock import Mock

import pytest

from slipagent.config import _preferences_path, ConfigError
from slipagent.uipreferences import load_ui_preferences, save_ui_preference, configured_folder_opener
from slipagent import explorer


def test_ui_preferences_preserve_unrelated_settings():
    path = _preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"provider": "nvidia", "models": {"nvidia": "test/model"}}))
    save_ui_preference("wrap", False)
    save_ui_preference("tab_width", 8)
    assert load_ui_preferences() == {"wrap": False, "tab_width": 8}
    saved = json.loads(path.read_text())
    assert saved["provider"] == "nvidia"
    assert saved["models"] == {"nvidia": "test/model"}


def test_configured_folder_opener_uses_literal_argument_placeholder(tmp_path, monkeypatch):
    from slipagent import uipreferences
    monkeypatch.setattr(uipreferences.shutil, "which", lambda name: "/bin/folders")
    opener = configured_folder_opener('folders --path {path} --label "My Folder"')
    assert opener == ("/bin/folders", "--path", "{path}", "--label", "My Folder")
    launch = Mock()
    launch.return_value.wait.return_value = 0
    monkeypatch.setattr(explorer.subprocess, "Popen", launch)
    path = tmp_path / "literal $(command); folder"
    assert explorer.open_folder(path, opener)
    assert launch.call_args.args[0] == ["/bin/folders", "--path", str(path), "--label", "My Folder"]
    with pytest.raises(ConfigError, match="separate"):
        configured_folder_opener("folders --path={path}")
