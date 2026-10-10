"""Native folder launchers use desktop availability and literal argument lists."""

from unittest.mock import Mock
import subprocess

import pytest

from slipagent import explorer


@pytest.mark.parametrize("platform,desktop,installed,expected", [
    ("linux", False, {"xdg-open"}, None),
    ("linux", True, set(), None),
    ("linux", True, {"xdg-open", "gio"}, ("/bin/xdg-open",)),
    ("linux", True, {"gio"}, ("/bin/gio", "open")),
    ("darwin", False, {"open"}, ("/bin/open",)),
    ("freebsd", True, {"xdg-open"}, None),
])
def test_folder_opener_requires_supported_desktop(monkeypatch, platform, desktop, installed, expected):
    monkeypatch.setattr(explorer.sys, "platform", platform)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    if desktop:
        monkeypatch.setenv("WAYLAND_DISPLAY", "test-display")
    monkeypatch.setattr(explorer.shutil, "which", lambda name: "/bin/" + name if name in installed else None)
    assert explorer.folder_opener() == expected


def test_windows_folder_opener_uses_startfile(tmp_path, monkeypatch):
    launch = Mock()
    monkeypatch.setattr(explorer.sys, "platform", "win32")
    monkeypatch.setattr(explorer.os, "startfile", launch, raising=False)
    assert explorer.folder_opener() == ()
    assert explorer.open_folder(tmp_path, ())
    launch.assert_called_once_with(str(tmp_path))


@pytest.mark.parametrize("outcome,expected", [(0, True), (1, False), ("running", True), ("error", False)])
def test_folder_launch_reports_immediate_failures_and_preserves_path(tmp_path, monkeypatch, outcome, expected):
    process = Mock()
    launch = Mock(return_value=process)
    monkeypatch.setattr(explorer.subprocess, "Popen", launch)
    if outcome == "error":
        launch.side_effect = OSError("unavailable")
    elif outcome == "running":
        process.wait.side_effect = subprocess.TimeoutExpired("opener", 1)
    else:
        process.wait.return_value = outcome
    directory = tmp_path / "a folder; literal"
    assert explorer.open_folder(directory, ("/bin/gio", "open")) is expected
    assert launch.call_args.args[0] == ["/bin/gio", "open", str(directory)]
