"""`--update` installs a newer GitHub release through its own installer."""

import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from slipagent import __version__, updater


def archive_with(files):
    def fetch(url, destination):
        assert url == updater.ARCHIVE_URL
        with tarfile.open(destination, "w:gz") as archive:
            for name, text in files.items():
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return fetch


@pytest.fixture
def managed(tmp_path, monkeypatch):
    root = tmp_path / "install"
    monkeypatch.setattr(updater, "installation", lambda: (root, {"launcher": str(tmp_path / "bin" / "slipagent")}))
    return root


def test_newer_release_runs_its_installer_for_this_installation(managed, tmp_path):
    calls = []
    fetch = archive_with({"SlipAgent-master/pyproject.toml": 'version = "99.0.0"\n',
                          "SlipAgent-master/install.py": "print('installer')\n"})
    def run(command, cwd):
        calls.append(command)
        assert (Path(cwd) / "install.py").is_file()
        return subprocess.CompletedProcess(command, 0)
    assert updater.update(fetch=fetch, run=run) == 0
    [command] = calls
    assert command[1].endswith("install.py")
    assert command[2:] == ["--install-dir", str(managed), "--bin-dir", str(tmp_path / "bin")]
    assert not list(updater.data_directory().glob("update-*"))


def test_current_release_does_not_reinstall(managed):
    fetch = archive_with({"SlipAgent-master/pyproject.toml": f'version = "{__version__}"\n'})
    assert updater.update(fetch=fetch, run=lambda *args, **kwargs: pytest.fail("installer ran")) == 0


def test_unsafe_archive_is_rejected(managed, capsys):
    fetch = archive_with({"SlipAgent-master/../escape.py": "x"})
    assert updater.update(fetch=fetch, run=lambda *args, **kwargs: pytest.fail("installer ran")) == 1
    assert "unsafe archive entry" in capsys.readouterr().err


def test_development_checkout_is_not_updated(monkeypatch, capsys):
    monkeypatch.setattr(updater, "installation", lambda: None)
    assert updater.update(fetch=lambda *args: pytest.fail("downloaded")) == 2
    assert "not a managed installation" in capsys.readouterr().err
