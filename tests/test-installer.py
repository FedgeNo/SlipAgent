"""Installer planning and PATH updates use pure helpers and mocked OS services."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


spec = importlib.util.spec_from_file_location("slipagent_install", Path(__file__).resolve().parents[1] / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.mark.parametrize("system, environment, root, bin_dir", [
    ("Linux", {}, "/users/alice/.local/share/slipagent", "/users/alice/.local/bin"),
    ("Linux", {"XDG_DATA_HOME": "/data"}, "/data/slipagent", "/users/alice/.local/bin"),
    ("Darwin", {}, "/users/alice/Library/Application Support/SlipAgent", "/users/alice/.local/bin"),
    ("Windows", {"LOCALAPPDATA": "/local"}, "/local/SlipAgent", "/local/SlipAgent/bin"),
    ("Windows", {}, "/users/alice/AppData/Local/SlipAgent", "/users/alice/AppData/Local/SlipAgent/bin"),
])
def test_platform_destinations(system, environment, root, bin_dir):
    anchor = Path(Path.cwd().anchor)
    absolute = lambda path: anchor / Path(path).relative_to(Path("/"))
    environment = {key: str(absolute(value)) for key, value in environment.items()}
    assert installer.install_paths(system, absolute("/users/alice"), environment) == (absolute(root), absolute(bin_dir))


@pytest.mark.parametrize("system, environment", [("Linux", {"XDG_DATA_HOME": "relative"}),
                                               ("Windows", {"LOCALAPPDATA": "relative"}), ("Plan9", {})])
def test_invalid_destination_configuration_is_reported(system, environment):
    with pytest.raises(ValueError):
        installer.install_paths(system, Path("/users/alice"), environment)


def test_windows_path_comparison_expands_variables_and_ignores_case(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Alice\AppData\Local")
    directory = Path(r"C:\Users\Alice\AppData\Local\SlipAgent\bin")
    assert installer.path_contains(directory, r"C:\other;%LOCALAPPDATA%\SLIPAGENT\bin" + "\\", windows=True)
    assert not installer.path_contains(directory, r"C:\other", windows=True)


def test_shell_snippets_quote_the_directory_and_guard_against_duplicates():
    directory = Path("/users/a b/.local/bin")
    snippet = installer.path_snippet(directory)
    assert "case" in snippet and "export PATH=" in snippet
    assert "'/users/a b/.local/bin'" in snippet
    fish = installer.path_snippet(directory, fish=True)
    assert "if not contains" in fish and "set -gx PATH" in fish


def test_fresh_download_replaces_application_directories_and_preserves_git(monkeypatch):
    checkout = Path("/mock/download")
    destination = Path("/mock/installed/source")
    monkeypatch.setattr(Path, "mkdir", MagicMock())
    monkeypatch.setattr(Path, "exists", lambda path: path.parent == destination)
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    monkeypatch.setattr(Path, "is_dir", lambda path: path.parent == checkout)
    monkeypatch.setattr(Path, "is_file", lambda path: path.parent == checkout)
    remove = MagicMock()
    copytree = MagicMock()
    copyfile = MagicMock()
    monkeypatch.setattr(installer.shutil, "rmtree", remove)
    monkeypatch.setattr(installer.shutil, "copytree", copytree)
    monkeypatch.setattr(installer.shutil, "copy2", copyfile)
    installer.copy_source(checkout, destination)
    assert [call.args[0] for call in remove.call_args_list] == [destination / name for name in ("src", "docs", "scripts", "tests", "prompts")]
    assert [call.args[:2] for call in copytree.call_args_list] == [(checkout / name, destination / name) for name in ("src", "docs", "scripts", "tests", "prompts")]
    assert any(call.args == (checkout / "install.py", destination / "install.py") for call in copyfile.call_args_list)
    assert any(call.args == (checkout / ".env.example", destination / ".env.example") for call in copyfile.call_args_list)


def test_configuration_template_is_included_in_application_hashes(monkeypatch):
    source = Path("/mock/source")
    monkeypatch.setattr(Path, "is_file", lambda path: path == source / ".env.example")
    monkeypatch.setattr(Path, "is_dir", lambda path: False)
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"OPENROUTER_API_KEY=\n")
    assert installer.source_hashes(source) == {
        ".env.example": installer.hashlib.sha256(b"OPENROUTER_API_KEY=\n").hexdigest(),
    }


def test_installer_preserves_and_tracks_application_logo(tmp_path):
    source = tmp_path / "download"
    source.mkdir()
    logo = source / "logo.png"
    logo.write_bytes(b"application logo")
    destination = tmp_path / "installed"
    installer.copy_source(source, destination)
    assert (destination / "logo.png").read_bytes() == logo.read_bytes()
    assert "logo.png" in installer.source_hashes(destination)


def test_updater_detects_edits_to_visible_prompt_files(tmp_path):
    directory = tmp_path / "prompts"
    directory.mkdir()
    prompt = directory / "system-prompt.txt"
    prompt.write_text("Original guidance.\n")
    baseline = installer.source_hashes(tmp_path)
    assert "prompts/system-prompt.txt" in baseline
    prompt.write_text("Edited guidance.\n")
    assert installer.changed_files(baseline, installer.source_hashes(tmp_path)) == ["prompts/system-prompt.txt"]


def test_dry_run_performs_no_installation_or_path_update(monkeypatch, capsys):
    install = MagicMock(side_effect=AssertionError("No installation allowed"))
    update = MagicMock(side_effect=AssertionError("No PATH mutation allowed"))
    monkeypatch.setattr(installer, "install", install)
    monkeypatch.setattr(installer, "update_posix_path", update)
    monkeypatch.setattr(installer, "update_windows_path", update)
    assert installer.main(["--dry-run"]) == 0
    assert "Installation:" in capsys.readouterr().out
    install.assert_not_called()
    update.assert_not_called()


@pytest.mark.parametrize("existing, kind, writes", [("C:\\other", 2, True),
                                                   ("C:\\TOOLS", 1, False), ("", 2, True)])
def test_windows_user_path_preserves_existing_entries_and_type(monkeypatch, existing, kind, writes):
    registry = MagicMock()
    registry.REG_SZ, registry.REG_EXPAND_SZ = 1, 2
    registry.QueryValueEx.return_value = (existing, kind)
    monkeypatch.setitem(sys.modules, "winreg", registry)
    import ctypes
    notify = MagicMock(return_value=1)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=SimpleNamespace(SendMessageTimeoutW=notify)), raising=False)
    installer.update_windows_path(Path(r"C:\tools"))
    assert registry.SetValueEx.called is writes
    if writes:
        args = registry.SetValueEx.call_args.args
        assert args[3] == kind
        assert args[4] == (existing + ";" if existing else "") + r"C:\tools"
        notify.assert_called_once()


@pytest.mark.parametrize("answer", ["", "n", "yes", " Y ", EOFError(), KeyboardInterrupt()])
def test_install_update_warning_and_confirmation(monkeypatch, capsys, answer):
    monkeypatch.setattr(installer, "update_changes", lambda root: (["src/edited.py", "src/deleted.py"], None))
    install = MagicMock(return_value=Path("/mock/bin/slipagent"))
    update = MagicMock()
    monkeypatch.setattr(installer, "install", install)
    monkeypatch.setattr(installer, "update_posix_path", update)
    monkeypatch.setattr(installer, "update_windows_path", update)

    def respond(prompt):
        output = capsys.readouterr().out
        assert "WARNING:" in output
        assert "Back up any application changes" in output
        assert "not merged" in output
        assert "src/edited.py" in output and "src/deleted.py" in output
        install.assert_not_called()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("builtins.input", respond)
    assert installer.main([]) == 0
    confirmed = isinstance(answer, str) and answer.strip().lower() in {"y", "yes"}
    assert install.called is confirmed
    assert update.called is confirmed


def test_unchanged_installation_updates_without_warning_or_confirmation(monkeypatch, capsys):
    monkeypatch.setattr(installer, "update_changes", lambda root: ([], None))
    install = MagicMock(return_value=Path("/mock/bin/slipagent"))
    monkeypatch.setattr(installer, "install", install)
    monkeypatch.setattr(installer, "update_posix_path", MagicMock())
    monkeypatch.setattr(installer, "update_windows_path", MagicMock())
    monkeypatch.setattr("builtins.input", MagicMock(side_effect=AssertionError("No confirmation expected")))
    assert installer.main([]) == 0
    install.assert_called_once()
    assert "WARNING" not in capsys.readouterr().out


def test_changed_files_includes_modifications_additions_and_deletions():
    assert installer.changed_files({"same": "a", "edited": "b", "deleted": "c"},
                                   {"same": "a", "edited": "d", "added": "e"}) == ["added", "deleted", "edited"]


@pytest.mark.parametrize("manifest", ['{"src/a.py": "' + "a" * 64 + '"}', '{}', 'invalid'])
def test_manifest_detection_and_unknown_baseline(monkeypatch, manifest):
    monkeypatch.setattr(Path, "exists", lambda path: True)
    monkeypatch.setattr(Path, "read_text", lambda path, **kwargs: manifest)
    monkeypatch.setattr(installer, "source_hashes", lambda source: {"src/a.py": "a" * 64})
    changes, problem = installer.update_changes(Path("/mock/root"))
    if manifest.startswith('{"src/'):
        assert changes == [] and problem is None
    else:
        assert changes == ["src/a.py"] and "unknown baseline" in problem
