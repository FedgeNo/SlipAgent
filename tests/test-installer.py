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
    ("Linux", {}, "/users/alice/.local/lib/slipagent", "/users/alice/.local/bin"),
    ("Linux", {"XDG_DATA_HOME": "/data"}, "/users/alice/.local/lib/slipagent", "/users/alice/.local/bin"),
    ("Darwin", {}, "/users/alice/Library/Application Support/SlipAgent", "/users/alice/.local/bin"),
    ("Windows", {"LOCALAPPDATA": "/local"}, "/local/Programs/SlipAgent", "/local/Programs/SlipAgent/bin"),
    ("Windows", {}, "/users/alice/AppData/Local/Programs/SlipAgent", "/users/alice/AppData/Local/Programs/SlipAgent/bin"),
])
def test_platform_destinations(system, environment, root, bin_dir):
    anchor = Path(Path.cwd().anchor)
    absolute = lambda path: anchor / Path(path).relative_to(Path("/"))
    environment = {key: str(absolute(value)) for key, value in environment.items()}
    assert installer.install_paths(system, absolute("/users/alice"), environment) == (absolute(root), absolute(bin_dir))


@pytest.mark.parametrize("system, environment", [("Windows", {"LOCALAPPDATA": "relative"}), ("Plan9", {})])
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


@pytest.mark.parametrize("shell, profiles", [
    ("bash", [".profile", ".bashrc"]),
    ("zsh", [".zprofile", ".zshrc"]),
    ("sh", [".profile"]),
    ("fish", [".config/fish/config.fish"]),
])
def test_posix_path_is_persistent_and_idempotent_even_when_in_current_path(tmp_path, monkeypatch, shell, profiles):
    directory = tmp_path / "bin"
    monkeypatch.setenv("PATH", str(directory))
    monkeypatch.setenv("SHELL", f"/bin/{shell}")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    installer.update_posix_path(directory, tmp_path)
    first = [(tmp_path / name).read_text() for name in profiles]
    assert all("SlipAgent installer PATH" in text for text in first)
    installer.update_posix_path(directory, tmp_path)
    assert [(tmp_path / name).read_text() for name in profiles] == first


def test_fish_path_uses_xdg_configuration_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/fish")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "configuration"))
    installer.update_posix_path(tmp_path / "bin", tmp_path)
    assert (tmp_path / "configuration/fish/config.fish").is_file()


@pytest.mark.parametrize("system, expected", [
    ("Linux", ".local/share/slipagent"),
    ("Darwin", "Library/Application Support/SlipAgent/data"),
    ("Windows", "AppData/Local/SlipAgent"),
])
def test_user_data_is_separate_from_application(tmp_path, system, expected):
    root, _ = installer.install_paths(system, tmp_path, {})
    data = installer.user_data_path(system, tmp_path, {})
    assert data == tmp_path / expected
    assert data != root and not data.is_relative_to(root / "source")


def test_xdg_changes_data_location_only(tmp_path):
    environment = {"XDG_DATA_HOME": str(tmp_path / "data")}
    assert installer.user_data_path("Linux", tmp_path, environment) == tmp_path / "data/slipagent"
    assert installer.install_paths("Linux", tmp_path, environment)[0] == tmp_path / ".local/lib/slipagent"


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
    directories = ("src", "docs", "scripts", "tests", "prompts", "benchmarks")
    assert [call.args[0] for call in remove.call_args_list] == [destination / name for name in directories]
    assert [call.args[:2] for call in copytree.call_args_list] == [(checkout / name, destination / name) for name in directories]
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
    prompt = directory / "system-prompt.md"
    prompt.write_text("Original guidance.\n")
    baseline = installer.source_hashes(tmp_path)
    assert "prompts/system-prompt.md" in baseline
    prompt.write_text("Edited guidance.\n")
    assert installer.changed_files(baseline, installer.source_hashes(tmp_path)) == ["prompts/system-prompt.md"]


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
        assert args[4] == r"C:\tools" + (";" + existing if existing else "")
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


@pytest.mark.parametrize("phase", ["copy", "dependencies", "verification", "launcher", "metadata"])
@pytest.mark.parametrize("windows", [False, True])
def test_failed_updates_restore_source_dependencies_and_activation(tmp_path, monkeypatch, phase, windows):
    import subprocess
    import os

    root, checkout, bin_dir = tmp_path / "installed", tmp_path / "download", tmp_path / "bin"
    source = root / "source"
    environment = source / ".venv"
    executables = environment / ("Scripts" if windows else "bin")
    executables.mkdir(parents=True)
    (source / "src").mkdir()
    application = source / "src/app.py"
    application.write_text("previous application")
    (source / ".git").mkdir()
    (source / ".git/HEAD").write_text("previous branch")
    dependency = environment / "dependency"
    dependency.write_text("previous dependency")
    executable = executables / ("python.exe" if windows else "python")
    executable.write_text("previous interpreter")
    entry = executables / ("slipagent.exe" if windows else "slipagent")
    entry.write_text("previous entry point")
    (root / ".slipagent-managed").write_text("managed fixture")
    metadata = root / "installation.json"
    metadata.write_text('{"previous": true}')
    baseline = root / "installed-file-hashes.json"
    baseline.write_text('{"previous": "hash"}')
    current = root / "current"
    if not windows:
        current.symlink_to(source)
    bin_dir.mkdir()
    launcher = bin_dir / ("slipagent.exe" if windows else "slipagent")
    launcher.symlink_to(entry)
    (checkout / "src").mkdir(parents=True)
    (checkout / "src/app.py").write_text("candidate application")
    monkeypatch.setattr(installer.shutil, "which", lambda name: "fixture-uv")
    def run(arguments, **options):
        dependency.write_text("candidate dependency")
        entry.write_text("candidate entry point")
        if ((phase == "dependencies" and "install" in arguments)
                or (phase == "verification" and "--help" in arguments)):
            raise subprocess.CalledProcessError(1, arguments)
    monkeypatch.setattr(installer.subprocess, "run", run)
    original_copy = installer.copy_source
    def copy(*arguments):
        original_copy(*arguments)
        if phase == "copy":
            raise OSError("copy failed")
    monkeypatch.setattr(installer, "copy_source", copy)
    original_launcher = installer.replace_launcher
    def replace(*arguments, **options):
        original_launcher(*arguments, **options)
        if phase == "launcher":
            raise OSError("launcher failed")
    monkeypatch.setattr(installer, "replace_launcher", replace)
    original_replace = os.replace
    failed = False
    def publish(before, after, *arguments, **options):
        nonlocal failed
        if phase == "metadata" and Path(after) == metadata and not failed:
            failed = True
            raise OSError("metadata failed")
        return original_replace(before, after, *arguments, **options)
    monkeypatch.setattr(installer.os, "replace", publish)
    with pytest.raises((OSError, subprocess.CalledProcessError)):
        installer.install(checkout, root, bin_dir, windows=windows, dev=False, python="fixture")
    assert application.read_text() == "previous application"
    assert dependency.read_text() == "previous dependency"
    assert entry.read_text() == "previous entry point"
    assert executable.read_text() == "previous interpreter"
    assert (source / ".git/HEAD").read_text() == "previous branch"
    assert metadata.read_text() == '{"previous": true}'
    assert baseline.read_text() == '{"previous": "hash"}'
    assert launcher.is_symlink() and launcher.resolve() == entry
    if not windows:
        assert current.is_symlink() and current.resolve() == source
    assert not list(tmp_path.glob(".installed-update-*"))


def test_failed_in_place_dependency_update_restores_environment(tmp_path, monkeypatch):
    import subprocess
    root = tmp_path / "installed"
    source = root / "source"
    dependency = source / ".venv/lib/dependency"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("old dependency")
    (root / ".slipagent-managed").write_text("managed fixture")
    monkeypatch.setattr(installer.shutil, "which", lambda name: "fixture-uv")
    def failed(arguments, **options):
        dependency.write_text("broken dependency")
        raise subprocess.CalledProcessError(1, arguments)
    monkeypatch.setattr(installer.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        installer.install(source, root, tmp_path / "bin", windows=False, dev=False, python="fixture")
    assert dependency.read_text() == "old dependency"


def test_failed_fresh_install_cleans_pending_files_and_can_be_retried(tmp_path, monkeypatch):
    import os
    root, checkout = tmp_path / "installed", tmp_path / "download"
    (checkout / "src").mkdir(parents=True)
    (checkout / "src/app.py").write_text("candidate")
    monkeypatch.setattr(installer.shutil, "which", lambda name: "fixture-uv")
    monkeypatch.setattr(installer.subprocess, "run", lambda *arguments, **options: None)
    original_replace = os.replace
    failed = False
    def publish(before, after, *arguments, **options):
        nonlocal failed
        if Path(after) == root / "installation.json" and not failed:
            failed = True
            raise OSError("activation failed")
        return original_replace(before, after, *arguments, **options)
    monkeypatch.setattr(installer.os, "replace", publish)
    with pytest.raises(OSError, match="activation failed"):
        installer.install(checkout, root, tmp_path / "bin", windows=False, dev=False, python="fixture")
    assert not (root / "source").exists()
    assert not list(root.glob("installation.*.json"))
    installer.install(checkout, root, tmp_path / "bin", windows=False, dev=False, python="fixture")
    assert (root / "installation.json").is_file()


def test_failed_rollback_retains_recoverable_backup(tmp_path, monkeypatch):
    import os
    import subprocess
    root, checkout = tmp_path / "installed", tmp_path / "download"
    source = root / "source"
    (source / "src").mkdir(parents=True)
    (source / "src/app.py").write_text("previous application")
    (root / ".slipagent-managed").write_text("managed fixture")
    (checkout / "src").mkdir(parents=True)
    (checkout / "src/app.py").write_text("candidate")
    monkeypatch.setattr(installer.shutil, "which", lambda name: "fixture-uv")
    def failed(*arguments, **options):
        raise subprocess.CalledProcessError(1, ["fixture"])
    monkeypatch.setattr(installer.subprocess, "run", failed)
    original_replace = os.replace
    def cannot_restore(before, after, *arguments, **options):
        if Path(after) == source:
            raise OSError("rollback rejected")
        return original_replace(before, after, *arguments, **options)
    monkeypatch.setattr(installer.os, "replace", cannot_restore)
    with pytest.raises(RuntimeError, match="Remaining backups"):
        installer.install(checkout, root, tmp_path / "bin", windows=False, dev=False, python="fixture")
    backups = list(tmp_path.glob(".installed-update-*"))
    assert len(backups) == 1
    assert (backups[0] / "0/src/app.py").read_text() == "previous application"
