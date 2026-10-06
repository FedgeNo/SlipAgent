#!/usr/bin/env python3
"""Install or update SlipAgent from this checkout without administrator privileges."""

from __future__ import annotations

import argparse
import hashlib
import fnmatch
import importlib
import json
import ntpath
import os
import platform
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from collections.abc import Mapping


SOURCE_DIRECTORIES = ("src", "docs", "scripts", "tests")
SOURCE_FILES = ("pyproject.toml", "README.md", "CONTRIBUTING.md", "AGENTS.md", "LICENSE", "install.py", ".gitignore", ".env.example")
HASH_MANIFEST = "file-hashes.json"
IGNORED_NAMES = (".git", ".venv", "venv", "__pycache__", "*.pyc", "*.pyo", "*.egg-info", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".env", ".env.*", ".mcp.json")


def source_hashes(source: Path) -> dict[str, str]:
    candidates = [source / name for name in SOURCE_FILES]
    for name in SOURCE_DIRECTORIES:
        directory = source / name
        if directory.is_symlink():
            candidates.append(directory)
        elif directory.is_dir():
            candidates.extend(directory.rglob("*"))
    hashes = {}
    for path in candidates:
        relative = path.relative_to(source)
        if relative.as_posix() != ".env.example" and any(fnmatch.fnmatch(part, pattern) for part in relative.parts for pattern in IGNORED_NAMES):
            continue
        if path.is_symlink():
            data = ("symlink:" + os.readlink(path)).encode("utf-8")
        elif path.is_file():
            data = path.read_bytes()
        else:
            continue
        hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    return dict(sorted(hashes.items()))


def changed_files(baseline: Mapping[str, str], actual: Mapping[str, str]) -> list[str]:
    return sorted(path for path in baseline.keys() | actual.keys() if baseline.get(path) != actual.get(path))


def update_changes(root: Path) -> tuple[list[str], str | None]:
    source = root / "source"
    if not source.exists():
        return [], None
    actual = source_hashes(source)
    manifest = root / "installed-file-hashes.json"
    if not manifest.exists():
        manifest = source / HASH_MANIFEST
    try:
        baseline = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict) or not baseline or any(
            not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for path, digest in baseline.items()
        ):
            raise ValueError("Invalid hash manifest")
    except (OSError, ValueError) as exc:
        return sorted(actual), f"Cannot verify installed files: {exc}. Listed files have an unknown baseline."
    return changed_files(baseline, actual), None


def install_paths(system: str, home: Path, environ: Mapping[str, str]) -> tuple[Path, Path]:
    if system == "Windows":
        base = Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
        if not base.is_absolute():
            raise ValueError("LOCALAPPDATA must be an absolute path")
        root = base / "SlipAgent"
        return root, root / "bin"
    if system == "Darwin":
        return home / "Library" / "Application Support" / "SlipAgent", home / ".local" / "bin"
    if system == "Linux":
        base = Path(environ.get("XDG_DATA_HOME") or home / ".local" / "share")
        if not base.is_absolute():
            raise ValueError("XDG_DATA_HOME must be an absolute path")
        return base / "slipagent", home / ".local" / "bin"
    raise ValueError(f"Unsupported operating system: {system}")


def path_contains(directory: Path, value: str, *, windows: bool = False) -> bool:
    if windows:
        def normalize(text: str) -> str:
            return ntpath.normcase(ntpath.normpath(ntpath.expandvars(text.strip('"'))))
        return normalize(str(directory)) in [normalize(part) for part in value.split(";") if part]
    return str(directory) in [str(Path(part).expanduser().absolute()) for part in value.split(os.pathsep) if part]


def path_snippet(directory: Path, *, fish: bool = False) -> str:
    quoted = shlex.quote(str(directory))
    if fish:
        return f"\n# SlipAgent installer PATH\nif not contains -- {quoted} $PATH\n    set -gx PATH {quoted} $PATH\nend\n"
    pattern = shlex.quote(":" + str(directory) + ":")
    return f'\n# SlipAgent installer PATH\ncase ":$PATH:" in\n    *{pattern}*) ;;\n    *) export PATH={quoted}:"$PATH" ;;\nesac\n'


def update_posix_path(directory: Path, home: Path) -> None:
    if path_contains(directory, os.environ.get("PATH", "")):
        return
    shell = Path(os.environ.get("SHELL", "/bin/sh")).name
    fish = shell == "fish"
    if fish:
        profiles = [home / ".config" / "fish" / "config.fish"]
    elif shell == "zsh":
        profiles = [home / ".zprofile", home / ".zshrc"]
    elif shell == "bash":
        login = next((home / name for name in (".bash_profile", ".bash_login") if (home / name).exists()), home / ".profile")
        profiles = [login, home / ".bashrc"]
    elif shell in {"sh", "dash", "ksh"}:
        profiles = [home / ".profile"]
    else:
        print(f"Add {directory} to PATH in your {shell} configuration.")
        return
    snippet = path_snippet(directory, fish=fish)
    for profile in profiles:
        existing = profile.read_text(encoding="utf-8") if profile.exists() else ""
        if snippet not in existing:
            profile.parent.mkdir(parents=True, exist_ok=True)
            with profile.open("a", encoding="utf-8") as handle:
                handle.write(snippet)
            print(f"Added PATH entry to {profile}")
    print("Open a new terminal to pick up the PATH entry.")


def update_windows_path(directory: Path) -> None:
    import ctypes
    from ctypes import wintypes

    winreg = importlib.import_module("winreg")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            value, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            value, kind = "", winreg.REG_EXPAND_SZ
        if not isinstance(value, str) or kind not in {winreg.REG_SZ, winreg.REG_EXPAND_SZ}:
            raise ValueError("The Windows user Path must be a string")
        if path_contains(directory, value, windows=True):
            return
        winreg.SetValueEx(key, "Path", 0, kind, value.rstrip(";") + (";" if value else "") + str(directory))
    notify = getattr(ctypes, "windll").user32.SendMessageTimeoutW
    notify.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPCWSTR,
                       wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
    notify.restype = wintypes.LPARAM
    result = ctypes.c_size_t()
    if not notify(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, ctypes.byref(result)):
        print("PATH is saved; Windows did not acknowledge the environment notification.")
    print("Added the launcher directory to your user PATH. Open a new terminal.")


def prepare_launcher(path: Path, root: Path, *, windows: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    owned = path.is_symlink() and path.resolve().is_relative_to(root)
    if path.is_file() and not owned:
        if windows:
            metadata = root / "installation.json"
            owned = metadata.is_file() and json.loads(metadata.read_text(encoding="utf-8")).get("launcher") == str(path)
        else:
            owned = "from slipagent.runtime import main" in path.read_text(encoding="utf-8", errors="replace")
    if not owned:
        raise ValueError(f"Refusing to replace an unrelated launcher: {path}")


def replace_launcher(target: Path, launcher: Path, root: Path, *, windows: bool) -> None:
    launcher.parent.mkdir(parents=True, exist_ok=True)
    candidate = launcher.with_name(launcher.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        try:
            candidate.symlink_to(target)
        except OSError:
            if not windows:
                raise
            # pip/uv's generated .exe embeds the managed interpreter path.
            shutil.copy2(target, candidate)
            print("Windows symlink privileges unavailable; installed the executable launcher instead.")
        if launcher.exists() or launcher.is_symlink():
            backup = root / "launcher-backups" / (launcher.name + "." + uuid.uuid4().hex)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if launcher.is_symlink():
                backup.symlink_to(os.readlink(launcher))
            else:
                shutil.copy2(launcher, backup)
            print(f"Saved the previous launcher to {backup}")
        os.replace(candidate, launcher)
    finally:
        if candidate.exists() or candidate.is_symlink():
            candidate.unlink()


def copy_source(checkout: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(*IGNORED_NAMES)
    for name in (*SOURCE_DIRECTORIES, ".git"):
        if name == ".git" and (destination / name).exists():
            continue
        if name != ".git" and (destination / name).exists():
            if (destination / name).is_symlink():
                (destination / name).unlink()
            else:
                shutil.rmtree(destination / name)
        if (checkout / name).is_dir():
            shutil.copytree(checkout / name, destination / name, ignore=ignore)
    for name in (*SOURCE_FILES, HASH_MANIFEST):
        if (checkout / name).is_file():
            shutil.copy2(checkout / name, destination / name)


def install(checkout: Path, root: Path, bin_dir: Path, *, windows: bool, dev: bool, python: str) -> Path:
    launcher = bin_dir / ("slipagent.exe" if windows else "slipagent")
    prepare_launcher(launcher, root, windows=windows)
    marker = root / ".slipagent-managed"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError(f"Installation directory is not an existing managed installation or empty: {root}")
    current = root / "current"
    if not windows and current.exists() and not current.is_symlink():
        raise ValueError(f"Cannot replace the installation shortcut: {current}")
    root.mkdir(parents=True, exist_ok=True)
    if not windows:
        root.chmod(0o700)
    marker.write_text("SlipAgent user installation\n", encoding="utf-8")
    source = root / "source"
    replacing_source = checkout.resolve() != source.resolve()
    activated = False
    try:
        if replacing_source:
            if source.exists():
                print("Updating the installed application from this checkout.")
            copy_source(checkout, source)
        environment = source / ".venv"
        executable = environment / ("Scripts/python.exe" if windows else "bin/python")
        uv = shutil.which("uv")
        package = str(source) + ("[dev]" if dev else "")
        if uv:
            if not executable.exists():
                subprocess.run([uv, "venv", "--python", python, str(environment)], check=True)
            subprocess.run([uv, "pip", "install", "--python", str(executable), "--editable", package], check=True)
        else:
            if not executable.exists():
                subprocess.run([python, "-m", "venv", str(environment)], check=True)
            subprocess.run([str(executable), "-m", "pip", "install", "--editable", package], check=True)
        entry = environment / ("Scripts/slipagent.exe" if windows else "bin/slipagent")
        subprocess.run([str(entry), "--help"], check=True,
                       stdout=subprocess.DEVNULL, env={**os.environ, "SLIPAGENT_NO_DOTENV": "1"})
        if replacing_source:
            pending_hashes = root / ("hashes." + uuid.uuid4().hex + ".json")
            pending_hashes.write_text(json.dumps(source_hashes(source), indent=2) + "\n", encoding="utf-8")
            os.replace(pending_hashes, root / "installed-file-hashes.json")
        # Preserve installed settings independently of source updates.
        configuration = root / ".env"
        if (checkout / ".env").is_file() and not configuration.exists():
            with configuration.open("xb") as handle:
                if not windows:
                    os.fchmod(handle.fileno(), 0o600)
                handle.write((checkout / ".env").read_bytes())
            print("Preserved configuration in the private installation directory.")
        replace_launcher(entry, launcher, root, windows=windows)
        activated = True
        metadata = {"source": str(source), "python": str(executable), "launcher": str(launcher), "checkout": str(source)}
        pending = root / ("installation." + uuid.uuid4().hex + ".json")
        pending.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(pending, root / "installation.json")
        if not windows:
            pending_link = root / ("current." + uuid.uuid4().hex)
            pending_link.symlink_to(source, target_is_directory=True)
            os.replace(pending_link, current)
        print(f"Installed source: {source}")
        print(f"Launcher: {launcher}")
        return launcher
    finally:
        if not activated:
            print("Installation did not activate. The permanent checkout and existing launcher have been retained.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, help="Override the OS-specific user installation directory")
    parser.add_argument("--bin-dir", type=Path, help="Override the launcher directory")
    parser.add_argument("--python", default=getattr(sys, "_base_executable", sys.executable), help="Python 3.11+ for the managed environment")
    parser.add_argument("--dev", action="store_true", help="Also install this project's development dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Show destinations without changing files or PATH")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or later is required")
    try:
        system = platform.system()
        home = Path.home()
        default_root, default_bin = install_paths(system, home, os.environ)
        root = (args.install_dir or default_root).expanduser().resolve()
        bin_dir = (args.bin_dir or (root / "bin" if system == "Windows" else default_bin)).expanduser().resolve()
        if args.dry_run:
            print(f"Installation: {root}\nLauncher directory: {bin_dir}\nPython: {args.python}")
            return 0
        changes, problem = update_changes(root)
        if changes or problem:
            print("WARNING: Installing or updating may result in loss of application edits.")
            if problem:
                print(problem)
            print("Changed files (including additions and deletions):" if not problem else "Files to inspect or back up:")
            for path in changes:
                print(f"  {path}")
            print("Back up any application changes before proceeding. Changes from separate checkouts are not merged.")
            try:
                answer = input("Proceed with installation/update? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer.strip().lower() not in {"y", "yes"}:
                print("Installation/update cancelled; no changes made.")
                return 0
        launcher = install(Path(__file__).resolve().parent, root, bin_dir, windows=system == "Windows", dev=args.dev, python=args.python)
        if system == "Windows":
            update_windows_path(bin_dir)
        else:
            update_posix_path(bin_dir, home)
        print(f"Ready: {launcher.name} --help")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        print("If Python cannot create a virtual environment, install your OS's Python venv support or uv and retry.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
