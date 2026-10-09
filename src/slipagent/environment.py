"""Project settings and validated Python discovery, independent of shell PATH.

Keep the lexical interpreter path: resolving a venv's `python` symlink to the
base binary would silently select the global environment. Probe with isolated
Python to identify what that exact executable actually uses.
"""

from __future__ import annotations

from .prompts import load_prompt

import asyncio
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError
from .workspace import Workspace

DEFAULT_LOG_QUOTA_BYTES = 100 * 1024 * 1024
PROJECT_SETTINGS_PATH = ".slipagent/project.json"
PROBE = (
    "import json,sys,importlib.util; print(json.dumps({"
    "'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
    "'version':sys.version.split()[0],"
    "'pip_available':importlib.util.find_spec('pip') is not None,"
    "'pytest_available':importlib.util.find_spec('pytest') is not None,"
    "'ensurepip_available':importlib.util.find_spec('ensurepip') is not None,"
    "'module_search_paths':sys.path}))"
)


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    python: str | None = None
    log_quota_bytes: int = DEFAULT_LOG_QUOTA_BYTES


def load_project_options(workspace: Workspace) -> dict[str, Any]:
    path = workspace.resolve(PROJECT_SETTINGS_PATH)
    try:
        with path.open("rb") as source:
            raw = source.read(16385)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"Cannot read {PROJECT_SETTINGS_PATH}: {exc}") from exc
    try:
        if len(raw) > 16384:
            raise ValueError("settings exceed 16 KiB")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.keys() - {"python", "log_quota_bytes", "python_syntax", "checks", "language_servers"}:
            raise ValueError("expected an object with only python, log_quota_bytes, python_syntax, checks and language_servers settings")
        python = data.get("python")
        if python is not None and (not isinstance(python, str) or not python.strip()):
            raise ValueError("python must be a nonempty interpreter path or null for discovery")
        quota = data.get("log_quota_bytes", DEFAULT_LOG_QUOTA_BYTES)
        if type(quota) is not int or quota < 1:
            raise ValueError("log_quota_bytes must be a positive integer")
        from .checks import validate_checks
        from .lsp import validate_servers
        validate_checks(data)
        validate_servers(data.get("language_servers", {}))
        return data
    except (ValueError, UnicodeError) as exc:
        raise ConfigError(f"Invalid {PROJECT_SETTINGS_PATH}: {exc}") from exc


def load_project_settings(workspace: Workspace) -> ProjectSettings:
    options = load_project_options(workspace)
    return ProjectSettings(options.get("python"), options.get("log_quota_bytes", DEFAULT_LOG_QUOTA_BYTES))


class ProjectEnvironment:
    """Refresh discovery on each request; repeat probes only when inputs change."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.python_override: str | None = None
        self._cached_key: tuple[Any, ...] | None = None
        self._cached: dict[str, Any] = {}

    def _module_stamp(self) -> tuple[tuple[str, int | None, int | None], ...]:
        """Recheck module availability when an import location changes.

        Installing/removing pip or pytest changes site-packages, not the Python
        executable or pyvenv.cfg. Retain cheap directory stamps between probes.
        Missing locations are tracked too, because an installer can create them.
        """
        stamps: list[tuple[str, int | None, int | None]] = []
        for location in self._cached.get("module_search_paths", []):
            try:
                stat = Path(location).stat()
                stamps.append((location, stat.st_mtime_ns, stat.st_size))
            except OSError:
                stamps.append((location, None, None))
        return tuple(stamps)

    async def snapshot(self) -> dict[str, Any]:
        from .tools.shell import capture_process

        settings = load_project_settings(self.workspace)
        selected = self.python_override
        source = "--python"
        if selected is None:
            selected = settings.python
            source = PROJECT_SETTINGS_PATH
        candidates = []
        executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        for name in (".venv", "venv"):
            directory = self.workspace.resolve(name)
            if directory.is_dir():
                candidates.append(str(directory / executable))
        info: dict[str, Any] = {
            "candidates": candidates,
            "selection_help": load_prompt("environment-selection-help.md", settings_path=PROJECT_SETTINGS_PATH),
            "shell_policy": load_prompt('environment-shell.md'),
        }
        if selected is None:
            if len(candidates) != 1:
                info["status"] = "ambiguous" if candidates else "unconfigured"
                info["instruction"] = load_prompt('environment-selection.md')
                return info
            selected = candidates[0]
            source = "discovery"
        path = Path(selected).expanduser()
        if not path.is_absolute():
            path = self.workspace.root / path
        path = Path(os.path.abspath(path))
        try:
            stat = path.stat()
            if not path.is_file() or not os.access(path, os.X_OK):
                raise OSError("not an executable file")
            config = path.parent.parent / "pyvenv.cfg"
            config_stamp = config.stat().st_mtime_ns if config.exists() else None
            # Include the probe contract so a live reload refreshes previously
            # cached properties when discovery gains new fields.
            key = (PROBE, str(path), source, stat.st_dev, stat.st_ino, stat.st_mtime_ns,
                   stat.st_size, config_stamp, self._module_stamp())
            if key != self._cached_key:
                process = await asyncio.create_subprocess_exec(
                    str(path), "-I", "-c", PROBE, cwd=self.workspace.root,
                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, start_new_session=os.name == "posix",
                )
                stdout, stderr, timed_out = await capture_process(process, 5)
                if timed_out or process.returncode:
                    raise ValueError("interpreter probe timed out" if timed_out else f"interpreter probe failed: {stderr[:500]}")
                properties = json.loads(stdout)
                if not isinstance(properties, dict) or any(not isinstance(properties.get(key), str) or not properties[key]
                                                         for key in ("executable", "prefix", "base_prefix", "version")):
                    raise ValueError("interpreter probe returned invalid Python properties")
                if any(type(properties.get(name)) is not bool for name in
                       ("pip_available", "pytest_available", "ensurepip_available")):
                    raise ValueError("interpreter probe returned invalid module availability")
                locations = properties.get("module_search_paths")
                if not isinstance(locations, list) or any(not isinstance(item, str) for item in locations):
                    raise ValueError("interpreter probe returned invalid module search paths")
                virtual = properties["prefix"] != properties["base_prefix"]
                if source == "discovery" and not virtual:
                    raise ValueError("discovered interpreter does not use a virtual environment")
                self._cached = {**properties, "is_virtual_environment": virtual}
                self._cached_key = key[:-1] + (self._module_stamp(),)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Invalid project Python interpreter {path}: {exc}. {info['selection_help']}") from exc
        quote = shlex.join if os.name != "nt" else subprocess.list2cmdline
        command = quote([str(path)])
        install_command = None
        bootstrap_command = None
        installer = None
        uv = shutil.which("uv")
        if self._cached["pip_available"]:
            installer = "pip"
            install_command = command + " -m pip install PACKAGE"
        elif uv is not None and self._cached["is_virtual_environment"]:
            installer = "uv"
            install_command = quote([os.path.abspath(uv), "pip", "install", "--python", str(path), "PACKAGE"])
        elif self._cached["ensurepip_available"] and self._cached["is_virtual_environment"]:
            bootstrap_command = command + " -m ensurepip --upgrade"
        properties = {name: value for name, value in self._cached.items() if name != "module_search_paths"}
        return {**info, **properties, "status": "selected", "source": source,
                "interpreter": str(path), "bin_directory": str(path.parent),
                "test_command": command + " -m pytest" if self._cached["pytest_available"] else None,
                "installer": installer, "install_command": install_command,
                "bootstrap_command": bootstrap_command,
                "instruction": (
                    load_prompt('environment-commands.md')
                )}
