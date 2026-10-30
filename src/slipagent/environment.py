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
import sys
import importlib
import importlib.metadata
import importlib.util
import getpass
import platform
import socket
import hashlib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError, command_environment, data_directory, save_private_text
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion
from .fileio import open_regular
from .workspace import Workspace

DEFAULT_LOG_QUOTA_BYTES = 1024 * 1024 * 1024
PROJECT_SETTINGS_PATH = ".slipagent/project.json"


def session_environment() -> dict[str, Any]:
    """Describe the local execution identity without collecting credentials."""
    try:
        user = getpass.getuser()
    except (OSError, KeyError):
        user = None
    directory_error = None
    try:
        directory = os.getcwd()
    except OSError as exc:
        directory = None
        directory_error = str(exc)
    return {"user": user, "machine": socket.gethostname(),
            "operating_system": platform.system(), "os_release": platform.release(),
            "architecture": platform.machine(), "process_id": os.getpid(),
            "process_working_directory": directory,
            "process_working_directory_error": directory_error,
            "local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "shell": os.environ.get("SHELL") or os.environ.get("COMSPEC")}


MATH_PACKAGES = {
    name: name for name in (
        "numpy", "scipy", "sympy", "mpmath", "numexpr", "numba", "matplotlib", "pandas",
        "networkx", "statsmodels", "seaborn", "pint", "cvxpy", "gmpy2", "symengine",
    )
} | {"scikit-learn": "sklearn", "Cython": "Cython"}
PROBE = (
    "import json,sys,importlib.util; print(json.dumps({"
    "'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
    "'version':sys.version.split()[0],"
    "'pip_available':importlib.util.find_spec('pip') is not None,"
    "'pytest_available':importlib.util.find_spec('pytest') is not None,"
    "'ensurepip_available':importlib.util.find_spec('ensurepip') is not None,"
    "'module_search_paths':sys.path}))"
)


def harness_python_environment() -> dict[str, Any]:
    """Report this process's environment without importing heavy math libraries."""
    importlib.invalidate_caches()
    libraries = {}
    for name, module in MATH_PACKAGES.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            available = version is not None and importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        libraries[name] = {"available": available, "version": version}
    quote = shlex.join if os.name != "nt" else subprocess.list2cmdline
    return {"interpreter": sys.executable, "version": sys.version.split()[0],
            "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "is_virtual_environment": sys.prefix != sys.base_prefix,
            "math_command": quote([sys.executable, "-I"]), "libraries": libraries}


def math_requirements() -> dict[str, Requirement]:
    names = {canonicalize_name(name) for name in MATH_PACKAGES}
    requirements: dict[str, Requirement] = {canonicalize_name(item.name): item
                    for raw in importlib.metadata.requires("slipagent") or []
                    if canonicalize_name((item := Requirement(raw)).name) in names
                    and (item.marker is None or item.marker.evaluate())}
    if requirements.keys() != names:
        raise ConfigError("Math dependency metadata is incomplete; refresh the SlipAgent installation.")
    return requirements


def math_verification_key() -> str:
    """Invalidate successful import checks when Python or installed packages change."""
    stamps: list[Any] = [sys.executable, sys.version, sys.prefix, platform.platform(), MATH_PACKAGES]
    for distribution in importlib.metadata.distributions():
        stamps.append((distribution.metadata["Name"], distribution.version))
        for entry in distribution.files or []:
            if entry.name not in ("METADATA", "RECORD") and entry.suffix not in (".so", ".pyd", ".dll"):
                continue
            path = Path(str(distribution.locate_file(entry)))
            try:
                stat = path.stat()
                stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                stamps.append((str(path), None))
    for module in MATH_PACKAGES.values():
        spec = importlib.util.find_spec(module)
        if spec is not None and spec.origin:
            path = Path(spec.origin)
            for location in (path, path.parent):
                try:
                    stat = location.stat()
                    stamps.append((str(location), stat.st_mtime_ns, stat.st_size))
                except OSError:
                    stamps.append((str(location), None))
    return hashlib.sha256(json.dumps(stamps, sort_keys=True).encode()).hexdigest()


async def verify_harness_math() -> None:
    from .tools.blocking import run_blocking
    from .tools.shell import capture_process

    key = await run_blocking(math_verification_key)
    cache = data_directory() / "math-environment-cache.json"
    try:
        if await run_blocking(cache.read_text, encoding="utf-8") == key:
            return
    except OSError:
        pass
    script = "import importlib; " + "; ".join(f"importlib.import_module({module!r})" for module in MATH_PACKAGES.values())
    # Shield process creation as well as capture, so cancellation cannot abandon
    # a process that has spawned but whose handle has not yet reached this task.
    from .lifecycle import finish_cleanup
    from .tools.shell import kill_and_drain
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        sys.executable, "-I", "-c", script, cwd=sys.prefix, env=command_environment(),
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=os.name == "posix"))
    try:
        process = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        async def reap() -> None:
            await kill_and_drain(await spawn)
        await finish_cleanup(asyncio.create_task(reap()))
        raise
    stdout, stderr, timed_out = await capture_process(process, 120)
    if timed_out or process.returncode:
        detail = "timed out after 120 seconds" if timed_out else (stderr or stdout)[-2000:]
        raise ConfigError(f"Harness math import verification failed: {detail}")
    try:
        await run_blocking(save_private_text, cache, key)
    except OSError:
        # Cache availability must not turn a successful verification into failure.
        pass


async def ensure_harness_math() -> None:
    """Install missing required libraries into the running harness environment."""
    from .tools.blocking import run_blocking
    from .tools.shell import capture_process

    info = await run_blocking(harness_python_environment)
    requirements = await run_blocking(math_requirements)
    missing = set()
    for name, data in info["libraries"].items():
        try:
            compatible = data["available"] and data.get("version") in requirements[canonicalize_name(name)].specifier
        except (InvalidVersion, TypeError):
            compatible = False
        if not compatible:
            missing.add(name)
    if not missing:
        await verify_harness_math()
        return
    if sys.prefix == sys.base_prefix:
        raise ConfigError("Math setup requires running SlipAgent in its own virtual environment.")
    # Read constraints from installed package metadata so startup and the installer
    # use the same dependency contract, including when running an editable checkout.
    selected = [str(requirements[canonicalize_name(name)]) for name in sorted(missing)]

    async def install(arguments: list[str]) -> None:
        process = await asyncio.create_subprocess_exec(
            *arguments, cwd=sys.prefix, env=command_environment(), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        stdout, stderr, timed_out = await capture_process(process, 600)
        if timed_out or process.returncode:
            detail = "timed out after 600 seconds" if timed_out else (stderr or stdout)[-2000:]
            raise ConfigError(f"Automatic harness math installation failed: {detail}")

    if importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-I", "-m", "pip", "install"]
    elif uv := shutil.which("uv"):
        command = [uv, "pip", "install", "--python", sys.executable]
    elif importlib.util.find_spec("ensurepip") is not None:
        await install([sys.executable, "-I", "-m", "ensurepip", "--upgrade"])
        command = [sys.executable, "-I", "-m", "pip", "install"]
    else:
        raise ConfigError("Cannot install harness math packages: pip, uv, and ensurepip are unavailable.")
    await install([*command, *selected])
    remaining = await run_blocking(harness_python_environment)
    unavailable = [name for name, data in remaining["libraries"].items() if not data["available"]]
    if unavailable:
        raise ConfigError(f"Harness math packages remain unavailable: {', '.join(unavailable)}")
    for name, data in remaining["libraries"].items():
        try:
            if data["version"] in requirements[canonicalize_name(name)].specifier:
                continue
        except (InvalidVersion, TypeError):
            pass
        raise ConfigError(f"Harness math package {name} remains incompatible: {data['version']}")
    await verify_harness_math()


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    python: str | None = None
    log_quota_bytes: int = DEFAULT_LOG_QUOTA_BYTES


def load_project_options(workspace: Workspace) -> dict[str, Any]:
    path = workspace.resolve(PROJECT_SETTINGS_PATH)
    try:
        with open_regular(path) as source:
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
        validate_checks(data)
        # Accepted but ignored: server commands come only from user settings.
        data.pop("language_servers", None)
        return data
    except (ValueError, UnicodeError) as exc:
        raise ConfigError(f"Invalid {PROJECT_SETTINGS_PATH}: {exc}") from exc


def load_project_settings(workspace: Workspace) -> ProjectSettings:
    options = load_project_options(workspace)
    return ProjectSettings(options.get("python"), options.get("log_quota_bytes", DEFAULT_LOG_QUOTA_BYTES))


class ProjectEnvironment:
    """Refresh discovery on each request; repeat probes only when inputs change."""

    override_source = "--python"

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
        source = self.override_source
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
                    str(path), "-I", "-c", PROBE, cwd=self.workspace.root, env=command_environment(),
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
            if source == "discovery":
                info["status"] = "invalid"
                info["error"] = f"Discovered interpreter {path} cannot be used: {exc}"
                info["instruction"] = load_prompt('environment-selection.md')
                return info
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
