"""Project environment creation and fresh harness-Python script execution."""

from __future__ import annotations

import asyncio
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import ConfigError, command_environment
from ..environment import ProjectEnvironment
from ..workspace import Workspace, WorkspaceError
from .base import Tool, ToolResult
from .output import CommandArchive
from .shell import capture_process, RunCommandTool, DEFAULT_TIMEOUT, MAX_TIMEOUT, _subprocess_env
from .blocking import run_blocking

if TYPE_CHECKING:
    from ..jobs import CommandInput


class CreatePythonEnvironmentTool(Tool):
    name = "create_python_environment"
    description = ""
    description_prompt = "tools/create-python-environment.md"
    parameter_prompts = "tools/create-python-environment-parameters.json"
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"}, "python": {"type": "string"},
    }}
    mutates_workspace = True
    instruction_path = "path"

    def __init__(self, environment: ProjectEnvironment) -> None:
        self.environment = environment

    async def _create(self, target: Path, interpreter: str) -> ToolResult | None:
        process = await asyncio.create_subprocess_exec(
            interpreter, "-I", "-m", "venv", str(target),
            cwd=self.environment.workspace.root, env=command_environment(),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=os.name == "posix",
        )
        output, error, timed_out = await capture_process(process, 120)
        if timed_out or process.returncode:
            return ToolResult.error({
                "error": "Environment creation failed. Partial files were retained; inspect them before retrying.",
                "path": str(target), "output": output, "stderr": error, "timed_out": timed_out,
            })
        return None

    async def run(self, path: str = ".venv", python: str | None = None) -> ToolResult:
        try:
            current = await self.environment.snapshot()
            if current["status"] == "selected":
                return ToolResult.ok({"created": False, "environment": current})
            if current["status"] == "ambiguous":
                return ToolResult.error("Multiple Python environments exist; select one explicitly before creating another.")
            if not path.strip():
                return ToolResult.error("Provide a nonempty environment directory path.")
            workspace = self.environment.workspace
            target = workspace.resolve(path)
            if target == workspace.root or target == Path(target.anchor):
                return ToolResult.error("The environment must have its own directory.")
            executable = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            candidate = ProjectEnvironment(workspace)
            candidate.python_override = str(executable)
            created = False
            if target.exists():
                # Existing directories are validated without running venv in them.
                if not (target / "pyvenv.cfg").is_file() or not executable.is_file():
                    return ToolResult.error("The target already exists and is not a complete virtual environment; it was left unchanged.")
            else:
                interpreter = sys.executable
                if python is not None:
                    if not python.strip():
                        return ToolResult.error("Provide a nonempty Python executable.")
                    located = shutil.which(python) if Path(python).name == python else None
                    requested = Path(python).expanduser()
                    if not requested.is_absolute():
                        requested = workspace.root / requested
                    interpreter = located or str(requested)
                if not Path(interpreter).is_file() or not os.access(interpreter, os.X_OK):
                    return ToolResult.error("The requested Python executable is unavailable.")
                # Exclusive creation also prevents two requests from overwriting a target.
                target.mkdir()
                created = True
                failure = await self._create(target, interpreter)
                if failure is not None:
                    return failure
            selected = await candidate.snapshot()
            if not selected["is_virtual_environment"]:
                return ToolResult.error("The target interpreter is not a virtual environment; selection was not changed.")
            self.environment.python_override = str(executable)
            self.environment.override_source = self.name
            selected = await self.environment.snapshot()
            return ToolResult.ok({"created": created, "path": str(target), "environment": selected})
        except (OSError, ValueError, ConfigError, WorkspaceError) as exc:
            return ToolResult.error(str(exc))


class RunPythonTool(Tool):
    name = 'run_python'
    description = ''
    description_prompt = 'tools/run-python.md'
    parameter_prompts = 'tools/run-python-parameters.json'
    parameters = {'type': 'object', 'properties': {
        'code': {'type': 'string', 'minLength': 1, 'maxLength': 262144},
        'timeout': {'type': 'number', 'minimum': .1, 'maximum': MAX_TIMEOUT},
    }, 'required': ['code']}

    def __init__(self, workspace: Workspace, archive: CommandArchive | None = None) -> None:
        self.workspace = workspace
        self.archive = archive

    async def run(self, code: str, timeout: float = DEFAULT_TIMEOUT) -> ToolResult:
        if not code.strip():
            return ToolResult.error('code must not be empty.')
        temporary = None

        def prepare() -> str:
            nonlocal temporary
            temporary = tempfile.TemporaryDirectory(prefix='slipagent-python-', dir=self.workspace.temporary_directory)
            script = Path(temporary.name) / 'script.py'
            script.write_text(code, encoding='utf-8')
            return str(script)

        try:
            script = await run_blocking(prepare)
            return await PythonCommandRunner(self.workspace, self.archive).run(script, timeout=timeout)
        finally:
            if temporary is not None:
                await run_blocking(temporary.cleanup)


class PythonCommandRunner(RunCommandTool):
    """Use shared capture, logs, and cleanup with direct Python execution."""

    def _command_label(self, command: str) -> str:
        arguments = [sys.executable, '-u', '-P', command]
        return subprocess.list2cmdline(arguments) if os.name == 'nt' else shlex.join(arguments)

    async def _spawn(self, command: str, *, input_channel: CommandInput | None = None) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable, '-u', '-P', command, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace.root), env=_subprocess_env(), start_new_session=os.name == 'posix')
