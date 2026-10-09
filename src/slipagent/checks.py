"""Run bounded project checks once files reach their final state in an edit batch."""

from __future__ import annotations

import asyncio
import math
import os
import shlex
from pathlib import Path
from typing import Any

from .tools.base import ToolRegistry, ToolResult
from .tools.files import EditFileTool, WriteFileTool
from .tools.shell import capture_process, _subprocess_env, _truncate
from .types import ToolCall

SYNTAX_CHECK = """import ast, sys, tokenize
failed = False
for name in sys.argv[1:]:
    try:
        with tokenize.open(name) as source:
            text = source.read(32000001)
        if len(text) > 32000000:
            raise ValueError('file exceeds the automatic syntax-check size limit')
        ast.parse(text, filename=name)
    except (SyntaxError, ValueError, OSError, RecursionError) as error:
        print(str(error))
        failed = True
sys.exit(1 if failed else 0)
"""


def validate_checks(options: dict[str, Any]) -> None:
    if type(options.get("python_syntax", True)) is not bool:
        raise ValueError("python_syntax must be a boolean")
    checks = options.get("checks", [])
    if not isinstance(checks, list) or len(checks) > 8:
        raise ValueError("checks must be an array of at most 8 commands")
    for check in checks:
        if not isinstance(check, dict) or check.keys() - {"name", "argv", "extensions", "timeout"}:
            raise ValueError("each check accepts only name, argv, extensions and timeout")
        if not isinstance(check.get("name"), str) or not check["name"].strip():
            raise ValueError("each check requires a nonempty name")
        argv = check.get("argv")
        if not isinstance(argv, list) or not 1 <= len(argv) <= 64 or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in argv):
            raise ValueError("each check requires argv: 1–64 nonempty arguments")
        if argv[0] == "{files}":
            raise ValueError("{files} cannot be the check executable")
        extensions = check.get("extensions", [])
        if not isinstance(extensions, list) or any(not isinstance(ext, str) or not ext.startswith(".") for ext in extensions):
            raise ValueError("check extensions must be an array of suffixes such as .py")
        timeout = check.get("timeout", 10)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not .1 <= timeout <= 120:
            raise ValueError("check timeout must be between 0.1 and 120 seconds")


async def check_edit_batch(registry: ToolRegistry, batch: list[tuple[ToolCall, ToolResult]], step_id: int) -> str:
    from .environment import load_project_options

    environment = registry.services.get("project_environment")
    if environment is None:
        return ""
    paths: set[str] = set()
    for call, result in batch:
        tool = registry.get(call.name)
        if not result.is_error and isinstance(tool, (EditFileTool, WriteFileTool)):
            paths.add(str(environment.workspace.resolve(call.arguments["path"])))
    if not paths:
        return ""
    try:
        options = load_project_options(environment.workspace)
        checks = list(options.get("checks", []))
        if options.get("python_syntax", True):
            checks.insert(0, {"name": "Python syntax", "argv": ["{python}", "-I", "-c", SYNTAX_CHECK, "{files}"], "extensions": [".py", ".pyi"]})
        reports: list[str] = []
        for check in checks:
            selected = sorted(path for path in paths if not check.get("extensions") or Path(path).suffix in check["extensions"])
            if not selected:
                continue
            interpreter = ""
            if "{python}" in check["argv"]:
                interpreter = (await environment.snapshot()).get("interpreter")
                if not isinstance(interpreter, str) or not interpreter:
                    reports.append(f"{check['name']}: SKIPPED — no selected project Python. No interpreter was substituted.")
                    continue
            argv: list[str] = [value for arg in check["argv"] for value in (selected if arg == "{files}" else [interpreter if arg == "{python}" else arg])]
            archive = registry.services.get("command_archive")
            log = archive.start(shlex.join(argv), step_id=step_id, call_id="batch-check") if archive is not None else None
            try:
                process = await asyncio.create_subprocess_exec(*argv, cwd=environment.workspace.root,
                    env=_subprocess_env(), stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    start_new_session=os.name == "posix")
            except OSError as exc:
                if log is not None:
                    log.finish(None, False)
                reports.append(f"{check['name']}: FAILED to start — {exc}")
                continue
            stdout, stderr, timed_out = await capture_process(process, check.get("timeout", 10), log=log)
            status = "TIMED OUT" if timed_out else ("PASSED" if process.returncode == 0 else f"FAILED (exit {process.returncode})")
            stdout, _ = _truncate(stdout, archived=log is not None)
            stderr, _ = _truncate(stderr, archived=log is not None)
            output = (stdout + "\n" + stderr).strip()
            reports.append(f"{check['name']}: {status}\n\n{output}" + ("\n\n" + log.notice() if log is not None else ""))
        return "\n\n".join(reports)
    except Exception as exc:
        # Files have already changed; preserve their results even if settings or
        # check infrastructure fails. Cancellation still propagates to the loop.
        return f"Step-batch checks unavailable: {type(exc).__name__}: {exc}"
