"""Read scoped guidance for permitted paths before model or tool activity."""

from __future__ import annotations

from .prompts import load_prompt

import fnmatch
import os
import json
from pathlib import Path
from typing import Any

from .workspace import Workspace, WorkspaceError

INSTRUCTION_FILES = (
    "CLAUDE.md", "AGENTS.md", "AGENTS.override.md", ".cursorrules", ".clinerules",
    ".windsurfrules", ".github/copilot-instructions.md",
)
INSTRUCTION_DIRECTORIES = (
    (".cursor/rules", "*.mdc"),
    (".claude/rules", "*.md"),
    (".github/instructions", "*.instructions.md"),
    (".clinerules", "*"),
    (".windsurf/rules", "*.md"),
)


def _present(workspace: Workspace, relative: str) -> bool:
    try:
        (workspace.root / relative).lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise WorkspaceError(f"cannot inspect project instructions {relative}: {exc}") from exc
    return True


def load_project_instructions(workspace: Workspace, scope: str = ".") -> str:
    """Read guidance within a scope, preserving path-labelled text.

    Also used by the session instruction tracker before each working request.
    """
    sources: list[str] = []
    seen: set[Path] = set()
    base = Path(workspace.relative(workspace.resolve(scope)))

    def read(relative: str) -> None:
        try:
            path = workspace.resolve(relative)
            if Path(relative).name == ".clinerules" and path.is_dir():
                return
            if not path.is_file():
                raise WorkspaceError("instruction path must be a regular file")
            if path in seen:
                return
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, RuntimeError, WorkspaceError) as exc:
            raise WorkspaceError(f"cannot read project instructions {relative}: {exc}") from exc
        seen.add(path)
        sources.append(load_prompt('instruction-file.md', path=relative, content=content))

    for filename in INSTRUCTION_FILES:
        relative = (base / filename).as_posix()
        if _present(workspace, relative):
            read(relative)

    for directory, pattern in INSTRUCTION_DIRECTORIES:
        relative = (base / directory).as_posix()
        if not _present(workspace, relative):
            continue
        try:
            folder = workspace.resolve(relative)
        except (OSError, RuntimeError, WorkspaceError) as exc:
            raise WorkspaceError(f"cannot inspect project instructions {relative}: {exc}") from exc
        if not folder.is_dir():
            if directory == ".clinerules":
                continue
            raise WorkspaceError(f"project instruction directory is not a directory: {relative}")

        def walk_error(exc: OSError) -> None:
            raise WorkspaceError(f"cannot discover project instructions in {relative}: {exc}") from exc

        paths = []
        for current, _, filenames in os.walk(folder, onerror=walk_error, followlinks=False):
            for filename in filenames:
                if fnmatch.fnmatchcase(filename, pattern):
                    suffix = (Path(current) / filename).relative_to(folder)
                    paths.append((Path(relative) / suffix).as_posix())
        for name in sorted(paths):
            read(name)
    return "\n\n".join(sources)


class ProjectInstructions:
    """Track visited scopes and which exact guidance reached a model request."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.scopes: set[str] = {"."}
        self.delivered: dict[str, str] = {}

    def snapshot(self) -> dict[str, str]:
        result = {".": load_project_instructions(self.workspace)}
        for scope in sorted(self.scopes - {"."}):
            # Retain visited external scopes for a later re-enable, but neither
            # read nor present them while confinement is restored.
            if Path(scope).is_absolute() and not self.workspace.access.danger:
                continue
            directory = self.workspace.resolve(scope)
            if directory.exists():
                # Scope controls applicability, while the original workspace
                # controls confinement, including links to shared parent rules.
                result[scope] = load_project_instructions(self.workspace, scope)
            else:
                result[scope] = ""
        return result

    def presented(self, snapshot: dict[str, str]) -> None:
        self.delivered = dict(snapshot)

    def guard(self, tool: Any, raw: str | dict[str, Any] | None) -> str | None:
        parameter = tool.instruction_path
        if parameter is None:
            return None
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw or {}
        except ValueError:
            return None  # The ordinary argument validator reports syntax errors.
        if not isinstance(arguments, dict):
            return None
        value = arguments.get(parameter) or "."
        if not isinstance(value, str):
            return None
        target = self.workspace.resolve(value)
        directory = target if target.is_dir() else target.parent
        relevant = {"."}
        while directory != self.workspace.root:
            relevant.add(self.workspace.relative(directory))
            if directory == directory.parent:
                break
            directory = directory.parent
        self.scopes.update(relevant)
        if tool.mutates_workspace:
            current = self.snapshot()
            changed = [scope for scope in sorted(relevant)
                       if current.get(scope, "") != self.delivered.get(scope, "")]
            if changed:
                return (load_prompt('project-instructions-changed.md', scopes=', '.join(changed)))
        return None

    @staticmethod
    def render(snapshot: dict[str, str]) -> str:
        sections = []
        for scope, content in snapshot.items():
            if content:
                sections.append(load_prompt('instruction-scope.md', scope=scope, content=content))
        if not sections:
            return ""
        return (load_prompt('project-instructions-current.md', scopes='\n\n'.join(sections)))
