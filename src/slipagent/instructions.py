"""Project guidance is read inside the workspace before model or tool activity."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

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


def load_project_instructions(workspace: Workspace) -> str:
    """Read root guidance and rule directories, preserving path-labelled text.

    Called at startup and /init. The returned text stays pinned across ordinary
    turns and reloads; it is not reread on every API request. Nested source-tree
    instruction files remain the model's responsibility before editing there.
    """
    sources: list[str] = []
    seen: set[Path] = set()

    def read(relative: str) -> None:
        try:
            path = workspace.resolve(relative)
            if relative == ".clinerules" and path.is_dir():
                return
            if not path.is_file():
                raise WorkspaceError("instruction path must be a regular file")
            if path in seen:
                return
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, RuntimeError, WorkspaceError) as exc:
            raise WorkspaceError(f"cannot read project instructions {relative}: {exc}") from exc
        seen.add(path)
        sources.append(f"### {relative}\n{content}\n")

    for relative in INSTRUCTION_FILES:
        if _present(workspace, relative):
            read(relative)

    for relative, pattern in INSTRUCTION_DIRECTORIES:
        if not _present(workspace, relative):
            continue
        try:
            folder = workspace.resolve(relative)
        except (OSError, RuntimeError, WorkspaceError) as exc:
            raise WorkspaceError(f"cannot inspect project instructions {relative}: {exc}") from exc
        if not folder.is_dir():
            if relative == ".clinerules":
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
    return "\n".join(sources)
