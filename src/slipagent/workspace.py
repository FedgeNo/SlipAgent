"""Path confinement for built-in file, navigation, search, and Git tools.

These tools funnel project paths through `Workspace.resolve`, which
resolves symlinks and rejects anything landing outside the project root. That
rejects ordinary attempts to reach outside the project. This is not an OS
sandbox: shell/MCP subprocesses have process permissions, and session log
storage is owned separately by the harness.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Prune expensive generated/dependency trees during recursive search. list_dir
# still exposes them, and an explicit search path can target their contents.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


class WorkspaceError(Exception):
    """Raised when a path escapes the sandbox or cannot be resolved."""


@dataclass(frozen=True, slots=True)
class Workspace:
    """A resolved project root that all filesystem tools are confined to."""

    root: Path

    def __init__(self, root: Path | str = ".") -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {resolved}")
        object.__setattr__(self, "root", resolved)

    def resolve(self, raw: str | Path) -> Path:
        """Resolve a model-supplied path to an absolute path inside the root.

        Relative paths are anchored to the root. Absolute paths are allowed but
        still sandboxed. Symlinks are followed *before* the containment check so
        a link inside the tree cannot point out of it.
        """
        text = str(raw)
        if not text.strip():
            raise WorkspaceError("path must be a non-empty string")

        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate

        resolved = candidate.resolve()
        if not self.contains(resolved):
            raise WorkspaceError(
                f"path is outside the workspace: {text!r} "
                f"(workspace root is {self.root})"
            )
        return resolved

    def contains(self, path: Path) -> bool:
        """True when `path` is the root itself or lives beneath it."""
        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        return True

    def relative(self, path: Path) -> str:
        """Display form of a path, relative to the root when contained."""
        if self.contains(path):
            relative = path.relative_to(self.root)
            return "." if str(relative) == "." else relative.as_posix()
        return str(path)

    def iter_files(self, start: Path | str | None = None) -> Iterator[Path]:
        """Yield candidate files beneath `start`, pruning ignored directories.

        Uses `os.walk` with in-place pruning so a large `.git` or `node_modules`
        is never descended into, rather than filtering an already-complete walk.

        Symlinks that point outside the root are skipped: `os.walk` does not
        descend into symlinked directories, but it does list symlinked *files* in
        `filenames`, and reading one would hand out content from beyond the
        sandbox even though `resolve` refuses the same path when asked directly.
        """
        base = self.resolve(start) if start is not None else self.root
        if base.is_file():
            yield base
            return

        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRS]
            current = Path(dirpath)
            for filename in filenames:
                candidate = current / filename
                if candidate.is_symlink() and not self.contains(candidate.resolve()):
                    continue
                yield candidate

    def iter_entries(self, start: Path | str | None = None) -> Iterator[tuple[Path, bool]]:
        """Yield `(path, is_dir)` for every entry beneath `start`, pruned.

        As in `iter_files`, entries whose symlink target lands outside the root
        are skipped so `glob` cannot be used to enumerate them.
        """
        base = self.resolve(start) if start is not None else self.root
        if not base.is_dir():
            yield base, base.is_dir()
            return

        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRS]
            current = Path(dirpath)
            for name in dirnames:
                entry = current / name
                if entry.is_symlink() and not self.contains(entry.resolve()):
                    continue
                yield entry, True
            for name in filenames:
                entry = current / name
                if entry.is_symlink() and not self.contains(entry.resolve()):
                    continue
                yield entry, False
