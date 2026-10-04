"""Path confinement for built-in file, navigation, search, and Git tools.

These tools funnel project paths through `Workspace.resolve`, which
resolves symlinks and, by default, rejects anything outside the project root. That
rejects ordinary attempts to reach outside the project. This is not an OS
sandbox: shell/MCP subprocesses have process permissions, and session log
storage is owned separately by the harness. Danger mode lifts path confinement
without changing the root used for relative paths or the process's permissions.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
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


@dataclass(slots=True)
class WorkspaceAccess:
    """Session-owned access mode shared by tools and retained across reloads."""

    danger: bool = False


@dataclass(frozen=True, slots=True)
class Workspace:
    """An immutable project root with an independently mutable access mode."""

    root: Path
    access: WorkspaceAccess = field(compare=False, repr=False)

    def __init__(self, root: Path | str = ".", *, danger: bool = False) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {resolved}")
        object.__setattr__(self, "root", resolved)
        object.__setattr__(self, "access", WorkspaceAccess(danger=danger))

    def resolve(self, raw: str | Path) -> Path:
        """Resolve a model-supplied path, enforcing the current access mode.

        Relative paths are anchored to the root. Absolute paths are allowed but
        confined unless danger mode is enabled. Symlinks are followed before
        the containment check so a link cannot bypass confinement.
        """
        text = str(raw)
        if not text.strip():
            raise WorkspaceError("path must be a non-empty string")

        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate

        resolved = candidate.resolve()
        if not self.access.danger and not self.contains(resolved):
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

        With confinement enabled, symlinks outside the root are skipped: `os.walk` does not
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
                if not self.access.danger and candidate.is_symlink() and not self.contains(candidate.resolve()):
                    continue
                yield candidate

    def iter_entries(self, start: Path | str | None = None) -> Iterator[tuple[Path, bool]]:
        """Yield `(path, is_dir)` for every entry beneath `start`, pruned.

        As in `iter_files`, confinement skips entries whose symlink target lands
        outside the root. Danger mode includes them.
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
                if not self.access.danger and entry.is_symlink() and not self.contains(entry.resolve()):
                    continue
                yield entry, True
            for name in filenames:
                entry = current / name
                if not self.access.danger and entry.is_symlink() and not self.contains(entry.resolve()):
                    continue
                yield entry, False
