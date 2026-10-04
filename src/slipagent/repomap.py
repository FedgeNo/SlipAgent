"""Bounded repository orientation; cached outlines are hints, never source reads."""

from __future__ import annotations

import ast
import asyncio
import os
import re
import subprocess
import time
from pathlib import Path

from .workspace import IGNORED_DIRS, Workspace, WorkspaceError

SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".java", ".php", ".rb", ".cs", ".swift", ".kt", ".sh", ".sql", ".html", ".css", ".md", ".toml"})
MAX_FILES = 2000
MAX_SOURCE_BYTES = 200_000


class RepositoryMap:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.cache: dict[str, tuple[tuple[int, int, int], str, set[str]]] = {}

    async def snapshot(self, query: str, limit: int = 8000) -> str:
        return await asyncio.to_thread(self._snapshot, query, limit)

    def _candidates(self) -> tuple[list[Path], bool]:
        paths: list[Path] = []
        pending = [self.workspace.root]
        visited = 0
        deadline = time.monotonic() + 1
        while pending and visited < 10000 and len(paths) < MAX_FILES and time.monotonic() < deadline:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        visited += 1
                        if visited >= 10000 or len(paths) >= MAX_FILES or time.monotonic() >= deadline:
                            return paths, True
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in IGNORED_DIRS and not entry.name.startswith("."):
                                pending.append(Path(entry.path))
                        elif Path(entry.name).suffix.lower() in SOURCE_SUFFIXES and not entry.name.startswith("."):
                            paths.append(Path(entry.path))
            except OSError:
                continue
        return paths, bool(pending)

    def _outline(self, path: Path) -> tuple[str, set[str]]:
        if path.suffix not in {".py", ".pyi"}:
            return "", set()
        try:
            with self.workspace.resolve(path).open("rb") as source:
                raw = source.read(MAX_SOURCE_BYTES + 1)
            if len(raw) > MAX_SOURCE_BYTES:
                return "[outline omitted: large file]", set()
            tree = ast.parse(raw)
        except (SyntaxError, ValueError, RecursionError, OSError, WorkspaceError):
            return "[outline unavailable; read source]", set()
        definitions: list[str] = []
        references: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                references.update(alias.name.split(".")[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                references.add(node.module.split(".")[-1])
        def describe(nodes: list[ast.stmt], prefix: str = "") -> None:
            for node in nodes:
                if len(definitions) >= 30:
                    return
                if isinstance(node, ast.ClassDef):
                    definitions.append(f"{node.lineno}: class {prefix}{node.name}")
                    describe(node.body, node.name + ".")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                    if node.args.vararg:
                        args.append("*" + node.args.vararg.arg)
                    elif node.args.kwonlyargs:
                        args.append("*")
                    args.extend(arg.arg for arg in node.args.kwonlyargs)
                    if node.args.kwarg:
                        args.append("**" + node.args.kwarg.arg)
                    definitions.append(f"{node.lineno}: {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {prefix}{node.name}({', '.join(args)})"[:240])
        describe(tree.body)
        return "\n".join(definitions), references

    def _snapshot(self, query: str, limit: int) -> str:
        paths, incomplete = self._candidates()
        names = [self.workspace.relative(path) for path in paths]
        # Delegate ignore semantics to Git, including nested rules and negations.
        # Outside Git the explicit source/dependency filters still apply.
        try:
            ignored = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}", "check-ignore", "-z", "--stdin"], input="\0".join(names).encode() + b"\0",
                cwd=self.workspace.root, capture_output=True, timeout=2,
            )
            if ignored.returncode in {0, 1}:
                excluded = set(ignored.stdout.decode(errors="replace").split("\0"))
                paths = [path for path in paths if self.workspace.relative(path) not in excluded]
        except (OSError, subprocess.TimeoutExpired):
            pass
        retained: dict[str, tuple[tuple[int, int, int], str, set[str]]] = {}
        deadline = time.monotonic() + 2
        parsed_bytes = 0
        for path in sorted(paths):
            name = self.workspace.relative(path)
            try:
                stat = self.workspace.resolve(path).stat()
                stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                cached = self.cache.get(name)
                if cached is None or cached[0] != stamp:
                    if time.monotonic() >= deadline or parsed_bytes + stat.st_size > 5_000_000:
                        incomplete = True
                        retained[name] = (stamp, "[outline deferred]", set())
                        continue
                    outline, references = self._outline(path)
                    parsed_bytes += min(stat.st_size, MAX_SOURCE_BYTES)
                    cached = stamp, outline, references
                retained[name] = cached
            except (OSError, WorkspaceError):
                continue
        self.cache = {name: entry for name, entry in retained.items() if entry[1] != "[outline deferred]"}
        words = set(re.findall(r"[a-z_][a-z_0-9]{2,}", query.casefold()))
        all_references = [reference for _, _, refs in retained.values() for reference in refs]
        def rank(name: str) -> tuple[int, str]:
            terms = set(re.findall(r"[a-z_][a-z_0-9]{2,}", (name + " " + retained[name][1]).casefold()))
            return -(len(words & terms) * 20 + min(10, all_references.count(Path(name).stem))), name
        header = (
            "\nRepository Map (Orientation Only):\n"
            "Paths and Python definitions with line numbers; signatures omit defaults and annotations. "
            "Other languages list paths only. Read files before editing. Dependencies, hidden directories, "
            "symlinks and Git-ignored files are excluded from this map; file tools can still inspect them.\n"
        )
        partial = "[Partial map: use glob/grep for files or definitions omitted here.]\n"
        if len(header) + len(partial) > limit:
            return ""
        result = header
        for name in sorted(retained, key=rank):
            entry = name + "\n" + retained[name][1] + "\n"
            if len(result) + len(entry) + len(partial) > limit:
                incomplete = True
                continue
            result += entry
        if incomplete:
            result += partial
        return result
