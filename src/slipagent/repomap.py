"""Bounded repository orientation; cached outlines are hints, never source reads."""

from __future__ import annotations

from .prompts import load_prompt

import ast
import asyncio
import os
import re
import subprocess
import time
from pathlib import Path
from collections import defaultdict

from .workspace import IGNORED_DIRS, Workspace, WorkspaceError
from .symbols import outline

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
        try:
            with self.workspace.resolve(path).open("rb") as source:
                raw = source.read(MAX_SOURCE_BYTES + 1)
            if len(raw) > MAX_SOURCE_BYTES:
                return "[outline omitted: large file]", set()
            if path.suffix.lower() not in {".py", ".pyi"}:
                return outline(raw, path.suffix.lower())
            tree = ast.parse(raw)
        except (SyntaxError, ValueError, RecursionError, OSError, WorkspaceError):
            return "[outline unavailable; read source]", set()
        definitions: list[str] = []
        references: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                references.update("module:" + alias.name.split(".")[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                references.add("module:" + node.module.split(".")[-1])
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                references.add("ref:" + node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                references.add("ref:" + node.attr)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                references.add("def:" + node.name)
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
        if getattr(self, "symbol_version", 0) != 1:
            self.cache.clear()
            self.symbol_version = 1
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
        definitions: dict[str, set[str]] = defaultdict(set)
        modules: dict[str, set[str]] = defaultdict(set)
        for name, (_, _, refs) in retained.items():
            modules[Path(name).stem].add(name)
            for tag in refs:
                if tag.startswith("def:"):
                    definitions[tag[4:]].add(name)
        edges: dict[str, set[str]] = {}
        relevance: dict[str, float] = {}
        for name, (_, _, refs) in retained.items():
            terms = set(re.findall(r"[a-z_][a-z_0-9]{2,}", (name + " " + retained[name][1]).casefold()))
            relevance[name] = 1 + len(words & terms) * 20
            targets: set[str] = set()
            for tag in sorted(refs)[:512]:
                matches = (definitions.get(tag[4:], set()) if tag.startswith("ref:")
                           else modules.get(tag[7:], set()) if tag.startswith("module:") else set())
                if len(matches) <= 20:
                    targets.update(matches)
                if len(targets) >= 256:
                    break
            edges[name] = targets - {name}
        total = sum(relevance.values()) or 1
        personal = {name: score / total for name, score in relevance.items()}
        scores = dict(personal)
        for _ in range(20):
            dangling = sum(scores[name] for name in edges if not edges[name])
            next_scores = {name: (.15 + .85 * dangling) * score for name, score in personal.items()}
            for name, targets in edges.items():
                if targets:
                    share = .85 * scores[name] / len(targets)
                    for target in targets:
                        next_scores[target] += share
            scores = next_scores
        def rank(name: str) -> tuple[float, str]:
            return -(relevance[name] + scores[name] * 20), name
        header = (
            load_prompt('repository-map.txt') + '\n\n'
        )
        partial = load_prompt('repository-map-partial.txt') + '\n'
        if len(header) + len(partial) > limit:
            return ""
        result = header
        for name in sorted(retained, key=rank):
            entry = name + "\n" + retained[name][1] + "\n\n"
            if len(result) + len(entry) + len(partial) > limit:
                incomplete = True
                continue
            result += entry
        if incomplete:
            result += partial
        return result
