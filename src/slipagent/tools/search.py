"""Search tools: regex `grep` and filename `glob`."""

from __future__ import annotations

from ..prompts import load_prompt

import fnmatch
import heapq
import asyncio
import json
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..workspace import Workspace, WorkspaceError
from .base import Tool, ToolResult
from .blocking import run_blocking
from .shell import _kill
from .grep import worker_source

MAX_GREP_RESULTS = 200
MAX_GREP_FILE_BYTES = 2_000_000
MAX_GLOB_RESULTS = 500
MAX_LINE_LENGTH = 400
GREP_TIMEOUT = 10.0


def _looks_binary(chunk: bytes) -> bool:
    return b"\x00" in chunk


class GrepTool(Tool):
    parameter_prompts = 'tools/grep-parameters.json'
    concurrent_safe = True
    instruction_path = "path"
    name = "grep"
    description_prompt = 'tools/grep.txt'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": ""},
            "path": {
                "type": "string",
                "description": "",
            },
            "include": {
                "type": "string",
                "description": "",
            },
            "ignore_case": {"type": "boolean", "description": ""},
            "max_results": {
                "type": "integer",
                "description": "",
                "minimum": 1,
                "maximum": MAX_GREP_RESULTS,
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._worker_source = worker_source()

    async def run(
        self,
        pattern: str,
        path: str | None = None,
        include: str | None = None,
        ignore_case: bool = False,
        max_results: int = MAX_GREP_RESULTS,
    ) -> ToolResult:
        # A thread cannot interrupt regex code holding the GIL. The child only
        # matches supplied text; workspace traversal stays in the confined parent.
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", self._worker_source.decode("utf-8"),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, cwd=self.workspace.root, limit=2_000_000,
            start_new_session=os.name == "posix",
        )
        try:
            async with asyncio.timeout(GREP_TIMEOUT):
                ready = await self._worker_request(process, {"pattern": pattern, "ignore_case": ignore_case,
                                                              "line_limit": MAX_LINE_LENGTH})
                if "error" in ready:
                    return ToolResult.error(str(ready["error"]))
                return await self._scan(process, pattern, path, include, max_results)
        except asyncio.TimeoutError:
            return ToolResult.error(f"Search timed out after {GREP_TIMEOUT:g}s. Narrow the path or simplify the regular expression.")
        finally:
            await _kill(process)

    @staticmethod
    async def _worker_request(process: asyncio.subprocess.Process, request: dict[str, Any]) -> dict[str, Any]:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        await process.stdin.drain()
        line = await process.stdout.readline()
        if not line:
            raise RuntimeError("Search worker exited without a result")
        result = json.loads(line)
        if not isinstance(result, dict):
            raise RuntimeError("Search worker returned an invalid result")
        return result

    async def _scan(
        self, process: asyncio.subprocess.Process, pattern: str, path: str | None,
        include: str | None, max_results: int,
    ) -> ToolResult:
        try:
            base = self.workspace.resolve(path) if path else self.workspace.root
        except WorkspaceError as exc:
            return ToolResult.error(str(exc))

        if not base.exists():
            return ToolResult.error(f"Path not found: {self.workspace.relative(base)}")

        matches: list[str] = []
        files_with_matches = 0
        files_scanned = 0
        truncated = False

        # validate_arguments rejects an oversized max_results, so this clamp
        # only matters for a direct run() call; the cap exists to keep a result
        # from flooding the context window either way.
        max_results = min(max_results, MAX_GREP_RESULTS)
        candidates = self.workspace.iter_files(base)
        while (candidate := await asyncio.to_thread(next, candidates, None)) is not None:
            if include and not _matches_include(candidate, base, include):
                continue
            files_scanned += 1
            text = await asyncio.to_thread(self._read_file, candidate)
            if text is None:
                continue
            result = await self._worker_request(process, {"text": text, "budget": max_results - len(matches)})
            line_hits = [f"{self.workspace.relative(candidate)}:{number}: {line}" for number, line in result["matches"]]
            if line_hits:
                files_with_matches += 1
                matches.extend(line_hits)
                if len(matches) >= max_results:
                    truncated = True
                    break

        if not matches:
            scope = self.workspace.relative(base)
            suffix = f" in files matching '{include}'" if include else ""
            return ToolResult.ok(
                f"No matches for {pattern!r}{suffix} under {scope} "
                f"({files_scanned} file(s) scanned)."
            )

        header = (
            f"{len(matches)}+ match(es) for {pattern!r} in "
            f"{files_with_matches} of {files_scanned} file(s) scanned"
            f"{' under ' + self.workspace.relative(base)}"
        )
        if truncated:
            header += f" — stopped at the {max_results}-result limit; narrow the search"

        return ToolResult.ok(header + "\n" + "\n".join(matches))

    def _read_file(self, candidate: Path) -> str | None:
        try:
            candidate = self.workspace.resolve(candidate)
            if candidate.stat().st_size > MAX_GREP_FILE_BYTES:
                return None
            with candidate.open("rb") as source:
                raw = source.read(MAX_GREP_FILE_BYTES + 1)
            if len(raw) > MAX_GREP_FILE_BYTES:
                return None
        except (OSError, WorkspaceError):
            return None
        if _looks_binary(raw[:4096]):
            return None

        return raw.decode("utf-8", errors="replace")


class GlobTool(Tool):
    parameter_prompts = 'tools/glob-parameters.json'
    concurrent_safe = True
    instruction_path = "path"
    name = "glob"
    description_prompt = 'tools/glob.txt'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "",
            },
            "path": {
                "type": "string",
                "description": "",
            },
            "max_results": {
                "type": "integer",
                "description": "",
                "minimum": 1,
                "maximum": MAX_GLOB_RESULTS,
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def run(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = MAX_GLOB_RESULTS,
    ) -> ToolResult:
        cancelled = threading.Event()
        return await run_blocking(self._scan, pattern, path, max_results, cancelled, on_cancel=cancelled.set)

    def _scan(self, pattern: str, path: str | None, max_results: int,
              cancelled: threading.Event) -> ToolResult:
        try:
            base = self.workspace.resolve(path) if path else self.workspace.root
        except WorkspaceError as exc:
            return ToolResult.error(str(exc))

        if not base.exists():
            return ToolResult.error(f"Path not found: {self.workspace.relative(base)}")

        # External searches in danger mode match relative to the requested
        # directory, while their displayed results remain absolute and reusable.
        match_root = self.workspace.root
        if not self.workspace.contains(base):
            match_root = base if base.is_dir() else base.parent
        count = 0

        def matches() -> Iterator[tuple[str, bool]]:
            nonlocal count
            for entry, is_dir in self.workspace.iter_entries(base):
                if cancelled.is_set():
                    return
                if _matches_glob(entry.relative_to(match_root).as_posix(), pattern):
                    count += 1
                    yield self.workspace.relative(entry), is_dir

        # Keep only the displayed paths, while retaining the exact total count
        # and lexicographic order on arbitrarily large directory trees.
        shown = heapq.nsmallest(min(max_results, MAX_GLOB_RESULTS), matches(), key=lambda pair: pair[0])
        if cancelled.is_set():
            return ToolResult.error("Glob scan cancelled.")
        if not count:
            return ToolResult.ok(
                f"No paths matching {pattern!r} under {self.workspace.relative(base)}."
            )

        lines = [
            entry + ("/" if is_dir else "")
            for entry, is_dir in shown
        ]
        header = f"{count} path(s) matching {pattern!r}"
        if count > len(shown):
            header += f" (showing first {len(shown)})"
        return ToolResult.ok(header + "\n" + "\n".join(lines))


def _matches_glob(relative_path: str, pattern: str) -> bool:
    if _match_components(relative_path.split("/"), pattern.split("/")):
        return True
    # fnmatch treats '*' as crossing '/', which differs from most people's
    # intuition for patterns like '*.py'. Try a basename match as well.
    return fnmatch.fnmatch(relative_path.rsplit("/", 1)[-1], pattern)


def _match_components(path_parts: list[str], pattern_parts: list[str]) -> bool:
    """Match a split path against a split pattern, honouring `**`.

    Plain `fnmatch` has no notion of `**`: it requires the literal `**` to be
    matched by one path component, so `fnmatch('a/b.py', '**/*.py')` is False.
    Since the tool description promises `**/test_*.py` and `src/**/*.ts` work,
    `**` is handled here as "zero or more path components", the usual shell and
    gitignore convention. Every other component is matched with plain fnmatch.
    """
    if not pattern_parts:
        return not path_parts
    head, *rest = pattern_parts
    if head == "**":
        # `**` may consume any number of components, including none.
        return any(_match_components(path_parts[index:], rest) for index in range(len(path_parts) + 1))
    if not path_parts:
        return False
    return fnmatch.fnmatch(path_parts[0], head) and _match_components(path_parts[1:], rest)


def _matches_include(candidate: Path, base: Path, include: str) -> bool:
    try:
        relative = candidate.relative_to(base).as_posix()
    except ValueError:
        relative = candidate.name
    return _matches_glob(relative, include) or fnmatch.fnmatch(candidate.name, include)


def search_tools(workspace: Workspace) -> list[Tool]:
    return [GrepTool(workspace), GlobTool(workspace)]
