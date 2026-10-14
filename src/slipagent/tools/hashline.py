"""Optional line-anchored edits validated against a complete file revision."""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from typing import Any

from ..fileio import read_regular
from ..workspace import Workspace, WorkspaceError
from ..artifacts import ArtifactStore
from .base import Tool, ToolRegistry, ToolResult
from .blocking import run_blocking
from .files import ReadFileTool, MAX_READ_BYTES, _atomic_write, _save_diff
from .editing import MAX_DIFF_CHARS, edit_diff

HASHLINE_NAMES = frozenset({"read_hashlines", "edit_hashlines"})


def source_lines(text: str) -> list[str]:
    return [line for line in re.split(r"(?<=\n)|(?<=\r)(?!\n)", text) if line]


def revision(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def anchor(number: int, line: str) -> str:
    content = line.rstrip("\r\n")
    return f"{number}:{revision(content.encode('utf-8'))[:10]}"


def anchored_view(data: bytes, offset: int = 1, limit: int = 200) -> dict[str, Any]:
    lines = source_lines(data.decode("utf-8"))
    end = min(len(lines), offset - 1 + limit)
    return {"revision": revision(data), "total_lines": len(lines),
            "lines": [{"anchor": anchor(index + 1, lines[index]), "text": lines[index].rstrip("\r\n")}
                      for index in range(offset - 1, end)],
            "next_offset": end + 1 if end < len(lines) else None}


def apply_hashline_edits(text: str, edits: list[dict[str, Any]]) -> tuple[str, int]:
    lines = source_lines(text)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    ending = next((match.group() for line in lines if (match := re.search(r"\r\n$|\r$|\n$", line))), "\n")

    def position(value: str) -> int:
        match = re.fullmatch(r"([1-9][0-9]*):([0-9a-fA-F]{10})", value)
        if match is None:
            raise ValueError("Invalid anchor; copy an anchor from read_hashlines.")
        number = int(match[1])
        if number > len(lines) or anchor(number, lines[number - 1]).lower() != value.lower():
            raise ValueError(f"Stale or invalid anchor {value!r}; read_hashlines again.")
        return number - 1

    if not 1 <= len(edits) <= 100:
        raise ValueError("Supply 1–100 edits against the same revision.")
    planned: list[tuple[int, int, str]] = []
    for edit in edits:
        op = edit["op"]
        start = position(edit["start"])
        replacement = edit["lines"]
        if any("\n" in line or "\r" in line for line in replacement):
            raise ValueError("Each lines entry must contain one line, without CR or LF.")
        if any(re.match(r"\s*[1-9][0-9]*:[0-9a-fA-F]{10}(?:\s|\|)", line) for line in replacement):
            raise ValueError("Replacement lines contain copied display anchors; supply source text only.")
        if op == "replace":
            end = position(edit.get("end", edit["start"]))
            if end < start:
                raise ValueError("The end anchor precedes the start anchor.")
            suffix = re.search(r"\r\n$|\r$|\n$", lines[end])
            body = ending.join(replacement) + (suffix.group() if suffix and replacement else "")
            left, right = offsets[start], offsets[end + 1]
        elif op in {"insert_before", "insert_after"}:
            if "end" in edit or not replacement:
                raise ValueError("Insertions require nonempty lines and one start anchor, without end.")
            left = right = offsets[start if op == "insert_before" else start + 1]
            body = ending.join(replacement) + ending
            if left == len(text) and text and not text.endswith(("\n", "\r")):
                body = ending + ending.join(replacement)
        else:
            raise ValueError("Unknown edit operation.")
        planned.append((left, right, body))
    planned.sort(key=lambda item: (item[0], item[1]))
    for index, (left, right, _) in enumerate(planned):
        for previous_left, previous_right, _ in planned[:index]:
            if (left < previous_right and right > previous_left or
                    left == right == previous_left == previous_right or
                    left == right and previous_left < left < previous_right):
                raise ValueError("Edits overlap or insert twice at the same boundary; no changes applied.")
    updated = text
    for left, right, body in reversed(planned):
        updated = updated[:left] + body + updated[right:]
    return updated, bisect_right(offsets, planned[0][0])


class ReadHashlinesTool(ReadFileTool):
    name = "read_hashlines"
    description = ""
    description_prompt = "tools/read-hashlines.md"
    parameter_prompts = ""
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"}, "offset": {"type": "integer", "minimum": 1},
        "limit": {"type": "integer", "minimum": 1, "maximum": 2000}}, "required": ["path"]}

    def _read(self, path: str, offset: int, limit: int | None) -> ToolResult:
        try:
            target = self.workspace.resolve(path)
            data = read_regular(target, MAX_READ_BYTES)
            return ToolResult.ok({"path": self.workspace.relative(target), **anchored_view(data, offset, limit or 200)})
        except (OSError, UnicodeError, WorkspaceError, ValueError) as exc:
            return ToolResult.error(f"Could not read anchored source: {exc}")


class EditHashlinesTool(Tool):
    name = "edit_hashlines"
    description = ""
    description_prompt = "tools/edit-hashlines.md"
    instruction_path = "path"
    mutates_workspace = True
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"}, "revision": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "edits": {"type": "array", "minItems": 1, "maxItems": 100, "items": {
            "type": "object", "additionalProperties": False, "required": ["op", "start", "lines"],
            "properties": {"op": {"type": "string", "enum": ["replace", "insert_before", "insert_after"]},
                           "start": {"type": "string"}, "end": {"type": "string"},
                           "lines": {"type": "array", "items": {"type": "string"}}}}}},
        "required": ["path", "revision", "edits"]}

    def __init__(self, workspace: Workspace, artifacts: ArtifactStore | None = None) -> None:
        self.workspace = workspace
        self.artifacts = artifacts

    async def run(self, path: str, revision: str, edits: list[dict[str, Any]]) -> ToolResult:
        return await run_blocking(self._edit, path, revision, edits)

    def _edit(self, path: str, expected: str, edits: list[dict[str, Any]]) -> ToolResult:
        try:
            target = self.workspace.resolve(path)
            data = read_regular(target, MAX_READ_BYTES)
            if revision(data) != expected:
                return ToolResult.error("File revision changed; no edits applied. Call read_hashlines again and copy fresh anchors/revision.")
            text = data.decode("utf-8")
            updated, first = apply_hashline_edits(text, edits)
            if read_regular(target, MAX_READ_BYTES) != data:
                return ToolResult.error("File changed during edit preparation; no edits applied. Call read_hashlines again.")
            if updated != text:
                _atomic_write(self.workspace, target, updated)
        except (OSError, UnicodeError, WorkspaceError, ValueError) as exc:
            return ToolResult.error(f"Anchored edit failed: {exc}. Inspect the file before retrying a storage/checkpoint failure.")
        relative = self.workspace.relative(target)
        diff = edit_diff(relative, text, updated, limit=None if self.artifacts is not None else MAX_DIFF_CHARS)
        notice = _save_diff(self.artifacts, f"edit_hashlines {path!r}", diff)
        return ToolResult.ok({"path": relative, "changed": updated != text,
                              "diff": edit_diff(relative, text, updated) + notice,
                              **anchored_view(updated.encode("utf-8"), max(1, first - 3), 30)})


def hashline_tools(registry: ToolRegistry) -> list[Tool]:
    registry.services.setdefault("request_only_definitions", set()).update(HASHLINE_NAMES)
    reader = ReadHashlinesTool(registry.services["workspace"])
    editor = EditHashlinesTool(reader.workspace, registry.services.get("artifact_store"))
    return [reader, editor]


def set_hashline(registry: ToolRegistry, enabled: bool) -> None:
    registry.services["hashline_enabled"] = enabled
    for name in HASHLINE_NAMES:
        registry.unregister(name)
    if enabled:
        for tool in hashline_tools(registry):
            registry.register(tool)
    registry.builtin_names = registry.builtin_names | HASHLINE_NAMES
    catalog = registry.services.get("tool_catalog")
    if catalog is not None:
        catalog.loaded.difference_update(HASHLINE_NAMES)
        for name in HASHLINE_NAMES:
            catalog.available.pop(name, None)
