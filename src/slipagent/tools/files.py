"""File tools: read, write, and precise-edit.

`edit_file` deliberately requires an exact string match and rejects ambiguous
matches. That strictness is what makes agent edits reliable: instead of
guessing at line numbers, the model has to quote real text from the file it
actually read, and a collision surfaces as an error it can fix.
"""

from __future__ import annotations

import os
import stat
import tempfile
import uuid
from pathlib import Path

from ..workspace import Workspace, WorkspaceError
from .base import Tool, ToolResult
from .editing import apply_edits, edit_diff

# Guard rails so a stray path or runaway file cannot blow up the context window.
MAX_READ_BYTES = 2_000_000
MAX_READ_LINES = 2_000


def _atomic_write(workspace: Workspace, target: Path, content: str) -> None:
    """Publish complete UTF-8 bytes without truncating files or following hard links.

    On POSIX, directory descriptors keep a swapped parent symlink from redirecting
    the write outside the workspace. Existing internal symlinks have already been
    resolved by Workspace. New files use the process umask; replacements keep mode.
    """
    encoded = content.encode("utf-8")
    workspace.resolve(target)
    # Outside targets in danger mode use their filesystem root as the anchor;
    # retain descriptor-based traversal and atomic replacement in both modes.
    anchor = workspace.root
    if not workspace.contains(target):
        anchor = Path(target.anchor)
    relative = target.relative_to(anchor)
    if os.name != "posix":
        target.parent.mkdir(parents=True, exist_ok=True)
        workspace.resolve(target)
        fd, temporary = tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            if target.exists():
                os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
            workspace.resolve(target)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(anchor, flags)
    temporary_name: str | None = None
    try:
        for part in relative.parts[:-1]:
            try:
                os.mkdir(part, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        try:
            original = os.stat(target.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            original = None
        if original is not None and not stat.S_ISREG(original.st_mode):
            raise OSError("Destination changed or is not a regular file; read it again before writing.")
        temporary_name = ".slipagent-write-" + uuid.uuid4().hex
        fd = os.open(temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=directory)
        with os.fdopen(fd, "wb") as output:
            output.write(encoded)
            output.flush()
            if original is not None:
                os.fchmod(output.fileno(), stat.S_IMODE(original.st_mode))
            os.fsync(output.fileno())
        os.replace(temporary_name, target.name, src_dir_fd=directory, dst_dir_fd=directory)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)


class ReadFileTool(Tool):
    concurrent_safe = True
    instruction_path = "path"
    name = "read_file"
    description = (
        "Read a text file allowed by the current Workspace Access mode.\n\n"
        "Returns numbered lines so they can be cited; omit the displayed line numbers when calling "
        "edit_file.\n\n"
        "Line endings are displayed as LF.\n\n"
        "Use offset/limit for large files.\n\n"
        "Binary files are rejected."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative or absolute path, subject to the current Workspace Access mode.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start from. Defaults to 1.",
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return. Defaults to 2000.",
                "minimum": 1,
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def run(
        self,
        path: str,
        offset: int = 1,
        limit: int = MAX_READ_LINES,
    ) -> ToolResult:
        try:
            target = self.workspace.resolve(path)
        except WorkspaceError as exc:
            return ToolResult.error(str(exc))

        if not target.exists():
            return ToolResult.error(f"File not found: {self.workspace.relative(target)}")
        if target.is_dir():
            return ToolResult.error(
                f"{self.workspace.relative(target)} is a directory; use list_dir instead."
            )

        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return ToolResult.error(
                f"File is too large to read ({size} bytes > {MAX_READ_BYTES}). "
                f"Read it with a shell command instead, e.g. `head -n 200 <path>`."
            )

        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.error(
                f"{self.workspace.relative(target)} appears to be binary; "
                f"it cannot be read as UTF-8 text."
            )
        except OSError as exc:
            return ToolResult.error(f"Could not read file: {exc}")

        lines = text.splitlines()
        total = len(lines)
        start = max(1, offset)
        window = lines[start - 1 : start - 1 + min(limit, MAX_READ_LINES)]

        if not window:
            return ToolResult.ok(
                f"{self.workspace.relative(target)} has {total} line(s); "
                f"offset {start} is past the end of the file."
            )

        body = "\n".join(f"{number:>6}\t{lines[number - 1]}" for number in
                         range(start, start + len(window)))
        shown_end = start + len(window) - 1

        notes = [f"{self.workspace.relative(target)} ({total} lines, showing "
                 f"{start}-{shown_end})"]
        if shown_end < total:
            notes.append(
                f"Truncated: {total - shown_end} more line(s). "
                f"Continue with offset={shown_end + 1}."
            )
        return ToolResult.ok("\n".join(notes) + "\n---\n" + body)


class WriteFileTool(Tool):
    instruction_path = "path"
    mutates_workspace = True
    name = "write_file"
    description = (
        "Write a file, creating or overwriting it.\n\n"
        "Parent directories are created automatically.\n\n"
        "Overwrites the whole file, so prefer edit_file for changes to existing code."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative or absolute path, subject to the current Workspace Access mode."},
            "content": {"type": "string", "description": "Full file contents to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def run(self, path: str, content: str) -> ToolResult:
        try:
            target = self.workspace.resolve(path)
        except WorkspaceError as exc:
            return ToolResult.error(str(exc))

        if target.is_dir():
            return ToolResult.error(
                f"{self.workspace.relative(target)} is a directory, not a file."
            )

        existed = target.exists()
        try:
            _atomic_write(self.workspace, target, content)
        except (OSError, UnicodeError) as exc:
            return ToolResult.error(f"Could not write file: {exc}")

        verb = "Updated" if existed else "Created"
        line_count = len(content.splitlines())
        return ToolResult.ok(
            f"{verb} {self.workspace.relative(target)} "
            f"({len(content.encode('utf-8'))} bytes, {line_count} line(s))."
        )


class EditFileTool(Tool):
    instruction_path = "path"
    mutates_workspace = True
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must match the file verbatim and must be "
        "unique unless replace_all is true; include surrounding lines to make it unique.\n\n"
        "Read the file first so old_string is exact.\n\n"
        "LF in copied text also matches CRLF; existing line endings are preserved outside the "
        "replacement.\n\n"
        "Omit read_file's line numbers.\n\n"
        "Alternatively supply edits=[{old_string,new_string}, ...] for several unique, "
        "non-overlapping replacements in this file.\n\n"
        "Every edit matches the ORIGINAL file; all are validated before one atomic write.\n\n"
        "Choose either edits or the single old_string/new_string pair, never both.\n\n"
        "On failure no edits are applied; nearby source is a suggestion only."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative or absolute path, subject to the current Workspace Access mode."},
            "old_string": {
                "type": "string",
                "description": "Exact text to replace, including indentation.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text. Use an empty string to delete.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring a unique match.",
            },
            "edits": {
                "type": "array", "minItems": 1, "maxItems": 100,
                "description": "Replacements matched against the original file. Merge overlapping targets.",
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["old_string", "new_string"],
                          "properties": {"old_string": {"type": "string", "minLength": 1},
                                         "new_string": {"type": "string"}}},
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def run(
        self,
        path: str,
        old_string: str | None = None,
        new_string: str | None = None,
        replace_all: bool = False,
        edits: list[dict[str, str]] | None = None,
    ) -> ToolResult:
        if edits is not None:
            if old_string is not None or new_string is not None or replace_all:
                return ToolResult.error("Use either edits or old_string/new_string/replace_all, never both.")
            if not 1 <= len(edits) <= 100:
                return ToolResult.error("edits must contain 1–100 replacements.")
            if any(set(edit) != {"old_string", "new_string"} or
                   not all(isinstance(value, str) for value in edit.values()) for edit in edits):
                return ToolResult.error("Each edit requires exactly old_string and new_string strings.")
            replacements = [(edit["old_string"], edit["new_string"], False) for edit in edits]
        else:
            if old_string is None or new_string is None:
                return ToolResult.error("missing required argument(s): old_string and new_string, or supply edits.")
            replacements = [(old_string, new_string, replace_all)]
        try:
            target = self.workspace.resolve(path)
        except WorkspaceError as exc:
            return ToolResult.error(str(exc))

        if not target.exists():
            return ToolResult.error(f"File not found: {self.workspace.relative(target)}")
        if target.is_dir():
            return ToolResult.error(f"{self.workspace.relative(target)} is a directory.")

        try:
            with target.open(encoding="utf-8", newline="") as source:
                text = source.read()
        except UnicodeDecodeError:
            return ToolResult.error(
                f"{self.workspace.relative(target)} is not UTF-8 text."
            )
        except OSError as exc:
            return ToolResult.error(f"Could not read file: {exc}")

        try:
            updated, replaced, first_line = apply_edits(text, replacements)
        except ValueError as exc:
            return ToolResult.error(f"{self.workspace.relative(target)}: {exc}\nNo changes were made.")

        try:
            _atomic_write(self.workspace, target, updated)
        except (OSError, UnicodeError) as exc:
            return ToolResult.error(f"Could not write file: {exc}")

        location = f" at line {first_line}" if first_line else ""
        return ToolResult.ok(
            f"Replaced {replaced} occurrence(s) in "
            f"{self.workspace.relative(target)}{location}.\n"
            + edit_diff(self.workspace.relative(target), text, updated)
        )


def _line_of(text: str, needle: str) -> int | None:
    index = text.find(needle)
    return None if index < 0 else text.count("\n", 0, index) + 1


def file_tools(workspace: Workspace) -> list[Tool]:
    return [ReadFileTool(workspace), WriteFileTool(workspace), EditFileTool(workspace)]
