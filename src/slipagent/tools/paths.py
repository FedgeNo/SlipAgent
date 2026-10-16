"""Dedicated filesystem operations with explicit targets and change records."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..prompts import load_prompt
from ..workspace import Workspace, WorkspaceError
from .base import Tool, ToolResult
from .blocking import run_blocking


class PathTool(Tool):
    mutates_workspace = True
    instruction_path = "path"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def target(self, raw: str) -> Path:
        target = self.workspace.resolve(raw)
        lexical = Path(raw).expanduser()
        if not lexical.is_absolute():
            lexical = self.workspace.root / lexical
        if any(part.is_symlink() for part in (lexical, *lexical.parents)):
            raise ValueError("Symlink paths are not supported by this operation.")
        if target == self.workspace.root or target == Path(target.anchor):
            raise ValueError("This operation cannot target the workspace or filesystem root.")
        return target

    def entries(self, target: Path) -> list[Path]:
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        entries = [target]
        if target.is_dir():
            for root, directories, files in os.walk(target, onerror=self.walk_error):
                entries.extend(Path(root) / name for name in directories + files)
        for entry in entries:
            self.target(str(entry))
            if not entry.is_dir() and not entry.is_file():
                raise ValueError(f"Not a regular file or directory: {entry}")
        return entries

    @staticmethod
    def walk_error(error: OSError) -> None:
        raise error

    def instruction_targets(self, arguments: dict[str, Any]) -> list[str]:
        targets = [arguments[key] for key in ("path", "source", "destination")
                   if isinstance(arguments.get(key), str)]
        source = arguments.get("source", arguments.get("path"))
        if isinstance(source, str):
            target = self.target(source)
            if target.is_dir():
                entries = self.entries(target)
                targets.extend(str(entry) for entry in entries)
                destination = arguments.get("destination")
                if isinstance(destination, str):
                    dest = self.target(destination)
                    targets.extend(str(dest / entry.relative_to(target)) for entry in entries)
        return targets

    def operate(self, **arguments: Any) -> ToolResult:
        changes: list[dict[str, str]] = []
        try:
            self.execute(changes, **arguments)
        except (OSError, ValueError, WorkspaceError) as exc:
            return ToolResult.error({"operation": self.name, "changes": changes, "error": str(exc)})
        return ToolResult.ok({"operation": self.name, "changes": changes})

    def execute(self, changes: list[dict[str, str]], **arguments: Any) -> None:
        raise NotImplementedError

    def changed(self, changes: list[dict[str, str]], path: Path, action: str) -> None:
        changes.append({"path": self.workspace.relative(path), "action": action,
                        "kind": "directory" if path.is_dir() else "file"})


class CopyPathTool(PathTool):
    name = "copy_path"
    description_prompt = "tools/copy-path.md"
    description = load_prompt(description_prompt)
    parameter_prompts = "tools/transfer-path-parameters.json"
    parameters = {"type": "object", "properties": {
        "source": {"type": "string"}, "destination": {"type": "string"}},
        "required": ["source", "destination"]}

    async def run(self, source: str, destination: str) -> ToolResult:
        return await run_blocking(self.operate, source=source, destination=destination)

    def execute(self, changes: list[dict[str, str]], **arguments: Any) -> None:
        source, destination = self.target(arguments["source"]), self.target(arguments["destination"])
        entries = self.entries(source)
        if destination.exists():
            raise FileExistsError("Destination already exists; no replacement or directory merging is performed.")
        if destination.is_relative_to(source):
            raise ValueError("Destination cannot be inside the source.")
        if not destination.parent.is_dir():
            raise FileNotFoundError("Destination parent is missing; use create_directory first.")
        for entry in entries:
            target = self.target(str(destination / entry.relative_to(source)))
            self.target(str(entry))
            if entry.is_dir():
                target.mkdir()
                self.changed(changes, target, "created")
            else:
                with entry.open("rb") as reader, target.open("xb") as writer:
                    self.changed(changes, target, "created")
                    shutil.copyfileobj(reader, writer)
                shutil.copymode(entry, target)


class MovePathTool(CopyPathTool):
    name = "move_path"
    description_prompt = "tools/move-path.md"
    description = load_prompt(description_prompt)

    def execute(self, changes: list[dict[str, str]], **arguments: Any) -> None:
        source, destination = self.target(arguments["source"]), self.target(arguments["destination"])
        entries = self.entries(source)
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Source and destination cannot contain one another.")
        # Rename keeps a same-filesystem move indivisible; cross-device moves fail.
        if destination.exists():
            raise FileExistsError("Destination already exists; no replacement or directory merging is performed.")
        source.rename(destination)
        for entry in entries:
            target = destination / entry.relative_to(source)
            kind = "directory" if target.is_dir() else "file"
            changes.extend([{"path": self.workspace.relative(entry), "action": "moved_from", "kind": kind},
                            {"path": self.workspace.relative(target), "action": "moved_to", "kind": kind}])


class DeletePathTool(PathTool):
    name = "delete_path"
    description_prompt = "tools/delete-path.md"
    description = load_prompt(description_prompt)
    parameter_prompts = "tools/delete-path-parameters.json"
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]}

    async def run(self, path: str, recursive: bool = False) -> ToolResult:
        return await run_blocking(self.operate, path=path, recursive=recursive)

    def execute(self, changes: list[dict[str, str]], **arguments: Any) -> None:
        target = self.target(arguments["path"])
        entries = self.entries(target)
        if len(entries) > 1 and not arguments.get("recursive", False):
            raise ValueError("Nonempty directory; recursive=true is required.")
        for entry in reversed(entries):
            self.target(str(entry))
            kind = "directory" if entry.is_dir() else "file"
            entry.rmdir() if kind == "directory" else entry.unlink()
            changes.append({"path": self.workspace.relative(entry), "action": "deleted", "kind": kind})


class CreateDirectoryTool(PathTool):
    name = "create_directory"
    description_prompt = "tools/create-directory.md"
    description = load_prompt(description_prompt)
    parameter_prompts = "tools/create-directory-parameters.json"
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"}, "parents": {"type": "boolean"}}, "required": ["path"]}

    async def run(self, path: str, parents: bool = False) -> ToolResult:
        return await run_blocking(self.operate, path=path, parents=parents)

    def execute(self, changes: list[dict[str, str]], **arguments: Any) -> None:
        target = self.target(arguments["path"])
        missing = [target]
        if arguments.get("parents", False):
            parent = target.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
        for directory in reversed(missing):
            self.target(str(directory)).mkdir()
            self.changed(changes, directory, "created")


def path_tools(workspace: Workspace) -> list[Tool]:
    return [CopyPathTool(workspace), MovePathTool(workspace), DeletePathTool(workspace), CreateDirectoryTool(workspace)]
