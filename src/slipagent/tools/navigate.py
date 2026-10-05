"""Navigation tool: list the contents of a workspace directory."""

from __future__ import annotations

from ..workspace import Workspace, WorkspaceError
from .base import Tool, ToolResult

MAX_ENTRIES = 300


class ListDirTool(Tool):
    concurrent_safe = True
    instruction_path = "path"
    name = "list_dir"
    description = (
        "List a directory allowed by the current Workspace Access mode.\n\n"
        "Directories are listed first and marked with a trailing '/'.\n\n"
        "Use this to orient yourself before glob, grep, or read_file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list. Defaults to the workspace root.",
            },
        },
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def run(self, path: str | None = None) -> ToolResult:
        try:
            target = self.workspace.resolve(path) if path else self.workspace.root
        except WorkspaceError as exc:
            return ToolResult.error(str(exc))

        if not target.exists():
            return ToolResult.error(f"Directory not found: {self.workspace.relative(target)}")
        if not target.is_dir():
            return ToolResult.error(
                f"{self.workspace.relative(target)} is a file; use read_file instead."
            )

        try:
            entries = sorted(
                target.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError as exc:
            return ToolResult.error(f"Could not list directory: {exc}")

        if not entries:
            return ToolResult.ok(f"{self.workspace.relative(target)} is empty.")

        directories = [e for e in entries if e.is_dir()]
        files = [e for e in entries if not e.is_dir()]

        lines = [
            f"{self.workspace.relative(entry)}/" if is_dir else self.workspace.relative(entry)
            for entry, is_dir in [(d, True) for d in directories] + [(f, False) for f in files]
        ]

        shown = lines[:MAX_ENTRIES]
        header = (
            f"{self.workspace.relative(target)}: "
            f"{len(directories)} director(ies), {len(files)} file(s)"
        )
        if len(lines) > len(shown):
            header += f" (showing first {len(shown)})"
        return ToolResult.ok(header + "\n" + "\n".join(shown))


def navigate_tools(workspace: Workspace) -> list[Tool]:
    return [ListDirTool(workspace)]
