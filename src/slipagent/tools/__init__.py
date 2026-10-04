"""Built-in tools and the default registry builder."""

from __future__ import annotations

from typing import Any

from ..workspace import Workspace
from ..environment import ProjectEnvironment, load_project_settings
from .base import Tool, ToolRegistry, ToolResult, validate_arguments
from .files import EditFileTool, ReadFileTool, WriteFileTool, file_tools
from .git import GitAddTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool, git_tools
from .navigate import ListDirTool, navigate_tools
from .search import GlobTool, GrepTool, search_tools
from .shell import RunCommandTool, shell_tools
from .web import FetchPageTool, WebSearchTool, web_tools
from .output import CommandArchive, ReadCommandOutputTool

__all__ = [
    "EditFileTool",
    "FetchPageTool",
    "GlobTool",
    "GitAddTool",
    "GitCommitTool",
    "GitDiffTool",
    "GitLogTool",
    "GitStatusTool",
    "GrepTool",
    "ListDirTool",
    "ReadFileTool",
    "RunCommandTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "WebSearchTool",
    "WriteFileTool",
    "build_default_registry",
    "file_tools",
    "git_tools",
    "navigate_tools",
    "search_tools",
    "shell_tools",
    "validate_arguments",
    "web_tools",
]


def build_default_registry(
    workspace: Workspace, exa_api_key: str | None = None, *, services: dict[str, Any] | None = None
) -> ToolRegistry:
    """Every tool the harness ships with, bound to `workspace`.

    Web tools are always registered so the model can see them; `web_search`
    reports a setup error at call time if no EXA_API_KEY is present.
    Passing services borrows live session owners during reload. Only initial
    construction creates the environment selector and command archive.
    """
    owns_services = services is None
    if services is None:
        settings = load_project_settings(workspace)
        services = {"project_environment": ProjectEnvironment(workspace),
                    "command_archive": CommandArchive(settings.log_quota_bytes)}
    archive = services["command_archive"]
    return ToolRegistry(
        [
            *file_tools(workspace),
            *search_tools(workspace),
            *navigate_tools(workspace),
            *shell_tools(workspace, archive),
            *git_tools(workspace, archive),
            ReadCommandOutputTool(archive),
            *web_tools(exa_api_key),
        ], services=services, owns_services=owns_services,
    )
