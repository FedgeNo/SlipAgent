"""Built-in tools and the default registry builder."""

from __future__ import annotations

from typing import Any

from ..workspace import Workspace
from ..instructions import ProjectInstructions
from ..environment import ProjectEnvironment, load_project_settings, load_project_options
from .base import Tool, ToolRegistry, ToolResult, validate_arguments
from .answer import AnswerTool
from .files import EditFileTool, ReadFileTool, WriteFileTool, file_tools
from .git import GitAddTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool, git_tools
from .navigate import ListDirTool, navigate_tools
from .search import GlobTool, GrepTool, search_tools
from .shell import RunCommandTool, shell_tools
from .web import FetchPageTool, WebSearchTool, web_tools
from .output import CommandArchive, ReadCommandOutputTool

__all__ = [
    "AnswerTool",
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
    """Build workspace tools; Agent attaches history and optional planning tools.

    Web tools are always registered so the model can see them; `web_search`
    reports a setup error at call time if no EXA_API_KEY is present.
    Passing services borrows live session owners during reload. Only initial
    construction creates the environment selector and command archive.
    Code navigation is included only when language servers are configured.
    """
    from ..jobs import CommandJobs, CommandJobsTool
    from ..diagnostics import RequestDiagnostics
    from ..lsp import LanguageServers, NavigateCodeTool
    owns_services = services is None
    if services is None:
        settings = load_project_settings(workspace)
        services = {"workspace": workspace,
                    "project_environment": ProjectEnvironment(workspace),
                    "project_instructions": ProjectInstructions(workspace),
                    "command_archive": CommandArchive(settings.log_quota_bytes),
                    "command_jobs": CommandJobs(),
                    "request_diagnostics": RequestDiagnostics(),
                    "language_servers": LanguageServers(workspace)}
    archive = services["command_archive"]
    registry = ToolRegistry(
        [
            AnswerTool(),
            *file_tools(workspace),
            *search_tools(workspace),
            *navigate_tools(workspace),
            *shell_tools(workspace, archive, services.get("command_jobs")),
            *([CommandJobsTool(services["command_jobs"])] if "command_jobs" in services else []),
            *([NavigateCodeTool(services["language_servers"])]
              if "language_servers" in services and load_project_options(workspace).get("language_servers") else []),
            *git_tools(workspace, archive),
            ReadCommandOutputTool(archive),
            *web_tools(exa_api_key),
        ], services=services, owns_services=owns_services,
    )
    registry.builtin_names = frozenset(registry.names)
    return registry
