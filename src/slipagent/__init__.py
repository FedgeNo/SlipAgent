"""SlipAgent: an agentic coding harness with selectable API providers."""

from __future__ import annotations

__version__ = "0.2.1"

from .agent import Agent, AgentEvent, build_system_prompt
from .api import APIClient, APIError
from .providers import create_client
from .nvidia import NvidiaClient
from .capabilities import RequestProfile
from .config import Config, ConfigError
from .mcp import MCPClient, MCPError, MCPManager, MCPTool, ServerSpec
from .openrouter import (
    OpenRouterAPIError,
    OpenRouterAuthError,
    OpenRouterClient,
    OpenRouterConfigError,
    OpenRouterError,
    OpenRouterLimitError,
    RetryPolicy,
)
from .tools import Tool, ToolRegistry, ToolResult, build_default_registry
from .types import Completion, Message, ModelInfo, ToolCall, ToolSpec, Usage
from .workspace import Workspace, WorkspaceError

__all__ = [
    "APIClient", "APIError", "create_client", "NvidiaClient", "RequestProfile",
    "Agent",
    "AgentEvent",
    "Completion",
    "Config",
    "ConfigError",
    "Message",
    "MCPClient",
    "MCPError",
    "MCPManager",
    "MCPTool",
    "ModelInfo",
    "OpenRouterAPIError",
    "OpenRouterAuthError",
    "OpenRouterClient",
    "OpenRouterConfigError",
    "OpenRouterError",
    "OpenRouterLimitError",
    "RetryPolicy",
    "ServerSpec",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "Workspace",
    "WorkspaceError",
    "__version__",
    "build_default_registry",
    "build_system_prompt",
]
