"""SlipAgent: an agentic coding harness driven by OpenRouter models."""

from __future__ import annotations

from .agent import Agent, AgentEvent, build_system_prompt
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

__version__ = "0.2.0"

__all__ = [
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
