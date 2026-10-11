"""Agent events shared by orchestration, request execution, and presentation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .tools.base import ToolResult
from .types import ToolCall, Usage

EventKind = Literal[
    "context",
    "step_start",
    "assistant_text",
    "suggestion",
    "assistant_delta",
    "reasoning_delta",
    "stream_end",
    "retry",
    "retry_wait",
    "user_message",
    "user_message_sent",
    "user_queue_reset",
    "tool_start",
    "tool_output",
    "tool_end",
    "artifact",
    "step_end",
    "warning",
    "notice",
    "error",
]


@dataclass(slots=True)
class AgentEvent:
    kind: EventKind
    step: int = 0
    text: str = ""
    tool_call: ToolCall | None = None
    result: ToolResult | None = None
    usage: Usage | None = None


EventHandler = Callable[[AgentEvent], None]
