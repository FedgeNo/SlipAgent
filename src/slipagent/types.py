"""Wire-level types shared by the OpenRouter client and the agent loop.

These mirror the OpenRouter Chat Completions schema closely enough that
`to_api()` is a straight projection, but they stay independent of httpx so the
agent loop can be tested without a network client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Self

Role = Literal["system", "user", "assistant", "tool"]


def decode_json_content(value: Any) -> Any:
    """Decode structured text at an external boundary; preserve ordinary text."""
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def content_text(value: Any) -> str:
    """Render structured content only when a destination needs text."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@dataclass(slots=True)
class ToolCall:
    """A single function invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        function = raw.get("function") or {}
        return cls(
            id=raw.get("id", ""),
            name=function.get("name", ""),
            arguments=_decode_arguments(function.get("arguments")),
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, separators=(",", ":")),
            },
        }

    def to_record(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}

    def brief(self) -> str:
        """Short human-readable summary used by the CLI event renderer."""
        rendered = json.dumps(self.arguments, separators=(",", ": "))
        if len(rendered) > 120:
            rendered = rendered[:117] + "..."
        return f"{self.name}({rendered})"


def _decode_arguments(raw: Any) -> dict[str, Any]:
    """Parse a tool call's `arguments` field, tolerating provider quirks.

    The spec says this is a JSON-encoded string, but some providers emit an
    already-decoded object or an empty string when no arguments are required.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None or isinstance(raw, str) and not raw.strip():
        return {}
    if not isinstance(raw, str):
        return {"__invalid_arguments__": raw}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Preserve the bad payload so the error surfaces in the tool result
        # rather than being silently swallowed.
        return {"__invalid_arguments__": raw}
    return parsed if isinstance(parsed, dict) else {"__invalid_arguments__": raw}


@dataclass(slots=True)
class Message:
    """One entry in the conversation history."""

    role: Role
    content: Any = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    # Provider reasoning blocks/signatures belong to the originating model.
    # Keep them opaque when replaying full tool steps; never render signatures.
    reasoning_details: list[dict[str, Any]] | None = None
    reasoning_model: str | None = None
    # Readable reasoning from this response only, concatenated before archival.
    reasoning: str | None = None

    def __post_init__(self) -> None:
        if self.role == "tool":
            self.content = decode_json_content(self.content)

    @classmethod
    def system(cls, content: str) -> Self:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: Any) -> Self:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: list[ToolCall] | None = None) -> Self:
        return cls(role="assistant", content=content, tool_calls=tool_calls or None)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: Any) -> Self:
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    def to_api(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": content_text(self.content) if self.content is not None else None}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_api() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        if self.reasoning_details:
            payload["reasoning_details"] = self.reasoning_details
        elif self.reasoning:
            payload["reasoning"] = self.reasoning
        return payload

    def to_record(self) -> dict[str, Any]:
        payload = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_record() for call in self.tool_calls]
        for key in ("tool_call_id", "name", "reasoning_details", "reasoning"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        raw_calls = raw.get("tool_calls") or []
        return cls(
            role=raw.get("role", "assistant"),
            content=raw.get("content"),
            tool_calls=[ToolCall.from_api(call) for call in raw_calls] or None,
            tool_call_id=raw.get("tool_call_id"),
            name=raw.get("name"),
            reasoning_details=raw.get("reasoning_details"),
            reasoning=raw.get("reasoning"),
        )


@dataclass(slots=True)
class ToolSpec:
    """A function definition advertised to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class Usage:
    """Token accounting for one completion, including OpenRouter's cost figure."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> Self:
        raw = raw or {}
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
            cost=_optional_float(raw.get("cost")),
        )

    def __add__(self, other: Usage) -> Usage:
        total = _optional_float(self.cost)
        if other.cost is not None:
            total = (total or 0.0) + other.cost
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=total,
        )

    def summary(self) -> str:
        parts = [
            f"in={self.prompt_tokens}",
            f"out={self.completion_tokens}",
        ]
        if self.cost is not None:
            parts.append(f"cost=${self.cost:.6f}")
        return " ".join(parts)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(slots=True)
class Completion:
    """A parsed, non-streaming chat completion."""

    message: Message
    model: str
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    # A complete but malformed model message is a retryable protocol failure,
    # not a transport failure. Preserve usage and a bounded diagnosis for it.
    response_error: str | None = None
    response_excerpt: str = ""

    @property
    def text(self) -> str:
        return self.message.content or ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls or []

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(slots=True)
class ModelInfo:
    """A single entry from the OpenRouter model catalog."""

    id: str
    name: str | None = None
    context_length: int | None = None
    pricing: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        context = raw.get("context_length")
        return cls(
            id=raw.get("id", ""),
            name=raw.get("name"),
            context_length=int(context) if context else None,
            pricing=raw.get("pricing") or {},
        )


@dataclass(slots=True)
class FreeQuota:
    """Daily allowance for zero-cost models."""

    used: int = 0
    limit: int = 0
    remaining: int = 0

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> FreeQuota | None:
        if not isinstance(raw, dict):
            return None
        return cls(
            used=int(raw.get("used") or 0),
            limit=int(raw.get("limit") or 0),
            remaining=int(raw.get("remaining") or 0),
        )


@dataclass(slots=True)
class KeyInfo:
    """Identity, spend limit, and quota for an OpenRouter API key.

    Fetched from `GET /api/v1/key`, which — unlike `GET /api/v1/models` —
    actually authenticates. Used both to verify a key and to warn when the key
    cannot reach paid models.
    """

    label: str = ""
    limit: float | None = None
    limit_remaining: float | None = None
    usage: float = 0.0
    is_free_tier: bool = False
    free_quota: FreeQuota | None = None
    expires_at: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        limit = raw.get("limit")
        remaining = raw.get("limit_remaining")
        return cls(
            label=raw.get("label") or "",
            limit=None if limit is None else float(limit),
            limit_remaining=None if remaining is None else float(remaining),
            usage=float(raw.get("usage") or 0.0),
            is_free_tier=bool(raw.get("is_free_tier")),
            free_quota=FreeQuota.from_api(raw.get("free_model_daily_requests")),
            expires_at=raw.get("expires_at"),
        )

    @property
    def cannot_reach_paid_models(self) -> bool:
        """True when the spend limit blocks everything above free tier."""
        return self.limit is not None and self.limit <= 0 and not self.is_free_tier
