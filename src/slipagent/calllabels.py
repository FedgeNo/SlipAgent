"""Optional plain-language labels the model attaches to its tool calls.

Each advertised tool accepts an extra `call_label` argument. The label stays in
the recorded call for display and history; tools never receive it.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .prompts import load_prompt
from .types import ToolCall, ToolSpec

LABEL = "call_label"
# The answer tool already speaks to the user; a label would only repeat it.
UNLABELED_TOOLS = frozenset({"answer"})


def labeled_spec(spec: ToolSpec) -> ToolSpec:
    """The spec with an optional `call_label`, unless the tool defines its own."""
    properties = spec.parameters.get("properties")
    if spec.name in UNLABELED_TOOLS or not isinstance(properties, dict) or LABEL in properties:
        return spec
    parameters = copy.copy(spec.parameters)
    parameters["properties"] = {**properties, LABEL: {
        "type": "string", "description": load_prompt("call-label.md").strip()}}
    return ToolSpec(spec.name, spec.description, parameters)


def tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Arguments as the tool itself receives them."""
    if LABEL not in arguments:
        return arguments
    return {key: value for key, value in arguments.items() if key != LABEL}


def call_label(call: ToolCall) -> str:
    label = call.arguments.get(LABEL)
    return " ".join(label.split()) if isinstance(label, str) else ""


def raw_call(call: ToolCall, *, limit: int = 120) -> str:
    """The exact call, without its label, compact enough for one feed line."""
    rendered = json.dumps(tool_arguments(call.arguments), separators=(",", ": "))
    if len(rendered) > limit:
        rendered = rendered[:limit - 1] + "…"
    return f"{call.name}({rendered})"
