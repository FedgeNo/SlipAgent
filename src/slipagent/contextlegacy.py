"""Compatibility readers for earlier inline summaries and bookkeeping echoes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .prompts import load_prompt
from .types import ToolCall, ToolSpec

SUMMARY_MAX_CHARS = 6_000
SUMMARY_START = "<slipagent_context>"
SUMMARY_END = "</slipagent_context>"
TOOL_SUMMARY_KEY = "_slipagent_context"
TOOL_PREVIOUS_KEY = "_slipagent_previous"
# Reserved labels from earlier context views, retained only to recognize echoes.
LEGACY_RESPONSE_HEADINGS = frozenset({
    "Agent Response (Full):", "Agent Response and Tool Calls (Full):",
})


@dataclass(slots=True)
class ResponseMemory:
    text: str
    summary: str | None = None
    previous: str | None = None


def memory_specs(specs: list[ToolSpec]) -> list[ToolSpec]:
    """Legacy schema adapter, retained for imported fixtures; not used by Agent."""
    result = []
    for spec in specs:
        properties = {**spec.parameters.get("properties", {}),
                      TOOL_SUMMARY_KEY: {"type": "string", "minLength": 1, "maxLength": SUMMARY_MAX_CHARS,
                                         "description": load_prompt("legacy-tool-summary.md")},
                      TOOL_PREVIOUS_KEY: {"type": "string", "maxLength": SUMMARY_MAX_CHARS,
                                          "description": load_prompt("legacy-tool-previous-summary.md")}}
        parameters = {**spec.parameters, "properties": properties,
                      "required": list(dict.fromkeys([*spec.parameters.get("required", []), TOOL_SUMMARY_KEY]))}
        result.append(ToolSpec(spec.name, spec.description, parameters))
    return result


def tool_response_memory(text: str, calls: list[ToolCall], step_id: int, previous_id: int | None) -> tuple[ResponseMemory, list[ToolCall]]:
    memory = response_memory(text, step_id, previous_id)
    cleaned = []
    for call in calls:
        summary = _valid_summary(call.arguments.get(TOOL_SUMMARY_KEY))
        previous = _valid_summary(call.arguments.get(TOOL_PREVIOUS_KEY))
        if memory.summary is None and summary is not None:
            memory.summary = summary
        if previous_id is not None and memory.previous is None and previous is not None:
            memory.previous = previous
        cleaned.append(ToolCall(call.id, call.name, {key: value for key, value in call.arguments.items()
                                                   if key not in {TOOL_SUMMARY_KEY, TOOL_PREVIOUS_KEY}}))
    return memory, cleaned


class VisibleStream:
    """Legacy prose-stream filter; current replies wait for action validation."""

    def __init__(self) -> None:
        self.content = ""
        self.emitted = 0

    def push(self, text: str) -> str:
        self.content += text
        markers = (SUMMARY_START, "[Harness context metadata]", "Previous step's current summary:")
        end = len(self.content)
        for marker in markers:
            index = self.content.find(marker)
            if index >= 0:
                end = min(end, index)
            for size in range(1, len(marker)):
                if self.content[:end].endswith(marker[:size]):
                    end = min(end, len(self.content[:end]) - size)
        end = len(self.content[:end].rstrip())
        chunk = self.content[self.emitted:end]
        self.emitted = max(self.emitted, end)
        return chunk

    def finish(self, visible: str) -> str:
        chunk = visible[self.emitted:] if visible.startswith(self.content[:self.emitted]) else ""
        self.emitted += len(chunk)
        return chunk


def response_memory(text: str, step_id: int, previous_id: int | None) -> ResponseMemory:
    """Hide the memory envelope and echoed bookkeeping; preserve prose/examples."""
    fallback = text
    # The summary itself may quote the marker. Try all candidates so a marker
    # inside a JSON string cannot hide or corrupt the actual outer envelope.
    for match in reversed(list(re.finditer(re.escape(SUMMARY_START), text))):
        start = match.start()
        fenced = False
        for line in text[:start].splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
        if fenced:
            continue
        body = text[match.end():].strip()
        if not body.startswith("{"):
            continue
        visible = text[:start].rstrip()
        if not body.endswith(SUMMARY_END):
            if SUMMARY_END not in body:
                fallback = visible
            continue
        fallback = visible
        try:
            raw = json.loads(body[:-len(SUMMARY_END)])
        except ValueError:
            continue
        if not isinstance(raw, dict) or type(raw.get("step")) is not int or raw["step"] != step_id:
            continue
        summary = _valid_summary(raw.get("summary"))
        previous = raw.get("previous")
        previous_summary = None
        if isinstance(previous, dict) and type(previous.get("step")) is int and previous["step"] == previous_id:
            previous_summary = _valid_summary(previous.get("summary"))
        return ResponseMemory(_visible_response(visible), summary, previous_summary)
    return ResponseMemory(_visible_response(fallback))


def _visible_response(text: str) -> str:
    """Remove reserved bookkeeping lines without hiding numbers in normal output."""
    lines = []
    fenced = False
    metadata = False
    changed = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            metadata = False
        elif not fenced:
            if stripped in LEGACY_RESPONSE_HEADINGS:
                changed = True
                continue
            if stripped == "[Harness context metadata]":
                metadata = True
                changed = True
                continue
            if stripped.startswith("Previous step's current summary:"):
                metadata = True
                changed = True
                continue
            if metadata and (not stripped or re.fullmatch(r"\d+|none", stripped)):
                changed = True
                continue
            metadata = False
        lines.append(line)
    return "".join(lines).strip() if changed else text


def _valid_summary(value: Any) -> str | None:
    if isinstance(value, str) and value.strip() and len(value) <= SUMMARY_MAX_CHARS:
        return value.strip()
    return None
