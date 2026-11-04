"""Render raw message data as JSON input or labelled instruction sections."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterator

from .sources import FileText, TextSource, json_chunks, text_chunks
from .prompts import load_prompt


@dataclass
class RenderedContent(TextSource):
    value: Any

    def chunks(self) -> Iterator[str]:
        yield from content_chunks(self.value)


@dataclass
class RenderedJSON(TextSource):
    value: Any

    def chunks(self) -> Iterator[str]:
        yield from json_chunks(self.value)


@dataclass
class MessageSections:
    """Unserialized text and data sections of one outgoing system message."""

    parts: list[Any]


@dataclass
class JSONInput:
    """Raw input fields, with historical context before the current request."""

    data: dict[str, Any]


def render_content(value: Any) -> str:
    return "".join(content_chunks(value))


def content_chunks(value: Any) -> Iterator[str]:
    if isinstance(value, TextSource):
        yield from value.chunks()
    elif isinstance(value, JSONInput):
        groups: list[tuple[dict[str, Any], list[tuple[Any, int, int]]]] = []
        block_count = 0

        payload = dict(value.data)
        if "history" in payload:
            history = []
            for record in payload["history"]:
                projected = dict(record)
                if "tool_calls" not in record:
                    history.append(projected)
                    continue
                calls = []
                for call in record["tool_calls"]:
                    result = dict(call)
                    blocks = []
                    for key in ("result", "error", "unclassified_result", "result_excerpt", "error_excerpt", "unclassified_result_excerpt"):
                        if key not in call:
                            continue
                        child = call[key]
                        characters = len(child) if isinstance(child, (str, FileText)) else sum(map(len, json_chunks(child)))
                        block_count += 1
                        blocks.append((child, characters, block_count))
                        result[key] = {"tool_result_block": block_count, "characters": characters,
                                       "format": "text" if isinstance(child, (str, FileText)) else "json"}
                    calls.append({"tool_call_block": call["call_id"]})
                    groups.append((result, blocks))
                projected["tool_calls"] = calls
                history.append(projected)
            payload["history"] = history
        if "history" in payload:
            yield load_prompt("input-history.md").strip() + "\n\n"
            yield from json_chunks({"history": payload.pop("history")})
        for call, blocks in groups:
            # Keep arguments beside their observations without copying payloads
            # into the history metadata or escaping literal tool output.
            yield "\n\nTool Call Block:\n"
            yield from json_chunks(call)
            for text, characters, index in blocks:
                yield f"\n\nTool Result Block {index} ({characters} characters):\n"
                yield from text_chunks(text) if isinstance(text, (str, FileText)) else json_chunks(text)
        if payload:
            # Keep the actual request after every historical observation, even
            # when observations stream from disk and contain apparent headings.
            yield "\n\n" + load_prompt("input-current-request.md").strip() + "\n\n"
            yield from json_chunks(payload)
    elif isinstance(value, MessageSections):
        for part in value.parts:
            yield from content_chunks(part)
    elif value is not None:
        yield from text_chunks(value) if isinstance(value, (str, FileText)) else data_chunks(value)


def render_data(value: Any, indent: int = 0) -> str:
    return "".join(data_chunks(value, indent))


def data_chunks(value: Any, indent: int = 0) -> Iterator[str]:
    """Stream labelled records without assembling their text values."""
    pad = " " * indent
    if isinstance(value, (str, FileText)):
        yield f"text ({len(value)} characters):\n" + pad + "  "
        for chunk in text_chunks(value):
            yield chunk.replace("\n", "\n" + pad + "  ")
    elif isinstance(value, dict):
        yield f"object ({len(value)} fields):"
        for key, item in value.items():
            if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9-]*", key):
                yield "\n" + pad + "  " + key + ": "
            else:
                yield "\n" + pad + "  ? "
                yield from data_chunks(key, indent + 2)
                yield "\n" + pad + "  : "
            yield from data_chunks(item, indent + 2)
    elif isinstance(value, (list, tuple)):
        yield f"list ({len(value)} items):"
        for item in value:
            yield "\n" + pad + "  - "
            yield from data_chunks(item, indent + 2)
    elif value is None:
        yield "null"
    elif isinstance(value, bool):
        yield "true" if value else "false"
    elif isinstance(value, (int, float)):
        yield str(value)
    else:
        raise TypeError(f"Unsupported model-input value: {type(value).__name__}")
