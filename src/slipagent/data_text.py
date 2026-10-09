"""Render raw message data as JSON input or labelled instruction sections."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class MessageSections:
    """Unserialized text and data sections of one outgoing system message."""

    parts: list[Any]


@dataclass
class JSONInput:
    """Raw user-input fields, encoded together only at the message boundary."""

    data: dict[str, Any]


def render_content(value: Any) -> str:
    if isinstance(value, JSONInput):
        return json.dumps(value.data, ensure_ascii=False)
    if isinstance(value, MessageSections):
        return "".join(render_content(part) for part in value.parts)
    if value is None:
        return ""
    return value if isinstance(value, str) else render_data(value)


def render_data(value: Any, indent: int = 0) -> str:
    """Use labelled, indented records with literal text blocks.

    Text is never parsed as JSON or interpreted as escape sequences. Added
    indentation belongs to the presentation; original values remain untouched.
    """
    pad = " " * indent
    if isinstance(value, str):
        # Length makes trailing newlines and embedded record-like text unambiguous.
        return f"text ({len(value)} characters):\n" + "\n".join(
            pad + "  " + line for line in value.split("\n")
        )
    if isinstance(value, dict):
        lines = [f"object ({len(value)} fields):"]
        for key, item in value.items():
            if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9-]*", key):
                lines.append(pad + "  " + key + ": " + render_data(item, indent + 2))
            else:
                lines.append(pad + "  ? " + render_data(key, indent + 2))
                lines.append(pad + "  : " + render_data(item, indent + 2))
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        return f"list ({len(value)} items):" + "".join(
            "\n" + pad + "  - " + render_data(item, indent + 2) for item in value
        )
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    raise TypeError(f"Unsupported model-input value: {type(value).__name__}")
