"""Saved-diff references retained inside ordinary queued user messages."""

from __future__ import annotations

import json
from typing import Any

from .types import Message

REFERENCE_MARKER = "\n\nAttached saved diff reference:\n"
REFERENCE_KEYS = {"record_id", "step_id", "call_id", "command"}


def diff_reference(record: dict[str, Any]) -> dict[str, Any]:
    if record["kind"] != "diff":
        raise ValueError("Only saved diffs accept diff comments.")
    return {"record_id": record["record_id"], "step_id": record["step_id"],
            "call_id": record["call_id"], "command": record["command"][:500]}


def attach_reference(comment: str, reference: dict[str, Any]) -> str:
    # Plain text follows normal queue, journal, resume and task-retention paths.
    return comment + REFERENCE_MARKER + json.dumps(reference, ensure_ascii=False)


def read_reference(text: str) -> dict[str, Any] | None:
    comment, marker, data = text.rpartition(REFERENCE_MARKER)
    if not marker or not comment.strip():
        return None
    try:
        reference = json.loads(data)
    except (ValueError, RecursionError):
        return None
    if (not isinstance(reference, dict) or reference.keys() != REFERENCE_KEYS
            or type(reference["record_id"]) is not int or reference["record_id"] < 1
            or reference["step_id"] is not None and (type(reference["step_id"]) is not int or reference["step_id"] < 1)
            or reference["call_id"] is not None and not isinstance(reference["call_id"], str)
            or not isinstance(reference["command"], str) or len(reference["command"]) > 500):
        return None
    return reference


def current_references(messages: list[Message]) -> list[dict[str, Any]]:
    """Use the latest user-input group until another user turn supersedes it."""
    references = []
    found_user = False
    for message in reversed(messages):
        if message.role != "user":
            if found_user:
                break
            continue
        found_user = True
        if isinstance(message.content, str):
            reference = read_reference(message.content)
            if reference is not None and reference not in references:
                references.append(reference)
    return list(reversed(references))
