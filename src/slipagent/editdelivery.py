"""Bounded saved-edit observations and revision-specific delivery receipts."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from typing import Any

from .data_text import JSONInput
from .types import Message

EDIT_PREVIEW_CHARS = 2048


def revision(entry: dict[str, Any]) -> str:
    value = entry.get("revision")
    if isinstance(value, str):
        return value
    # Older journals retain the full diff without a revision identity.
    return "legacy-" + hashlib.sha256(entry.get("diff", "").encode("utf-8")).hexdigest()


def notifications(context: list[Message]) -> list[dict[str, Any]]:
    for message in reversed(context):
        if isinstance(message.content, JSONInput):
            return copy.deepcopy(message.content.data.get("user_edits", []))
    return []


def bounded_notifications(entries: list[dict[str, Any]], budget: int,
                          measure: Callable[[list[dict[str, Any]], int], int]) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    previews: list[str] = []
    for entry in entries:
        preview = entry.get("diff_preview", entry.get("diff", ""))[:EDIT_PREVIEW_CHARS]
        total = entry.get("diff_characters", len(entry.get("diff", preview)))
        record = {"edit_id": entry.get("edit_id", entry["path"]), "revision": revision(entry),
                  "path": entry["path"], "source": "user", "saved_before_step": entry.get("saved_before_step"),
                  "diff": "", "diff_characters": total, "diff_truncated": bool(total), "next_offset": 0}
        artifact = entry.get("artifact")
        if isinstance(artifact, dict):
            record["artifact"] = {key: artifact[key] for key in ("record_id", "kind", "step_id", "call_id") if key in artifact}
        elif total:
            record["archive_unavailable"] = True
        if entry.get("reverted_to_editor_baseline"):
            record["reverted_to_editor_baseline"] = True
        proposed = [*selected, record]
        if measure(proposed, len(entries) - len(proposed)) > budget:
            break
        selected.append(record)
        previews.append(preview)
    omitted = len(entries) - len(selected)
    for index, preview in enumerate(previews):
        available = max(0, budget - measure(selected, omitted)) // (len(selected) - index)
        limit = measure(selected, omitted) + available
        low, high = 0, len(preview)
        best = dict(selected[index])
        while low <= high:
            middle = (low + high) // 2
            selected[index].update(diff=preview[:middle], next_offset=middle,
                                   diff_truncated=middle < selected[index]["diff_characters"])
            if measure(selected, omitted) <= limit:
                best = dict(selected[index])
                low = middle + 1
            else:
                high = middle - 1
        selected[index] = best
    return selected, omitted


def acknowledge(pending: dict[str, Any], delivered: list[dict[str, Any]]) -> None:
    for notice in delivered:
        entry = pending.get(notice["edit_id"])
        if entry is not None and revision(entry) == notice["revision"]:
            # Failed archival cannot retire the only complete retained diff.
            if notice.get("archive_unavailable") and notice["diff_truncated"]:
                continue
            entry["delivered"] = True


def validate_notifications(value: Any) -> None:
    if not isinstance(value, list) or any(
        not isinstance(entry, dict) or entry.get("source") != "user"
        or not all(isinstance(entry.get(key), str) for key in ("edit_id", "revision", "path", "diff"))
        or type(entry.get("diff_characters")) is not int or entry["diff_characters"] < len(entry["diff"])
        or type(entry.get("diff_truncated")) is not bool
        or entry["diff_truncated"] != (len(entry["diff"]) < entry["diff_characters"])
        or type(entry.get("next_offset")) is not int or entry["next_offset"] != len(entry["diff"])
        or ("artifact" in entry and (not isinstance(entry["artifact"], dict)
            or type(entry["artifact"].get("record_id")) is not int or entry["artifact"]["record_id"] < 1))
        for entry in value
    ):
        raise ValueError("Invalid saved-edit delivery record")
