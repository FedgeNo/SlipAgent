"""JSON input records, separate from the model's response/tool-call protocol.

Each API user message carries one object. Descriptions belong in the system
prompt; original user, reply, and tool text remains inside JSON string values.
"""

from __future__ import annotations

import json
from typing import Any

from .types import Message


def record_message(record: dict[str, Any]) -> Message:
    return Message.user(json.dumps(record, ensure_ascii=False))


def turn_record(messages: list[Message], *, current: bool = False,
                retained_prompt: str = "") -> dict[str, Any]:
    prompts = [message.content or "" for message in messages if message.role == "user"]
    calls = [call for message in messages for call in message.tool_calls or []]
    names = {call.id: call.name for call in calls}
    results = []
    for message in messages:
        if message.role != "tool":
            continue
        result = {"call_id": message.tool_call_id, "tool_name": names.get(message.tool_call_id or "", message.name),
                  "status": "unknown", "content": message.content}
        try:
            saved = json.loads(message.content or "")
        except ValueError:
            saved = None
        # Unwrap only the harness's exact observation envelope. Arbitrary JSON
        # returned by a tool stays its content, even when it contains these keys.
        if (isinstance(saved, dict) and saved.keys() == {"tool", "call_id", "status", "content"}
                and saved["call_id"] == message.tool_call_id
                and saved["tool"] == result["tool_name"]
                and saved["status"] in ("success", "error") and isinstance(saved["content"], str)):
            result.update(status=saved["status"], content=saved["content"])
        results.append(result)
    record: dict[str, Any] = {
        "record_type": "current_turn" if current else "history_turn",
        "representation": "full",
        "user_prompt": prompts or ([retained_prompt] if retained_prompt else []),
        "agent_response": next((message.content for message in messages if message.role == "assistant"), None),
        "tool_calls": [{"call_id": call.id, "tool_name": call.name, "arguments": call.arguments} for call in calls],
        "tool_results": results,
    }
    if current:
        record["continue_current_task"] = not prompts
    return record
