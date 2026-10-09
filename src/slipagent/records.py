"""Structured input records, separate from the model's response/tool-call protocol.

The outgoing system prompt carries current input before selected completed-step
objects; user messages carry plain input or an empty string. Originals retain their
roles in storage. Text fields render literally at the API boundary.
"""

from __future__ import annotations

from typing import Any

from .types import Message


def record_message(record: dict[str, Any]) -> Message:
    return Message.user(record)


def step_record(messages: list[Message], *, current: bool = False,
                retained_prompt: str = "", step_id: int | None = None) -> dict[str, Any]:
    prompts = [message.content or "" for message in messages if message.role == "user"]
    calls = [call for message in messages for call in message.tool_calls or []]
    names = {call.id: call.name for call in calls}
    results = []
    for message in messages:
        if message.role != "tool":
            continue
        result = {"call_id": message.tool_call_id, "tool_name": names.get(message.tool_call_id or "", message.name),
                  "status": "unknown", "content": message.content}
        saved = message.content
        # Unwrap only the harness's exact observation envelope. Arbitrary JSON
        # returned by a tool stays its content, even when it contains these keys.
        if (isinstance(saved, dict) and saved.keys() == {"tool", "call_id", "status", "content"}
                and saved["call_id"] == message.tool_call_id
                and saved["tool"] == result["tool_name"]
                and saved["status"] in ("success", "error")):
            result.update(status=saved["status"], content=saved["content"])
        results.append(result)
    record: dict[str, Any] = {
        "record_type": "current_step" if current else "history_step",
        "representation": "full",
        "user_prompt": prompts or ([retained_prompt] if retained_prompt else []),
        "agent_response": next((message.content for message in messages if message.role == "assistant"), None),
        "tool_calls": [{"call_id": call.id, "tool_name": call.name, "arguments": call.arguments} for call in calls],
        "tool_results": results,
    }
    if current:
        # This marks the absence of a new user message, not unfinished work.
        record["is_tool_result_response"] = not prompts
    if step_id is not None:
        record["step_id"] = step_id
    return record
