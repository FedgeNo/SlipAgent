"""Structured input records, separate from the model's response/tool-call protocol.

The user-input JSON carries new text and selected history, with tool outcomes
attached to their calls. Saved originals retain their roles and separate results.
Values remain structured until the outgoing message is rendered.
"""

from __future__ import annotations

from typing import Any

from .types import Message


def linked_tool_results(record: dict[str, Any]) -> dict[str, Any]:
    """Project each observation under its call without altering saved records."""
    if record.get("representation") == "excerpt":
        projected = {key: value for key, value in record.items() if key != "messages"}
        calls = []
        responses = []
        for entry in record["messages"]:
            if entry["role"] == "tool":
                key = {"success": "result_excerpt", "error": "error_excerpt"}.get(entry["status"], "unclassified_result_excerpt")
                calls.append({"call_id": entry["call_id"], "tool_name": entry["tool_name"], key: entry["content_excerpt"]})
            else:
                responses.append(entry)
        projected.update(tool_calls=calls, response_excerpts=responses)
        return projected
    if "tool_results" not in record:
        return record
    projected = {key: value for key, value in record.items() if key != "tool_results"}
    calls = [dict(call) for call in record.get("tool_calls", [])]
    for result in record["tool_results"]:
        call = next((call for call in calls if call["call_id"] == result["call_id"]), None)
        if call is None:
            call = {"call_id": result["call_id"], "tool_name": result["tool_name"]}
            calls.append(call)
        key = {"success": "result", "error": "error"}.get(result["status"], "unclassified_result")
        call[key] = result["content"]
    projected["tool_calls"] = calls
    return projected


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
