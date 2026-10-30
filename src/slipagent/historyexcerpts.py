"""Clip oversized fields in a history projection without removing its turn."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from .sources import FileText, json_chunks, text_chunks


def excerpt_fields(record: dict[str, Any], budget: int,
                   measure: Callable[[dict[str, Any]], int], *, results_only: bool = False) -> dict[str, Any] | None:
    """Prefer trimming tool content; retain all other fields that still fit.

    When the requested budget cannot hold even minimal metadata, return the
    smallest useful projection so the caller can reduce another large turn.
    """
    original_cost = measure(record)
    if original_cost <= budget:
        return None
    projected = copy.deepcopy(record)
    projected["representation"] = "excerpt"
    projected["excerpted_fields"] = []

    def candidates(value: Any, path: list[str | int]) -> list[tuple[int, list[str | int], Any]]:
        found: list[tuple[int, list[str | int], Any]] = []
        if isinstance(value, (str, FileText)) and len(value):
            size = sum(len(part.encode("utf-8")) for part in text_chunks(value))
            found.append((size, path, value))
        elif isinstance(value, dict):
            for key, child in value.items():
                found.extend(candidates(child, [*path, key]))
        elif isinstance(value, list) and value:
            # Lists of records/numbers can themselves be the oversized content.
            size = sum(len(part.encode("utf-8")) for part in json_chunks(value))
            found.append((size, path, value))
            for index, child in enumerate(value):
                found.extend(candidates(child, [*path, index]))
        return found

    def parent(path: list[str | int]) -> tuple[Any, str | int]:
        container: Any = projected
        for part in path[:-1]:
            container = container[part]
        return container, path[-1]

    def clipped(value: Any, keep: int, path: list[str | int]) -> dict[str, Any]:
        reference: dict[str, Any] = {"step_id": record["step_id"]}
        if path[0] == "tool_results":
            result = record["tool_results"][path[1]]
            wrapper = ["content"] if result["status"] in {"success", "error"} else []
            reference.update(call_id=result["call_id"], section="all", field_path=[*wrapper, *path[3:]])
        elif path[0] == "tool_calls":
            call = record["tool_calls"][path[1]]
            reference.update(call_id=call["call_id"], section="tool_calls",
                             field_path=[0, "function", "arguments", *path[3:]])
        else:
            section = {"user_prompts": "user", "agent_response": "response"}.get(str(path[0]), str(path[0]))
            reference.update(section=section, field_path=path[1:])
        left, right = (keep + 1) // 2, keep // 2
        if isinstance(value, list):
            return {"prefix_items": value[:left], "suffix_items": value[-right:] if right else [],
                    "original_items": len(value), "omitted_items": len(value) - keep, "original": reference}
        head, tail = "", ""
        for part in text_chunks(value):
            if len(head) < left:
                head += part[:left - len(head)]
            if right:
                tail = (tail + part)[-right:]
            if not right and len(head) >= left:
                break
        return {"prefix": head, "suffix": tail, "original_characters": len(value),
                "omitted_characters": len(value) - keep, "original": reference}

    groups = []
    results = []
    for index, result in enumerate(record.get("tool_results", [])):
        results.extend(candidates(result["content"], ["tool_results", index, "content"]))
    groups.append(results)
    for field in ("reasoning", "agent_response", "user_edits", "background_commands"):
        groups.append(candidates(record.get(field), [field]))
    arguments = []
    for index, call in enumerate(record.get("tool_calls", [])):
        arguments.extend(candidates(call.get("arguments"), ["tool_calls", index, "arguments"]))
    prompts = []
    for index, prompt in enumerate(record.get("user_prompts", [])):
        prompts.extend(candidates(prompt, ["user_prompts", index]))
    groups.extend([arguments, prompts])
    if results_only:
        groups = groups[:1]
    changed: list[list[str | int]] = []
    cost = original_cost
    for group in groups:
        for _, path, value in sorted(group, key=lambda item: item[0], reverse=True):
            if any(path[:len(prior)] == prior for prior in changed):
                continue
            container, key = parent(path)
            previous = container[key]
            container[key] = clipped(value, 0, path)
            projected["excerpted_fields"].append(path)
            minimal = measure(projected)
            if minimal >= cost:
                container[key] = previous
                projected["excerpted_fields"].pop()
                continue
            changed.append(path)
            if minimal <= budget:
                low, high = 0, len(value) - 1
                best = container[key]
                while low <= high:
                    middle = (low + high) // 2
                    container[key] = clipped(value, middle, path)
                    if measure(projected) <= budget:
                        best = container[key]
                        low = middle + 1
                    else:
                        high = middle - 1
                container[key] = best
                return projected
            cost = minimal
    return projected if changed else None
