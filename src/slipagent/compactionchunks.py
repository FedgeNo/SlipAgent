"""Disk-backed structural chunks and original-text seams for oversized turns."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterator
from pathlib import Path

from .context import message_tokens
from .prompts import load_prompt
from .sources import FileText, TextSource, json_chunks, spool_directory, text_chunks, write_indexed_json
from .types import Message

OVERLAP_CHARACTERS = 1024
SHARED_REQUEST_CHARACTERS = 2048


def prefix(value: str | TextSource, limit: int) -> str:
    parts = []
    for part in text_chunks(value):
        parts.append(part[:limit])
        limit -= len(parts[-1])
        if limit <= 0:
            break
    return "".join(parts)


def edge(value: Any, *, last: bool) -> str:
    """Bounded original JSON context; never materialize the full neighboring chunk."""
    result = ""
    for part in json_chunks(value):
        result = (result + part)[-OVERLAP_CHARACTERS:] if last else (result + part)[:OVERLAP_CHARACTERS]
        if not last and len(result) == OVERLAP_CHARACTERS:
            break
    return result


@dataclass
class ChunkPlan:
    directory: tempfile.TemporaryDirectory[str]
    shared: dict[str, Any]
    chunks: list[dict[str, Any]]

    def close(self) -> None:
        self.directory.cleanup()


def chunk_messages(shared: dict[str, Any], records: Any, index: int, count: int) -> list[Message]:
    return [Message.system(load_prompt("background-summary-prompt.md") + load_prompt("compaction-chunk.md")),
            Message.user({"phase": "chunk", "shared_context": shared, "chunk_index": index,
                          "chunk_count": count, "records": records})]


def plan_chunks(step_id: int, source: dict[str, Any], budget: int) -> ChunkPlan:
    """Keep call/result pairs intact when possible; split oversized structures losslessly."""
    directory = tempfile.TemporaryDirectory(prefix="compaction-chunks-", dir=spool_directory())
    request = source["user_prompt"]
    shared = {"step_id": step_id, "user_request": prefix(request, SHARED_REQUEST_CHARACTERS),
              "user_request_truncated": len(request) > SHARED_REQUEST_CHARACTERS,
              "original": {"tool": "recall_history", "step_id": step_id},
              "contextual_history_omitted": source.get("history_omitted", 0) + len(source.get("history", []))}
    plan = ChunkPlan(directory, shared, [])

    def fits(records: list[dict[str, Any]]) -> bool:
        # Reserve room for final indices before the number of chunks is known.
        return message_tokens(chunk_messages(shared, records, 1_000_000_000, 1_000_000_000)) <= budget

    def split(path: list[str | int], value: Any, call: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        record = {"path": path, "value": value}
        if call is not None:
            record["tool_identity"] = call
        if fits([record]):
            yield record
        elif isinstance(value, dict) and value:
            for key, child in value.items():
                yield from split([*path, key], child, call)
        elif isinstance(value, (list, tuple)) and value:
            for index, child in enumerate(value):
                yield from split([*path, index], child, call)
        elif isinstance(value, (str, TextSource)):
            yield from fragments(path, value, call)
        else:
            raise ValueError("The compaction model cannot fit chunk metadata and one source value")

    def fragments(path: list[str | int], value: str | TextSource,
                  call: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
        offset, before, buffer = 0, "", ""
        length = len(value) if isinstance(value, (str, FileText)) else sum(map(len, text_chunks(value)))
        # Bounded text buffering, even for file-backed results with enormous lines.
        target = max(1, budget // 2)
        stream = iter(text_chunks(value))
        finished = False
        while not finished or buffer:
            while not finished and len(buffer) < target + OVERLAP_CHARACTERS:
                part = next(stream, None)
                if part is None:
                    finished = True
                else:
                    buffer += part
            if not buffer:
                break

            def candidate(end: int) -> dict[str, Any]:
                result = {"path": path, "start": offset, "end": offset + end,
                          "total_characters": length, "text": buffer[:end],
                          "overlap_before": before, "overlap_after": buffer[end:end + OVERLAP_CHARACTERS]}
                if call is not None:
                    result["tool_identity"] = call
                return result

            low, high = 0, min(target, len(buffer))
            while low < high:
                mid = (low + high + 1) // 2
                if fits([candidate(mid)]):
                    low = mid
                else:
                    high = mid - 1
            if not low:
                raise ValueError("The compaction model cannot fit the request, source overlap, and one character")
            # Prefer paragraph/line seams near the end; oversized lines still progress.
            cut = buffer.rfind("\n\n", low // 2, low)
            if cut >= 0:
                low = cut + 2
            else:
                cut = buffer.rfind("\n", low // 2, low)
                if cut >= 0:
                    low = cut + 1
            record = candidate(low)
            if not fits([record]):
                raise ValueError("Source overlap does not fit the compaction model")
            yield record
            before = (before + buffer[:low])[-OVERLAP_CHARACTERS:]
            buffer = buffer[low:]
            offset += low
        if not offset:
            raise ValueError("The compaction model cannot fit an empty source field and its metadata")

    def records() -> Iterator[dict[str, Any]]:
        for field in ("user_prompt", "agent_response", "reasoning", "user_edits", "background_commands"):
            if field in source:
                yield from split([field], source[field])
        used: set[int] = set()
        for index, call in enumerate(source["tool_calls"]):
            identity = {"call_id": call["id"], "name": call["function"]["name"]}
            results = [(n, result) for n, result in enumerate(source["tool_results"])
                       if result["call_id"] == call["id"]]
            paired = [{"path": ["tool_calls", index], "value": call},
                      *[{"path": ["tool_results", n], "value": result} for n, result in results]]
            if fits([{"call_and_results": paired}]):
                yield {"call_and_results": paired}
            else:
                yield from split(["tool_calls", index], call, identity)
                for n, result in results:
                    yield from split(["tool_results", n], result, identity)
            used.update(n for n, _ in results)
        for n, result in enumerate(source["tool_results"]):
            if n not in used:
                yield from split(["tool_results", n], result)

    def save(records: list[dict[str, Any]]) -> None:
        path = Path(directory.name) / f"{len(plan.chunks):08d}.json"
        plan.chunks.append(write_indexed_json(path, {"records": records}))

    try:
        current: list[dict[str, Any]] = []
        for record in records():
            # Paired records are kept in one group, including their wrapper overhead.
            if current and not fits([*current, record]):
                save(current)
                current = []
            if not fits([record]):
                raise ValueError("A structural chunk exceeds the compaction budget")
            current.append(record)
        if current:
            save(current)
        if not plan.chunks:
            raise ValueError("No source records were available for chunked compaction")
        return plan
    except BaseException:
        plan.close()
        raise
