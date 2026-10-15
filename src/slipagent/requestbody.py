"""Immutable disk snapshots and replayable, bounded HTTP request streams."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx

from .data_text import RenderedContent, RenderedJSON
from .sources import json_chunks, materialize, spool_directory, write_indexed_json
from .types import Message


def wire_message(message: Message) -> dict[str, Any]:
    stored = getattr(message, "_stored", None)
    record = stored if stored is not None else message.to_record()
    content = record.get("content")
    result: dict[str, Any] = {"role": message.role, "content": RenderedContent(content) if content is not None else None}
    for key in ("tool_call_id", "name"):
        if record.get(key) is not None:
            result[key] = record[key]
    if record.get("tool_calls"):
        result["tool_calls"] = [{**call, "function": {**call["function"],
            "arguments": RenderedJSON(call["function"]["arguments"])}}
            for call in record["tool_calls"]]
    return result


def value_tokens(value: Any, *, ensure_ascii: bool = False) -> int:
    size = sum(len(chunk.encode("utf-8")) for chunk in json_chunks(value, indent=None, ensure_ascii=ensure_ascii))
    return (size + 2) // 3 + 32


class RequestBody(httpx.AsyncByteStream):
    """The temporary directory lives as long as its request/event references."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="request-", dir=spool_directory())
        self.path = Path(self.temporary.name) / "request.json"
        try:
            self.index = write_indexed_json(self.path, payload)
            self.size = self.path.stat().st_size
            self.estimate = sum(value_tokens(message) for message in self.index.get("messages", []))
            for key in ("tools", "response_format"):
                if key in self.index:
                    self.estimate += value_tokens(self.index[key], ensure_ascii=True)
        except BaseException:
            self.temporary.cleanup()
            raise

    def chunks(self) -> Iterator[bytes]:
        with self.path.open("rb") as source:
            while chunk := source.read(65536):
                yield chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks():
            yield chunk

    def read(self) -> dict[str, Any]:
        result: dict[str, Any] = materialize(self.index)
        return result
