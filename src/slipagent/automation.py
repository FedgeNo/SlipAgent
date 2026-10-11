"""Opt-in, flushed JSON Lines records for headless one-shot consumers."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from typing import Any, TextIO

from .events import AgentEvent


class JsonEventStream:
    """Write structured observations without buffering an entire run."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.started = time.monotonic()
        self.sequence = 0

    def write(self, record_type: str, **values: Any) -> None:
        self.sequence += 1
        record = {"schema_version": 1, "record_type": record_type,
                  "sequence": self.sequence, "elapsed_seconds": time.monotonic() - self.started,
                  **values}
        # ASCII JSON also works with redirected legacy Windows encodings.
        self.stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.stream.flush()

    def event(self, event: AgentEvent) -> None:
        # Completed observations and saved sessions retain the useful data;
        # provider token fragments would swamp an evaluation trace.
        if event.kind in {"reasoning_delta", "assistant_delta", "tool_output"}:
            return
        values = asdict(event)
        if event.kind == "context":
            values["request"] = json.loads(values.pop("text"))
        self.write("event", **values)
