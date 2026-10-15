"""Raw turns, individual summaries, and grouped summaries as request sources."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .sources import FileText, materialize, spool_directory, write_indexed_json
from .types import Message, ToolCall

if TYPE_CHECKING:
    from .context import CompletedStep, ConversationHistory


class ArchivedMessage(Message):
    """Keep message identity fields resident and retrieve archived values on demand."""

    def __init__(self, record: dict[str, Any] | None = None, **values: Any) -> None:
        if record is None:
            super().__init__(**values)
            record = super().to_record()
            record["reasoning_model"] = super().__getattribute__("reasoning_model")
        else:
            super().__init__(role=record["role"], tool_call_id=record.get("tool_call_id"), name=record.get("name"))
        self._stored = record

    def __getattribute__(self, name: str) -> Any:
        if name in {"content", "reasoning", "reasoning_details", "reasoning_model", "tool_calls"} and "_stored" in self.__dict__:
            stored = object.__getattribute__(self, "_stored")
            value = materialize(stored.get(name))
            if name == "tool_calls":
                return [ToolCall.from_api(call) for call in value] if value else None
            return value
        return super().__getattribute__(name)

    def __copy__(self) -> ArchivedMessage:
        return ArchivedMessage(self._stored)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Message):
            return NotImplemented
        return self.to_record() == other.to_record() and self.reasoning_model == other.reasoning_model


def _relocate(value: Any, path: Path) -> Any:
    if isinstance(value, FileText):
        return FileText(path, value.offset, value.size, value.characters, value.quoted)
    if isinstance(value, dict):
        return {key: _relocate(item, path) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item, path) for item in value]
    return value


class TurnStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="turns-", dir=spool_directory()) if directory is None else None
        self.directory = Path(self.temporary.name) if self.temporary is not None else directory
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.snapshots = tempfile.TemporaryDirectory(prefix=".snapshots-", dir=self.directory)
        self.sequence = 0

    def write(self, name: str, value: Any) -> Any:
        assert self.directory is not None
        path = self.directory / name
        fd, temporary = tempfile.mkstemp(prefix=".turn-", dir=self.directory)
        os.close(fd)
        pending = Path(temporary)
        pending.unlink()
        try:
            indexed = write_indexed_json(pending, value)
            self.sequence += 1
            pinned = Path(self.snapshots.name) / str(self.sequence)
            os.link(pending, pinned)
            os.replace(pending, path)
            return _relocate(indexed, pinned)
        finally:
            pending.unlink(missing_ok=True)

    def archive(self, step: CompletedStep, messages: list[Message]) -> None:
        if "_disk" not in step.__dict__:
            originals = step.messages
            source = {"version": 1, "step_id": step.id, "request": step.request,
                      "messages": [{**message.to_record(), "reasoning_model": message.reasoning_model} for message in originals],
                      "parts": step.parts(), "model": step.context_messages()[0].content}
            step._disk = self.write(f"{step.id:08d}.raw.json", source)
            replacements = {id(message): ArchivedMessage(record) for message, record in zip(originals, step._disk["messages"])}
            step.messages = [replacements[id(message)] for message in originals]
            for index, message in enumerate(messages):
                if id(message) in replacements:
                    messages[index] = replacements[id(message)]
            for name in ("_request", "user_prompt", "agent_response", "reasoning", "tool_calls", "tool_results", "_compaction_messages"):
                step.__dict__.pop(name, None)
            if "user_edits" in step._disk["parts"]:
                step.user_edits = step._disk["parts"]["user_edits"]
        self.summarized(step)

    def summarized(self, step: CompletedStep) -> None:
        state = (step.summary, getattr(step, "reasoning_summary", None), step.compaction_status, step.compaction_error)
        if step.__dict__.get("_disk_summary_state") != state:
            record = {"version": 1, "step_id": step.id, "status": step.compaction_status,
                      "summary": step.summary, "reasoning_summary": getattr(step, "reasoning_summary", None),
                      "model": step.context_messages(compressed=True)[0].content}
            step._disk_summary = self.write(f"{step.id:08d}.compressed.json", record)
            step._disk_summary_state = state
        grouped = getattr(step, "historical_summary", None)
        if grouped is not None and step.__dict__.get("_disk_group_state") is not grouped:
            first, last = grouped["step_ids"][0], grouped["step_ids"][-1]
            step._disk_group = self.write(f"{first:08d}-{last:08d}.grouped.json", grouped)
            step._disk_group_state = grouped

    def close(self) -> None:
        self.snapshots.cleanup()
        if self.temporary is not None:
            self.temporary.cleanup()

    async def aclose(self) -> None:
        self.close()

    def attach_artifact(self, step: CompletedStep, call_id: str, reference: dict[str, Any]) -> None:
        for result in step._disk["model"].get("tool_results", []):
            if result["call_id"] == call_id:
                result.setdefault("artifacts", []).append(reference)
        step._disk["model"].setdefault("artifacts", []).append(reference)
        for result in step._disk["parts"].get("tool_results", []):
            if result["call_id"] == call_id and isinstance(result["content"], dict):
                result["content"].setdefault("artifacts", []).append(reference)
        step._disk = self.write(f"{step.id:08d}.raw.json", step._disk)
        for message, record in zip(step.messages, step._disk["messages"]):
            if isinstance(message, ArchivedMessage):
                message._stored = record
        step.__dict__.pop("_disk_summary_state", None)
        self.summarized(step)
