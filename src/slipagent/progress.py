"""Detect repeated tool batches with unchanged results, without deduplicating calls."""

from __future__ import annotations

from .prompts import load_prompt

import hashlib
import json
import re
from collections.abc import Sequence

from .openrouter import OpenRouterAPIError
from .tools.base import ToolRegistry, ToolResult
from .types import ToolCall


class LoopGuard:
    def __init__(self) -> None:
        self.signature: str | None = None
        self.count = 0

    def reset(self) -> None:
        self.signature = None
        self.count = 0
        self.recent: list[tuple[str, int, tuple[str, ...]]] = []

    def observe(self, batch: Sequence[tuple[ToolCall, ToolResult]], registry: ToolRegistry,
                step_id: int) -> tuple[str, bool]:
        records = []
        for call, result in batch:
            tool = registry.get(call.name)
            if getattr(tool, "progress_exempt", False):
                continue
            if call.name == "run_command" and call.arguments.get("poll") is True:
                continue
            content = result.content
            archive = registry.services.get("command_archive")
            if archive is not None and isinstance(content, str):
                # Log IDs vary even when stdout/stderr are unchanged. Remove
                # only the actual archive's exact invocation-specific notice.
                for log in reversed(list(archive.logs.values())):
                    if log.step_id == step_id and log.call_id == call.id:
                        content = content.removesuffix("\n" + log.notice())
                        break
            records.append((call.name, call.arguments, result.is_error, content))
        if not records:
            self.reset()
            return "", False
        encoded = json.dumps(records, sort_keys=True, ensure_ascii=True).encode()
        signature = hashlib.sha256(encoded).hexdigest()
        if not hasattr(self, "recent"):
            self.recent = []  # Migrate guards retained by an older generation.
        self.recent.append((signature, step_id, tuple(record[0] for record in records)))
        self.recent = self.recent[-24:]
        self.count = self.count + 1 if signature == self.signature else 1
        self.signature = signature
        signatures = [item[0] for item in self.recent]
        period = next((size for size in range(1, 7)
                       if len(signatures) >= size * 3
                       and signatures[-size:] == signatures[-size * 2:-size]
                       == signatures[-size * 3:-size * 2]), 0)
        if not period:
            return "", False
        stopping = (len(signatures) >= period * 4
                    and signatures[-period:] == signatures[-period * 4:-period * 3])
        names = ", ".join(dict.fromkeys(name for item in self.recent[-period:] for name in item[2]))
        if not stopping:
            return (
                load_prompt('repeated-tools.txt', tool_names=names,
                            first_step=self.recent[-period * 3][1], last_step=step_id), False,
            )
        return (
            "Stopped: the same tool batch or cycle returned unchanged results again after recovery guidance. "
            "History is preserved. Give a correction or continue to try another approach.", True,
        )


class StreamLoopError(OpenRouterAPIError):
    """A response repeats itself; its partial calls must never execute."""


class StreamLoopGuard:
    """Detect sustained exact prose repetition independently of SSE chunk sizes."""

    def __init__(self) -> None:
        self.buffers: dict[str, str] = {}
        self.checked: dict[str, int] = {}
        self.pending = ""
        self.fence = ""

    @staticmethod
    def _prose(line: str) -> bool:
        return not re.match(r"^(?: {4}|\t|\s*(?:[|#]|[-*+]\s|\d+[.)]\s|[={}\[\]]|`{3,}|~{3,}))", line)

    def feed(self, kind: str, text: str) -> None:
        if kind not in {"content", "reasoning"} or not text:
            return
        if kind == "content":
            lines = (self.pending + text).split("\n")
            self.pending = lines.pop()
            if len(self.pending) > 32768:
                self.pending = self.pending[:256] + self.pending[-32512:]
            for line in lines:
                marker = re.match(r"^\s*(`{3,}|~{3,})", line)
                if marker:
                    if not self.fence:
                        self.fence = marker[1]
                    elif marker[1][0] == self.fence[0] and len(marker[1]) >= len(self.fence):
                        self.fence = ""
                elif not self.fence and self._prose(line):
                    self.buffers[kind] = (self.buffers.get(kind, "") + line + "\n")[-32768:]
        else:
            self.buffers[kind] = (self.buffers.get(kind, "") + text)[-32768:]
        self.checked[kind] = self.checked.get(kind, 0) + len(text)
        if self.checked[kind] >= 256:
            self.check(kind)
            self.checked[kind] = 0

    def check(self, kind: str) -> None:
        text = self.buffers.get(kind, "")
        if kind == "content" and not self.fence and self._prose(self.pending):
            text += self.pending
        text = re.sub(r"\s+", " ", text).strip()[-16384:]
        if len(text) < 480:
            return
        anchor = text[-64:]
        start = max(0, len(text) - 2112)
        end = len(text) - 64
        while (match := text.rfind(anchor, start, end)) >= start:
            period = len(text) - 64 - match
            if 80 <= period <= 2048 and len(text) >= period * 6:
                unit = text[-period:]
                if text[-period * 6:] == unit * 6:
                    raise StreamLoopError(
                        f"Stopped: sustained repetition in the model's {kind} response. "
                        "No tools from this response ran; accepted history is preserved."
                    )
            end = match

    def finish(self) -> None:
        for kind in self.checked:
            self.check(kind)
