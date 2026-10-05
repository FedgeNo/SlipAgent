"""Detect repeated tool batches with unchanged results, without deduplicating calls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from .tools.base import ToolRegistry, ToolResult
from .types import ToolCall


class LoopGuard:
    def __init__(self) -> None:
        self.signature: str | None = None
        self.count = 0

    def reset(self) -> None:
        self.signature = None
        self.count = 0

    def observe(self, batch: Sequence[tuple[ToolCall, ToolResult]], registry: ToolRegistry,
                post_id: int) -> tuple[str, bool]:
        records = []
        for call, result in batch:
            tool = registry.get(call.name)
            if getattr(tool, "progress_exempt", False):
                continue
            if call.name == "run_command" and call.arguments.get("poll") is True:
                continue
            content = result.content
            archive = registry.services.get("command_archive")
            if archive is not None:
                # Log IDs vary even when stdout/stderr are unchanged. Remove
                # only the actual archive's exact invocation-specific notice.
                for log in reversed(list(archive.logs.values())):
                    if log.post_id == post_id and log.call_id == call.id:
                        content = content.removesuffix("\n" + log.notice())
                        break
            records.append((call.name, call.arguments, result.is_error, content))
        if not records:
            self.reset()
            return "", False
        encoded = json.dumps(records, sort_keys=True, ensure_ascii=True).encode()
        signature = hashlib.sha256(encoded).hexdigest()
        self.count = self.count + 1 if signature == self.signature else 1
        self.signature = signature
        if self.count < 3:
            return "", False
        names = ", ".join(record[0] for record in records)
        if self.count == 3:
            return (
                f"The same tool batch ({names}) returned unchanged results in posts {post_id - 2}–{post_id}.\n\n"
                "Those tools already ran. Change the approach: inspect the specific error, choose a different "
                "query or target, or report the blocker.\n\nIf intentionally polling an external change with "
                "run_command, set poll=true; ordinary retries are not polling.", False,
            )
        return (
            "Stopped: the same tool batch returned unchanged results again after recovery guidance. "
            "History is preserved. Give a correction or continue to try another approach.", True,
        )
