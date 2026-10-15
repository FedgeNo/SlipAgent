"""Separate leading thought blocks without interpreting tags inside answers."""

from collections.abc import Callable


class ThoughtStream:
    def __init__(self, emit: Callable[[str, str], None]) -> None:
        self.emit = emit
        self.pending = ""
        self.mode = "prefix"

    def add(self, text: str) -> None:
        self.pending += text
        while self.pending:
            if self.mode == "response":
                self.emit("content", self.pending)
                self.pending = ""
            elif self.mode == "prefix":
                candidate = self.pending.lstrip()
                if candidate.startswith("<thought>"):
                    self.pending = candidate[len("<thought>"):]
                    self.mode = "thought"
                elif not candidate or "<thought>".startswith(candidate):
                    return
                else:
                    self.mode = "response"
            else:
                before, closing, following = self.pending.partition("</thought>")
                if closing:
                    if before:
                        self.emit("reasoning", before)
                    self.pending = following
                    self.mode = "prefix"
                    continue
                # Hold only a possible closing-tag prefix across chunks.
                keep = next((size for size in range(min(len(self.pending), len("</thought>") - 1), 0, -1)
                             if "</thought>".startswith(self.pending[-size:])), 0)
                ready = self.pending[:-keep] if keep else self.pending
                if ready:
                    self.emit("reasoning", ready)
                self.pending = self.pending[-keep:] if keep else ""
                return

    def finish(self) -> None:
        if self.mode == "thought":
            raise ValueError("Unclosed <thought> block in provider response")
        if self.pending:
            self.emit("content", self.pending)
            self.pending = ""
