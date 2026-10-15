"""Identity and cancellation for terminal input awaiting ordered dispatch."""

from __future__ import annotations

import uuid


class Submission(str):
    identifier: str
    cancelled: bool
    suggested: bool

    def __new__(cls, text: str, *, suggested: bool = False) -> Submission:
        value = super().__new__(cls, text)
        value.identifier = uuid.uuid4().hex
        value.cancelled = False
        value.suggested = suggested
        return value

    def strip(self, chars: str | None = None) -> str:
        if self.suggested and chars is None:
            return self
        text = super().strip(chars)
        if text == self:
            return self
        # Keep cancellation tied to the submitted row after normalization.
        value = type(self)(text, suggested=self.suggested)
        value.identifier = self.identifier
        value.__dict__["_source"] = self
        return value


def cancelled(line: str | None) -> bool:
    return isinstance(line, Submission) and (line.cancelled or cancelled(line.__dict__.get("_source")))


def suggested(line: str) -> bool:
    return isinstance(line, Submission) and line.suggested
