"""Harness-owned retention of original user prompts and their source steps."""

from __future__ import annotations


from .types import Message
from .records import record_message, step_record


class TaskMemory:
    def __init__(self, start_message: int = 0) -> None:
        self.start_message = start_message
        self.last_message = start_message - 1
        self.sources: dict[int, list[str]] = {}

    def note_user(self, message_index: int, step_id: int, text: str) -> None:
        """Register each actual user message once, including unarchived tail input."""
        if message_index <= self.last_message or message_index < self.start_message:
            return
        self.last_message = message_index
        self.sources.setdefault(step_id, []).append(text)

    @property
    def current_prompt_step(self) -> int | None:
        """The call receiving the latest user input, independent of compression."""
        return next(reversed(self.sources), None)

    def prompt_supplement(self, messages: list[Message]) -> list[Message]:
        """Supply the current prompt if history selection has not already done so.

        Queued inputs received together share a step. Continuation steps may
        carry their joined wording; neither form needs a duplicate prompt.
        """
        step_id = self.current_prompt_step
        if step_id is None:
            return []
        prompts = self.sources[step_id]
        joined = "\n".join(prompts)
        supplied: set[str] = set()
        current = step_record([], current=True)
        for message in messages:
            if message.role != "user":
                continue
            record = message.content
            supplied.update(record.get("user_prompt", []))
            if record.get("record_type") == "current_step":
                current = record
        if joined in supplied or all(prompt in supplied for prompt in prompts):
            return []
        return [record_message({**current, "user_prompt": prompts})]
