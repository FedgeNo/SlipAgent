"""Inspect and discard user input that has not reached a working request."""

from __future__ import annotations

import copy
import json
import uuid
from typing import TYPE_CHECKING, Any

from .context import CompletedStep, ConversationHistory
from .events import AgentEvent
from .task import TaskMemory
from .submissions import Submission

if TYPE_CHECKING:
    from .agent import Agent


def discarded_state(state: dict[str, Any], index: int | None, pending_index: int | None) -> dict[str, Any]:
    """Apply a journalled discard to its saved queue and message indices."""
    result = copy.deepcopy(state)
    queued = result.get("queued_message_indices", [])
    if index is None:
        if type(pending_index) is not int or not 0 <= pending_index < len(result["pending"]):
            raise ValueError("invalid pending input discard")
        del result["pending"][pending_index]
    else:
        if type(index) is not int or index not in queued or pending_index is not None:
            raise ValueError("input discard must identify an unsent user message")
        result["queued_message_indices"] = [value - (value > index) for value in queued if value != index]
        result["task_start"] -= index < result["task_start"]
    return result


class InputQueue:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.entries: list[tuple[str, str]] = []
        self.prepared: tuple[str, ...] | None = None

    def snapshot(self) -> list[tuple[str, str]]:
        values = [*self.agent.queued_messages.values(), *self.agent.pending]
        # Resume and older live generations have queue text but no UI identities.
        if [text for _, text in self.entries] != values:
            self.entries = [(uuid.uuid4().hex, text) for text in values]
        return list(self.entries)

    def appended(self, text: str) -> None:
        self.entries.append((text.identifier if isinstance(text, Submission) else uuid.uuid4().hex, text))

    def sent(self, count: int) -> None:
        self.snapshot()
        del self.entries[:count]
        # Delivered input was part of the prepared context; only later changes count.
        if self.prepared is not None:
            self.prepared = self.prepared[count:]

    def signature(self) -> tuple[str, ...]:
        return tuple(identifier for identifier, _ in self.snapshot())

    def changed(self) -> bool:
        return self.prepared is not None and self.prepared != self.signature()

    def discard(self, identifier: str) -> bool:
        if self.agent.registry.services.get("input_queue") is not self:
            return False
        entries = self.snapshot()
        position = next((index for index, entry in enumerate(entries) if entry[0] == identifier), None)
        if position is None:
            return False
        agent = self.agent
        indices = list(agent.queued_messages)
        index = indices[position] if position < len(indices) else None
        pending_index = position - len(indices) if index is None else None
        journal = agent.registry.resources.journal
        if journal is not None:
            # Persist the removal before changing memory; one event also updates
            # queue/task indices, so a crash cannot resurrect discarded input.
            journal.discard_input(agent, index, pending_index)
        if index is None:
            assert pending_index is not None
            del agent.pending[pending_index]
        else:
            del agent.messages[index]
            remaining = {value - (value > index): text for value, text in agent.queued_messages.items()
                         if value != index}
            agent.queued_messages.clear()
            agent.queued_messages.update(remaining)
            self._refresh_history(index)
        del self.entries[position]
        agent._persist()
        agent._emit(AgentEvent(kind="user_queue_changed", text=json.dumps(
            [text for _, text in self.snapshot()], ensure_ascii=False)))
        return True

    def _refresh_history(self, removed: int) -> None:
        history = self.agent.history
        task = history.task
        start = task.start_message - (removed < task.start_message)
        refreshed = ConversationHistory(task=TaskMemory(start_message=start))
        refreshed.sync(self.agent.messages)
        # An unsent message can belong to a locally recorded request failure.
        # Preserve step identities, summaries, metadata, and compaction owners.
        affected = False
        for saved, current in zip(history.steps, refreshed.steps, strict=True):
            affected = affected or len(saved.messages) != len(current.messages)
            if not affected:
                continue
            saved.messages = current.messages
            saved.request = current.request
            if history.store is not None and isinstance(saved, CompletedStep) and "_disk" in saved.__dict__:
                history.store.refresh_input(saved)
        history.cursor = refreshed.cursor
        history._request = refreshed._request
        task.start_message = refreshed.task.start_message
        task.last_message = refreshed.task.last_message
        task.sources = refreshed.task.sources


def input_queue(agent: Agent) -> InputQueue:
    queue: InputQueue = agent.registry.services.setdefault("input_queue", InputQueue(agent))
    return queue
