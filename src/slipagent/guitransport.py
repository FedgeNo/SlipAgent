"""Bounded live observations; durable history remains under the session owners."""

from __future__ import annotations

import asyncio
import secrets
from typing import Any

from .application import ApplicationState, SessionApplication
from .events import AgentEvent

MAX_CLIENTS = 8
QUEUE_SIZE = 128
TEXT_LIMIT = 2048


def state_record(state: ApplicationState) -> dict[str, Any]:
    return {'sessionId': state.session_id, 'workspace': state.workspace, 'model': state.model,
            'running': state.running, 'stopped': state.stopped, 'queuedInputs': state.queued_inputs,
            'changing': state.changing, 'closed': state.closed}


class GuiTransport:
    """Publish small notifications without retaining tool payloads or replay history."""

    def __init__(self, controller: SessionApplication) -> None:
        self.controller = controller
        self.epoch = secrets.token_hex(16)
        self.sequence = 0
        self.last_state = controller.state()
        self.clients: set[asyncio.Queue[dict[str, Any] | None]] = set()
        self.closed = False
        self.unsubscribe = controller.subscribe(self.event)
        self.unsubscribe_state = controller.states.subscribe(self.state)

    def snapshot(self) -> dict[str, Any]:
        state = self.controller.state()
        activity = self.controller.session.registry.services.get('activity_phase', '') if state.running else ''
        return self.envelope('snapshot', state=state_record(state), activity=activity, resync=True)

    def envelope(self, kind: str, **values: Any) -> dict[str, Any]:
        return {'schemaVersion': 1, 'epoch': self.epoch, 'sequence': self.sequence,
                'type': kind, **values}

    def attach(self) -> asyncio.Queue[dict[str, Any] | None]:
        if self.closed or len(self.clients) >= MAX_CLIENTS:
            raise RuntimeError('GUI connection limit reached or host closed.')
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(QUEUE_SIZE)
        self.clients.add(queue)
        queue.put_nowait(self.snapshot())
        return queue

    def detach(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self.clients.discard(queue)

    def publish(self, kind: str, **values: Any) -> None:
        self.sequence += 1
        record = self.envelope(kind, **values)
        for queue in tuple(self.clients):
            if queue.full():
                # Evict slow consumers instead of blocking the engine or retaining output.
                self.detach(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)
            else:
                queue.put_nowait(record)

    def event(self, event: AgentEvent) -> None:
        current = self.controller.state()
        if current != self.last_state:
            self.state(current)
        if event.kind == 'context':
            # Context bodies can contain megabytes, credentials and provider requests.
            return
        text = event.text[:TEXT_LIMIT]
        if len(event.text) > TEXT_LIMIT:
            text += '…'
        call = event.tool_call
        self.publish('event', event={'kind': event.kind, 'step': event.step, 'text': text,
                                    'truncated': len(event.text) > TEXT_LIMIT,
                                    'toolCall': {'id': call.id, 'name': call.name} if call else None,
                                    'hasResult': event.result is not None})

    def state(self, state: ApplicationState) -> None:
        self.last_state = state
        self.publish('state', state=state_record(state))

    def close(self) -> None:
        self.closed = True
        self.unsubscribe()
        self.unsubscribe_state()
        for queue in tuple(self.clients):
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(None)
        self.clients.clear()
