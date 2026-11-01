"""Bounded local phase durations, separate from model context and accounting."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.base import ToolRegistry


@dataclass(frozen=True)
class PhaseStart:
    phase: str
    step: int
    call_id: str
    started: float


class PhaseTimings:
    def __init__(self, *, limit: int = 512, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.enabled = True
        self.sequence = 0
        self.records: deque[dict[str, Any]] = deque(maxlen=limit)
        self.active: dict[int, PhaseStart] = {}

    def clear(self) -> None:
        self.records.clear()
        self.active.clear()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.active.clear()

    def start(self, phase: str, step: int, call_id: str = '') -> int | None:
        if not self.enabled:
            return None
        self.sequence += 1
        self.active[self.sequence] = PhaseStart(phase, step, call_id, self.clock())
        return self.sequence

    def record(self, phase: str, seconds: float, step: int, call_id: str = '', outcome: str = 'completed') -> None:
        if self.enabled:
            self.records.append({'phase': phase, 'step': step, 'call_id': call_id,
                                 'seconds': max(0.0, seconds), 'outcome': outcome})

    def finish(self, ticket: int | None, outcome: str = 'completed') -> None:
        source = self.active.pop(ticket, None) if ticket is not None else None
        if source is not None:
            self.record(source.phase, self.clock() - source.started, source.step, source.call_id, outcome)

    @contextmanager
    def measure(self, phase: str, step: int, call_id: str = '') -> Iterator[None]:
        ticket = self.start(phase, step, call_id)
        try:
            yield
        except BaseException as exc:
            self.finish(ticket, 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'error')
            raise
        else:
            self.finish(ticket)

    def snapshot(self, step: int | None = None) -> list[dict[str, Any]]:
        rows = [dict(record) for record in self.records if step is None or record['step'] == step]
        now = self.clock()
        rows.extend({'phase': source.phase, 'step': source.step, 'call_id': source.call_id,
                     'seconds': max(0.0, now - source.started), 'outcome': 'running'}
                    for source in self.active.values() if step is None or source.step == step)
        return rows


def timing_log(registry: ToolRegistry) -> PhaseTimings:
    result = registry.services.get('phase_timings')
    if result is None:
        result = PhaseTimings()
        registry.services['phase_timings'] = result
    return result


async def save_timings(registry: ToolRegistry) -> None:
    timings = registry.services.get('phase_timings')
    diagnostics = registry.resources.diagnostics
    if timings is not None and timings.enabled and diagnostics is not None:
        from .tools.blocking import run_blocking
        await run_blocking(diagnostics.update_timings, timings.snapshot())
