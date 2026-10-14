"""Bounded reusable transcript wrapping at the feed and Inspect widths."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from .transcript import WrappedTranscript


@dataclass
class WrappedLayout:
    flow: WrappedTranscript
    count: int = 0


class TranscriptLayouts:
    def __init__(self) -> None:
        self.key: tuple[object, ...] = ()
        self.states: OrderedDict[int, WrappedLayout] = OrderedDict()
        self.tasks: dict[int, asyncio.Task[None]] = {}

    def close(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        self.tasks.clear()
        for state in self.states.values():
            state.flow.close()
        self.states.clear()

    def configure(self, key: tuple[object, ...], *, reset: bool = False) -> None:
        if reset or self.key != key:
            self.close()
            self.key = key

    def state(self, width: int, create: Callable[[int], WrappedTranscript]) -> WrappedLayout:
        state = self.states.get(width)
        if state is None:
            state = WrappedLayout(create(width))
            self.states[width] = state
            if len(self.states) > 2:
                previous_width, previous = self.states.popitem(last=False)
                task = self.tasks.pop(previous_width, None)
                if task is not None:
                    task.cancel()
                previous.flow.close()
        self.states.move_to_end(width)
        return state

    def update(self, state: WrappedLayout, total: int, append: Callable[[WrappedTranscript, int], None]) -> None:
        while state.count < total:
            append(state.flow, state.count)
            state.count += 1

    def warm(self, width: int, total: int, create: Callable[[int], WrappedTranscript],
             append: Callable[[WrappedTranscript, int], None],
             spawn: Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]]) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        state = self.state(width, create)
        if state.count >= total or width in self.tasks:
            return

        async def prepare() -> None:
            try:
                # Rendering gets the current layout first. Prewarming yields
                # between records so input and agent work remain responsive.
                await asyncio.sleep(0)
                while self.states.get(width) is state and state.count < total:
                    self.update(state, state.count + 1, append)
                    await asyncio.sleep(0)
                if self.states.get(width) is state:
                    state.flow.content_rows()
            finally:
                if self.tasks.get(width) is asyncio.current_task():
                    self.tasks.pop(width, None)

        self.tasks[width] = spawn(prepare())
