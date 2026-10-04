"""Explicit ownership and cancellation-safe, idempotent asynchronous teardown.

Closing runs in a retained task. Cancelling a waiter never cancels cleanup;
the waiter propagates cancellation only after owned work has been drained.
This module is part of the stable frame so a reload cannot replace an active
resource's ownership bookkeeping.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


async def finish_cleanup(task: asyncio.Task[T]) -> T:
    """Await one owned cleanup to completion even if this waiter is cancelled."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except BaseException:
            break
    if cancellation is not None:
        try:
            task.result()
        except BaseException as exc:
            raise cancellation from exc
        raise cancellation
    return task.result()


class Lifetime:
    """Own pending tasks and LIFO cleanups; all close callers share settlement."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._callbacks: list[Callable[[], Any]] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closing: asyncio.Task[None] | None = None

    @property
    def active(self) -> bool:
        return self._closing is None

    def defer(self, callback: Callable[[], Any]) -> None:
        if not self.active:
            raise RuntimeError(f"Lifetime {self.name} is closed")
        self._callbacks.append(callback)

    def spawn(self, operation: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        if not self.active:
            operation.close()
            raise RuntimeError(f"Lifetime {self.name} is closed")
        task = asyncio.create_task(operation, name=self.name)
        self._tasks.add(task)
        task.add_done_callback(self._settled)
        return task

    def _settled(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            task.get_loop().call_exception_handler({
                "message": f"Unhandled owned task failure in {self.name}",
                "exception": task.exception(), "task": task,
            })

    async def _close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        failure: BaseException | None = None
        while self._callbacks:
            callback = self._callbacks.pop()
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure

    async def aclose(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(), name=f"close {self.name}")
        await finish_cleanup(self._closing)
