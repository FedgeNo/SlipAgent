"""Run synchronous local tool work without blocking the terminal event loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from ..lifecycle import finish_cleanup

T = TypeVar("T")


async def run_blocking(operation: Callable[..., T], *arguments: Any,
                       on_cancel: Callable[[], None] | None = None, **keywords: Any) -> T:
    """Own a worker until it finishes, even when its async caller is cancelled.

    A thread cannot be forcibly stopped. Long scans can supply a cooperative
    cancellation callback; atomic edits finish before cancellation propagates.
    This keeps batch ordering, access policy, and reload boundaries valid.
    """
    worker = asyncio.create_task(asyncio.to_thread(operation, *arguments, **keywords))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancellation:
        if on_cancel is not None:
            on_cancel()
        try:
            await finish_cleanup(worker)
        finally:
            raise cancellation
