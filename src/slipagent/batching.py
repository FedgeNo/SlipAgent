"""Bounded read concurrency with exclusive barriers and ordered observations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .lifecycle import finish_cleanup
from .tools.base import ToolRegistry, ToolResult
from .types import ToolCall


async def run_batch(
    calls: list[ToolCall], registry: ToolRegistry,
    invoke: Callable[[ToolCall], Awaitable[ToolResult]],
    commit: Callable[[ToolCall, ToolResult], None], *, concurrency: int = 4,
) -> None:
    if concurrency < 1:
        raise ValueError("Tool concurrency must be positive")
    results: dict[int, ToolResult] = {}
    started: set[int] = set()
    committed = 0

    async def run(index: int) -> None:
        started.add(index)
        results[index] = await invoke(calls[index])

    def safe(index: int) -> bool:
        tool = registry.get(calls[index].name)
        return tool is not None and tool.allows_concurrency(calls[index].arguments)

    tasks: list[asyncio.Task[None]] = []
    try:
        while committed < len(calls):
            end = committed + 1
            if safe(committed):
                while end < min(len(calls), committed + concurrency) and safe(end):
                    end += 1
            tasks = [asyncio.create_task(run(index)) for index in range(committed, end)]
            await asyncio.gather(*tasks)
            while committed < end:
                index = committed
                committed += 1
                commit(calls[index], results[index])
    except BaseException:
        async def drain() -> None:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await finish_cleanup(asyncio.create_task(drain()))
        except asyncio.CancelledError:
            pass  # Drain finished; preserve every result before propagating cancellation.
        for index in range(committed, len(calls)):
            result = results.get(index)
            if result is None:
                detail = ("Tool interrupted; effects may be partial." if index in started else
                          "Tool was not run because the turn was interrupted.")
                result = ToolResult.error(detail)
            commit(calls[index], result)
        raise
