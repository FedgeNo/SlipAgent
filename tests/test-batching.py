
from slipagent.types import content_text
import asyncio

import pytest

from slipagent.batching import run_batch
from slipagent.tools.base import Tool, ToolRegistry, ToolResult
from slipagent.types import ToolCall


class Example(Tool):
    name = "read"
    description = "read"
    concurrent_safe = True

    async def run(self):
        return ToolResult.ok("read")


async def test_parallel_reads_finish_before_exclusive_write_and_commit_in_order():
    read, write = Example(), Example()
    write.name, write.concurrent_safe = "write", False
    registry = ToolRegistry([read, write])
    calls = [ToolCall(str(i), name, {}) for i, name in enumerate(["read", "read", "write", "read"])]
    started, committed = [], []
    both = asyncio.Event()

    async def invoke(call):
        started.append(call.id)
        if call.id == "1":
            both.set()
        if call.id == "0":
            await asyncio.wait_for(both.wait(), 1)
        if call.id == "2":
            assert committed == ["0", "1"]
        if call.id == "3":
            assert committed == ["0", "1", "2"]
        return ToolResult.ok(call.id)

    await run_batch(calls, registry, invoke, lambda call, result: committed.append(result.content))
    assert committed == ["0", "1", "2", "3"]


async def test_cancellation_drains_started_calls_and_pairs_every_result():
    registry = ToolRegistry([Example()])
    started, finished, committed = [], [], []
    ready = asyncio.Event()

    async def invoke(call):
        started.append(call.id)
        if len(started) == 2:
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(call.id)

    task = asyncio.create_task(run_batch([ToolCall(str(i), "read", {}) for i in range(6)], registry,
                                        invoke, lambda call, result: committed.append((call.id, result)), concurrency=2))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(finished) == started == ["0", "1"]
    assert [key for key, _ in committed] == [str(i) for i in range(6)]
    assert all(result.is_error for _, result in committed)
    assert "not run" in content_text(committed[-1][1].content)
