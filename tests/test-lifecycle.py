import asyncio

import pytest

from slipagent.lifecycle import Lifetime


async def test_close_is_shared_and_finishes_despite_repeated_cancellation():
    lifetime = Lifetime("test")
    entered, release = asyncio.Event(), asyncio.Event()
    closed = []

    async def cleanup():
        entered.set()
        await release.wait()
        closed.append("resource")

    lifetime.defer(cleanup)
    first = asyncio.create_task(lifetime.aclose())
    await entered.wait()
    first.cancel()
    await asyncio.sleep(0)
    first.cancel()
    second = asyncio.create_task(lifetime.aclose())
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    await lifetime.aclose()
    assert closed == ["resource"]


async def test_failed_cleanup_does_not_skip_other_resources():
    lifetime = Lifetime("test")
    closed = []
    lifetime.defer(lambda: closed.append("first"))

    def fail():
        closed.append("second")
        raise ValueError("cleanup failed")

    lifetime.defer(fail)
    with pytest.raises(ValueError, match="cleanup failed"):
        await lifetime.aclose()
    assert closed == ["second", "first"]
    with pytest.raises(RuntimeError, match="closed"):
        lifetime.defer(lambda: None)


async def test_scope_cancels_and_drains_owned_tasks_before_resource_close():
    lifetime = Lifetime("test")
    entered, release = asyncio.Event(), asyncio.Event()
    events = []

    async def worker():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await release.wait()
            events.append("worker finished")

    lifetime.spawn(worker())
    lifetime.defer(lambda: events.append("resource closed"))
    await entered.wait()
    closing = asyncio.create_task(lifetime.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    assert events == ["worker finished", "resource closed"]
