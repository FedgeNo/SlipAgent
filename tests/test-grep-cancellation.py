"""Grep retains its filesystem worker until cancellation has settled."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from slipagent.tools import search
from slipagent.workspace import Workspace


@pytest.fixture
async def blocked_scan(workspace, monkeypatch, request):
    phase = request.param
    loop = asyncio.get_running_loop()
    entered, finished = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    paths = [workspace.root / f"file{index}.txt" for index in range(3)]
    for path in paths:
        path.write_bytes(b"\0needle" if phase == "binary" else b"needle")
    visited, reads = [], []
    tool = search.GrepTool(workspace)
    read_regular, matches_include, read_batch = search.read_regular, search._matches_include, tool._read_batch

    def block():
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(3), "Test did not release the filesystem operation"

    def candidates(self, base=None):
        for path in paths:
            visited.append(path)
            if phase == "traversal":
                block()
            yield path

    def include_matches(*arguments):
        if phase == "include":
            block()
        return matches_include(*arguments)

    def read(path, limit):
        reads.append(path)
        if phase in {"read", "binary", "unreadable"}:
            block()
        if phase == "unreadable":
            raise OSError("Fixture file is unreadable")
        return read_regular(path, limit)

    def batch(*arguments, **keywords):
        try:
            return read_batch(*arguments, **keywords)
        finally:
            loop.call_soon_threadsafe(finished.set)

    monkeypatch.setattr(Workspace, "iter_files", candidates)
    monkeypatch.setattr(search, "_matches_include", include_matches)
    monkeypatch.setattr(search, "read_regular", read)
    monkeypatch.setattr(tool, "_read_batch", batch)
    try:
        yield SimpleNamespace(tool=tool, entered=entered, finished=finished, release=release,
                              visited=visited, reads=reads, paths=paths, phase=phase,
                              include="*.py" if phase == "include" else None)
    finally:
        release.set()
        if entered.is_set():
            await asyncio.wait_for(finished.wait(), 3)


@pytest.mark.parametrize("blocked_scan", ["traversal", "include", "read", "binary", "unreadable"], indirect=True)
@pytest.mark.parametrize("interrupt", ["cancel", "timeout"])
async def test_scan_interrupt_stops_and_drains_filesystem_work(blocked_scan, monkeypatch, interrupt):
    scan = blocked_scan
    deadline = asyncio.timeout(None)

    async def worker_request(*arguments):
        pytest.fail("An interrupted batch reached the regex worker")

    monkeypatch.setattr(scan.tool, "_worker_request", worker_request)

    async def run():
        async with deadline:
            return await scan.tool._scan(None, "needle", None, scan.include, 100)

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(scan.entered.wait(), 2)
        if interrupt == "timeout":
            deadline.reschedule(asyncio.get_running_loop().time())
        else:
            task.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done(), "Cancellation returned while filesystem work was still running"
        if interrupt == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        scan.release.set()
        with pytest.raises(TimeoutError if interrupt == "timeout" else asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert scan.finished.is_set()
        assert scan.visited == scan.paths[:1]
        assert scan.reads == ([] if scan.phase in {"traversal", "include"} else scan.paths[:1])
    finally:
        scan.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("blocked_scan", ["read"], indirect=True)
async def test_cancelled_grep_drains_scan_and_reaps_regex_process(blocked_scan, monkeypatch):
    scan = blocked_scan
    spawn = asyncio.create_subprocess_exec
    processes = []

    async def record_process(*arguments, **keywords):
        process = await spawn(*arguments, **keywords)
        processes.append(process)
        return process

    monkeypatch.setattr(search.asyncio, "create_subprocess_exec", record_process)
    task = asyncio.create_task(scan.tool.invoke({"pattern": "needle"}))
    try:
        await asyncio.wait_for(scan.entered.wait(), 2)
        task.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        task.cancel()
        scan.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert scan.finished.is_set()
        assert scan.visited == scan.paths[:1]
        assert len(processes) == 1 and processes[0].returncode is not None
    finally:
        scan.release.set()
        await asyncio.gather(task, return_exceptions=True)
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.parametrize("skipped", ["include", "binary", "unreadable"])
async def test_scan_batches_count_rejected_candidates(workspace, monkeypatch, skipped):
    paths = [workspace.root / f"file{index}.txt" for index in range(10)]
    for path in paths:
        path.write_bytes(b"\0needle" if skipped == "binary" else b"needle")
    visited, batch_sizes = [], []
    tool = search.GrepTool(workspace)
    read_batch = tool._read_batch

    def candidates(self, base=None):
        for path in paths:
            visited.append(path)
            yield path

    def batch(*arguments, **keywords):
        before = len(visited)
        result = read_batch(*arguments, **keywords)
        batch_sizes.append(len(visited) - before)
        return result

    def unreadable(*arguments):
        raise OSError("Fixture file is unreadable")

    monkeypatch.setattr(Workspace, "iter_files", candidates)
    monkeypatch.setattr(tool, "_read_batch", batch)
    monkeypatch.setattr(search, "GREP_BATCH_FILES", 3)
    if skipped == "unreadable":
        monkeypatch.setattr(search, "read_regular", unreadable)
    result = await tool.run("needle", include="*.py" if skipped == "include" else None)
    assert not result.is_error and result.content.startswith("No matches")
    assert f"({0 if skipped == 'include' else 10} file(s) scanned)" in result.content
    assert visited == paths
    assert all(size <= 3 for size in batch_sizes)
