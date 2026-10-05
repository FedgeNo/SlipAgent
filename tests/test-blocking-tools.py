"""Local tool I/O must not hold the terminal's asyncio event loop."""

import asyncio
import threading

import pytest

from slipagent.tools.files import EditFileTool, ReadFileTool, WriteFileTool
from slipagent.tools.navigate import ListDirTool
from slipagent.tools.search import GlobTool
from slipagent.tools import build_default_registry
from slipagent.workspace import Workspace


@pytest.mark.parametrize("tool_class, arguments", [
    (ReadFileTool, {"path": "file.txt"}),
    (WriteFileTool, {"path": "file.txt", "content": "updated"}),
    (EditFileTool, {"path": "file.txt", "old_string": "original", "new_string": "updated"}),
    (ListDirTool, {"path": "."}),
    (GlobTool, {"path": ".", "pattern": "*.txt"}),
])
@pytest.mark.parametrize("through_registry", [False, True])
async def test_blocked_filesystem_call_keeps_input_loop_responsive(workspace, monkeypatch, tool_class, arguments, through_registry):
    (workspace.root / "file.txt").write_text("original")
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    resolve = Workspace.resolve

    def blocked_resolve(self, path):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(1):
            raise RuntimeError("The terminal loop could not release the filesystem operation.")
        return resolve(self, path)

    registry = build_default_registry(workspace)
    tracker = registry.services["project_instructions"]
    tracker.presented(tracker.snapshot())
    monkeypatch.setattr(Workspace, "resolve", blocked_resolve)
    operation = registry.invoke(tool_class.name, arguments) if through_registry else tool_class(workspace).invoke(arguments)
    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        # This represents input/redraw work serviced while disk access waits.
        release.set()
        result = await asyncio.wait_for(task, 2)
        assert not result.is_error, result.content
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await registry.aclose()


async def test_cancelled_write_finishes_owned_worker_before_returning(workspace, monkeypatch):
    from slipagent.tools import files
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    atomic_write = files._atomic_write

    def blocked_write(*arguments):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(2)
        atomic_write(*arguments)
        finished.set()

    monkeypatch.setattr(files, "_atomic_write", blocked_write)
    task = asyncio.create_task(WriteFileTool(workspace).run("saved.txt", "complete"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert finished.is_set()
        assert (workspace.root / "saved.txt").read_text() == "complete"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
