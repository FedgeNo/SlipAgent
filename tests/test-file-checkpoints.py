"""Rewind preserves user changes, exact original bytes, and session ownership."""

import io
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from slipagent import cli
from slipagent.agent import Agent
from slipagent.checkpoints import CheckpointError, FileCheckpoints, current_checkpoint
from slipagent.sessions import SessionJournal
from slipagent.tools import build_default_registry
from slipagent.tools.files import _atomic_write
from slipagent.types import ToolCall
from test_agent import StubClient, completion


def write_batch(store, step, writes):
    store.begin(step)
    token = current_checkpoint.set(store)
    try:
        for path, text in writes:
            _atomic_write(store.workspace, path, text)
    finally:
        current_checkpoint.reset(token)
        store.finish()


async def test_rewind_restores_exact_bytes_modes_and_removes_created_files(workspace):
    original = workspace.root / "original.txt"
    original.write_bytes(b"first\r\nsecond\r\n")
    original.chmod(0o640)
    created = workspace.root / "new.txt"
    store = FileCheckpoints(workspace)
    try:
        write_batch(store, 1, [(original, "changed"), (created, "created")])
        checkpoint = store.listing()[0]["id"]
        preview = store.preview(checkpoint)
        assert "original.txt" in preview and "-changed\n" in preview and "+first\n" in preview
        assert set(store.rewind(checkpoint)) == {"original.txt", "new.txt"}
        assert original.read_bytes() == b"first\r\nsecond\r\n"
        assert not created.exists() and not store.listing()
        if os.name == "posix":
            assert original.stat().st_mode & 0o777 == 0o640
    finally:
        await store.aclose()


async def test_rewind_older_batch_restores_all_later_edits(workspace):
    path = workspace.root / "file.txt"
    path.write_text("original")
    store = FileCheckpoints(workspace)
    try:
        write_batch(store, 1, [(path, "first"), (path, "second")])
        first = store.listing()[0]["id"]
        write_batch(store, 2, [(path, "third")])
        assert "2 edit batch(es)" in store.preview(first)
        store.rewind(first)
        assert path.read_text() == "original" and not store.listing()
    finally:
        await store.aclose()


@pytest.mark.parametrize("change", ["edit", "delete", "mode", "symlink"])
async def test_rewind_refuses_conflicts_before_changing_any_file(workspace, change):
    first, second = workspace.root / "first.txt", workspace.root / "second.txt"
    first.write_text("one")
    second.write_text("two")
    store = FileCheckpoints(workspace)
    try:
        write_batch(store, 1, [(first, "edited one"), (second, "edited two")])
        checkpoint = store.listing()[0]["id"]
        if change == "edit":
            second.write_text("user edit")
        elif change == "delete":
            second.unlink()
        elif change == "mode":
            if os.name != "posix":
                pytest.skip("POSIX mode bits")
            second.chmod(0o400)
        else:
            second.unlink()
            second.symlink_to(first)
        with pytest.raises(CheckpointError, match="changed since the edit"):
            store.rewind(checkpoint)
        assert first.read_text() == "edited one"
        assert store.listing()
    finally:
        await store.aclose()


async def test_pending_checkpoint_recovers_only_a_published_edit(workspace):
    path = workspace.root / "file.txt"
    path.write_text("original")
    store = FileCheckpoints(workspace)
    try:
        for published in (False, True):
            store.clear()
            store.begin(1)
            store.prepare(path, b"edited")
            if published:
                path.write_bytes(b"edited")
            store.finish()
            store.rewind(store.listing()[0]["id"])
            assert path.read_text() == "original"
    finally:
        await store.aclose()


async def test_agent_checkpoints_complete_batches_and_preserves_history(workspace):
    registry = build_default_registry(workspace)
    client = StubClient([
        completion("", [ToolCall("one", "write_file", {"path": "file.txt", "content": "one"}),
                        ToolCall("bad", "edit_file", {"path": "file.txt", "old_string": "missing", "new_string": "two"}),
                        ToolCall("two", "edit_file", {"path": "file.txt", "old_string": "one", "new_string": "three"})]),
        completion("Done"),
    ])
    agent = Agent(client, registry, "test")
    try:
        assert await agent.run("edit file") == "Done"
        await agent.wait_for_compaction()
        store = agent._checkpoints()
        assert len(store.listing()) == 1
        assert len(store.listing()[0]["files"]) == 2
        history = list(agent.messages)
        store.rewind(store.listing()[0]["id"])
        assert not (workspace.root / "file.txt").exists()
        assert agent.messages == history
    finally:
        await registry.aclose()


async def test_resume_fork_and_delete_keep_checkpoint_storage_separate(workspace, tmp_path):
    registry = build_default_registry(workspace)
    client = StubClient([completion("", [ToolCall("one", "write_file", {"path": "file.txt", "content": "one"})]), completion("Done")])
    agent = Agent(client, registry, "test")
    journal = SessionJournal(str(workspace.root), tmp_path / "state")
    registry.services["session_journal"] = journal
    try:
        await agent.run("edit")
        await agent.wait_for_compaction()
        parent = journal.session_id
        store = agent._checkpoints()
        parent_index = (store.directory / "index.json").read_bytes()
        data = journal.load(parent)
        journal.restore(agent, data)
        assert len(store.listing()) == 1
        journal.restore(agent, journal.load(parent), fork=True)
        child = journal.session_id
        assert child != parent and len(store.listing()) == 1
        store.rewind(store.listing()[0]["id"])
        assert (journal.directory / (parent + "-checkpoints") / "index.json").read_bytes() == parent_index
        agent.reset(new_session=False)
        journal.delete_current()
        assert not (journal.directory / (child + "-checkpoints")).exists()
        assert (journal.directory / (parent + "-checkpoints")).exists()
    finally:
        await registry.aclose()


@pytest.mark.parametrize("confirmation", [None, "cancel", "restore"])
async def test_rewind_command_requires_confirmation_and_reports_current_files(workspace, confirmation):
    registry = build_default_registry(workspace)
    client = StubClient([])
    agent = Agent(client, registry, "test")
    renderer = cli.Renderer(cli.Style(False), io.StringIO(), False)
    session = cli.Session(agent, registry, client, renderer, workspace, "test", "http://unused.invalid", None, "test")
    terminal = MagicMock()
    renderer.terminal = terminal
    path = workspace.root / "file.txt"
    path.write_text("original")
    store = agent._checkpoints()
    write_batch(store, 1, [(path, "edited")])
    terminal.choose = AsyncMock(side_effect=[store.listing()[0]["id"], confirmation])
    try:
        await cli._handle_command(session, "/rewind")
        assert terminal.choose.await_count == 2
        assert path.read_text() == ("original" if confirmation == "restore" else "edited")
        assert bool(registry.context_notes.get("file_rewind")) == (confirmation == "restore")
    finally:
        renderer.terminal = None
        await registry.aclose()


async def test_checkpoint_storage_failure_prevents_the_edit(workspace, monkeypatch):
    store = FileCheckpoints(workspace)
    path = workspace.root / "file.txt"
    path.write_text("original")
    try:
        monkeypatch.setattr(store, "_persist", lambda: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            write_batch(store, 1, [(path, "changed")])
        assert path.read_text() == "original"
        assert not store.listing()
    finally:
        await store.aclose()


async def test_damaged_backup_blocks_whole_restore(workspace):
    store = FileCheckpoints(workspace)
    first, second = workspace.root / "first.txt", workspace.root / "second.txt"
    first.write_text("one")
    second.write_text("two")
    try:
        write_batch(store, 1, [(first, "edited one"), (second, "edited two")])
        batch = store.listing()[0]
        (store.directory / "blobs" / batch["files"][0]["before"]).write_bytes(b"corrupt")
        with pytest.raises(CheckpointError, match="backup is damaged"):
            store.rewind(batch["id"])
        assert first.read_text() == "edited one" and second.read_text() == "edited two"
    finally:
        await store.aclose()


async def test_new_file_permission_change_blocks_rewind(workspace):
    if os.name != "posix":
        pytest.skip("POSIX mode bits")
    store = FileCheckpoints(workspace)
    path = workspace.root / "new.txt"
    try:
        write_batch(store, 1, [(path, "new")])
        path.chmod(0o400)
        with pytest.raises(CheckpointError, match="changed since the edit"):
            store.rewind(store.listing()[0]["id"])
        assert path.exists()
    finally:
        await store.aclose()
