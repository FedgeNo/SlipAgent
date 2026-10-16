"""Explicit filesystem operations track real changes and preserve boundaries."""

import pytest

from slipagent.instructions import ProjectInstructions
from slipagent.tools.paths import CopyPathTool, MovePathTool, DeletePathTool, CreateDirectoryTool


async def test_binary_tree_copy_move_delete_and_creation(workspace):
    root = workspace.root
    created = await CreateDirectoryTool(workspace).run("source/nested", parents=True)
    assert not created.is_error
    (root / "source/nested/data.bin").write_bytes(b"\0\xffhello")
    copied = await CopyPathTool(workspace).run("source", "copy")
    assert not copied.is_error
    assert (root / "copy/nested/data.bin").read_bytes() == b"\0\xffhello"
    assert {item["path"] for item in copied.content["changes"]} == {"copy", "copy/nested", "copy/nested/data.bin"}
    assert (await CopyPathTool(workspace).run("source", "copy")).is_error
    moved = await MovePathTool(workspace).run("copy", "renamed")
    assert not moved.is_error and not (root / "copy").exists()
    assert {item["action"] for item in moved.content["changes"]} == {"moved_from", "moved_to"}
    assert (await DeletePathTool(workspace).run("renamed")).is_error
    assert (root / "renamed/nested/data.bin").exists()
    deleted = await DeletePathTool(workspace).run("renamed", recursive=True)
    assert not deleted.is_error and not (root / "renamed").exists()
    assert len(deleted.content["changes"]) == 3


async def test_exact_filenames_and_no_overwrite(workspace):
    root = workspace.root
    (root / "--source with spaces").write_bytes(b"original")
    copied = await CopyPathTool(workspace).run("--source with spaces", "literal*.txt")
    assert not copied.is_error and (root / "literal*.txt").read_bytes() == b"original"
    assert (await MovePathTool(workspace).run("--source with spaces", "literal*.txt")).is_error
    assert (root / "--source with spaces").read_bytes() == b"original"
    assert not (await DeletePathTool(workspace).run("literal*.txt")).is_error
    assert not (await CreateDirectoryTool(workspace).run("empty")).is_error
    assert not (await DeletePathTool(workspace).run("empty")).is_error


async def test_move_checks_destination_after_containment_validation(workspace, monkeypatch):
    from pathlib import Path

    source, destination = workspace.root / "source", workspace.root / "destination"
    source.write_text("original")
    is_relative_to = Path.is_relative_to

    def create_destination(path, other):
        result = is_relative_to(path, other)
        if path == source and other == destination:
            destination.write_text("concurrent file")
        return result

    monkeypatch.setattr(Path, "is_relative_to", create_destination)
    result = await MovePathTool(workspace).run("source", "destination")
    assert result.is_error and result.content["changes"] == []
    assert source.read_text() == "original"
    assert destination.read_text() == "concurrent file"


@pytest.mark.parametrize("tool,args", [
    (CopyPathTool, {"source": ".", "destination": "copy"}),
    (MovePathTool, {"source": ".", "destination": "move"}),
    (DeletePathTool, {"path": ".", "recursive": True}),
    (CreateDirectoryTool, {"path": "../outside"}),
])
async def test_reject_roots_and_escape(workspace, tool, args):
    result = await tool(workspace).run(**args)
    assert result.is_error and result.content["changes"] == []


async def test_tree_symlinks_rejected_before_changes(workspace):
    root = workspace.root
    (root / "tree").mkdir()
    (root / "tree/file").write_text("keep")
    (root / "tree/link").symlink_to(root / "tree/file")
    for tool, args in [(CopyPathTool, {"source": "tree", "destination": "copy"}),
                       (MovePathTool, {"source": "tree", "destination": "move"}),
                       (DeletePathTool, {"path": "tree", "recursive": True})]:
        result = await tool(workspace).run(**args)
        assert result.is_error and result.content["changes"] == []
    assert (root / "tree/file").read_text() == "keep"


def test_instructions_cover_nested_source_and_destination(workspace):
    root = workspace.root
    (root / "source/deep").mkdir(parents=True)
    (root / "source/deep/AGENTS.md").write_text("Preserve source rules.")
    (root / "destination").mkdir()
    (root / "destination/AGENTS.md").write_text("Destination rules.")
    instructions = ProjectInstructions(workspace)
    instructions.presented(instructions.snapshot())
    tool = CopyPathTool(workspace)
    args = {"source": "source", "destination": "destination/new"}
    assert instructions.guard(tool, args) is not None
    snapshot = instructions.snapshot()
    assert "source/deep" in snapshot and "destination" in snapshot
    instructions.presented(snapshot)
    assert instructions.guard(tool, args) is None


async def test_partial_copy_reports_created_paths(workspace, monkeypatch):
    import slipagent.tools.paths as paths
    (workspace.root / "source").write_bytes(b"hello")
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(paths.shutil, "copyfileobj", fail)
    result = await CopyPathTool(workspace).run("source", "destination")
    assert result.is_error
    assert result.content["changes"] == [{"path": "destination", "action": "created", "kind": "file"}]
    assert "disk full" in result.content["error"]


async def test_outside_source_requires_danger_mode(tmp_path):
    from slipagent.workspace import Workspace
    (tmp_path / "project").mkdir()
    workspace = Workspace(tmp_path / "project")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    tool = CopyPathTool(workspace)
    assert (await tool.run(str(outside), "copied.bin")).is_error
    workspace.access.danger = True
    assert not (await tool.run(str(outside), "copied.bin")).is_error
    assert (workspace.root / "copied.bin").read_bytes() == b"outside"
