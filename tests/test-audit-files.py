"""File safety regressions use disposable workspace files only."""

import os
import stat

import pytest

from slipagent.tools import files
from slipagent.tools.files import EditFileTool, WriteFileTool


@pytest.mark.parametrize("edit", [False, True])
async def test_failed_encoding_preserves_original(workspace, edit):
    target = workspace.root / "source.txt"
    target.write_bytes(b"original\r\n")
    tool = EditFileTool(workspace) if edit else WriteFileTool(workspace)
    args = {"old_string": "original", "new_string": "\ud800"} if edit else {"content": "\ud800"}
    result = await tool.invoke({"path": target.name, **args})
    assert result.is_error
    assert target.read_bytes() == b"original\r\n"


@pytest.mark.parametrize("edit", [False, True])
async def test_replace_does_not_modify_other_hard_links(workspace, tmp_path, edit):
    outside = tmp_path / "outside"
    outside.write_text("original")
    root = workspace.root / "project"
    root.mkdir()
    from slipagent.workspace import Workspace
    confined = Workspace(root)
    target = root / "link.txt"
    os.link(outside, target)
    target.chmod(0o751)
    tool = EditFileTool(confined) if edit else WriteFileTool(confined)
    args = {"old_string": "original", "new_string": "replacement"} if edit else {"content": "replacement"}
    result = await tool.invoke({"path": target.name, **args})
    assert not result.is_error, result.content
    assert target.read_text() == "replacement"
    assert outside.read_text() == "original"
    assert stat.S_IMODE(target.stat().st_mode) == 0o751


async def test_replace_failure_keeps_original_and_cleans_temporary(workspace, monkeypatch):
    target = workspace.root / "source.txt"
    target.write_text("original")
    def fail(*args, **kwargs):
        raise OSError("simulated replacement failure")
    monkeypatch.setattr(files.os, "replace", fail)
    result = await WriteFileTool(workspace).invoke({"path": target.name, "content": "replacement"})
    assert result.is_error
    assert target.read_text() == "original"
    assert list(workspace.root.iterdir()) == [target]


async def test_lf_edit_of_crlf_read_preserves_other_line_endings(workspace):
    target = workspace.root / "source.txt"
    target.write_bytes(b"first\r\nalpha\r\nbeta\r\nlast\n")
    result = await EditFileTool(workspace).invoke({
        "path": target.name, "old_string": "alpha\nbeta", "new_string": "one\ntwo",
    })
    assert not result.is_error, result.content
    assert target.read_bytes() == b"first\r\none\r\ntwo\r\nlast\n"


@pytest.mark.skipif(os.name != "posix", reason="directory descriptor protection is POSIX")
async def test_parent_symlink_swap_cannot_redirect_atomic_write(workspace, monkeypatch):
    root = workspace.root / "root"
    parent = root / "sub"
    parent.mkdir(parents=True)
    (parent / "file").write_text("original")
    outside = workspace.root / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside")
    from slipagent.workspace import Workspace
    original_open = files.os.open
    swapped = False
    def swap(path, *args, **kwargs):
        nonlocal swapped
        if path == root and not swapped:
            swapped = True
            parent.rename(root / "saved")
            parent.symlink_to(outside, target_is_directory=True)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(files.os, "open", swap)
    result = await WriteFileTool(Workspace(root)).invoke({"path": "sub/file", "content": "replacement"})
    assert result.is_error
    assert (outside / "file").read_text() == "outside"
    assert (root / "saved" / "file").read_text() == "original"
