"""Danger mode changes path access, using only disposable files and repositories."""

from slipagent.types import content_text

from pathlib import Path

import pytest

from slipagent.agent import Agent, build_system_prompt
from slipagent.tools import build_default_registry
from slipagent.workspace import Workspace, WorkspaceError
from test_agent import StubClient


@pytest.mark.parametrize("path_kind", ["absolute", "parent", "symlink"])
async def test_danger_allows_external_file_tools_and_can_restore_confinement(tmp_path, path_kind):
    root, outside = tmp_path / "project", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "external").symlink_to(outside, target_is_directory=True)
    target = outside / "sub" / "file.txt"
    path = {"absolute": str(target), "parent": "../outside/sub/file.txt", "symlink": "external/sub/file.txt"}[path_kind]
    workspace = Workspace(root)
    registry = build_default_registry(workspace)
    try:
        assert (await registry.invoke("write_file", {"path": path, "content": "original"})).is_error
        assert not target.exists()
        workspace.access.danger = True
        result = await registry.invoke("write_file", {"path": path, "content": "original"})
        assert not result.is_error, result.content
        target.chmod(0o640)
        result = await registry.invoke("edit_file", {"path": path, "old_string": "original", "new_string": "updated"})
        assert not result.is_error, result.content
        assert target.read_text() == "updated"
        assert target.stat().st_mode & 0o777 == 0o640
        result = await registry.invoke("read_file", {"path": path})
        assert not result.is_error and "updated" in content_text(result.content)
        assert workspace.root == root and not workspace.contains(target)
        assert workspace.relative(target) == str(target)
        workspace.access.danger = False
        for tool, arguments in [("read_file", {"path": path}),
                                ("write_file", {"path": path, "content": "blocked"})]:
            assert (await registry.invoke(tool, arguments)).is_error
        assert target.read_text() == "updated"
        assert registry.services["project_instructions"].snapshot() == {".": ""}
    finally:
        await registry.aclose()


async def test_danger_searches_outside_and_through_file_symlinks(tmp_path):
    root, outside = tmp_path / "project", tmp_path / "outside"
    root.mkdir()
    (outside / "src").mkdir(parents=True)
    target = outside / "src" / "module.py"
    target.write_text("searchable needle\n")
    (root / "linked.py").symlink_to(target)
    workspace = Workspace(root, danger=True)
    registry = build_default_registry(workspace)
    try:
        for tool, arguments in [
            ("list_dir", {"path": str(outside / "src")}),
            ("glob", {"path": str(outside), "pattern": "src/**/*.py"}),
            ("grep", {"path": str(outside), "pattern": "needle"}),
        ]:
            result = await registry.invoke(tool, arguments)
            assert not result.is_error and str(target) in content_text(result.content), result.content
        result = await registry.invoke("grep", {"pattern": "needle"})
        assert not result.is_error and "linked.py" in content_text(result.content)
        assert root / "linked.py" in list(workspace.iter_files())
        workspace.access.danger = False
        assert root / "linked.py" not in list(workspace.iter_files())
        assert root / "linked.py" not in [entry for entry, _ in workspace.iter_entries()]
        assert (await registry.invoke("glob", {"path": str(outside), "pattern": "*.py"})).is_error
    finally:
        await registry.aclose()


async def test_external_instructions_and_access_mode_reach_each_request(tmp_path):
    root, outside = tmp_path / "project", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "AGENTS.md").write_text("External rule: preserve CRLF.")
    workspace = Workspace(root, danger=True)
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([]), registry, "test", system_prompt=build_system_prompt(str(root)))
    try:
        target = outside / "new.txt"
        result = await registry.invoke("write_file", {"path": str(target), "content": "first"})
        assert result.is_error and "instructions" in content_text(result.content)
        assert not target.exists()
        view = await agent._context_view(registry.specs(), 1)
        text = "\n".join(content_text(message.content) for message in view)
        assert "Danger Mode ON" in text and "External rule: preserve CRLF." in text
        tracker = registry.services["project_instructions"]
        tracker.presented(tracker.snapshot())
        result = await registry.invoke("write_file", {"path": str(target), "content": "first"})
        assert not result.is_error, result.content
        workspace.access.danger = False
        view = await agent._context_view(registry.specs(), 1)
        text = "\n".join(content_text(message.content) for message in view)
        assert "Danger Mode OFF" in text and "Danger Mode ON" not in text
        assert "External rule: preserve CRLF." not in text
        with pytest.raises(WorkspaceError, match="outside the workspace"):
            workspace.resolve(target)
        assert Workspace(root).access.danger is False
    finally:
        await registry.aclose()
