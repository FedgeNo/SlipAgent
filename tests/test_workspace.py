"""The sandbox boundary must hold — these are the security-critical tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slipagent.workspace import Workspace, WorkspaceError


def test_resolves_relative_against_root(workspace: Workspace) -> None:
    target = workspace.root / "a" / "b.txt"
    target.parent.mkdir(parents=True)
    target.write_text("hi", encoding="utf-8")

    assert workspace.resolve("a/b.txt") == target.resolve()
    assert workspace.resolve("./a/b.txt") == target.resolve()


def test_resolves_root_itself(workspace: Workspace) -> None:
    assert workspace.resolve(".") == workspace.root


def test_empty_path_is_rejected(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="non-empty"):
        workspace.resolve("   ")


@pytest.mark.parametrize(
    "escape",
    [
        "../outside.txt",
        "../../etc/passwd",
        "src/../../outside.txt",
        "..",
        "a/b/../../../outside.txt",
    ],
)
def test_traversal_is_rejected(workspace: Workspace, tmp_path: Path, escape: str) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.resolve(escape)


def test_absolute_path_outside_root_is_rejected(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.resolve("/etc/passwd")


def test_absolute_path_inside_root_is_allowed(workspace: Workspace) -> None:
    inside = workspace.root / "ok.txt"
    inside.write_text("fine", encoding="utf-8")

    assert workspace.resolve(str(inside)) == inside.resolve()


def test_symlink_escape_is_rejected(workspace: Workspace, tmp_path: Path) -> None:
    """A link inside the tree must not become a way out of it."""
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "secret.txt").write_text("secret", encoding="utf-8")

    link = workspace.root / "escape"
    link.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(WorkspaceError, match="outside the workspace"):
        workspace.resolve("escape/secret.txt")


def test_workspace_rejects_non_directory(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="not a directory"):
        Workspace(target)


def test_relative_display_is_rooted(project: Workspace) -> None:
    assert project.relative(project.root / "src" / "app.py") == "src/app.py"
    assert project.relative(project.root) == "."


def test_iter_files_prunes_ignored_directories(project: Workspace) -> None:
    found = {project.relative(path) for path in project.iter_files()}

    assert "src/app.py" in found
    assert "README.md" in found
    assert not any(name.startswith(".git/") for name in found)
    assert not any(name.startswith("node_modules/") for name in found)


def test_iter_files_skips_symlinks_that_escape_the_root(
    workspace: Workspace, tmp_path: Path
) -> None:
    """A link inside the tree must not become a way to read outside it.

    Regression: `os.walk` lists symlinked files in `filenames`, so grep would
    happily read a link pointing at `~/.ssh/id_rsa` even though `resolve`
    refuses that exact path when the model asks for it directly.
    """
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret", encoding="utf-8")

    (workspace.root / "ok.txt").write_text("fine", encoding="utf-8")
    (workspace.root / "escape.txt").symlink_to(outside / "secret.txt")

    found = {workspace.relative(path) for path in workspace.iter_files()}

    assert "ok.txt" in found
    assert "escape.txt" not in found


def test_iter_entries_skips_symlinks_that_escape_the_root(
    workspace: Workspace, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret", encoding="utf-8")

    (workspace.root / "real").mkdir()
    (workspace.root / "escape_file").symlink_to(outside / "secret.txt")
    (workspace.root / "escape_dir").symlink_to(outside, target_is_directory=True)

    entries = {workspace.relative(path) for path, _ in workspace.iter_entries()}

    assert "real" in entries
    assert "escape_file" not in entries
    assert "escape_dir" not in entries


def test_iter_files_keeps_symlinks_that_stay_inside(workspace: Workspace) -> None:
    """Internal links are harmless and should not be dropped."""
    (workspace.root / "real.txt").write_text("content", encoding="utf-8")
    (workspace.root / "alias.txt").symlink_to(workspace.root / "real.txt")

    found = {workspace.relative(path) for path in workspace.iter_files()}

    assert "alias.txt" in found


def test_iter_entries_reports_directory_kind(project: Workspace) -> None:
    entries = {
        project.relative(path): is_dir
        for path, is_dir in project.iter_entries()
    }

    assert entries["src"] is True
    assert entries["README.md"] is False
    assert "node_modules" not in entries


def test_resolve_preserves_filename_spaces(workspace: Workspace) -> None:
    assert workspace.resolve(" file.txt ") == workspace.root / " file.txt "
