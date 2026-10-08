"""Git behavior and confinement are exercised only in disposable repositories."""

from __future__ import annotations

from slipagent.types import content_text

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from slipagent.tools import build_default_registry
from slipagent.tools import git as git_module
from slipagent.tools.git import GitAddTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from slipagent.workspace import Workspace


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout


@pytest.fixture()
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    for key in list(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test User")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "app.txt").write_text("original\n")
    git(root, "add", "app.txt")
    git(root, "commit", "-qm", "Initial commit")
    return Workspace(root)


async def test_registry_registers_git_tools(repository: Workspace) -> None:
    registry = build_default_registry(repository)
    try:
        assert {"git_status", "git_diff", "git_log", "git_add", "git_commit"} <= set(registry.names)
        assert not (await registry.invoke("git_status", {})).is_error
    finally:
        await registry.aclose()


async def test_danger_git_allows_external_parent_and_worktree_metadata(repository, tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    workspace = Workspace(unrelated, danger=True)
    tools = {tool.name: tool for tool in [GitStatusTool(workspace), GitDiffTool(workspace),
             GitLogTool(workspace), GitAddTool(workspace), GitCommitTool(workspace)]}
    target = repository.root / "app.txt"
    target.write_text("outside change\n")
    for name, args in [
        ("git_status", {}), ("git_diff", {}),
        ("git_add", {"paths": [str(target)]}),
        ("git_commit", {"message": "Outside change"}), ("git_log", {}),
    ]:
        result = await tools[name].invoke({"repo": str(repository.root), **args})
        assert not result.is_error, result.content
    child = repository.root / "child"
    child.mkdir()
    parent_tool = GitStatusTool(Workspace(child, danger=True))
    assert not (await parent_tool.invoke({})).is_error
    worktree = tmp_path / "worktree"
    git(repository.root, "worktree", "add", "-q", "-b", "danger-test", str(worktree))
    assert not (await GitStatusTool(Workspace(worktree, danger=True)).invoke({})).is_error
    workspace.access.danger = False
    assert (await tools["git_status"].invoke({"repo": str(repository.root)})).is_error


async def test_danger_repository_discovery_stops_at_filesystem_root(tmp_path, monkeypatch):
    tool = GitStatusTool(Workspace(tmp_path, danger=True))
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path.name == ".git" else original(path))
    result = await tool.invoke({})
    assert result.is_error and "No Git repository" in content_text(result.content)


async def test_log_never_runs_signature_verification_helper(repository, tmp_path):
    root = repository.root
    marker = tmp_path / "helper-was-run"
    helper = tmp_path / "verify"
    helper.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n")
    helper.chmod(0o700)
    git(root, "config", "log.showSignature", "true")
    git(root, "config", "gpg.program", str(helper))
    commit = (f"tree {git(root, 'rev-parse', 'HEAD^{tree}').strip()}\n"
              "author Test <test@example.invalid> 0 +0000\n"
              "committer Test <test@example.invalid> 0 +0000\n"
              "gpgsig -----BEGIN PGP SIGNATURE-----\n fake\n -----END PGP SIGNATURE-----\n\nSigned fixture\n")
    result = subprocess.run(["git", "-C", str(root), "hash-object", "-t", "commit", "-w", "--stdin"],
                            input=commit, text=True, capture_output=True, check=True)
    git(root, "update-ref", "HEAD", result.stdout.strip())
    output = await GitLogTool(repository).invoke({})
    assert not output.is_error, output.content
    assert "Signed fixture" in content_text(output.content)
    assert not marker.exists()


async def test_status_diff_add_commit_and_log(repository: Workspace) -> None:
    root = repository.root
    (root / "app.txt").write_text("staged version\n")
    (root / "new.txt").write_text("untracked\n")
    status = await GitStatusTool(repository).invoke({})
    assert not status.is_error
    assert " M app.txt" in content_text(status.content) and "?? new.txt" in content_text(status.content)
    assert not (await GitAddTool(repository).invoke({"paths": ["app.txt"]})).is_error
    (root / "app.txt").write_text("unstaged version\n")
    diff = GitDiffTool(repository)
    staged = await diff.invoke({"staged": True})
    unstaged = await diff.invoke({})
    assert not staged.is_error and "+staged version" in content_text(staged.content)
    assert "+unstaged version" not in content_text(staged.content)
    assert not unstaged.is_error and "+unstaged version" in content_text(unstaged.content)
    message = "Change text; $(touch unwanted) 'quoted'"
    committed = await GitCommitTool(repository).invoke({"message": message})
    assert not committed.is_error, committed.content
    assert git(root, "show", "HEAD:app.txt") == "staged version\n"
    assert not (root / "unwanted").exists()
    log = await GitLogTool(repository).invoke({"limit": 1})
    assert not log.is_error and message in content_text(log.content)
    assert "Initial commit" not in content_text(log.content)
    assert "?? new.txt" in content_text((await GitStatusTool(repository).invoke({})).content)


async def test_literal_paths_deletions_and_add_all(repository: Workspace) -> None:
    root = repository.root
    names = ["a*.txt", "abc.txt", "-odd name.txt", ":(glob)*.txt"]
    for name in names:
        (root / name).write_text(name)
    tool = GitAddTool(repository)
    assert not (await tool.invoke({"paths": ["a*.txt", "-odd name.txt", ":(glob)*.txt"]})).is_error
    staged = git(root, "diff", "--cached", "--name-only", "-z").split("\0")
    assert "a*.txt" in staged and "abc.txt" not in staged
    assert "-odd name.txt" in staged and ":(glob)*.txt" in staged
    (root / "app.txt").unlink()
    assert not (await tool.invoke({"paths": ["."]})).is_error
    assert "D\tapp.txt" in git(root, "diff", "--cached", "--name-status")
    assert "abc.txt" in git(root, "diff", "--cached", "--name-only")


async def test_nested_repository_and_workspace_relative_paths(repository: Workspace) -> None:
    nested = repository.root / "nested"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "file.txt").write_text("nested")
    result = await GitAddTool(repository).invoke({"repo": "nested", "paths": ["nested/file.txt"]})
    assert not result.is_error, result.content
    assert "file.txt" in git(nested, "diff", "--cached", "--name-only")
    assert not (await GitStatusTool(repository).invoke({"repo": "nested"})).is_error
    wrong = await GitAddTool(repository).invoke({"repo": "nested", "paths": ["app.txt"]})
    assert wrong.is_error and "outside this repository" in content_text(wrong.content)


async def test_errors_are_returned(repository: Workspace, tmp_path: Path) -> None:
    commit = GitCommitTool(repository)
    assert (await commit.invoke({"message": "   "})).is_error
    nothing = await commit.invoke({"message": "Nothing"})
    assert nothing.is_error and "nothing to commit" in content_text(nothing.content)
    git(repository.root, "config", "user.name", "")
    (repository.root / "app.txt").write_text("changed")
    git(repository.root, "add", "app.txt")
    assert (await commit.invoke({"message": "Missing identity"})).is_error
    assert (await GitAddTool(repository).invoke({"paths": []})).is_error
    assert (await GitAddTool(repository).invoke({"paths": ["missing.txt"]})).is_error
    assert (await GitLogTool(repository).invoke({"limit": 101})).is_error
    empty = tmp_path / "empty"
    empty.mkdir()
    result = await GitStatusTool(Workspace(empty)).invoke({})
    assert result.is_error and "No Git repository" in content_text(result.content)
    git(empty, "init", "-q")
    assert (await GitLogTool(Workspace(empty)).invoke({})).is_error


async def test_diff_paths_and_subdirectory_repo(repository: Workspace) -> None:
    (repository.root / "sub").mkdir()
    (repository.root / "app.txt").write_text("changed\n")
    (repository.root / "other.txt").write_text("other\n")
    await GitAddTool(repository).invoke({"paths": ["other.txt"]})
    diff = await GitDiffTool(repository).invoke({"paths": ["app.txt"], "repo": "sub"})
    assert not diff.is_error and "+changed" in content_text(diff.content)
    assert "other.txt" not in content_text(diff.content)


@pytest.mark.parametrize("tool", [GitStatusTool, GitDiffTool, GitLogTool, GitAddTool, GitCommitTool])
async def test_reject_parent_repository(repository: Workspace, tool: type) -> None:
    child = repository.root / "child"
    child.mkdir()
    args = {"paths": ["."]} if tool is GitAddTool else {"message": "No"} if tool is GitCommitTool else {}
    result = await tool(Workspace(child)).invoke(args)
    assert result.is_error and "Parent repositories" in content_text(result.content)


@pytest.mark.parametrize("path", ["../outside.txt", "/tmp", ".git/index"])
async def test_reject_invalid_paths(repository: Workspace, path: str) -> None:
    result = await GitAddTool(repository).invoke({"paths": [path]})
    assert result.is_error
    assert git(repository.root, "diff", "--cached", "--name-only") == ""


async def test_reject_symlink_escape_in_directory(repository: Workspace, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (repository.root / "escape").symlink_to(outside)
    for paths in (["escape"], ["."]):
        result = await GitAddTool(repository).invoke({"paths": paths})
        assert result.is_error and "outside the workspace" in content_text(result.content)
    assert git(repository.root, "diff", "--cached", "--name-only") == ""
    assert outside.read_text() == "outside"


@pytest.mark.parametrize("metadata", ["index", "HEAD", "objects", "refs"])
async def test_reject_metadata_escape(repository: Workspace, tmp_path: Path, metadata: str) -> None:
    source = repository.root / ".git" / metadata
    target = tmp_path / f"external-{metadata}"
    source.rename(target)
    source.symlink_to(target, target_is_directory=target.is_dir())
    result = await GitStatusTool(repository).invoke({})
    assert result.is_error and "outside the workspace" in content_text(result.content)


async def test_reject_external_gitdir_and_worktree(repository: Workspace, tmp_path: Path) -> None:
    worktree = tmp_path / "linked"
    git(repository.root, "worktree", "add", "-qb", "linked", str(worktree))
    result = await GitStatusTool(Workspace(worktree)).invoke({})
    assert result.is_error and "outside the workspace" in content_text(result.content)
    marker = repository.root / ".git"
    target = tmp_path / "external-git"
    marker.rename(target)
    marker.write_text(f"gitdir: {target}\n")
    assert (await GitStatusTool(repository).invoke({})).is_error


async def test_add_directory_rejects_nested_external_metadata(repository: Workspace, tmp_path: Path) -> None:
    external = tmp_path / "external-repo"
    external.mkdir()
    git(external, "init", "-q")
    git(external, "config", "user.name", "Test User")
    git(external, "config", "user.email", "test@example.invalid")
    (external / "file.txt").write_text("content")
    git(external, "add", "file.txt")
    git(external, "commit", "-qm", "Nested commit")
    nested = repository.root / "nested"
    nested.mkdir()
    (nested / ".git").write_text(f"gitdir: {external / '.git'}\n")
    result = await GitAddTool(repository).invoke({"paths": ["."]})
    assert result.is_error and "outside the workspace" in content_text(result.content)
    assert git(repository.root, "diff", "--cached", "--name-only") == ""


async def test_internal_metadata_link_cannot_hide_escape(repository: Workspace, tmp_path: Path) -> None:
    root = repository.root
    objects = root / ".git/objects"
    moved = root / "object-store"
    objects.rename(moved)
    objects.symlink_to(moved, target_is_directory=True)
    outside = tmp_path / "outside-object"
    outside.write_text("unchanged")
    (moved / "escape").symlink_to(outside)
    result = await GitStatusTool(repository).invoke({})
    assert result.is_error and "outside the workspace" in content_text(result.content)


async def test_reject_external_alternate_objects(repository: Workspace, tmp_path: Path) -> None:
    alternates = repository.root / ".git/objects/info/alternates"
    alternates.write_text(str(tmp_path / "external-objects") + "\n")
    result = await GitStatusTool(repository).invoke({})
    assert result.is_error and "outside the workspace" in content_text(result.content)


async def test_ignore_environment_and_config_worktree_redirects(repository: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    git(outside, "init", "-q")
    (outside / "outside.txt").write_text("outside")
    monkeypatch.setenv("GIT_DIR", str(outside / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(outside))
    monkeypatch.setenv("GIT_INDEX_FILE", str(outside / "redirected-index"))
    # Write config directly through a clean subprocess environment, not the redirected Git.
    with (repository.root / ".git/config").open("a") as config:
        config.write(f"\n[core]\n\tworktree = {outside}\n")
    (repository.root / "new.txt").write_text("inside")
    result = await GitAddTool(repository).invoke({"paths": ["new.txt"]})
    assert not result.is_error, result.content
    assert not (outside / "redirected-index").exists()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(key)
    assert git(outside, "diff", "--cached", "--name-only") == ""


async def test_disable_hooks_diff_and_signing_helpers(repository: Workspace) -> None:
    root = repository.root
    hook = root / ".git/hooks/pre-commit"
    hook.write_text("#!/bin/sh\ntouch hook-ran\nexit 1\n")
    hook.chmod(0o755)
    git(root, "config", "commit.gpgSign", "true")
    git(root, "config", "gpg.program", "/missing-signing-helper")
    git(root, "config", "diff.external", "touch diff-ran")
    git(root, "config", "core.fsmonitor", "touch fsmonitor-ran")
    (root / "app.txt").write_text("changed\n")
    assert not (await GitDiffTool(repository).invoke({})).is_error
    assert not (await GitAddTool(repository).invoke({"paths": ["app.txt"]})).is_error
    result = await GitCommitTool(repository).invoke({"message": "Confined commit"})
    assert not result.is_error, result.content
    for name in ("hook-ran", "diff-ran", "fsmonitor-ran"):
        assert not (root / name).exists()


@pytest.mark.parametrize("kind", ["clean", "process"])
async def test_reject_active_filters_without_running_them(repository: Workspace, kind: str) -> None:
    root = repository.root
    git(root, "config", f"filter.custom.{kind}", "touch filter-ran")
    # Unused filter configuration does not prevent normal staging.
    assert not (await GitAddTool(repository).invoke({"paths": ["app.txt"]})).is_error
    (root / ".gitattributes").write_text("*.txt filter=custom\n")
    (root / "app.txt").write_text("changed")
    result = await GitAddTool(repository).invoke({"paths": ["."]})
    assert result.is_error and "filter 'custom'" in content_text(result.content)
    assert not (root / "filter-ran").exists()
    assert git(root, "diff", "--cached", "--name-only") == ""


async def test_index_lock_error(repository: Workspace) -> None:
    lock = repository.root / ".git/index.lock"
    lock.write_text("busy")
    result = await GitAddTool(repository).invoke({"paths": ["app.txt"]})
    assert result.is_error and "index.lock" in content_text(result.content)
    assert lock.read_text() == "busy"


async def test_missing_objects_do_not_launch_remote_helpers(repository: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    root = repository.root
    binary = root / "bin"
    binary.mkdir()
    helper = binary / "git-remote-danger"
    helper.write_text("#!/bin/sh\ntouch remote-helper-ran\nexit 1\n")
    helper.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    blob = git(root, "rev-parse", "HEAD:app.txt").strip()
    git(root, "config", "remote.origin.url", "danger::unused")
    git(root, "config", "remote.origin.promisor", "true")
    git(root, "config", "protocol.danger.allow", "always")
    (root / ".git/objects" / blob[:2] / blob[2:]).unlink()
    (root / "app.txt").write_text("changed\n")
    result = await GitDiffTool(repository).invoke({})
    assert result.is_error
    assert not (root / "remote-helper-ran").exists()


async def test_validation_never_uses_truncated_listing(repository: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    for number in range(10):
        (repository.root / f"file-{number}.txt").write_text("content")
    monkeypatch.setattr(git_module, "MAX_OUTPUT_CHARS", 30)
    result = await GitAddTool(repository).invoke({"paths": ["."]})
    assert result.is_error and "select fewer paths" in content_text(result.content)
    assert git(repository.root, "diff", "--cached", "--name-only") == ""
    monkeypatch.setattr(git_module, "MAX_OUTPUT_CHARS", 10)
    log = await GitLogTool(repository).invoke({})
    assert "… [output truncated]" in content_text(log.content)
async def test_missing_git(repository: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/missing-git-bin")
    result = await GitStatusTool(repository).invoke({})
    assert result.is_error and "Git is not installed" in content_text(result.content)


async def test_timeout_and_cancellation(repository: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    original = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []

    async def sleeping_git(*args, **kwargs):
        process = await original("/bin/sleep", "30", **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", sleeping_git)
    result = await GitStatusTool(repository).invoke({"timeout": 0.1})
    assert result.is_error and "timed out" in content_text(result.content)
    assert processes[0].returncode is not None
    task = asyncio.create_task(GitStatusTool(repository).invoke({}))
    while len(processes) < 2:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[1].returncode is not None
