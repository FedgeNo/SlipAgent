"""grep, glob, and list_dir behaviour."""

from __future__ import annotations

from slipagent.types import content_text

import asyncio

from pathlib import Path

import pytest

from slipagent.tools.navigate import ListDirTool
from slipagent.tools.search import GlobTool, GrepTool
from slipagent.workspace import Workspace


async def test_pathological_regex_times_out_without_blocking_input(workspace, monkeypatch):
    from slipagent.tools import search
    (workspace.root / "sample.txt").write_text("a" * 35 + "!")
    monkeypatch.setattr(search, "GREP_TIMEOUT", .25)
    ticks = 0
    async def heartbeat():
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(.02)
            ticks += 1
    beat = asyncio.create_task(heartbeat())
    result = await asyncio.wait_for(search.GrepTool(workspace).invoke({"pattern": "(a+)+$"}), 2)
    await beat
    assert result.is_error and "timed out" in content_text(result.content)
    assert ticks == 10


# --------------------------------------------------------------------------- #
# grep
# --------------------------------------------------------------------------- #


async def test_grep_finds_matches_with_line_numbers(project: Workspace) -> None:
    result = await GrepTool(project).invoke({"pattern": r"def (\w+)\("})

    assert not result.is_error
    assert "src/app.py:1: def main():" in content_text(result.content)
    assert "src/app.py:5: def helper(value):" in content_text(result.content)


async def test_grep_include_filter_narrows_files(project: Workspace) -> None:
    result = await GrepTool(project).invoke({"pattern": "def", "include": "test_*.py"})

    assert not result.is_error
    assert "tests/test_app.py" in content_text(result.content)
    assert "src/app.py" not in content_text(result.content)


async def test_grep_ignore_case(project: Workspace) -> None:
    sensitive = await GrepTool(project).invoke({"pattern": "VALUE"})
    insensitive = await GrepTool(project).invoke({"pattern": "VALUE", "ignore_case": True})

    assert "src/util.py:1" in content_text(sensitive.content)
    assert "src/util.py:1" in content_text(insensitive.content)

    result = await GrepTool(project).invoke({"pattern": "HELLO", "ignore_case": True})
    assert "src/app.py:2" in content_text(result.content)


async def test_grep_reports_no_matches_clearly(project: Workspace) -> None:
    result = await GrepTool(project).invoke({"pattern": "zzz-not-here"})

    assert not result.is_error
    assert "No matches" in content_text(result.content)
    assert "file(s) scanned" in content_text(result.content)


async def test_grep_rejects_invalid_regex(project: Workspace) -> None:
    result = await GrepTool(project).invoke({"pattern": "([unclosed"})

    assert result.is_error
    assert "Invalid regular expression" in content_text(result.content)


async def test_grep_skips_ignored_directories(project: Workspace) -> None:
    result = await GrepTool(project).invoke({"pattern": "module.exports"})

    assert "No matches" in content_text(result.content)
    assert ".git" not in content_text(result.content)


async def test_grep_skips_binary_files(workspace: Workspace) -> None:
    (workspace.root / "a.bin").write_bytes(b"\x00secret\x00")

    result = await GrepTool(workspace).invoke({"pattern": "secret"})

    assert "No matches" in content_text(result.content)


async def test_grep_respects_max_results(project: Workspace) -> None:
    result = await GrepTool(project).invoke({"pattern": ".", "max_results": 1})

    body = result.content.splitlines()[1:]
    assert len(body) == 1
    assert "stopped at the 1-result limit" in content_text(result.content)


async def test_grep_clamps_an_oversized_max_results(workspace: Workspace) -> None:
    """A tool call can carry any integer, so the cap is enforced here too.

    Regression: the schema only declared a minimum, so `max_results: 100000`
    returned every match and could flood the context window these limits exist
    to protect.
    """
    from slipagent.tools.search import MAX_GREP_RESULTS

    for index in range(MAX_GREP_RESULTS + 50):
        (workspace.root / f"mod{index}.py").write_text("x = 1\n", encoding="utf-8")

    result = await GrepTool(workspace).invoke({"pattern": "x = 1", "max_results": 100_000})

    assert result.is_error
    assert "must be <= 200" in content_text(result.content)


async def test_grep_run_clamps_directly(workspace: Workspace) -> None:
    """The clamp holds even when validation is bypassed by calling run()."""
    from slipagent.tools.search import MAX_GREP_RESULTS

    for index in range(MAX_GREP_RESULTS + 50):
        (workspace.root / f"mod{index}.py").write_text("x = 1\n", encoding="utf-8")

    result = await GrepTool(workspace).run(pattern="x = 1", max_results=100_000)

    body = result.content.splitlines()[1:]
    assert len(body) == MAX_GREP_RESULTS
    assert f"stopped at the {MAX_GREP_RESULTS}-result limit" in content_text(result.content)


async def test_grep_cannot_escape(project: Workspace) -> None:
    result = await GrepTool(project).invoke({"pattern": "x", "path": "../.."})

    assert result.is_error
    assert "outside the workspace" in content_text(result.content)


async def test_grep_does_not_read_through_an_escaping_symlink(
    workspace: Workspace, tmp_path: Path
) -> None:
    """Regression: a symlink in the tree must not leak outside content.

    The model cannot name the escaping path directly — `path` is resolved and
    rejected — but it can let grep walk the tree and hit the link with a pattern
    instead, so the walk itself has to stay inside the sandbox.
    """
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("PRIVATE KEY MATERIAL", encoding="utf-8")
    (workspace.root / "innocent.txt").symlink_to(outside / "secret.txt")

    result = await GrepTool(workspace).invoke({"pattern": "PRIVATE"})

    # The pattern itself is echoed in the header, so assert on the leaked body.
    assert "PRIVATE KEY MATERIAL" not in content_text(result.content)
    assert "No matches" in content_text(result.content)


async def test_glob_does_not_enumerate_an_escaping_symlink(
    workspace: Workspace, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("PRIVATE", encoding="utf-8")
    (workspace.root / "escape_file").symlink_to(outside / "secret.txt")
    (workspace.root / "escape_dir").symlink_to(outside, target_is_directory=True)

    result = await GlobTool(workspace).invoke({"pattern": "**"})

    assert "escape_file" not in content_text(result.content)
    assert "escape_dir" not in content_text(result.content)


# --------------------------------------------------------------------------- #
# glob
# --------------------------------------------------------------------------- #


async def test_glob_finds_python_files(project: Workspace) -> None:
    result = await GlobTool(project).invoke({"pattern": "*.py"})

    assert not result.is_error
    assert "src/app.py" in content_text(result.content)
    assert "tests/test_app.py" in content_text(result.content)
    assert "README.md" not in content_text(result.content)


async def test_glob_supports_recursive_patterns(project: Workspace) -> None:
    result = await GlobTool(project).invoke({"pattern": "**/test_*.py"})

    assert "tests/test_app.py" in content_text(result.content)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("*.py", "top.py"),
        ("**/*.py", "top.py"),
        ("**/test_*.py", "test_top.py"),
        ("src/**/*.ts", "src/nested/app.ts"),
    ],
)
async def test_glob_recursive_patterns_match_at_every_depth(
    workspace: Workspace, pattern: str, expected: str
) -> None:
    """`**` spans zero or more directories, as the tool description promises.

    Plain `fnmatch` treats the slash after `**` literally, so `**/*.py` misses
    top-level files and `src/**/*.ts` misses files directly inside `src`.
    """
    (workspace.root / "top.py").write_text("x", encoding="utf-8")
    (workspace.root / "test_top.py").write_text("x", encoding="utf-8")
    (workspace.root / "src" / "nested").mkdir(parents=True)
    (workspace.root / "src" / "nested" / "app.ts").write_text("x", encoding="utf-8")

    result = await GlobTool(workspace).invoke({"pattern": pattern})

    assert expected in result.content.splitlines()[1:]


async def test_glob_marks_directories(project: Workspace) -> None:
    result = await GlobTool(project).invoke({"pattern": "src"})

    assert "src/" in content_text(result.content)


async def test_glob_reports_no_paths(project: Workspace) -> None:
    result = await GlobTool(project).invoke({"pattern": "*.rs"})

    assert not result.is_error
    assert "No paths matching" in content_text(result.content)


async def test_glob_ignores_vendor_directories(project: Workspace) -> None:
    result = await GlobTool(project).invoke({"pattern": "**/index.js"})

    assert "No paths matching" in content_text(result.content)


async def test_glob_clamps_an_oversized_max_results(workspace: Workspace) -> None:
    from slipagent.tools.search import MAX_GLOB_RESULTS

    for index in range(MAX_GLOB_RESULTS + 50):
        (workspace.root / f"mod{index}.py").write_text("x", encoding="utf-8")

    result = await GlobTool(workspace).invoke({"pattern": "*.py", "max_results": 100_000})

    assert result.is_error
    assert "must be <= 500" in content_text(result.content)


async def test_glob_run_clamps_directly(workspace: Workspace) -> None:
    from slipagent.tools.search import MAX_GLOB_RESULTS

    for index in range(MAX_GLOB_RESULTS + 50):
        (workspace.root / f"mod{index}.py").write_text("x", encoding="utf-8")

    result = await GlobTool(workspace).run(pattern="*.py", max_results=100_000)

    body = result.content.splitlines()[1:]
    assert len(body) == MAX_GLOB_RESULTS


# --------------------------------------------------------------------------- #
# list_dir
# --------------------------------------------------------------------------- #


async def test_list_dir_lists_directories_first(project: Workspace) -> None:
    result = await ListDirTool(project).invoke({})

    body = result.content.splitlines()[1:]
    assert body[:4] == [".git/", "node_modules/", "src/", "tests/"]
    assert "README.md" in body


async def test_list_dir_shows_environment_and_ignored_directories(project: Workspace) -> None:
    (project.root / ".venv").mkdir()
    result = await ListDirTool(project).invoke({})

    body = result.content.splitlines()[1:]
    assert ".venv/" in body
    assert ".git/" in body
    assert "node_modules/" in body


async def test_list_dir_on_subdirectory(project: Workspace) -> None:
    result = await ListDirTool(project).invoke({"path": "src"})

    assert "src: 0 director(ies), 2 file(s)" in content_text(result.content)
    assert "src/app.py" in content_text(result.content)


async def test_list_dir_rejects_files(project: Workspace) -> None:
    result = await ListDirTool(project).invoke({"path": "README.md"})

    assert result.is_error
    assert "read_file" in content_text(result.content)


async def test_list_dir_missing_path(workspace: Workspace) -> None:
    result = await ListDirTool(workspace).invoke({"path": "nowhere"})

    assert result.is_error
    assert "Directory not found" in content_text(result.content)


async def test_list_dir_empty(workspace: Workspace) -> None:
    result = await ListDirTool(workspace).invoke({})

    assert "is empty" in content_text(result.content)
