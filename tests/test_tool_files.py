"""read_file / write_file / edit_file behaviour."""

from __future__ import annotations

from slipagent.types import content_text

import pytest

from slipagent.tools.files import EditFileTool, ReadFileTool, WriteFileTool
from slipagent.workspace import Workspace


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


async def test_read_numbers_lines(project: Workspace) -> None:
    result = await ReadFileTool(project).invoke({"path": "src/app.py"})

    assert not result.is_error
    lines = result.content.splitlines()
    assert lines[0].startswith("src/app.py (6 lines, showing 1-6)")
    assert lines[2].strip().startswith("1") and "def main():" in lines[2]


async def test_read_offset_and_limit(project: Workspace) -> None:
    result = await ReadFileTool(project).invoke(
        {"path": "src/app.py", "offset": 5, "limit": 2}
    )

    assert "showing 5-6" in content_text(result.content)
    assert "def helper(value):" in content_text(result.content)
    assert "def main():" not in content_text(result.content)


async def test_read_truncation_is_announced(project: Workspace) -> None:
    result = await ReadFileTool(project).invoke({"path": "src/app.py", "limit": 2})

    assert "Truncated: 4 more line(s)" in content_text(result.content)
    assert "offset=3" in content_text(result.content)


async def test_read_missing_file(workspace: Workspace) -> None:
    result = await ReadFileTool(workspace).invoke({"path": "nope.txt"})

    assert result.is_error
    assert "File not found" in content_text(result.content)


async def test_read_directory_suggests_list_dir(project: Workspace) -> None:
    result = await ReadFileTool(project).invoke({"path": "src"})

    assert result.is_error
    assert "list_dir" in content_text(result.content)


async def test_read_rejects_binary(workspace: Workspace) -> None:
    (workspace.root / "blob.bin").write_bytes(b"\x00\x01\x02\xff")

    result = await ReadFileTool(workspace).invoke({"path": "blob.bin"})

    assert result.is_error
    assert "binary" in content_text(result.content)


async def test_read_cannot_escape(project: Workspace) -> None:
    result = await ReadFileTool(project).invoke({"path": "../../etc/passwd"})

    assert result.is_error
    assert "outside the workspace" in content_text(result.content)


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #


async def test_write_creates_file_and_parents(workspace: Workspace) -> None:
    result = await WriteFileTool(workspace).invoke(
        {"path": "deep/nested/new.py", "content": "x = 1\n"}
    )

    assert not result.is_error
    assert "Created" in content_text(result.content)
    assert (workspace.root / "deep" / "nested" / "new.py").read_text() == "x = 1\n"


async def test_write_overwrite_reports_updated(project: Workspace) -> None:
    result = await WriteFileTool(project).invoke(
        {"path": "README.md", "content": "# Replaced\n"}
    )

    assert "Updated" in content_text(result.content)
    assert (project.root / "README.md").read_text() == "# Replaced\n"


async def test_write_cannot_escape(workspace: Workspace) -> None:
    result = await WriteFileTool(workspace).invoke(
        {"path": "../evil.txt", "content": "x"}
    )

    assert result.is_error
    assert "outside the workspace" in content_text(result.content)


# --------------------------------------------------------------------------- #
# edit_file
# --------------------------------------------------------------------------- #


async def test_edit_replaces_unique_match(project: Workspace) -> None:
    result = await EditFileTool(project).invoke(
        {
            "path": "src/app.py",
            "old_string": "    return value * 2",
            "new_string": "    return value**2",
        }
    )

    assert not result.is_error
    assert "line 6" in content_text(result.content)
    assert "return value**2" in (project.root / "src" / "app.py").read_text()


async def test_edit_reports_ambiguous_match(project: Workspace) -> None:
    result = await EditFileTool(project).invoke(
        {
            "path": "src/app.py",
            "old_string": "\n",
            "new_string": "\n",
        }
    )

    assert result.is_error
    assert "matched 6 times" in content_text(result.content)
    assert "replace_all=true" in content_text(result.content)


async def test_edit_rejects_missing_match(project: Workspace) -> None:
    result = await EditFileTool(project).invoke(
        {
            "path": "src/app.py",
            "old_string": "def nonexistent():",
            "new_string": "x",
        }
    )

    assert result.is_error
    assert "No match for old_string" in content_text(result.content)


async def test_edit_replace_all(project: Workspace) -> None:
    result = await EditFileTool(project).invoke(
        {
            "path": "src/app.py",
            "old_string": "\n",
            "new_string": "\n\n",
            "replace_all": True,
        }
    )

    assert not result.is_error
    assert "6 occurrence(s)" in content_text(result.content)
    assert "\n\n\n" in (project.root / "src" / "app.py").read_text()


async def test_edit_can_delete_with_empty_string(project: Workspace) -> None:
    result = await EditFileTool(project).invoke(
        {
            "path": "src/app.py",
            "old_string": "\n\ndef helper(value):\n    return value * 2\n",
            "new_string": "",
        }
    )

    assert not result.is_error
    text = (project.root / "src" / "app.py").read_text()
    assert "helper" not in text
    assert "def main():" in text


async def test_edit_rejects_empty_old_string(project: Workspace) -> None:
    result = await EditFileTool(project).invoke(
        {"path": "src/app.py", "old_string": "", "new_string": "x"}
    )

    assert result.is_error
    assert "must not be empty" in content_text(result.content)


@pytest.mark.parametrize("bad", [{"path": "src/app.py"}, {"old_string": "x"}])
async def test_edit_requires_both_strings(project: Workspace, bad: dict) -> None:
    result = await EditFileTool(project).invoke(bad)

    assert result.is_error
    assert "missing required argument" in content_text(result.content)


async def test_edit_preserves_crlf(workspace: Workspace) -> None:
    target = workspace.root / "windows.txt"
    target.write_bytes(b"first\r\nsecond\r\n")
    result = await EditFileTool(workspace).invoke(
        {"path": "windows.txt", "old_string": "second", "new_string": "updated"}
    )
    assert not result.is_error
    assert target.read_bytes() == b"first\r\nupdated\r\n"


async def test_write_reports_utf8_byte_count(workspace: Workspace) -> None:
    result = await WriteFileTool(workspace).invoke({"path": "text.txt", "content": "é"})
    assert "2 bytes" in content_text(result.content)
