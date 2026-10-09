"""Edits recover from model mistakes without guessing what to write."""

from slipagent.types import content_text

import pytest

from slipagent.tools.files import EditFileTool


async def test_missing_match_reports_actual_numbered_source_without_writing(workspace):
    target = workspace.root / "app.py"
    original = "def compute(value):\n    return value * 2\n"
    target.write_text(original)
    result = await EditFileTool(workspace).invoke({
        "path": "app.py", "old_string": "    return value * 3", "new_string": "    return value * 4",
    })
    assert result.is_error
    assert "2\t    return value * 2" in content_text(result.content)
    assert "No changes" in content_text(result.content)
    assert target.read_text() == original


async def test_missing_match_identifies_existing_replacement(workspace):
    target = workspace.root / "app.py"
    target.write_text("new_value = 2\n")
    result = await EditFileTool(workspace).invoke({
        "path": "app.py", "old_string": "old_value = 1", "new_string": "new_value = 2",
    })
    assert result.is_error
    assert "already exists at line 1" in content_text(result.content)
    assert target.read_text() == "new_value = 2\n"


async def test_multiple_edits_use_original_file_and_return_diff(workspace):
    target = workspace.root / "app.py"
    target.write_bytes(b"first = 1\r\nsecond = 2\r\n")
    result = await EditFileTool(workspace).invoke({"path": "app.py", "edits": [
        {"old_string": "second = 2", "new_string": "last = 3"},
        {"old_string": "first = 1", "new_string": "second = 2\nextra = 0"},
    ]})
    assert not result.is_error, result.content
    assert target.read_bytes() == b"second = 2\r\nextra = 0\r\nlast = 3\r\n"
    assert "-first = 1" in content_text(result.content) and "+last = 3" in content_text(result.content)


@pytest.mark.parametrize("edits", [
    [{"old_string": "first", "new_string": "changed"}, {"old_string": "absent", "new_string": "new"}],
    [{"old_string": "first", "new_string": "changed"}, {"old_string": "first = 1", "new_string": "new"}],
    [{"old_string": "first", "new_string": "changed"}, {"old_string": "first", "new_string": "different"}],
    [], [{"old_string": "", "new_string": "empty match"}],
])
async def test_invalid_multi_edit_leaves_original_intact(workspace, edits):
    target = workspace.root / "app.py"
    target.write_bytes(b"first = 1\nsecond = 2\n")
    result = await EditFileTool(workspace).invoke({"path": "app.py", "edits": edits})
    assert result.is_error
    assert target.read_bytes() == b"first = 1\nsecond = 2\n"


async def test_edit_forms_cannot_be_mixed(workspace):
    target = workspace.root / "app.py"
    target.write_text("original")
    result = await EditFileTool(workspace).invoke({
        "path": "app.py", "old_string": "original", "new_string": "legacy",
        "edits": [{"old_string": "original", "new_string": "batch"}],
    })
    assert result.is_error
    assert target.read_text() == "original"


async def test_edit_returns_diff_beyond_old_preview_limit(workspace):
    before = "\n".join(f"old_{index}" for index in range(2000))
    after = "\n".join(f"new_{index}" for index in range(2000))
    (workspace.root / "large.txt").write_text(before)
    result = await EditFileTool(workspace).invoke({"path": "large.txt", "old_string": before, "new_string": after})
    assert not result.is_error
    assert "+new_1999" in result.content and "-old_1999" in result.content
    assert "Diff truncated" not in result.content


async def test_diagnostic_preserves_long_source_lines(workspace):
    target = workspace.root / "app.py"
    target.write_text("value = " + "x" * 50000)
    result = await EditFileTool(workspace).invoke({"path": "app.py", "old_string": "value = y", "new_string": "value = z"})
    assert result.is_error
    assert "value = " + "x" * 50000 in result.content
