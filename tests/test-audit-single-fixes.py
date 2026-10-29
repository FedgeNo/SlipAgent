"""Regression tests for the single-solution audit findings fixed in this pass.

Each test names the finding it covers and fails against the code as it was
when the audit was written.
"""

import pytest

from slipagent.agent import Agent
from slipagent.context import ContextError, RecallHistoryTool
from slipagent.requests import MAX_MEMORY_ATTEMPTS
from slipagent.tools.base import ToolRegistry
from slipagent.tools.editing import apply_edits
from slipagent.tools.files import EditFileTool, ReadFileTool
from slipagent.tools.search import GrepTool
from slipagent.types import content_text
from test_agent import RecordingTool, StubClient, build_agent, completion


def test_edit_error_lines_count_cr_only_breaks() -> None:
    """L-13: CR-only files number their lines like every other reader does."""
    text = "first = 1\rsecond = 2\rthird = 3\r"
    with pytest.raises(ValueError) as missing:
        apply_edits(text, [("missing = 0", "second = 2", False)])
    assert "Replacement text already exists at line 2" in str(missing.value)

    with pytest.raises(ValueError) as repeated:
        apply_edits("value = 1\rvalue = 2\rvalue = 3\r", [("value", "x", False)])
    assert "old_string matched 3 times (lines 1, 2, 3)" in str(repeated.value)


async def test_grep_reports_truncation_only_when_more_matches_exist(workspace) -> None:
    """L-16: a search with exactly max_results matches is complete."""
    (workspace.root / "a.py").write_text("needle\nneedle\n", encoding="utf-8")
    result = await GrepTool(workspace).run(pattern="needle", max_results=2)
    body = content_text(result.content)
    assert not result.is_error
    assert body.startswith("2 match(es) for 'needle'")
    assert "stopped at the" not in body


async def test_grep_still_reports_truncation_when_more_matches_exist(workspace) -> None:
    """L-16: a search with more matches than the limit still says so."""
    (workspace.root / "a.py").write_text("needle\nneedle\nneedle\n", encoding="utf-8")
    result = await GrepTool(workspace).run(pattern="needle", max_results=2)
    body = content_text(result.content)
    assert body.startswith("2+ match(es) for 'needle'")
    assert "stopped at the 2-result limit" in body


async def test_recall_history_rejects_step_ids_below_one() -> None:
    """L-25: a direct caller cannot ask for another step by accident."""
    agent, _ = build_agent([completion("Done")])
    await agent.run("go")
    tool = RecallHistoryTool(agent.history)
    for step_id in (0, -1):
        result = await tool.run(step_id=step_id)
        assert result.is_error
        assert f"No history step {step_id}" in content_text(result.content)
    assert not (await tool.run(step_id=1)).is_error


async def test_edit_file_reports_the_read_limit_like_read_file(workspace, monkeypatch) -> None:
    """L-11: both file tools refuse an oversized file with the same message."""
    from slipagent.tools import files
    monkeypatch.setattr(files, "MAX_READ_BYTES", 64)
    target = workspace.root / "big.py"
    original = "x = 1\n" * 20
    target.write_text(original, encoding="utf-8")
    read = await ReadFileTool(workspace).run(path="big.py")
    edit = await EditFileTool(workspace).run(path="big.py", old_string="x = 1", new_string="x = 2")
    assert read.is_error and edit.is_error
    assert content_text(edit.content) == content_text(read.content)
    assert target.read_text(encoding="utf-8") == original


async def test_invalid_output_message_counts_the_attempts_in_this_step() -> None:
    """L-04: the message names the attempt budget it actually used."""
    client = StubClient([completion('{"response":"","tool_calls":[{"name":"record","arguments":"["}]}')
                         for _ in range(200)])
    agent = Agent(client, ToolRegistry([RecordingTool()]), "test")
    with pytest.raises(ContextError, match=f"{MAX_MEMORY_ATTEMPTS} attempts in this step"):
        await agent.run("work")
