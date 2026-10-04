"""Repository orientation and checks remain bounded, current, and project-scoped."""

import json
import subprocess
import sys

import pytest

from slipagent.agent import Agent
from slipagent.checks import check_edit_batch
from slipagent.environment import load_project_settings
from slipagent.repomap import RepositoryMap
from slipagent.tools import build_default_registry
from slipagent.tools.base import ToolResult
from slipagent.types import ToolCall
from test_agent import StubClient, completion


async def test_map_refreshes_edits_and_deletions_without_reading_outside_workspace(workspace, tmp_path):
    source = workspace.root / "module.py"
    source.write_text("def first(value=123):\n    return value\n")
    mapping = RepositoryMap(workspace)
    text = await mapping.snapshot("first")
    assert "def first(value)" in text and "123" not in text
    source.write_text("class Second:\n    pass\n")
    text = await mapping.snapshot("Second")
    assert "class Second" in text and "first" not in text
    source.unlink()
    assert "module.py" not in await mapping.snapshot("")
    (workspace.root / "escape.py").symlink_to(tmp_path.parent / "outside.py")
    assert "escape.py" not in await mapping.snapshot("")


async def test_map_respects_gitignore_and_keeps_text_within_budget(workspace):
    subprocess.run(["git", "init", "-q", str(workspace.root)], check=True)
    (workspace.root / ".gitignore").write_text("private.py\n")
    (workspace.root / "private.py").write_text("def secret(): pass")
    (workspace.root / "public.py").write_text("def visible(): pass")
    (workspace.root / ".venv").mkdir()
    (workspace.root / ".venv" / "hidden.py").write_text("def hidden(): pass")
    text = await RepositoryMap(workspace).snapshot("visible", 1000)
    assert "visible" in text
    assert "private.py" not in text and "hidden.py" not in text
    assert len(text) <= 1000
    for limit in (100, 400, 480, 550):
        assert len(await RepositoryMap(workspace).snapshot("visible", limit)) <= limit


def configure(workspace, **settings):
    (workspace.root / ".slipagent").mkdir(exist_ok=True)
    (workspace.root / ".slipagent" / "project.json").write_text(json.dumps(settings))


async def test_python_check_uses_final_batch_state_and_does_not_import_code(workspace):
    configure(workspace, python=sys.executable)
    registry = build_default_registry(workspace)
    client = StubClient([
        completion("Editing", [ToolCall("1", "write_file", {"path": "code.py", "content": "def incomplete("}),
                               ToolCall("2", "write_file", {"path": "code.py", "content": "raise RuntimeError('must not execute')\n"})]),
        completion("Done"),
    ])
    try:
        agent = Agent(client, registry, "test")
        assert await agent.run("edit") == "Done"
        results = agent.history.posts[0].parts()["tool_results"]
        assert "Python syntax: PASSED" in str(results)
        assert "Python syntax: FAILED" not in str(results)
        assert "Python syntax: PASSED" in str(client.calls[1]["messages"])
        await agent.wait_for_compaction()
    finally:
        await registry.aclose()


async def test_syntax_failures_and_absent_environment_are_reported(workspace):
    registry = build_default_registry(workspace)
    (workspace.root / "code.py").write_text("def broken(")
    batch = [(ToolCall("1", "write_file", {"path": "code.py"}), ToolResult.ok("written"))]
    try:
        assert "SKIPPED" in await check_edit_batch(registry, batch, 1)
        configure(workspace, python=sys.executable)
        result = await check_edit_batch(registry, batch, 1)
        assert "FAILED" in result and "never closed" in result
        assert "Command log" in result
    finally:
        await registry.aclose()


@pytest.mark.parametrize("settings", [
    {"python_syntax": 1}, {"checks": [{}]}, {"checks": [{"name": "bad", "argv": ["x"], "timeout": 1000}]},
])
def test_invalid_check_configuration_fails_at_load(workspace, settings):
    configure(workspace, **settings)
    with pytest.raises(Exception, match="Invalid .slipagent/project.json"):
        load_project_settings(workspace)


async def test_configured_checks_expand_paths_as_arguments_without_shell(workspace):
    configure(workspace, python=sys.executable, python_syntax=False, checks=[{
        "name": "Arguments", "argv": ["{python}", "-I", "-c", "import sys; print(repr(sys.argv[1:]))", "{files}"],
        "extensions": [".py"],
    }])
    name = "space ; file.py"
    (workspace.root / name).write_text("")
    registry = build_default_registry(workspace)
    try:
        result = await check_edit_batch(registry, [(ToolCall("1", "write_file", {"path": name}), ToolResult.ok("written"))], 1)
        assert "Arguments: PASSED" in result and str(workspace.root / name) in result
    finally:
        await registry.aclose()
