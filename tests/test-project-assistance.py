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
from test_cli_e2e import cli_environment


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


@pytest.mark.parametrize("suffix,source,expected", [
    (".js", "export class Box { run(x) { return convert(x); } } function convert(x) { return x; }", ["class Box", "run(x)", "function convert(x)"]),
    (".ts", "interface Box { run(x: number): number; } export const convert = (x: number): number => x;", ["interface Box", "run(x: number)", "convert = (x: number)"]),
    (".tsx", "export function View() { return <div>Hello</div>; }", ["function View()"]),
    (".php", "<?php class Box { public function run($x) {return helper($x);} } function helper($x) {return $x;}", ["class Box", "function run($x)", "function helper($x)"]),
    (".go", "package main\ntype Box struct {}\nfunc (b Box) Run(x int) int { return x }", ["Box struct", "Run(x int) int"]),
    (".rs", "struct Box {} impl Box { fn run(&self, x: i32) -> i32 { x } }", ["struct Box", "fn run(&self, x: i32)"]),
    (".java", "class Box { int run(int x) { return x; } }", ["class Box", "int run(int x)"]),
    (".cpp", "struct Box {}; int convert(int x) { return x; }", ["struct Box", "int convert(int x)"]),
    (".cs", "class Box { int Run(int x) { return x; } }", ["class Box", "int Run(int x)"]),
    (".rb", "class Box\n def run(x)\n x\n end\nend", ["class Box", "def run(x)"]),
])
async def test_map_extracts_multilanguage_declarations_offline(workspace, suffix, source, expected):
    (workspace.root / ("code" + suffix)).write_text(source)
    mapping = RepositoryMap(workspace)
    text = await mapping.snapshot("Box")
    for declaration in expected:
        assert declaration in text
    assert await mapping.snapshot("Box") == text


async def test_map_ranks_referenced_symbols_above_unrelated_files(workspace):
    (workspace.root / "entry.ts").write_text("export function start() { return convert(); }")
    (workspace.root / "z-helper.ts").write_text("export function convert() { return 1; }")
    (workspace.root / "a-unrelated.ts").write_text("export function unrelated() { return 2; }")
    text = await RepositoryMap(workspace).snapshot("start")
    assert text.index("entry.ts") < text.index("z-helper.ts") < text.index("a-unrelated.ts")


async def test_map_labels_partial_syntax_trees(workspace):
    (workspace.root / "broken.ts").write_text("export function unfinished(x: number) {")
    text = await RepositoryMap(workspace).snapshot("unfinished")
    assert "partial outline: syntax errors" in text


def test_outline_high_line_numbers_do_not_corrupt_memory(tmp_path):
    # Isolate native parser failures so a regression cannot crash the test runner.
    probe = """
import sys
if sys.platform != "win32":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from slipagent.symbols import outline
raw = b"\\n" * 300 + b"function later() { return value; }\\n" * 100
for _ in range(10):
    text, tags = outline(raw, ".js")
    assert text.splitlines() == [f"{line}: function later()" for line in range(301, 401)]
    assert "def:later" in tags and "ref:value" in tags
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", probe],
        cwd=tmp_path, env=cli_environment(), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
        results = agent.history.steps[0].parts()["tool_results"]
        assert "Python syntax: PASSED" in str(results)
        assert "Python syntax: FAILED" not in str(results)
        assert "Python syntax: PASSED" in str(client.calls[1]["messages"])
        await agent.wait_for_compaction()
    finally:
        await registry.aclose()


async def test_large_source_outline_syntax_and_check_output(workspace):
    configure(workspace, python=sys.executable, checks=[{
        "name": "Full output", "argv": ["{python}", "-I", "-c", "print('x' * 100000 + ' END')"],
    }])
    source = "# " + "x" * 2_100_000 + "\n" + "\n".join(f"def item_{index}(): pass" for index in range(100))
    (workspace.root / "large.py").write_text(source)
    mapping = await RepositoryMap(workspace).snapshot("item_99")
    assert "def item_99()" in mapping
    registry = build_default_registry(workspace)
    try:
        result = await check_edit_batch(registry, [(ToolCall("1", "write_file", {"path": "large.py"}), ToolResult.ok("written"))], 1)
        assert "Python syntax: PASSED" in result
        assert "Full output: PASSED" in result
        assert "x" * 100000 + " END" in result
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


async def test_missing_python_check_is_retained_for_agent_without_chat_warning(workspace):
    registry = build_default_registry(workspace)
    events = []
    client = StubClient([
        completion("Editing", [ToolCall("1", "write_file", {"path": "code.py", "content": "value = 1\n"})]),
        completion("Done"),
    ])
    try:
        agent = Agent(client, registry, "test", on_event=events.append)
        assert await agent.run("edit") == "Done"
        await agent.wait_for_compaction()
        assert "SKIPPED — no selected project Python." in str(client.calls[1]["messages"])
        assert not any(event.kind == "warning" and "no selected project Python" in event.text for event in events)
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
