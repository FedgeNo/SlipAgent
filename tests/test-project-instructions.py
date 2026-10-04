"""Instruction discovery stays scoped and reaches the model before its first action."""

from __future__ import annotations

from pathlib import Path

import pytest

from slipagent.agent import Agent, build_system_prompt
from slipagent.instructions import load_project_instructions
from slipagent.tools import ToolRegistry
from slipagent.workspace import Workspace, WorkspaceError


def test_harness_operating_guidance_does_not_depend_on_project_notes(workspace):
    prompt = build_system_prompt(str(workspace.root))
    assert "read_command_output" in prompt and "next_offset" in prompt
    assert "Project Python Environment" in prompt
    assert "src/slipagent/" not in prompt
    assert "test-runtime.py" not in prompt
    project_guidance = "Project-specific check: run ./verify-project."
    combined = build_system_prompt(str(workspace.root), project_instructions=project_guidance)
    assert combined.index("read_command_output") < combined.index("## Project instructions")
    assert project_guidance in combined


@pytest.mark.parametrize("name", [
    "CLAUDE.md", "AGENTS.md", "AGENTS.override.md", ".cursorrules", ".clinerules",
    ".windsurfrules", ".github/copilot-instructions.md",
])
def test_reads_recognized_instruction_files(workspace, name):
    path = workspace.root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Project rule: preserve existing behavior.", encoding="utf-8")
    content = load_project_instructions(workspace)
    assert f"### {name}\nProject rule: preserve existing behavior." in content


@pytest.mark.parametrize("folder, suffix", [
    (".cursor/rules", ".mdc"), (".claude/rules", ".md"),
    (".github/instructions", ".instructions.md"), (".clinerules", ".md"),
    (".windsurf/rules", ".md"),
])
def test_reads_nested_rule_directory_files_in_stable_order(workspace, folder, suffix):
    parent = workspace.root / folder
    (parent / "nested").mkdir(parents=True)
    (parent / f"z-rule{suffix}").write_text("last rule", encoding="utf-8")
    (parent / "nested" / f"a-rule{suffix}").write_text("first rule", encoding="utf-8")
    content = load_project_instructions(workspace)
    assert content.index("first rule") < content.index("last rule")
    assert f"{folder}/nested/a-rule{suffix}" in content


def test_discovery_does_not_scan_arbitrary_project_files_or_wrong_extensions(workspace):
    (workspace.root / "src").mkdir()
    (workspace.root / "src" / "AGENTS.md").write_text("nested guidance", encoding="utf-8")
    folder = workspace.root / ".cursor" / "rules"
    folder.mkdir(parents=True)
    (folder / "readme.md").write_text("wrong extension", encoding="utf-8")
    (workspace.root / ".env").write_text("NOT_A_RULE=private", encoding="utf-8")
    assert load_project_instructions(workspace) == ""


def test_rule_content_is_verbatim_and_survives_reset(workspace):
    text = '---\nglobs: src/*.tsx\nalwaysApply: false\n---\nUse {"setting": true}.\n'
    (workspace.root / "CLAUDE.md").write_text(text, encoding="utf-8")
    prompt = build_system_prompt(str(workspace.root), project_instructions=load_project_instructions(workspace))
    assert text in prompt
    assert "path and glob restrictions" in prompt
    assert "Explicit user instructions take precedence" in prompt
    assert "read any nested instruction files" in prompt
    agent = Agent(client=None, registry=ToolRegistry(), model="test", system_prompt=prompt)
    agent.reset()
    assert agent.messages[0].content == prompt


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_instruction_symlinks_cannot_escape_workspace(tmp_path, kind):
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace(root)
    outside = tmp_path / "outside"
    if kind == "file":
        outside.write_text("outside rules", encoding="utf-8")
        link = workspace.root / "AGENTS.md"
    else:
        outside.mkdir()
        (outside / "rule.mdc").write_text("outside rules", encoding="utf-8")
        link = workspace.root / ".cursor" / "rules"
        link.parent.mkdir()
    link.symlink_to(outside, target_is_directory=kind == "directory")
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        load_project_instructions(workspace)


def test_duplicate_symlinked_instruction_content_is_loaded_once(workspace):
    source = workspace.root / "CLAUDE.md"
    source.write_text("a unique rule", encoding="utf-8")
    (workspace.root / "AGENTS.md").symlink_to(source)
    assert load_project_instructions(workspace).count("a unique rule") == 1


@pytest.mark.parametrize("failure", ["permission", "encoding", "directory"])
def test_unreadable_instruction_file_reports_its_path(workspace, monkeypatch, failure):
    path = workspace.root / "AGENTS.md"
    if failure == "directory":
        path.mkdir()
    elif failure == "encoding":
        path.write_bytes(b"\xff")
    else:
        path.write_text("a rule", encoding="utf-8")
        original = Path.read_text
        def read_text(self, *args, **kwargs):
            if self == path:
                raise PermissionError("permission denied")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(WorkspaceError, match="cannot read project instructions AGENTS.md"):
        load_project_instructions(workspace)
