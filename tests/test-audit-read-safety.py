"""Workspace readers reject files swapped after path validation."""

import os

import pytest

from slipagent.config import ConfigError
from slipagent.environment import load_project_options
from slipagent.instructions import load_project_instructions
from slipagent.lsp import _document
from slipagent.repomap import RepositoryMap
from slipagent.workspace import Workspace, WorkspaceError


@pytest.mark.skipif(os.name != "posix", reason="Descriptor-based no-follow traversal")
@pytest.mark.parametrize("reader,name,content", [
    ("instructions", "AGENTS.md", "External instructions"),
    ("settings", ".slipagent/project.json", '{"python":"external-python"}'),
    ("outline", "sample.py", "def external(): pass\n"),
    ("navigation", "sample.py", "def external(): pass\n"),
])
def test_swapped_instruction_settings_and_outline_files(tmp_path, monkeypatch, reader, name, content):
    root = tmp_path / "project"
    root.mkdir()
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    outside = tmp_path / "outside"
    outside.write_text(content)
    workspace = Workspace(root)
    resolve = Workspace.resolve

    def swapped(owner, path):
        resolved = resolve(owner, path)
        if resolved == target:
            target.unlink()
            target.symlink_to(outside)
        return resolved

    monkeypatch.setattr(Workspace, "resolve", swapped)
    if reader == "instructions":
        with pytest.raises(WorkspaceError):
            load_project_instructions(workspace)
    elif reader == "settings":
        with pytest.raises(ConfigError):
            load_project_options(workspace)
    elif reader == "navigation":
        with pytest.raises(OSError):
            _document(workspace.resolve(name))
    else:
        assert RepositoryMap(workspace)._outline(target) == ("[outline unavailable; read source]", set())
