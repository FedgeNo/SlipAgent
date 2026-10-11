"""Environment setup preserves existing data and does not repeat creation."""

import asyncio
import os
import sys
import venv
from unittest.mock import AsyncMock

import pytest

from slipagent.environment import ProjectEnvironment
from slipagent.tools import build_default_registry
from slipagent.tools.base import ToolResult
from slipagent.tools.python import CreatePythonEnvironmentTool
from slipagent.workspace import Workspace


@pytest.fixture
def setup_tool(tmp_path):
    environment = ProjectEnvironment(Workspace(tmp_path))
    tool = CreatePythonEnvironmentTool(environment)

    async def create(target, interpreter):
        # Exercise real discovery without bootstrapping packages in the fixture.
        venv.EnvBuilder(with_pip=False).create(target)

    tool._create = AsyncMock(side_effect=create)
    return tool, environment


async def test_repeated_calls_reuse_selected_environment_without_creating_others(setup_tool, tmp_path):
    tool, environment = setup_tool
    result = await tool.run()
    assert not result.is_error and result.content["created"] is True
    selected = await environment.snapshot()
    assert selected["is_virtual_environment"] is True
    assert selected["source"] == "create_python_environment"
    assert selected["prefix"] == str(tmp_path / ".venv")
    sentinel = tmp_path / ".venv" / "user-data"
    sentinel.write_text("keep")
    for _ in range(5):
        again = await tool.run(path="another-environment", python="unavailable-python")
        assert not again.is_error and again.content["created"] is False
        assert again.content["environment"]["interpreter"] == selected["interpreter"]
    tool._create.assert_awaited_once()
    assert sentinel.read_text() == "keep" and not (tmp_path / "another-environment").exists()


async def test_nested_existing_environment_is_reused_without_modification(setup_tool, tmp_path):
    tool, environment = setup_tool
    target = tmp_path / "nested" / ".venv"
    venv.EnvBuilder(with_pip=False).create(target)
    sentinel = target / "user-data"
    sentinel.write_text("keep")
    result = await tool.run(path="nested/.venv")
    assert not result.is_error and result.content["created"] is False
    assert (await environment.snapshot())["prefix"] == str(target)
    assert sentinel.read_text() == "keep"
    tool._create.assert_not_awaited()


@pytest.mark.parametrize("target", ["partial", "unrelated", "."])
async def test_existing_directories_are_never_recreated(setup_tool, tmp_path, target):
    tool, environment = setup_tool
    folder = tmp_path / target
    folder.mkdir(exist_ok=True)
    sentinel = folder / "user-data"
    sentinel.write_text("keep")
    if target == "partial":
        (folder / "pyvenv.cfg").write_text("incomplete")
    result = await tool.run(path=target)
    assert result.is_error and sentinel.read_text() == "keep"
    assert environment.python_override is None
    tool._create.assert_not_awaited()


async def test_failed_creation_is_not_retried_over_partial_directory(setup_tool, tmp_path):
    tool, environment = setup_tool
    tool._create = AsyncMock(return_value=ToolResult.error("creation failed"))
    assert (await tool.run()).is_error
    assert (tmp_path / ".venv").is_dir()
    assert (await tool.run()).is_error
    tool._create.assert_awaited_once()
    assert environment.python_override is None


async def test_interruption_leaves_partial_directory_untouched_on_retry(setup_tool, tmp_path):
    tool, environment = setup_tool
    tool._create = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await tool.run()
    assert (tmp_path / ".venv").is_dir()
    assert (await tool.run()).is_error
    tool._create.assert_awaited_once()
    assert environment.python_override is None


async def test_confined_paths_and_invalid_python_create_nothing(setup_tool, tmp_path):
    tool, _ = setup_tool
    assert (await tool.run(path="../outside")).is_error
    assert (await tool.run(python="missing-python-executable")).is_error
    assert not (tmp_path / ".venv").exists()
    tool._create.assert_not_awaited()


async def test_configured_interpreter_is_preserved(setup_tool, tmp_path):
    tool, environment = setup_tool
    environment.python_override = sys.executable
    result = await tool.run()
    assert not result.is_error and result.content["created"] is False
    assert environment.python_override == sys.executable
    assert not (tmp_path / ".venv").exists()
    tool._create.assert_not_awaited()


async def test_ambiguous_environments_are_not_changed(setup_tool, tmp_path):
    tool, _ = setup_tool
    for name in (".venv", "venv"):
        venv.EnvBuilder(with_pip=False).create(tmp_path / name)
    assert (await tool.run()).is_error
    tool._create.assert_not_awaited()


async def test_creation_uses_venv_without_dependency_installation(setup_tool, tmp_path, monkeypatch):
    tool, _ = setup_tool
    process = AsyncMock(returncode=0)
    spawn = AsyncMock(return_value=process)
    capture = AsyncMock(return_value=("", "", False))
    monkeypatch.setattr("slipagent.tools.python.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("slipagent.tools.python.capture_process", capture)
    assert await CreatePythonEnvironmentTool._create(tool, tmp_path / ".venv", sys.executable) is None
    assert spawn.call_args.args == (sys.executable, "-I", "-m", "venv", str(tmp_path / ".venv"))
    capture.assert_awaited_once_with(process, 120)


async def test_tool_borrows_environment_owner_across_registry_rebuild(workspace):
    registry = build_default_registry(workspace)
    candidate = build_default_registry(workspace, services=registry.services)
    try:
        assert candidate.get("create_python_environment").environment is registry.resources.environment
        assert "create_python_environment" in candidate.builtin_names
    finally:
        await candidate.aclose()
        await registry.aclose()
