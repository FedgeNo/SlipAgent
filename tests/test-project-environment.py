import json
import os
import shlex
import subprocess
import sys
import venv

import pytest

from slipagent.config import ConfigError
from slipagent.environment import ProjectEnvironment, load_project_settings
from slipagent.workspace import Workspace
from slipagent.agent import Agent
from slipagent.tools import build_default_registry
from slipagent.capabilities import ModelCapabilities


async def test_pipless_environment_does_not_advertise_unavailable_commands(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    venv.EnvBuilder(with_pip=False).create(tmp_path / ".venv")
    info = await ProjectEnvironment(Workspace(tmp_path)).snapshot()
    assert info["pip_available"] is False and info["pytest_available"] is False
    assert info["install_command"] is None
    assert info["test_command"] is None
    if info["ensurepip_available"]:
        quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
        assert info["bootstrap_command"] == quote([info["interpreter"], "-m", "ensurepip", "--upgrade"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell quoting")
async def test_pipless_environment_uses_uv_with_the_exact_interpreter(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    root.mkdir()
    venv.EnvBuilder(with_pip=False).create(root / ".venv")
    uv = str(tmp_path / "tools with spaces" / "uv")
    monkeypatch.setattr("shutil.which", lambda name: uv if name == "uv" else None)
    info = await ProjectEnvironment(Workspace(root)).snapshot()
    assert info["pip_available"] is False
    assert shlex.split(info["install_command"]) == [uv, "pip", "install", "--python", info["interpreter"], "PACKAGE"]
    assert info["bootstrap_command"] is None


async def test_installed_modules_refresh_cached_environment_guidance(tmp_path, monkeypatch):
    import asyncio
    from pathlib import Path
    monkeypatch.setattr("shutil.which", lambda name: None)
    venv.EnvBuilder(with_pip=False).create(tmp_path / ".venv")
    environment = ProjectEnvironment(Workspace(tmp_path))
    before = await environment.snapshot()
    process = await asyncio.create_subprocess_exec(
        before["interpreter"], "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    site = Path(stdout.decode().strip())
    assert site.is_relative_to(tmp_path)
    for name in ("pip", "pytest"):
        (site / (name + ".py")).write_text("print('scratch module')\n")
    after = await environment.snapshot()
    assert after["pip_available"] is True and after["pytest_available"] is True
    assert "-m pip install" in after["install_command"]
    assert "-m pytest" in after["test_command"]
    for name in ("pip", "pytest"):
        (site / (name + ".py")).unlink()
    removed = await environment.snapshot()
    assert removed["pip_available"] is False and removed["pytest_available"] is False


async def test_discovers_project_interpreter_without_changing_path(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    venv.EnvBuilder(with_pip=False).create(root / ".venv")
    original = os.environ.get("PATH")
    info = await ProjectEnvironment(Workspace(root)).snapshot()
    assert info["status"] == "selected"
    assert info["interpreter"].startswith(str(root / ".venv"))
    assert info["prefix"] == str(root / ".venv")
    assert info["is_virtual_environment"] is True
    assert info["test_command"] is None  # This disposable venv has no pytest.
    assert info["pip_available"] is False
    assert os.environ.get("PATH") == original


async def test_missing_and_ambiguous_environments_need_explicit_selection(tmp_path):
    environment = ProjectEnvironment(Workspace(tmp_path))
    assert (await environment.snapshot())["status"] == "unconfigured"
    for name in (".venv", "venv"):
        venv.EnvBuilder(with_pip=False).create(tmp_path / name)
    info = await environment.snapshot()
    assert info["status"] == "ambiguous"
    assert len(info["candidates"]) == 2
    assert "interpreter" not in info


async def test_config_changes_are_revalidated_and_override_is_explicit(tmp_path):
    folder = tmp_path / ".slipagent"
    folder.mkdir()
    config = folder / "project.json"
    config.write_text(json.dumps({"python": "missing/python"}))
    environment = ProjectEnvironment(Workspace(tmp_path))
    with pytest.raises(ConfigError, match="interpreter"):
        await environment.snapshot()
    environment.python_override = sys.executable
    assert (await environment.snapshot())["source"] == "--python"
    environment.python_override = None
    config.write_text(json.dumps({"python": sys.executable}))
    assert (await environment.snapshot())["source"] == ".slipagent/project.json"


@pytest.mark.parametrize("settings", [{"python": 12}, {"log_quota_bytes": 0}, {"log_quota_bytes": True}, {"unknown": 1}])
def test_invalid_project_settings_are_not_silently_ignored(tmp_path, settings):
    (tmp_path / ".slipagent").mkdir()
    (tmp_path / ".slipagent/project.json").write_text(json.dumps(settings))
    with pytest.raises(ConfigError):
        load_project_settings(Workspace(tmp_path))


async def test_environment_is_explicit_in_json_object_model_context(tmp_path):
    registry = build_default_registry(Workspace(tmp_path))
    registry.services["project_environment"].python_override = sys.executable
    profile = ModelCapabilities({}, [{"tag": "stub", "status": 0, "context_length": 1000000,
                                     "supported_parameters": ["tools", "response_format"]}])
    try:
        agent = Agent(object(), registry, "test")
        view = await agent._context_view(registry.specs(), 1, capabilities=profile)
        system = "\n".join(message.content or "" for message in view if message.role == "system")
        assert "Project Python Environment" in system
        assert sys.executable in system and "-m pytest" in system
        assert "ordinary plain text" in system
    finally:
        await registry.aclose()
