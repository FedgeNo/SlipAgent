"""The model receives the actual harness math environment on every tool channel."""

import importlib.metadata
import os
import shlex
import subprocess
import sys

import pytest
from packaging.requirements import Requirement

from slipagent.agent import Agent
from slipagent.capabilities import RequestProfile
from slipagent.environment import harness_python_environment
from slipagent.environment import ensure_harness_math
from slipagent.config import ConfigError
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, content_text
from test_agent import StubClient


@pytest.fixture(autouse=True)
def math_setup_contract(monkeypatch):
    monkeypatch.setattr("slipagent.environment.math_requirements", lambda: {
        "numpy": Requirement("numpy>=1.26,<3"), "scipy": Requirement("scipy>=1.12,<2")})
    async def verify():
        pass
    monkeypatch.setattr("slipagent.environment.verify_harness_math", verify)


async def test_math_setup_installs_missing_dependency_with_own_python_and_constraints(monkeypatch):
    snapshots = iter([
        {"libraries": {"numpy": {"available": False}, "scipy": {"available": True, "version": "1.14"}}},
        {"libraries": {"numpy": {"available": True, "version": "2.0"}, "scipy": {"available": True, "version": "1.14"}}},
    ])
    monkeypatch.setattr("slipagent.environment.harness_python_environment", lambda: next(snapshots))
    monkeypatch.setattr(sys, "prefix", "/harness-venv")
    monkeypatch.setattr("slipagent.environment.importlib.metadata.requires",
                        lambda name: ["numpy>=1.26,<3", "scipy>=1.12,<2"])
    monkeypatch.setattr("slipagent.environment.importlib.util.find_spec", lambda name: object())
    commands = []

    async def spawn(*args, **kwargs):
        commands.append(args)
        return type("Process", (), {"returncode": 0})()

    async def capture(process, timeout):
        return "installed", "", False

    monkeypatch.setattr("slipagent.environment.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("slipagent.tools.shell.capture_process", capture)
    await ensure_harness_math()
    assert commands == [(sys.executable, "-I", "-m", "pip", "install", str(Requirement("numpy>=1.26,<3")))]


async def test_math_setup_skips_installer_when_ready(monkeypatch):
    monkeypatch.setattr("slipagent.environment.harness_python_environment",
                        lambda: {"libraries": {"numpy": {"available": True, "version": "2.0"}}})
    await ensure_harness_math()


async def test_math_setup_reports_missing_installer_without_global_install(monkeypatch):
    monkeypatch.setattr("slipagent.environment.harness_python_environment",
                        lambda: {"libraries": {"numpy": {"available": False}}})
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    with pytest.raises(ConfigError, match="own virtual environment"):
        await ensure_harness_math()


@pytest.mark.parametrize("timed_out", [False, True])
async def test_math_setup_reports_installation_failure(monkeypatch, timed_out):
    monkeypatch.setattr("slipagent.environment.harness_python_environment",
                        lambda: {"libraries": {"numpy": {"available": False}}})
    monkeypatch.setattr(sys, "prefix", "/harness-venv")
    monkeypatch.setattr("slipagent.environment.importlib.metadata.requires", lambda name: ["numpy>=1.26,<3"])
    monkeypatch.setattr("slipagent.environment.importlib.util.find_spec", lambda name: object())

    async def spawn(*args, **kwargs):
        return type("Process", (), {"returncode": 1})()

    async def capture(process, timeout):
        return "", "package index unreachable", timed_out

    monkeypatch.setattr("slipagent.environment.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("slipagent.tools.shell.capture_process", capture)
    with pytest.raises(ConfigError, match="timed out" if timed_out else "package index unreachable"):
        await ensure_harness_math()


def test_math_metadata_uses_the_running_interpreter_and_reports_missing_packages(monkeypatch):
    def version(name):
        if name == "scipy":
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.2.3"

    monkeypatch.setattr("slipagent.environment.importlib.metadata.version", version)
    monkeypatch.setattr("slipagent.environment.importlib.util.find_spec",
                        lambda name: None if name == "numba" else object())
    info = harness_python_environment()
    assert info["interpreter"] == sys.executable
    assert info["prefix"] == sys.prefix
    assert info["version"] == sys.version.split()[0]
    quote = shlex.join if os.name != "nt" else subprocess.list2cmdline
    assert info["math_command"] == quote([sys.executable, "-I"])
    assert info["libraries"]["numpy"] == {"available": True, "version": "1.2.3"}
    assert info["libraries"]["scipy"] == {"available": False, "version": None}
    assert info["libraries"]["numba"] == {"available": False, "version": "1.2.3"}


@pytest.mark.parametrize("base", [None, "Custom project instructions."])
@pytest.mark.parametrize("model,native", [("test/math", True), ("test/math", False), ("z-ai/glm-5.3", True)])
async def test_math_guidance_and_metadata_reach_custom_native_embedded_and_glm_contexts(base, model, native):
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, model, system_prompt=base)
    agent.messages.append(Message.user("Calculate the roots and residuals."))
    profile = RequestProfile(parameters={"tools"} if native else {"response_format"},
                             context_length=1_000_000, format=None if native else "json_object")
    try:
        messages = await agent._context_view([], 0, capabilities=profile, preview=True)
        delivered = "\n".join(content_text(message.content) for message in messages if message.role == "system")
        assert delivered.count("Use Python for challenging arithmetic") == 1
        assert "Harness Python and Mathematics" in delivered
        assert "Session Environment" in delivered
        assert "operating_system" in delivered and "machine" in delivered and "user" in delivered
        assert sys.executable in delivered
        assert "math_command" in delivered and "-I" in delivered
        assert "sympy" in delivered and "numexpr" in delivered and "numba" in delivered
        assert "do not install project dependencies into the harness environment" in delivered
        assert registry.resources.environment is None
    finally:
        await registry.aclose()
