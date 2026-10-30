"""Verify package constraints and cache isolated import checks without installing."""

import sys

import pytest
from packaging.requirements import Requirement

from slipagent import environment
from slipagent.config import ConfigError


async def test_incompatible_installed_math_version_is_repaired(monkeypatch):
    versions = iter(["0.0.1", "2.0"])
    monkeypatch.setattr(environment, "harness_python_environment", lambda: {
        "libraries": {"numpy": {"available": True, "version": next(versions)}}})
    monkeypatch.setattr(environment, "math_requirements", lambda: {"numpy": Requirement("numpy>=1.26,<3")})
    monkeypatch.setattr(sys, "prefix", "/harness-venv")
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: object())
    commands, verified = [], []
    async def spawn(*args, **kwargs):
        assert kwargs["cwd"] == sys.prefix
        commands.append(args)
        return type("Process", (), {"returncode": 0})()
    async def capture(process, timeout):
        return "", "", False
    async def verify():
        verified.append(True)
    monkeypatch.setattr(environment.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr("slipagent.tools.shell.capture_process", capture)
    monkeypatch.setattr(environment, "verify_harness_math", verify)
    await environment.ensure_harness_math()
    assert commands == [(sys.executable, "-I", "-m", "pip", "install", str(Requirement("numpy>=1.26,<3")))]
    assert verified == [True]


async def test_successful_import_probe_is_cached_and_environment_change_invalidates_it(monkeypatch, tmp_path):
    monkeypatch.setattr(environment, "data_directory", lambda: tmp_path)
    identity = ["first"]
    monkeypatch.setattr(environment, "math_verification_key", lambda: identity[0])
    commands = []
    async def spawn(*args, **kwargs):
        assert kwargs["cwd"] == sys.prefix
        commands.append(args)
        return type("Process", (), {"returncode": 0})()
    async def capture(process, timeout):
        return "", "", False
    monkeypatch.setattr(environment.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr("slipagent.tools.shell.capture_process", capture)
    await environment.verify_harness_math()
    await environment.verify_harness_math()
    assert len(commands) == 1
    assert commands[0][:3] == (sys.executable, "-I", "-c")
    assert "importlib.import_module('numpy')" in commands[0][3]
    identity[0] = "second"
    await environment.verify_harness_math()
    assert len(commands) == 2


@pytest.mark.parametrize("timed_out", [False, True])
async def test_broken_imports_are_reported_and_never_cached(monkeypatch, tmp_path, timed_out):
    monkeypatch.setattr(environment, "data_directory", lambda: tmp_path)
    monkeypatch.setattr(environment, "math_verification_key", lambda: "first")
    async def spawn(*args, **kwargs):
        return type("Process", (), {"returncode": 1})()
    async def capture(process, timeout):
        return "", "binary dependency failed to load", timed_out
    monkeypatch.setattr(environment.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr("slipagent.tools.shell.capture_process", capture)
    with pytest.raises(ConfigError, match="timed out" if timed_out else "binary dependency failed"):
        await environment.verify_harness_math()
    assert not (tmp_path / "math-environment-cache.json").exists()


def test_requirements_use_installed_contract_and_reject_missing_entries(monkeypatch):
    monkeypatch.setattr(environment, "MATH_PACKAGES", {"numpy": "numpy", "scikit-learn": "sklearn"})
    monkeypatch.setattr(environment.importlib.metadata, "requires", lambda name: ["numpy>=2", "scikit-learn>=1.4"])
    assert str(environment.math_requirements()["numpy"]) == "numpy>=2"
    monkeypatch.setattr(environment.importlib.metadata, "requires", lambda name: ["numpy>=2"])
    with pytest.raises(ConfigError, match="metadata is incomplete"):
        environment.math_requirements()
