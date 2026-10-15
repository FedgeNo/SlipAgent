"""Offline demonstration commands use disposable configuration and credentials."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("module_name", ["demo_run", "mcp_drive", "repl_render_demo"])
def test_offline_demo_isolates_its_child_environment(monkeypatch, module_name):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    module = importlib.import_module(module_name)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-inherited-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-inherited-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-inherited-key")
    monkeypatch.setenv("KILO_API_KEY", "test-inherited-key")
    monkeypatch.setenv("SLIPAGENT_PROVIDER", "nvidia")
    inherited_python_path = os.environ["PYTHONPATH"].split(os.pathsep)
    server = SimpleNamespace(server_address=("127.0.0.1", 1), serve_forever=lambda: None,
                             shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(module, "serve" if module_name == "repl_render_demo" else "make_server", lambda _: server)
    monkeypatch.setattr(sys, "argv", [module_name])
    calls = []

    def capture(arguments, **options):
        calls.append(options)
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", capture)
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0
    assert len(calls) == 1
    options = calls[0]
    env = options["env"]
    assert Path(env["SLIPAGENT_STATE_DIR"]).is_relative_to(options["cwd"])
    assert env["SLIPAGENT_PROVIDER"] == "openrouter"
    assert env["SLIPAGENT_NO_DOTENV"] == "1"
    for name in ("NVIDIA_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "KILO_API_KEY"):
        assert name not in env
    assert set(inherited_python_path).issubset(env["PYTHONPATH"].split(os.pathsep))
