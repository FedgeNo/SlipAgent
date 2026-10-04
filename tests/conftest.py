"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
import os

import pytest

from slipagent.workspace import Workspace
from offline.sitecustomize import GUARD_DIRECTORY


@pytest.fixture(autouse=True)
def isolated_session_storage(tmp_path: Path, monkeypatch):
    """Tests and children use dummy credentials, guarded networking and scratch state."""
    for name in tuple(os.environ):
        if name.startswith(("OPENROUTER_", "SLIPAGENT_", "EXA_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("SLIPAGENT_STATE_DIR", str(tmp_path / "session-state"))
    python_path = [GUARD_DIRECTORY, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in python_path if path))


@pytest.fixture()
def workspace(tmp_path: Path) -> Workspace:
    """An empty sandbox rooted in a fresh temp directory."""
    return Workspace(tmp_path)


@pytest.fixture()
def project(workspace: Workspace) -> Workspace:
    """A workspace pre-populated with a small, realistic project tree."""
    (workspace.root / "src").mkdir()
    (workspace.root / "tests").mkdir()
    (workspace.root / "node_modules" / "junk").mkdir(parents=True)
    (workspace.root / ".git").mkdir()

    (workspace.root / "src" / "app.py").write_text(
        "def main():\n"
        "    print('hello')\n"
        "\n"
        "\n"
        "def helper(value):\n"
        "    return value * 2\n",
        encoding="utf-8",
    )
    (workspace.root / "src" / "util.py").write_text(
        "VALUE = 42\n\n\ndef unused():\n    return None\n",
        encoding="utf-8",
    )
    (workspace.root / "tests" / "test_app.py").write_text(
        "from src.app import main\n\n\ndef test_main():\n    main()\n",
        encoding="utf-8",
    )
    (workspace.root / "README.md").write_text("# Project\n", encoding="utf-8")
    (workspace.root / "node_modules" / "junk" / "index.js").write_text(
        "module.exports = {};\n", encoding="utf-8"
    )
    (workspace.root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    return workspace
