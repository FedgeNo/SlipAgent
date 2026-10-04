"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from slipagent.workspace import Workspace


@pytest.fixture(autouse=True)
def isolated_session_storage(tmp_path: Path, monkeypatch):
    """CLI tests and inherited subprocesses must never write the user's sessions."""
    monkeypatch.setenv("SLIPAGENT_STATE_DIR", str(tmp_path / "session-state"))


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
