"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
import compileall
import os
import shutil
import tempfile

import pytest

from slipagent.workspace import Workspace
from offline.sitecustomize import GUARD_DIRECTORY


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--slow", action="store_true", help="Also run tests marked slow.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long, stable tests that run only with --slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--slow") or "slow" in (config.getoption("-m") or ""):
        return
    slow = [item for item in items if item.get_closest_marker("slow") is not None]
    if slow:
        config.hook.pytest_deselected(items=slow)
        items[:] = [item for item in items if item.get_closest_marker("slow") is None]


@pytest.fixture(autouse=True)
def isolated_session_storage(tmp_path: Path, system_temp_root: Path, monkeypatch):
    """Tests and children use dummy credentials, guarded networking and scratch state."""
    for name in tuple(os.environ):
        if name.startswith(("OPENROUTER_", "NVIDIA_", "KILO_", "MISTRAL_", "GEMINI_", "ANTHROPIC_", "SLIPAGENT_", "EXA_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("SLIPAGENT_STATE_DIR", str(tmp_path / "session-state"))
    # tmp_path already has a unique name. Avoid rescanning every earlier test's
    # numbered temp directories, while keeping system temp outside its workspace.
    system_temp = system_temp_root / tmp_path.name
    system_temp.mkdir()
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(system_temp))
    monkeypatch.setattr(tempfile, "tempdir", str(system_temp))
    python_path = [GUARD_DIRECTORY, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in python_path if path))
    # MCP servers inherit a minimal environment; test servers keep the offline guard.
    from slipagent import mcp
    monkeypatch.setattr(mcp, "INHERITED_ENV", (*mcp.INHERITED_ENV, "PYTHONPATH"))


@pytest.fixture(scope="session")
def system_temp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("system-temp")


@pytest.fixture(scope="session")
def reload_source(tmp_path_factory):
    """Compile one immutable template per worker; each test edits its own copy."""
    package_root = tmp_path_factory.mktemp("reload-source")
    repository = Path(__file__).resolve().parents[1]
    # Each reload test copies this template into system temp storage, which may be
    # RAM-backed; the bundled embedding model is not needed for reloads.
    shutil.copytree(repository / "src" / "slipagent", package_root / "slipagent",
                    ignore=lambda directory, names: {"__pycache__", *(name for name in names if name.endswith(".pyc"))}
                    | ({"model"} if Path(directory).name == "semantic" else set()))
    shutil.copytree(repository / "prompts", package_root / "slipagent" / "prompts")
    shutil.copyfile(Path(__file__).with_name("data_text_reader.py"), package_root / "data_text_reader.py")
    assert compileall.compile_dir(package_root, quiet=2)
    return package_root


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


@pytest.fixture(autouse=True)
def alternate_provider_catalogs(monkeypatch):
    """Unrelated tests have no live alternate catalog; provider tests supply fixtures."""
    from slipagent import cli
    async def empty_catalog(provider, **options):
        return []
    monkeypatch.setattr(cli, "fetch_catalog", empty_catalog)
    import slipagent.sessionstate as sessionstate
    monkeypatch.setattr(sessionstate, "fetch_catalog", empty_catalog)
