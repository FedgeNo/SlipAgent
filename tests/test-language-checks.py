"""Automatic language checks select installed tools and preserve diagnostics."""

import asyncio
import os
from pathlib import Path
import sys

from slipagent.checks import _automatic_checks, check_edit_batch
from slipagent.tools import build_default_registry
from slipagent.tools.base import ToolResult
from slipagent.types import ToolCall


def test_missing_tools_are_skipped_only_for_edited_languages(workspace, monkeypatch):
    monkeypatch.setattr("slipagent.checks.shutil.which", lambda *args, **kwargs: None)
    checks = _automatic_checks({str(workspace.root / name) for name in ["a.js", "b.php", "c.ts", "d.txt"]}, workspace.root)
    assert len(checks) == 3
    assert all("skip" in check and "argv" not in check for check in checks)
    assert _automatic_checks({str(workspace.root / "a.py")}, workspace.root) == []


def test_typescript_uses_nearest_project_and_local_compiler(workspace, monkeypatch):
    root = workspace.root
    package = root / "packages" / "site"
    package.mkdir(parents=True)
    (root / "tsconfig.json").write_text("{}")
    (package / "tsconfig.json").write_text("{}")
    compiler = package / "node_modules" / ".bin" / "tsc"

    def which(name, path=None):
        assert name == "tsc"
        assert path.split(os.pathsep)[0] == str(compiler.parent)
        return str(compiler)

    monkeypatch.setattr("slipagent.checks.shutil.which", which)
    checks = _automatic_checks({str(package / name) for name in ["one.ts", "two.tsx"]}, root)
    assert len(checks) == 1
    assert checks[0]["cwd"] == package
    argv = checks[0]["argv"]
    assert argv[0] == str(compiler)
    assert argv[-2:] == ["--project", str(package / "tsconfig.json")]
    assert "--noEmit" in argv
    assert argv[argv.index("--tsBuildInfoFile") + 1] == os.devnull
    assert not any(arg.endswith(("one.ts", "two.tsx")) for arg in argv)


def test_typescript_without_config_checks_edited_files(workspace, monkeypatch):
    monkeypatch.setattr("slipagent.checks.shutil.which", lambda *args, **kwargs: "/fake/tsc")
    names = [str(workspace.root / "one.mts"), str(workspace.root / "two.cts")]
    checks = _automatic_checks(set(names), workspace.root)
    assert len(checks) == 1
    assert checks[0]["argv"][-2:] == names
    assert "--project" not in checks[0]["argv"]


async def test_checks_run_per_file_and_deliver_failures(workspace, monkeypatch):
    monkeypatch.setattr("slipagent.checks.shutil.which", lambda name, **kwargs: f"/fake/{name}")
    start = asyncio.create_subprocess_exec
    invocations = []

    async def fake_checker(*argv, **kwargs):
        invocations.append(argv)
        failing = argv[-1].endswith("bad.php")
        return await start(sys.executable, "-I", "-c",
                           "import sys; print('syntax error' if sys.argv[1] == '1' else 'ok'); sys.exit(int(sys.argv[1]))",
                           str(int(failing)), **kwargs)

    monkeypatch.setattr("slipagent.checks.asyncio.create_subprocess_exec", fake_checker)
    names = ["one.js", "two.cjs", "ok.php", "bad.php"]
    batch = []
    for index, name in enumerate(names):
        (workspace.root / name).write_text("test")
        batch.append((ToolCall(str(index), "write_file", {"path": name}), ToolResult.ok("written")))
    batch.append((ToolCall("failed", "write_file", {"path": "failed.mjs"}), ToolResult.error("not written")))
    registry = build_default_registry(workspace)
    try:
        report = await check_edit_batch(registry, batch, 1)
        assert len(invocations) == 4
        assert all(len(argv) == 3 for argv in invocations)
        assert all(argv[1] == ("-l" if argv[0].endswith("php") else "--check") for argv in invocations)
        assert {Path(argv[-1]).name for argv in invocations} == set(names)
        assert "PHP syntax (bad.php): FAILED (exit 1)" in report
        assert "syntax error" in report
        assert "JavaScript syntax (one.js): PASSED" in report
        assert "Command log" in report
    finally:
        await registry.aclose()
