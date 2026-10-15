"""Dev-only: drive the REPL's /mcp command against the stub MCP server.

Usage:
    python scripts/mcp_drive.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import shlex
from pathlib import Path
from demo_run import demo_environment, make_server

ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "tests" / "mcp_stub_server.py"


def main() -> None:
    temporary = tempfile.TemporaryDirectory(prefix="slipagent-mcp-")
    workspace = Path(temporary.name)
    server = make_server({"n": 0})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    py = sys.executable
    commands = [
        "/mcp",
        f"/mcp add stub {shlex.quote(py)} {shlex.quote(str(STUB))} legacy",
        "/mcp",
        "/tools",
        f"/mcp save persisted {shlex.quote(py)} {shlex.quote(str(STUB))} modern",
        "/mcp",
        "/mcp remove persisted",
        "/mcp remove nope",
        "/mcp bogus",
        "/exit",
    ]

    try:
        run_demo(workspace, py, commands, server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        temporary.cleanup()


def run_demo(workspace: Path, py: str, commands: list[str], port: int) -> None:
    proc = subprocess.run(
        [py, "-m", "slipagent.cli", "--workspace", str(workspace),
         "--model", "demo/model", "--base-url", f"http://127.0.0.1:{port}/api/v1"],
        cwd=workspace,
        input="\n".join(commands) + "\n",
        capture_output=True,
        text=True,
        timeout=180,
        env={**demo_environment(workspace), "NO_COLOR": "1"},
    )
    print(proc.stderr)
    print(f"--- exit {proc.returncode}")
    print("--- demo MCP configuration ---")
    configuration = workspace / ".demo-state" / "mcp.json"
    if configuration.exists():
        print(configuration.read_text())
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
