"""Dev-only: drive the REPL using real provider configuration.

This is a live manual utility; use demo_run.py for an offline stub session.

Usage:
    python scripts/repl_drive.py <workspace> "/model; /models free; /exit"

Commands are separated by semicolons so multi-word commands such as
`/models free` stay a single command.
"""

from __future__ import annotations

import subprocess
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMMANDS = "/model; /models free; /model nope/missing; /key show; /exit"


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    raw = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else DEFAULT_COMMANDS
    commands = [part.strip() for part in raw.split(";") if part.strip()]

    proc = subprocess.run(
        [
            sys.executable, "-m", "slipagent.cli",
            "--workspace", str(workspace),
            "--model", "nvidia/nemotron-3.5-lightning:free",
        ],
        cwd=workspace,
        input="\n".join(commands) + "\n",
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
        },
    )
    print(proc.stderr)
    print(f"--- exit {proc.returncode}")


if __name__ == "__main__":
    main()
