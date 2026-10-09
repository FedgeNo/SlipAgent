"""Dev-only: show the REPL rendering — spacing, queued input, the prompt line.

Runs a local stub server in a thread and the CLI in a subprocess. Prints the
captured plain-text transcript; this does not exercise interactive redraws.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from demo_run import metadata_response, structured_message, summary_response

ROOT = Path(__file__).resolve().parents[1]


def step(tool: str | None, arguments: dict, text: str | None = None) -> dict:
    if tool:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"c{tool}", "type": "function",
                 "function": {"name": tool, "arguments": json.dumps(arguments)}}
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish = "stop"
    return {
        "id": "gen", "model": "stub/model",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40,
                  "total_tokens": 160, "cost": 0.0001},
    }


SCRIPT = [
    step("read_file", {"path": "notes.txt"}),
    step("list_dir", {}),
    step(None, {}, text="The notes say to buy milk, and there is one file here."),
]


def serve(requests: list[dict]) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            body = summary_response(request)
            if body is None:
                requests.append(request)
                body = json.loads(json.dumps(SCRIPT[min(len(requests) - 1, len(SCRIPT) - 1)]))
                message = body["choices"][0]["message"]
                body["choices"][0]["message"] = structured_message(message, request["messages"])
                body["choices"][0]["finish_reason"] = "stop"
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            raw = json.dumps(metadata_response(self.path, "stub/model")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return HTTPServer(("127.0.0.1", 0), Handler)


def main() -> None:
    requests: list[dict] = []
    server = serve(requests)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    temporary = tempfile.TemporaryDirectory(prefix="slipagent-render-")
    workspace = Path(temporary.name)
    (workspace / "notes.txt").write_text("remember the milk\n", encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "slipagent.cli", "--workspace", str(workspace),
             "--model", "stub/model", "-v",
             "--base-url", f"http://127.0.0.1:{port}/api/v1"],
            cwd=workspace,
            # Submit all lines through a pipe; their arrival is not paced.
            input="read the notes\nalso list the dir\n/exit\n",
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "OPENROUTER_API_KEY": "sk-or-v1-test",
                 "PYTHONPATH": str(ROOT / "src"), "NO_COLOR": "1", "SLIPAGENT_NO_DOTENV": "1"},
        )
        time.sleep(0.2)
        print(proc.stderr)
        print(f"--- exit {proc.returncode}")
        print(f"--- {len(requests)} completion request(s)")
        raise SystemExit(proc.returncode)
    finally:
        server.shutdown()
        server.server_close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
