"""Dev-only: run the CLI against an in-process stub OpenRouter server.

Lets you watch the harness work end to end without spending tokens or needing
a second terminal:

    python scripts/demo_run.py              # one-shot, verbose
    python scripts/demo_run.py --repl       # drive the REPL
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT: list[dict[str, object]] = [
    {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "list_dir", "arguments": "{}"}}
            ],
        },
        "finish_reason": "tool_calls",
    },
    {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_2", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": '{"path": "src/calc.py"}'}}
            ],
        },
        "finish_reason": "tool_calls",
    },
    {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_3", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": '{"path": "../../../etc/passwd"}'}}
            ],
        },
        "finish_reason": "tool_calls",
    },
    {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_4", "type": "function",
                 "function": {"name": "nope", "arguments": "{}"}}
            ],
        },
        "finish_reason": "tool_calls",
    },
    {
        "message": {
            "role": "assistant",
            "content": (
                "Done. I listed the project, read src/calc.py, tried to reach a "
                "file outside the sandbox (refused), and called a tool that does "
                "not exist (refused).\n\nBoth refusals came back as messages to "
                "me rather than crashing the run, which is the point: the loop "
                "stays alive while the model corrects itself."
            ),
            "tool_calls": None,
        },
        "finish_reason": "stop",
    },
]


def structured_message(message: dict, messages: list[dict]) -> dict:
    """A schema-capable demo sends only reply text in content and native calls."""
    return {**message, "content": json.dumps({"response": message.get("content") or ""})}


def summary_response(request: dict) -> dict | None:
    if not request["messages"][0]["content"].startswith("Summarize one completed SlipAgent turn"):
        return None
    source = json.loads(request["messages"][1]["content"])
    text = f"Demo turn {source['post_id']}: {source['agent_response'][:200]}; {len(source['tool_results'])} tool observations saved."
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0}}


def metadata_response(path: str, model: str = "demo/model") -> dict:
    parameters = ["tools", "response_format", "structured_outputs", "temperature", "max_tokens"]
    if path.endswith("/models"):
        return {"data": [{"id": model, "context_length": 1_000_000,
                          "supported_parameters": parameters, "pricing": {"prompt": "0", "completion": "0"}}]}
    if path.endswith("/endpoints"):
        return {"data": {"endpoints": [{"tag": "demo", "status": 0, "context_length": 1_000_000,
                                       "max_completion_tokens": 8192, "supported_parameters": parameters}]}}
    if path.endswith("/key"):
        return {"data": {"label": "offline demo", "limit": 0, "limit_remaining": 0,
                         "usage": 0, "is_free_tier": True,
                         "free_model_daily_requests": {"used": 88, "limit": 1000, "remaining": 912}}}
    return {"error": {"message": "Unknown demo endpoint"}}


def make_server(counter: dict[str, int]) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def _send(self, payload: dict[str, object], status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            summary = summary_response(request)
            if summary is not None:
                self._send(summary)
                return
            step = json.loads(json.dumps(SCRIPT[min(counter["n"], len(SCRIPT) - 1)]))
            step["message"] = structured_message(step["message"], request["messages"])
            step["finish_reason"] = "stop"
            counter["n"] += 1
            self._send(
                {
                    "id": "demo",
                    "model": "demo/model",
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 180,
                              "total_tokens": 1380, "cost": 0.00412},
                    "choices": [{"index": 0, **step}],
                }
            )

        def do_GET(self) -> None:
            self._send(metadata_response(self.path))

    return HTTPServer(("127.0.0.1", 0), Handler)


def main() -> None:
    counter = {"n": 0}
    server = make_server(counter)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    temporary = tempfile.TemporaryDirectory(prefix="slipagent-demo-")
    project = Path(temporary.name)
    (project / "src").mkdir(parents=True, exist_ok=True)
    (project / "src" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )

    use_repl = "--repl" in sys.argv[1:]
    args = [
        sys.executable, "-m", "slipagent.cli",
        "--base-url", f"http://127.0.0.1:{port}/api/v1",
        "--workspace", str(project),
        "--model", "demo/model",
        *([] if use_repl else ["-p", "explore this project", "-v"]),
    ]

    print(f"--- demo workspace: {project}")
    print(f"--- stub OpenRouter on port {port}\n")

    env = {
        **os.environ,
        "OPENROUTER_API_KEY": "demo-key",
        "PYTHONPATH": str(ROOT / "src"),
        "SLIPAGENT_NO_DOTENV": "1",
    }
    try:
        result = subprocess.run(args, env=env, cwd=project, timeout=120)
    finally:
        server.shutdown()
        server.server_close()
        temporary.cleanup()
    print(f"\n--- exit code: {result.returncode}")
    print(f"--- stub received {counter['n']} working completion request(s)")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
