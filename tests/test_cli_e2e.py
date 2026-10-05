"""End-to-end tests: the installed CLI against a stub OpenRouter server.

These exercise the whole stack — argument parsing, config, transport, the agent
loop, real tool execution on a real filesystem, and rendering — so a regression
in any layer surfaces here rather than hiding behind unit test doubles.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from test_agent import structured_message, context_body, summary_response, context_records, unpack_api_context
from slipagent.config import DEFAULT_MODEL
from offline.sitecustomize import GUARD_DIRECTORY


class StubOpenRouter:
    """A tiny HTTP server that speaks just enough of the OpenRouter API."""

    def __init__(self, script: list[dict[str, Any]], *, include_memory: bool = True) -> None:
        self._script = list(script)
        self.include_memory = include_memory
        self.requests: list[dict[str, Any]] = []
        self.summary_requests: list[dict[str, Any]] = []
        self._server = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self.port = self._server.server_address[1]
        # Teardown should not spend the server's default half-second poll
        # interval waiting for a request that the test has finished sending.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": .01}, daemon=True,
        )

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence stderr noise
                pass

            def _send(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
                if self.path != "/api/v1/chat/completions":
                    self._send({"error": {"message": "unknown endpoint"}}, status=404)
                    return

                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                summary = summary_response(request)
                if summary is not None:
                    stub.summary_requests.append(request)
                    self._send(summary)
                    return
                stub.requests.append(request)

                step = json.loads(json.dumps(stub._script[min(len(stub.requests) - 1, len(stub._script) - 1)]))
                message = step["choices"][0]["message"]
                if request.get("response_format", {}).get("type") == "json_schema":
                    step["choices"][0]["message"] = structured_message(message, request["messages"], include_memory=stub.include_memory)
                    if step["choices"][0].get("finish_reason") == "tool_calls":
                        step["choices"][0]["finish_reason"] = "stop"
                self._send(step)

            def do_GET(self) -> None:  # noqa: N802
                params = ["tools", "response_format", "structured_outputs"]
                if self.path.endswith("/endpoints"):
                    self._send({"data": {"endpoints": [{"tag": "stub-provider", "supported_parameters": params,
                                "context_length": 32000 if "stub/two" in self.path else 128000}]}})
                    return
                if self.path != "/api/v1/models":
                    self._send({"error": {"message": "unknown endpoint"}}, status=404)
                    return
                self._send({"data": [{"id": "stub/one", "context_length": 128000, "supported_parameters": params},
                                     {"id": "stub/two", "context_length": 32000, "supported_parameters": params},
                                     {"id": "stub/model", "context_length": 128000, "supported_parameters": params},
                                     {"id": DEFAULT_MODEL, "context_length": 128000, "supported_parameters": params}]})

        return Handler

    def __enter__(self) -> StubOpenRouter:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1"


def text_step(content: str, cost: float = 0.0001) -> dict[str, Any]:
    return {
        "id": "gen",
        "model": "stub/model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": cost,
        },
    }


def tool_step(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen",
        "model": "stub/model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                  "total_tokens": 15, "cost": 0.0001},
    }


def cli_environment(**overrides: str) -> dict[str, str]:
    """Keep child CLIs off real configuration and inside this test's state store."""
    env = {name: os.environ[name] for name in ("PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP")
           if name in os.environ}
    env.update({
        "OPENROUTER_API_KEY": "test-key", "SLIPAGENT_NO_DOTENV": "1", "NO_COLOR": "1",
        "SLIPAGENT_STATE_DIR": os.environ["SLIPAGENT_STATE_DIR"],
        "PYTHONPATH": os.pathsep.join([GUARD_DIRECTORY, str(Path(__file__).resolve().parents[1] / "src")]),
    })
    env.update(overrides)
    return env


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "slipagent.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        env=cli_environment(),
    )


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    return root


@pytest.fixture()
def metadata_server(monkeypatch):
    with StubOpenRouter([text_step("unused")]) as stub:
        monkeypatch.setenv("OPENROUTER_BASE_URL", stub.base_url)
        monkeypatch.setenv("OPENROUTER_MODEL", "stub/one")
        yield stub


# --------------------------------------------------------------------------- #
# One-shot mode
# --------------------------------------------------------------------------- #


def test_one_shot_answers_on_stdout(project_dir: Path) -> None:
    with StubOpenRouter([text_step("Everything is fine.")]) as stub:
        result = run_cli(
            "-p", "check the project",
            "--base-url", stub.base_url,
            "--model", "stub/model",
            cwd=project_dir,
        )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Everything is fine."
    assert stub.requests[0]["model"] == "stub/model"


@pytest.mark.parametrize("danger", [False, True])
def test_danger_flag_controls_unattended_external_write(project_dir, danger):
    target = project_dir.parent / "external.txt"
    with StubOpenRouter([tool_step("write_file", {"path": str(target), "content": "outside"}),
                         text_step("Finished")]) as stub:
        result = run_cli("-p", "Write the requested file", "--base-url", stub.base_url,
                         "--model", "stub/model", *(["--danger"] if danger else []), cwd=project_dir)
    assert result.returncode == 0, result.stderr
    assert target.exists() is danger
    assert result.stdout.strip() == "Finished"
    system = "\n".join(m["content"] for m in stub.requests[0]["messages"] if m["role"] == "system")
    assert ("Danger mode ON" if danger else "Danger mode OFF") in system
    if danger:
        assert target.read_text() == "outside"
    else:
        assert "outside the workspace" in result.stderr


@pytest.mark.parametrize("mode", ["one-shot", "repl"])
def test_project_instructions_are_present_before_first_model_request(project_dir: Path, mode: str) -> None:
    instructions = {
        "CLAUDE.md": 'Use the project formatter. Keep this literal: {"setting": true}.',
        "AGENTS.md": "Run the project tests before finishing.",
        ".cursorrules": "Use existing components before adding new ones.",
        ".cursor/rules/frontend.mdc": "---\nglobs: src/*.tsx\n---\nFollow the frontend conventions.",
        ".github/copilot-instructions.md": "Preserve public interfaces.",
    }
    for name, content in instructions.items():
        path = project_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    with StubOpenRouter([text_step("done")]) as stub:
        if mode == "one-shot":
            proc = run_cli("-p", "go", "--base-url", stub.base_url, cwd=project_dir)
        else:
            proc = run_repl_commands(project_dir, ["go"], "--base-url", stub.base_url)
    assert proc.returncode == 0, proc.stderr
    prompt = stub.requests[0]["messages"][0]["content"]
    for name, content in instructions.items():
        assert name in prompt
        assert content in prompt


def test_unreadable_project_instructions_stop_before_model_actions(project_dir: Path) -> None:
    (project_dir / "AGENTS.md").write_bytes(b"\xff")
    with StubOpenRouter([text_step("should never run")]) as stub:
        proc = run_cli("-p", "go", "--base-url", stub.base_url, cwd=project_dir)
    assert proc.returncode == 2
    assert "cannot read project instructions AGENTS.md" in proc.stderr
    assert not stub.requests


def test_one_shot_prints_the_answer_exactly_once(project_dir: Path) -> None:
    """The answer goes to stdout; stderr must not repeat it."""
    with StubOpenRouter([text_step("Only once.")]) as stub:
        result = run_cli("-p", "hi", "--base-url", stub.base_url, cwd=project_dir)

    assert result.stdout.count("Only once.") == 1
    assert "Only once." not in result.stderr


def test_one_shot_executes_a_read_tool(project_dir: Path) -> None:
    with StubOpenRouter(
        [
            tool_step("read_file", {"path": "src/app.py"}),
            text_step("The file returns 1."),
        ]
    ) as stub:
        result = run_cli(
            "-p", "what does src/app.py do?",
            "--base-url", stub.base_url,
            cwd=project_dir,
        )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "The file returns 1."

    # The second request must carry the tool result back to the model.
    second = stub.requests[1]
    tool_messages = context_records(second["messages"])[0]["tool_results"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["call_id"] == "call_1"
    assert "def main():" in tool_messages[0]["content"]


def test_one_shot_applies_a_real_edit(project_dir: Path) -> None:
    with StubOpenRouter(
        [
            tool_step(
                "edit_file",
                {
                    "path": "src/app.py",
                    "old_string": "    return 1",
                    "new_string": "    return 42",
                },
            ),
            text_step("Updated."),
        ]
    ) as stub:
        result = run_cli("-p", "make it return 42", "--base-url", stub.base_url,
                         cwd=project_dir)

    assert result.returncode == 0, result.stderr
    assert "return 42" in (project_dir / "src" / "app.py").read_text()


def test_one_shot_refuses_to_write_outside_workspace(project_dir: Path, tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("original", encoding="utf-8")

    with StubOpenRouter(
        [
            tool_step("write_file", {"path": "../victim.txt", "content": "pwned"}),
            text_step("I could not write there."),
        ]
    ) as stub:
        result = run_cli("-p", "overwrite the victim file", "--base-url", stub.base_url,
                         cwd=project_dir)

    assert result.returncode == 0, result.stderr
    assert victim.read_text() == "original"

    tool_messages = context_records(stub.requests[1]["messages"])[0]["tool_results"]
    assert "outside the workspace" in tool_messages[0]["content"]


def test_workspace_flag_changes_the_sandbox_root(project_dir: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "only_here.txt").write_text("data", encoding="utf-8")

    with StubOpenRouter(
        [
            tool_step("read_file", {"path": "only_here.txt"}),
            text_step("Read it."),
        ]
    ) as stub:
        result = run_cli(
            "-p", "read it", "--workspace", str(other),
            "--base-url", stub.base_url, cwd=project_dir,
        )

    assert result.returncode == 0, result.stderr
    assert "data" in context_records(stub.requests[1]["messages"])[0]["tool_results"][0]["content"]


def test_positional_prompt_is_accepted(project_dir: Path) -> None:
    with StubOpenRouter([text_step("ok")]) as stub:
        result = run_cli("hello", "world", "--base-url", stub.base_url, cwd=project_dir)

    assert result.returncode == 0, result.stderr
    assert context_records(stub.requests[0]["messages"])[-1]["user_prompt"] == ["hello world"]


def test_verbose_shows_tool_output_on_stderr(project_dir: Path) -> None:
    with StubOpenRouter(
        [tool_step("read_file", {"path": "src/app.py"}), text_step("done")]
    ) as stub:
        result = run_cli("-p", "read", "-v", "--base-url", stub.base_url, cwd=project_dir)

    assert "read_file" in result.stderr
    assert "def main():" in result.stderr


def test_step_limit_is_reported(project_dir: Path) -> None:
    with StubOpenRouter([tool_step("list_dir", {})]) as stub:
        result = run_cli(
            "-p", "loop forever", "--max-steps", "2",
            "--base-url", stub.base_url, cwd=project_dir,
        )

    assert result.returncode == 1, result.stderr
    assert "step limit" in result.stdout
    assert len(stub.requests) == 2


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_missing_api_key_exits_with_config_error(project_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "slipagent.cli", "-p", "hi", "--workspace", str(project_dir)],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=60,
        env=cli_environment(OPENROUTER_API_KEY=""),
    )

    assert result.returncode == 2
    assert "OPENROUTER_API_KEY" in result.stderr


def test_api_error_exits_nonzero(project_dir: Path) -> None:
    with StubOpenRouter([text_step("never reached")]) as stub:
        # A path the stub does not serve; the client gets a 404.
        result = run_cli("-p", "hi", "--base-url", f"{stub.base_url}/nope",
                         cwd=project_dir)

    assert result.returncode == 2
    assert "slipagent:" in result.stderr


def test_list_models_prints_catalog(project_dir: Path) -> None:
    with StubOpenRouter([]) as stub:
        result = run_cli("--list-models", "--base-url", stub.base_url, cwd=project_dir)

    assert result.returncode == 0, result.stderr
    assert "stub/one" in result.stdout
    assert "stub/two" in result.stdout


def test_list_models_reports_api_failure(project_dir: Path) -> None:
    with StubOpenRouter([]) as stub:
        result = run_cli("--list-models", "--base-url", f"{stub.base_url}/nope",
                         cwd=project_dir)

    assert result.returncode == 1
    assert "slipagent:" in result.stderr


def test_repl_handles_slash_commands(project_dir: Path) -> None:
    """The REPL must not need a TTY to exercise its command handling."""
    with StubOpenRouter([text_step("hi from model")]) as stub:
        proc = subprocess.run(
            [sys.executable, "-m", "slipagent.cli", "--base-url", stub.base_url,
             "--model", "stub/model"],
            cwd=project_dir,
            input="/tools\n/model\n/cost\n/exit\n",
            capture_output=True,
            text=True,
            timeout=60,
            env=cli_environment(),
        )

    assert proc.returncode == 0, proc.stderr
    assert "read_file" in proc.stderr
    assert "stub/model" in proc.stderr


def test_repl_prints_the_answer_once_per_turn(project_dir: Path) -> None:
    with StubOpenRouter([text_step("Replied once.")]) as stub:
        proc = subprocess.run(
            [sys.executable, "-m", "slipagent.cli", "--base-url", stub.base_url,
             "--model", "stub/model"],
            cwd=project_dir,
            input="hello\n/exit\n",
            capture_output=True,
            text=True,
            timeout=60,
            env=cli_environment(),
        )

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.count("Replied once.") == 1


def test_repl_reports_api_failure_without_crashing(project_dir: Path) -> None:
    """An unreachable server must produce an error, not a traceback."""
    # Port 1 is reserved and closed; connection is refused immediately.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead_port = sock.getsockname()[1]
    sock.close()

    proc = subprocess.run(
        [sys.executable, "-m", "slipagent.cli",
         "--base-url", f"http://127.0.0.1:{dead_port}/api/v1"],
        cwd=project_dir,
        input="hello\n/exit\n",
        capture_output=True,
        text=True,
        timeout=90,
        env=cli_environment(),
    )

    assert proc.returncode == 2, proc.stderr
    assert "Network error" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_help_exits_zero(project_dir: Path) -> None:
    result = run_cli("--help", cwd=project_dir)

    assert result.returncode == 0
    assert "--workspace" in result.stdout


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #

MCP_STUB = str(Path(__file__).parent / "mcp_stub_server.py")


def run_repl_commands(project_dir: Path, commands: list[str], *extra: str) -> subprocess.CompletedProcess[str]:
    with ExitStack() as resources:
        metadata = None
        if not any(arg == "--base-url" or arg.startswith("--base-url=") for arg in extra):
            # Slash-command tests still fetch model metadata and quota at
            # startup. They must use a local server, even without inference.
            metadata = resources.enter_context(StubOpenRouter([text_step("unused")]))
            extra = ("--base-url", metadata.base_url, *extra)
        result = subprocess.run(
            [sys.executable, "-m", "slipagent.cli", *extra],
            cwd=project_dir,
            input="\n".join([*commands, "/exit"]) + "\n",
            capture_output=True,
            text=True,
            timeout=120,
            env=cli_environment(),
        )
        if metadata is not None:
            assert not metadata.requests, "Supply a scripted server for tests that request inference"
        return result


def test_mcp_status_is_empty_without_config(project_dir: Path) -> None:
    proc = run_repl_commands(project_dir, ["/mcp"])

    assert proc.returncode == 0, proc.stderr
    assert "no MCP servers configured" in proc.stderr


def test_mcp_add_connects_and_registers_tools(project_dir: Path) -> None:
    proc = run_repl_commands(
        project_dir,
        [f"/mcp add stub {sys.executable} {MCP_STUB} legacy", "/tools"],
    )

    assert proc.returncode == 0, proc.stderr
    assert "connected" in proc.stderr
    # The remote tools must join the registry the model draws from.
    assert "stub__echo" in proc.stderr
    assert "stub__boom" in proc.stderr


def test_mcp_save_persists_to_dot_json(project_dir: Path) -> None:
    proc = run_repl_commands(
        project_dir,
        [f"/mcp save db {sys.executable} {MCP_STUB} legacy"],
    )

    assert proc.returncode == 0, proc.stderr
    config = json.loads((project_dir / ".mcp.json").read_text())
    assert config["mcpServers"]["db"]["command"] == sys.executable
    assert MCP_STUB in config["mcpServers"]["db"]["args"]


def test_saved_server_connects_automatically_on_the_next_run(project_dir: Path) -> None:
    """A server in .mcp.json is connected when the REPL starts, with no command."""
    (project_dir / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"db": {"command": sys.executable, "args": [MCP_STUB, "legacy"]}}}),
        encoding="utf-8",
    )

    proc = run_repl_commands(project_dir, ["/mcp"])

    assert proc.returncode == 0, proc.stderr
    assert "db: connected" in proc.stderr
    assert "db__echo" in proc.stderr


def test_no_mcp_flag_skips_configured_servers(project_dir: Path) -> None:
    (project_dir / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"db": {"command": sys.executable, "args": [MCP_STUB, "legacy"]}}}),
        encoding="utf-8",
    )

    proc = run_repl_commands(project_dir, ["/mcp"], "--no-mcp")

    assert proc.returncode == 0, proc.stderr
    assert "disabled" in proc.stderr


def test_broken_server_does_not_stop_the_session(project_dir: Path) -> None:
    """A server that cannot start is reported, but the REPL still works."""
    (project_dir / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"ghost": {"command": "/nonexistent/mcp-binary"}}}),
        encoding="utf-8",
    )

    proc = run_repl_commands(project_dir, ["/mcp", "/tools"])

    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    # Built-in tools are still available.
    assert "read_file" in proc.stderr


def test_mcp_remove_reports_unknown_server(project_dir: Path) -> None:
    proc = run_repl_commands(project_dir, ["/mcp remove nope"])

    assert proc.returncode == 0, proc.stderr
    assert "no MCP server named" in proc.stderr


def test_mcp_bad_usage_explains_itself(project_dir: Path) -> None:
    proc = run_repl_commands(project_dir, ["/mcp add lonely"])

    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stderr


def test_mcp_tool_is_callable_by_the_model(project_dir: Path) -> None:
    """The whole point: a remote tool is usable through a normal agent turn."""
    (project_dir / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"stub": {"command": sys.executable, "args": [MCP_STUB, "legacy"]}}}),
        encoding="utf-8",
    )

    with StubOpenRouter(
        [
            tool_step("stub__echo", {"message": "from the model"}),
            text_step("The MCP tool replied."),
        ]
    ) as stub:
        proc = run_repl_commands(project_dir, ["use the echo tool"], "--base-url", stub.base_url)

    assert proc.returncode == 0, proc.stderr
    # The result must come back to the model on the next request.
    tool_messages = context_records(stub.requests[1]["messages"])[0]["tool_results"]
    assert context_body(tool_messages[0]["content"]) == "echo: from the model"


# --------------------------------------------------------------------------- #
# REPL presentation
# --------------------------------------------------------------------------- #


def test_repl_never_prints_a_step_counter(project_dir: Path) -> None:
    """Per-step progress lines are noise; the tool lines say enough."""
    with StubOpenRouter([tool_step("read_file", {"path": "src/app.py"}),
                         text_step("done")]) as stub:
        proc = subprocess.run(
            [sys.executable, "-m", "slipagent.cli", "--base-url", stub.base_url,
             "--model", "stub/model"],
            cwd=project_dir, input="go\n/exit\n", capture_output=True, text=True, timeout=60,
            env=cli_environment(),
        )

    assert proc.returncode == 0, proc.stderr
    assert "step 1" not in proc.stderr
    assert "step 2" not in proc.stderr


def test_model_reply_starts_with_a_gap_after_tool_output(project_dir: Path) -> None:
    """A reply starts a new unit even after a step containing only tools."""
    with StubOpenRouter([tool_step("read_file", {"path": "src/app.py"}),
                         text_step("Here is the answer.")]) as stub:
        proc = subprocess.run(
            [sys.executable, "-m", "slipagent.cli", "--base-url", stub.base_url,
             "--model", "stub/model", "-v"],
            cwd=project_dir, input="go\n/exit\n", capture_output=True, text=True, timeout=60,
            env=cli_environment(),
        )

    lines = proc.stderr.splitlines()
    answer = lines.index("Here is the answer.")

    assert lines[answer - 1] == ""
    assert any("⚙ read_file" in line for line in lines[:answer])


def test_status_bar_shows_cwd_and_free_calls(project_dir: Path) -> None:
    with StubOpenRouter([text_step("ok")]) as stub:
        proc = run_repl_commands(project_dir, ["go"], "--base-url", stub.base_url)

    assert proc.returncode == 0, proc.stderr
    assert "free: " in proc.stderr
    assert "│" in proc.stderr


def test_status_bar_is_separated_from_the_prompt(project_dir: Path) -> None:
    with StubOpenRouter([text_step("ok")]) as stub:
        proc = run_repl_commands(project_dir, ["go"], "--base-url", stub.base_url)

    assert proc.returncode == 0, proc.stderr
    lines = proc.stderr.splitlines()
    prompt = next(i for i, line in enumerate(lines) if line.strip() == ">")
    assert lines[prompt + 1] == ""
    assert "free: " in lines[prompt + 2]
    assert "model: " in lines[prompt + 2]
    assert ">" not in lines[prompt + 2]
    submitted = lines.index("> go")
    assert lines[submitted - 1] == ""
    assert lines[submitted + 1] == ""


def test_status_bar_names_the_working_directory(project_dir: Path) -> None:
    """At the workspace root the readout must name the directory, not ".".

    `Workspace.relative` answers "." for the root, which reads as a stray dot on
    the prompt line and tells the user nothing about where they are.
    """
    with StubOpenRouter([text_step("ok")]) as stub:
        proc = run_repl_commands(project_dir, ["go"], "--base-url", stub.base_url)

    assert proc.returncode == 0, proc.stderr
    line = next(line for line in proc.stderr.splitlines() if "free: " in line)
    cwd, _, _ = line.partition("│")
    cwd = cwd.strip()
    assert cwd, proc.stderr
    assert cwd not in {".", "./", ".."}, proc.stderr
    # A name, not a path: the tail of the project directory is what identifies it.
    assert project_dir.name in cwd


def test_prompt_leaves_the_cursor_at_the_marker(project_dir: Path) -> None:
    """The prompt row contains no readouts or padding after its marker."""
    with StubOpenRouter([text_step("ok")]) as stub:
        proc = subprocess.run(
            [sys.executable, "-m", "slipagent.cli", "--base-url", stub.base_url,
             "--model", "stub/model"],
            cwd=project_dir, input="go\n/exit\n", capture_output=True, text=True,
            timeout=60,
            env=cli_environment(COLUMNS="200"),
        )

    assert proc.returncode == 0, proc.stderr
    drawn = [line for line in proc.stderr.splitlines() if line.strip() == ">"]
    assert drawn, proc.stderr
    for line in drawn:
        # Nothing follows the marker but the user's own keystrokes: the row is
        # not padded out to the terminal width, so the cursor stays on the
        # marker however wide the terminal claims to be.
        assert line.rstrip() == ">", repr(line)
        assert len(line) < 200, repr(line)


def test_status_bar_elides_the_cwd_when_the_terminal_is_tight(
    project_dir: Path,
) -> None:
    """Too narrow for path and readouts: the path yields, the prompt survives."""
    with StubOpenRouter([text_step("ok")]) as stub:
        proc = subprocess.run(
            [sys.executable, "-m", "slipagent.cli", "--base-url", stub.base_url,
             "--model", "stub/model"],
            cwd=project_dir, input="go\n/exit\n", capture_output=True, text=True,
            timeout=60,
            env=cli_environment(COLUMNS="30"),
        )

    assert proc.returncode == 0, proc.stderr
    line = next(line for line in proc.stderr.splitlines() if "free: " in line)
    assert "free: " in line
    assert "model: " in line
    assert len(line) <= 30
    assert any(line.strip() == ">" for line in proc.stderr.splitlines())


def test_prompt_and_status_never_share_a_line_with_output(project_dir: Path) -> None:
    """Agent output must not be written onto the prompt's row."""
    with StubOpenRouter(
        [tool_step("read_file", {"path": "src/app.py"}), text_step("The answer.")]
    ) as stub:
        proc = run_repl_commands(
            project_dir, ["go"], "--base-url", stub.base_url, "-v"
        )

    assert proc.returncode == 0, proc.stderr
    for line in proc.stderr.splitlines():
        if re.search(r"(?:^|\s)>(?:\s|$)", line) is None:
            continue
        # Either a prompt row (readouts, then the marker and the user's text)
        # or the echo of a queued line — never both mashed together.
        assert line.lstrip().startswith(">") or "free: " in line, line


def test_mid_turn_input_is_queued_and_labelled(project_dir: Path) -> None:
    """Text typed while the agent works is accepted, and visibly queued."""
    with StubOpenRouter(
        [tool_step("list_dir", {}), text_step("done"), text_step("also done")]
    ) as stub:
        proc = run_repl_commands(
            project_dir, ["go", "also do this"], "--base-url", stub.base_url
        )

    assert proc.returncode == 0, proc.stderr
    assert "also do this" in proc.stderr
    assert "queued" in proc.stderr
    lines = proc.stderr.splitlines()
    submitted = lines.index("> also do this")
    assert lines[submitted - 1] == ""
    assert "queued" in lines[submitted + 1]
    assert lines[submitted + 2] == ""
    # It must have reached the model as ordinary user input.
    users = [prompt for record in context_records(stub.requests[-1]["messages"]) for prompt in record.get("user_prompt", [])]
    assert "also do this" in users


@pytest.mark.parametrize("command", ["/exit", "/quit"])
def test_exit_during_a_turn_still_prints_the_answer(project_dir: Path, command: str) -> None:
    """Exit commands must not discard work already in flight."""
    with StubOpenRouter([text_step("finished anyway")]) as stub:
        proc = run_repl_commands(
            project_dir, ["go", command], "--base-url", stub.base_url
        )

    assert proc.returncode == 0, proc.stderr
    assert "finished anyway" in proc.stderr


def test_quit_is_advertised_and_exits_without_sending_later_input(project_dir: Path) -> None:
    with StubOpenRouter([text_step("should never run")]) as stub:
        proc = run_repl_commands(
            project_dir, ["/help", "/quit", "should never run"], "--base-url", stub.base_url,
        )
    assert proc.returncode == 0, proc.stderr
    assert "/quit" in proc.stderr
    assert "same as /exit" in proc.stderr
    assert not stub.requests


def test_turn_completes_when_input_ends_mid_turn(project_dir: Path) -> None:
    """Closing the pipe must not cancel the answer in progress."""
    with StubOpenRouter([text_step("the answer")]) as stub:
        proc = run_repl_commands(project_dir, ["go"], "--base-url", stub.base_url)

    assert proc.returncode == 0, proc.stderr
    assert "the answer" in proc.stderr


def test_one_shot_step_limit_returns_failure(project_dir: Path) -> None:
    with StubOpenRouter([tool_step("list_dir", {})]) as stub:
        result = run_cli("go", "--max-steps", "1", "--base-url", stub.base_url, cwd=project_dir)
    assert result.returncode == 1
    assert "step limit" in result.stdout


def test_missing_key_for_list_models_is_reported_without_traceback(project_dir: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "slipagent.cli", "--list-models"],
        cwd=project_dir, capture_output=True, text=True, timeout=5,
        env=cli_environment(OPENROUTER_API_KEY=""))
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_mcp_remove_without_name_reports_usage(project_dir: Path) -> None:
    proc = run_repl_commands(project_dir, ["/mcp remove"])
    assert proc.returncode == 0
    assert "usage:" in proc.stderr


async def test_build_session_honors_workspace_environment(tmp_path, monkeypatch, metadata_server) -> None:
    from slipagent.cli import build_parser, build_session, _shutdown
    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("SLIPAGENT_WORKSPACE", str(tmp_path))
    session = await build_session(build_parser().parse_args(["--no-mcp"]))
    try:
        assert session.workspace.root == tmp_path
    finally:
        await _shutdown(session)


async def test_cancelled_input_reader_does_not_lose_next_line(tmp_path, monkeypatch, metadata_server) -> None:
    import asyncio
    import io
    from slipagent.cli import build_parser, build_session, _shutdown, _read_line

    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    session = await build_session(build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    reading = threading.Event()
    release = threading.Event()
    class Input:
        calls = 0
        def readline(self):
            self.calls += 1
            if self.calls == 1:
                reading.set()
                release.wait(2)
                return "next task\n"
            return ""
    source = Input()
    monkeypatch.setattr(sys, "stdin", source)
    task = asyncio.create_task(_read_line(session, session.renderer.style))
    try:
        await asyncio.to_thread(reading.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        line = await _read_line(session, session.renderer.style)
        assert line == "next task\n"
    finally:
        release.set()
        await _shutdown(session)


def test_followup_during_final_response_is_answered(project_dir: Path) -> None:
    import time
    requested, release, queued = threading.Event(), threading.Event(), threading.Event()
    class WaitingStub(StubOpenRouter):
        def _make_handler(self):
            handler = super()._make_handler()
            send = handler._send
            def paused_send(instance, payload, status=200):
                if payload.get("choices") and not requested.is_set():
                    requested.set()
                    assert release.wait(5)
                send(instance, payload, status)
            handler._send = paused_send
            return handler

    with WaitingStub([text_step("first answer"), text_step("followup answer")]) as stub:
        # Keep stdin open: /quit and EOF intentionally prevent another queued
        # run. Submit quit only after the owed follow-up has actually started.
        proc = subprocess.Popen(
            [sys.executable, "-m", "slipagent.cli", "--base-url", stub.base_url,
             "--model", "stub/model", "--workspace", str(project_dir)],
            cwd=project_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=cli_environment(),
        )
        output = []
        def read_output():
            for line in proc.stderr:
                output.append(line)
                if "(queued" in line:
                    queued.set()
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            proc.stdin.write("go\n")
            proc.stdin.flush()
            assert requested.wait(5)
            proc.stdin.write("followup task\n")
            proc.stdin.flush()
            assert queued.wait(5)
            release.set()
            deadline = time.monotonic() + 5
            while len(stub.requests) < 2 and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            proc.stdin.write("/quit\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        finally:
            release.set()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            reader.join(timeout=5)
            proc.stdin.close()
            proc.stdout.close()
            proc.stderr.close()
        error = "".join(output)
    assert proc.returncode == 0, error
    assert "followup answer" in error
    assert len(stub.requests) == 2


@pytest.mark.parametrize("command", ["/reset", "/resume", "/danger", "/danger off", "/key secret", "/mcp remove stub", "/model new/model"])
async def test_mutating_commands_wait_until_turn_finishes(tmp_path, monkeypatch, command, metadata_server) -> None:
    import io
    from slipagent.cli import build_parser, build_session, _shutdown, _handle_command
    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    session = await build_session(build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    out = io.StringIO()
    session.renderer.stream = out
    session.agent.running = True
    try:
        await _handle_command(session, command)
        assert "Queued /" in out.getvalue()
        assert session.extensions["deferred_commands"] == [command]
        assert session.agent.messages
        assert session.api_key == "test"
    finally:
        session.agent.running = False
        await _shutdown(session)


async def test_danger_commands_preserve_mode_through_reset_and_resume(tmp_path, metadata_server):
    import io
    from slipagent import cli
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    output = io.StringIO()
    session.renderer.stream = output
    try:
        assert not session.workspace.access.danger
        await cli._handle_command(session, "/danger")
        assert session.workspace.access.danger
        await cli._handle_command(session, "/danger")
        assert session.workspace.access.danger  # Repeating the command keeps it enabled.
        await cli._handle_command(session, "/reset")
        await cli._handle_command(session, "/resume latest")
        assert session.workspace.access.danger
        session.agent.running = True
        await cli._handle_command(session, "/danger status")
        assert "Danger mode ON" in output.getvalue()
        session.agent.running = False
        await cli._handle_command(session, "/danger off")
        assert not session.workspace.access.danger
        await cli._handle_command(session, "/danger invalid")
        assert "usage: /danger" in output.getvalue()
        assert not session.workspace.access.danger
    finally:
        session.agent.running = False
        await cli._shutdown(session)


@pytest.mark.parametrize("selection", [None, "/tools", "/reset", "/quit", "/models"])
async def test_menu_dispatch_preserves_command_checks(tmp_path, monkeypatch, metadata_server, selection):
    import asyncio
    import io
    from unittest.mock import AsyncMock, Mock
    from slipagent import cli
    from slipagent.terminal import TerminalUI

    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    output = io.StringIO()
    terminal = Mock(spec=TerminalUI)
    terminal.choose = AsyncMock(return_value=selection)
    terminal.write.side_effect = lambda text, **kwargs: output.write(text)
    session.renderer.terminal = terminal
    models = AsyncMock()
    monkeypatch.setattr(cli, "_models_command", models)
    session.agent.running = True
    try:
        assert await cli._handle_command(session, "/menu") is (selection == "/quit")
        terminal.choose.assert_awaited_once_with("Commands", cli.MENU_OPTIONS)
        if selection == "/tools":
            assert "read_file" in output.getvalue()
        elif selection == "/reset":
            assert "Queued /reset" in output.getvalue()
            assert session.agent.messages
        elif selection == "/models":
            tasks = list(session.extensions["command_tasks"])
            assert tasks
            await asyncio.gather(*tasks)
            models.assert_awaited_once()
        else:
            assert output.getvalue() == ""
    finally:
        session.agent.running = False
        await cli._shutdown(session)


def test_menu_without_terminal_reports_how_to_get_commands(project_dir):
    proc = run_repl_commands(project_dir, ["/menu extra", "/menu", "/resume"])
    assert proc.returncode == 0, proc.stderr
    assert "usage: /menu" in proc.stderr
    assert "/menu requires an interactive terminal; use /help" in proc.stderr
    assert "use /resume <id|latest>" in proc.stderr


@pytest.mark.parametrize("cancel", [False, True])
async def test_resume_picker_lists_all_sessions_and_restores_only_the_selected_one(
    tmp_path, metadata_server, cancel,
):
    import io
    from unittest.mock import AsyncMock, Mock
    from slipagent import cli
    from slipagent.terminal import TerminalUI
    from slipagent.types import Message

    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    journal = session.registry.services["session_journal"]
    output = io.StringIO()
    terminal = Mock(spec=TerminalUI)
    terminal.write.side_effect = lambda text, **kwargs: output.write(text)
    session.renderer.terminal = terminal
    try:
        ids = []
        for index in range(8):
            journal.rename(f"Saved Session {index}")
            session.agent.messages.extend([
                Message.user(f"Question {index}"), Message.assistant(f"Answer {index}"),
            ])
            session.agent._persist()
            ids.append(journal.session_id)
            session.agent.reset()
        entries = journal.listing()
        originals = {path: path.read_bytes() for path in journal.directory.glob("*.jsonl")}
        current_id = journal.session_id
        current_messages = list(session.agent.messages)
        terminal.choose = AsyncMock(return_value=None if cancel else ids[2])

        assert await cli._handle_command(session, "/resume") is False
        terminal.choose.assert_awaited_once()
        title, options = terminal.choose.call_args.args
        assert title == "Resume Session"
        assert [value for value, label in options] == [entry["id"] for entry in entries]
        for entry, (value, label) in zip(entries, options):
            assert entry["title"] in label and entry["created"] in label
            assert ("(current)" in label) == (value == current_id)
        assert all(path.read_bytes() == content for path, content in originals.items())
        if cancel:
            assert journal.session_id == current_id
            assert session.agent.messages == current_messages
            assert len(journal.listing()) == 9
            assert output.getvalue() == ""
            terminal.clear_transcript.assert_not_called()
        else:
            assert journal.session_id not in {*ids, current_id}
            assert journal.title == "Saved Session 2 | SlipAgent"
            assert [message.content for message in session.agent.messages if message.role != "system"] == ["Question 2", "Answer 2"]
            assert "Question 2" in output.getvalue() and "Answer 2" in output.getvalue()
            terminal.clear_transcript.assert_called_once()
            terminal.scroll_to_end.assert_called_once()
            terminal.set_title.assert_called_with("Saved Session 2 | SlipAgent")
    finally:
        await cli._shutdown(session)


@pytest.mark.parametrize("scenario", ["empty", "unreadable", "invalid", "disabled"])
async def test_resume_picker_errors_preserve_active_session(tmp_path, monkeypatch, metadata_server, scenario):
    import io
    from unittest.mock import AsyncMock, Mock
    from slipagent import cli
    from slipagent.sessions import SessionError
    from slipagent.terminal import TerminalUI

    args = ["--no-mcp", "-w", str(tmp_path)]
    if scenario == "disabled":
        args.append("--no-session")
    session = await cli.build_session(cli.build_parser().parse_args(args))
    output = io.StringIO()
    terminal = Mock(spec=TerminalUI)
    terminal.choose = AsyncMock(return_value="f" * 32)
    terminal.write.side_effect = lambda text, **kwargs: output.write(text)
    session.renderer.terminal = terminal
    journal = session.registry.services.get("session_journal")
    current_id = session.agent.session_id
    current_messages = list(session.agent.messages)
    try:
        if scenario == "empty":
            monkeypatch.setattr(journal, "listing", lambda: [])
        elif scenario == "unreadable":
            monkeypatch.setattr(journal, "listing", Mock(side_effect=SessionError("Cannot list saved sessions: unreadable")))
        assert await cli._handle_command(session, "/resume") is False
        expected = {"empty": "no saved sessions", "unreadable": "Cannot list saved sessions",
                    "invalid": "Cannot resume", "disabled": "session persistence is disabled"}
        assert expected[scenario] in output.getvalue()
        assert session.agent.session_id == current_id
        assert session.agent.messages == current_messages
        terminal.clear_transcript.assert_not_called()
        if scenario != "invalid":
            terminal.choose.assert_not_awaited()
    finally:
        await cli._shutdown(session)


def test_mcp_command_preserves_quoted_arguments() -> None:
    from slipagent.cli import _split_mcp_command
    _, spec, _ = _split_mcp_command('add stub python "path with spaces/server.py"')
    assert spec.args == ["path with spaces/server.py"]


def test_rename_is_saved_listed_and_restored_in_another_process(project_dir):
    first = run_repl_commands(project_dir, ["/rename", "/rename Review café changes", "/sessions"])
    assert first.returncode == 0, first.stderr
    assert "usage: /rename <name>" in first.stderr
    assert first.stderr.count("Review café changes | SlipAgent") >= 2
    second = run_repl_commands(project_dir, ["/sessions"], "--resume", "latest")
    assert second.returncode == 0, second.stderr
    assert second.stderr.count("Review café changes | SlipAgent") >= 2
    assert "\x1b]2;" not in first.stderr + second.stderr


async def test_interrupted_repl_turn_cancels_agent(tmp_path, monkeypatch, metadata_server) -> None:
    import asyncio
    import io
    from slipagent import cli
    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    session.renderer.stream = io.StringIO()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    running = []
    async def run(agent, prompt):
        running.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    async def read(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(cli.Agent, "run", run)
    monkeypatch.setattr(cli, "_read_line", read)
    turn = asyncio.create_task(cli._run_turn(session, "go", session.renderer.style))
    try:
        await started.wait()
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        assert cancelled.is_set()
    finally:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        await cli._shutdown(session)


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX pseudoterminal")
def test_tty_footer_stop_and_explicit_resume(project_dir: Path) -> None:
    import codecs
    import fcntl
    import select
    import struct
    import termios
    import time
    import pyte

    started, release = threading.Event(), threading.Event()
    class SlowStub(StubOpenRouter):
        def _make_handler(self):
            base = super()._make_handler()
            class Handler(base):
                def do_POST(self):
                    if not started.is_set():
                        started.set()
                        release.wait(5)
                    super().do_POST()
            return Handler

    step = tool_step("write_file", {"path": "finished.txt", "content": "completed"})
    step["choices"][0]["message"]["content"] = "Writing the file and checking the directory."
    step["choices"][0]["message"]["tool_calls"].append({
        "id": "call_2", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}})
    master, slave = os.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    original_modes = termios.tcgetattr(slave)
    screen = pyte.Screen(100, 24)
    parser = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with SlowStub([step, text_step("resumed successfully")]) as stub:
        proc = subprocess.Popen(
            [sys.executable, "-m", "slipagent.cli", "--base-url", stub.base_url,
             "--workspace", str(project_dir), "--model", "stub/one", "--no-mcp"],
            cwd=project_dir, stdin=slave, stdout=slave, stderr=slave,
            env=cli_environment(TERM="xterm-256color", NO_COLOR=""),
        )
        def wait_for(predicate):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if select.select([master], [], [], .05)[0]:
                    raw = os.read(master, 65536)
                    if b"\x1b[6n" in raw:
                        os.write(master, b"\x1b[1;1R")
                    parser.feed(decoder.decode(raw))
                if predicate():
                    return
                assert proc.poll() is None, "CLI exited before the expected state"
            pytest.fail("terminal state timed out: " + repr(screen.display))
        try:
            wait_for(lambda: "model: stub/one" in screen.display[23])
            assert screen.title == f"{project_dir} | SlipAgent"
            assert screen.display[22].strip() == ""
            startup = "\n".join(screen.display[:17])
            for command in ["/help", "/menu", "/danger", "/tools", "/model", "/models", "/key", "/cost", "/mcp", "/rename", "/reset", "/reload", "/generations", "/init", "/stop", "/exit", "/quit"]:
                assert command in startup
            assert "Follow-ups queue" in startup
            assert "Ctrl-D quits" in startup
            os.write(master, b"/danger\r")
            wait_for(lambda: screen.display[23].rstrip().endswith(" | Danger Mode"))
            start = screen.display[23].index("Danger Mode")
            assert all(screen.buffer[23][column].fg == "ff6666" for column in range(start, start + len("Danger Mode")))
            os.write(master, b"/danger off\r")
            wait_for(lambda: "Danger Mode" not in screen.display[23])
            os.write(master, b"/menu\r")
            wait_for(lambda: "Enter = select | Esc = back" in screen.display[23])
            assert screen.display[18].strip() == "Commands"
            os.write(master, b"\x1b")
            wait_for(lambda: "Ready" in screen.display[18])
            assert not stub.requests
            os.write(master, b"/menu\r")
            wait_for(lambda: "Enter = select | Esc = back" in screen.display[23])
            os.write(master, b"\x1b[B\r")
            wait_for(lambda: "Ready" in screen.display[18] and "read_file" in "\n".join(screen.display[:17]))
            assert not stub.requests
            os.write(master, b"go\r")
            wait_for(started.is_set)
            os.write(master, b"/rename Work in progress\r")
            wait_for(lambda: screen.title == "Work in progress | SlipAgent")
            os.write(master, b"/stop\r")
            wait_for(lambda: "Stopping After This Turn" in screen.display[18])
            os.write(master, b"queued followup\r")
            wait_for(lambda: "queued followup" in "\n".join(screen.display[:17]))
            release.set()
            wait_for(lambda: "Stopped after the current turn" in "\n".join(screen.display[:17]) and "Ready" in screen.display[18])
            assert (project_dir / "finished.txt").read_text() == "completed"
            assert len(stub.requests) == 1
            os.write(master, b"continue\r")
            wait_for(lambda: "resumed successfully" in "\n".join(screen.display[:17]))
            assert len(stub.requests) == 2
            history = unpack_api_context(stub.requests[1]["messages"])
            assert [m["tool_call_id"] for m in history if m["role"] == "tool"] == ["call_1", "call_2"]
            assert "queued followup" in [context_body(m["content"]) for m in history if m["role"] == "user"]
            os.write(master, b"/model stub/two\r")
            wait_for(lambda: "model: stub/two" in screen.display[23])
            assert screen.title == "Work in progress | SlipAgent"
            os.write(master, b"/reset\r")
            wait_for(lambda: screen.title == f"{project_dir} | SlipAgent")
            wait_for(lambda: "conversation cleared" in "\n".join(screen.display[:17]))
            os.write(master, b"/resume\r")
            wait_for(lambda: screen.display[18].strip() == "Resume Session")
            os.write(master, b"\x1b")
            wait_for(lambda: "Ready" in screen.display[18])
            assert screen.title == f"{project_dir} | SlipAgent"
            assert "conversation cleared" in "\n".join(screen.display[:17])
            os.write(master, b"/resume\r")
            wait_for(lambda: screen.display[18].strip() == "Resume Session")
            os.write(master, b"\x1b[B\r")
            wait_for(lambda: screen.title == "Work in progress | SlipAgent")
            wait_for(lambda: "restored " in "\n".join(screen.display[:17]))
            assert "resumed successfully" in "\n".join(screen.display[:17])
            assert "conversation cleared" not in "\n".join(screen.display[:17])
            os.write(master, b"/exit\r")
            assert proc.wait(timeout=5) == 0
            assert termios.tcgetattr(slave) == original_modes
        finally:
            release.set()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            os.close(master)
            os.close(slave)
