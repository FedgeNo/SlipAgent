
from slipagent.types import content_text

from slipagent.types import decode_json_content
import asyncio
import json
import sys

import pytest

from slipagent.lsp import LanguageServers, NavigateCodeTool, validate_servers


SERVER = '''
import json, sys
from pathlib import Path
opened = None
events = []
def send(payload):
    raw = json.dumps({"jsonrpc":"2.0", **payload}).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(raw)}\\r\\n\\r\\n".encode() + raw)
    sys.stdout.buffer.flush()
while True:
    header = b""
    while not header.endswith(b"\\r\\n\\r\\n"):
        part = sys.stdin.buffer.read(1)
        if not part: sys.exit(0)
        header += part
    size = int(header.split(b":", 1)[1].strip())
    message = json.loads(sys.stdin.buffer.read(size))
    method = message.get("method")
    events.append(method)
    if method == "initialize":
        send({"id":message["id"],"result":{"capabilities":{
            "definitionProvider":True,"referencesProvider":True,"implementationProvider":True,"hoverProvider":True}}})
    elif method == "textDocument/didOpen": opened = message["params"]["textDocument"]
    elif method == "textDocument/didClose": opened = None
    elif method == "textDocument/hover":
        send({"id":message["id"],"result":{"contents":{"kind":"plaintext","value":opened["text"]}}})
    elif method in ("textDocument/definition", "textDocument/references", "textDocument/implementation"):
        position = message["params"]["position"]
        send({"id":message["id"], "result":[
            {"uri": opened["uri"], "range":{"start":position,"end":position}},
            {"uri": Path.cwd().parent.as_uri()+"/outside.py", "range":{"start":position,"end":position}}]})
    elif method == "shutdown": send({"id":message["id"],"result":None})
    elif method == "exit":
        Path("events.json").write_text(json.dumps(events))
        sys.exit(0)
'''


def setup(workspace, source=SERVER):
    script = workspace.root / "server.py"
    script.write_text(source)
    directory = workspace.root / ".slipagent"
    directory.mkdir()
    settings = {"language_servers": {"test": {"command": [sys.executable, str(script)],
                                            "extensions": [".py"], "language_id": "python"}}}
    (directory / "project.json").write_text(json.dumps(settings))
    return settings


async def test_navigation_confinement_unicode_and_fresh_disk_contents(workspace):
    setup(workspace)
    source = workspace.root / "source.py"
    source.write_text("a😀target = 1\n")
    servers = LanguageServers(workspace)
    tool = NavigateCodeTool(servers)
    try:
        for operation in ["definition", "references", "implementation"]:
            result = await tool.invoke({"operation": operation, "path": "source.py", "line": 1, "column": 3})
            assert not result.is_error, result.content
            data = decode_json_content(result.content)
            assert data["locations"] == [{"path": "source.py", "line": 1, "column_utf16": 4}]
            assert data["excluded_outside_workspace"] == 1
        source.write_text("updated = 2\n")
        result = await tool.run("hover", "source.py", 1, 1)
        assert result.content == "updated = 2\n"
        process = servers.clients["test"].process
        assert process.returncode is None
    finally:
        await servers.aclose()
    assert process.returncode == 0
    events = json.loads((workspace.root / "events.json").read_text())
    assert events[0] == "initialize" and events[-2:] == ["shutdown", "exit"]
    assert "workspace/didChangeWatchedFiles" in events


async def test_unconfigured_navigation_reports_setup_without_spawning(workspace):
    (workspace.root / "source.py").write_text("x = 1")
    servers = LanguageServers(workspace)
    result = await NavigateCodeTool(servers).run("definition", "source.py", 1, 1)
    assert result.is_error and "Configure language_servers" in content_text(result.content)
    assert not servers.clients
    await servers.aclose()
    closed = await NavigateCodeTool(servers).invoke({"operation": "definition", "path": "source.py", "line": 1, "column": 1})
    assert closed.is_error and "closed" in content_text(closed.content)


async def test_cancelled_initialize_reaps_server(workspace):
    setup(workspace, "import time; time.sleep(60)")
    (workspace.root / "source.py").write_text("x = 1")
    servers = LanguageServers(workspace)
    task = asyncio.create_task(NavigateCodeTool(servers).run("hover", "source.py", 1, 1))
    try:
        while not servers.clients or servers.clients["test"].process is None:
            await asyncio.sleep(.01)
        process = servers.clients["test"].process
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.returncode is not None
        assert not servers.clients
    finally:
        await servers.aclose()


async def test_oversized_startup_output_cannot_deadlock_cleanup(workspace):
    setup(workspace, "import sys,time; sys.stdout.write('x' * 1000000); sys.stdout.flush(); time.sleep(60)")
    (workspace.root / "source.py").write_text("x = 1")
    servers = LanguageServers(workspace)
    task = asyncio.create_task(NavigateCodeTool(servers).invoke({"operation": "hover", "path": "source.py", "line": 1, "column": 1}))
    try:
        done, _ = await asyncio.wait({task}, timeout=1)
        assert done, "Malformed startup output must fail and reap the server promptly"
        assert task.result().is_error
    finally:
        if not task.done():
            process = servers.clients["test"].process
            if process.returncode is None:
                process.kill()
            # The failed frame reader is finished. Drain the pipes so even the
            # pre-fix implementation can finish cleanup without orphaning it.
            await asyncio.gather(process.stdout.read(), process.stderr.read())
            await asyncio.wait_for(task, 2)
        await servers.aclose()


@pytest.mark.parametrize("config", [[], {"x": {}}, {"x": {"command": "bad"}}])
def test_invalid_server_configuration_is_explicit(config):
    with pytest.raises(ValueError):
        validate_servers(config)


async def test_server_configuration_removal_closes_existing_process(workspace):
    setup(workspace)
    (workspace.root / "source.py").write_text("x = 1")
    servers = LanguageServers(workspace)
    try:
        await NavigateCodeTool(servers).run("hover", "source.py", 1, 1)
        process = servers.clients["test"].process
        (workspace.root / ".slipagent" / "project.json").write_text("{}")
        await servers.refresh()
        assert process.returncode is not None
        assert not servers.clients
    finally:
        await servers.aclose()


async def test_request_timeout_does_not_leave_pending_response_waiter(workspace):
    from slipagent.lsp import LanguageServer
    settings = setup(workspace, SERVER.replace('elif method == "textDocument/hover":', 'elif method == "ignored-hover":'))
    server = LanguageServer(workspace, settings["language_servers"]["test"])
    try:
        await server.start()
        with pytest.raises(TimeoutError):
            await server.request("textDocument/hover", {}, timeout=.01)
        assert not server.pending
    finally:
        await server.aclose()


async def test_navigation_tool_is_exposed_only_when_configured(workspace):
    from slipagent.tools import build_default_registry
    registry = build_default_registry(workspace)
    try:
        assert registry.get("navigate_code") is None
        setup(workspace)
        configured = build_default_registry(workspace, services=registry.services)
        try:
            assert configured.get("navigate_code") is not None
        finally:
            await configured.aclose()
    finally:
        await registry.aclose()
