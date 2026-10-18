"""State-changing commands apply only after the active tool batch has settled."""

from slipagent.types import content_text
import asyncio
import io

import pytest

from slipagent import cli
from slipagent.tools.base import Tool, ToolResult
from slipagent.tools.discovery import tool_catalog
from test_agent import StubClient, call, completion, unpack_context
from test_cli_e2e import metadata_server


@pytest.mark.parametrize("commands, stop, max_steps, expected_danger, expected_calls", [
    (["/danger"], False, 2, True, 2),
    (["/danger", "/danger off"], False, 2, False, 2),
    (["/danger"], True, 2, True, 1),
    (["/danger"], False, 1, True, 1),
    (["/reset", "/danger"], False, 2, True, 1),
])
async def test_commands_wait_for_tools_and_preserve_stop_and_request_budget(
    tmp_path, monkeypatch, metadata_server, commands, stop, max_steps, expected_danger, expected_calls,
):
    session = await cli.Launcher(cli.build_parser().parse_args([
        "--no-mcp", "-w", str(tmp_path), "--max-steps", str(max_steps),
    ])).build_session()
    session.renderer.stream = io.StringIO()
    entered = asyncio.Event()
    release = asyncio.Event()

    class GatedTool(Tool):
        name = "gated_read"
        description = "Read once the test releases the disk wait."
        parameters = {"type": "object", "properties": {}}

        async def run(self):
            entered.set()
            await release.wait()
            assert not session.workspace.access.danger
            return ToolResult.ok("The current tool finished before the mode changed.")

    session.registry.register(GatedTool())
    tool_catalog(session.registry).loaded.add("gated_read")
    client = StubClient([
        completion(tool_calls=[call("gated_read")]),
        completion("Finished with the selected access mode."),
    ])
    session.agent.client = client

    async def read(*arguments):
        await asyncio.Event().wait()

    monkeypatch.setattr(cli.Repl, "read_line", read)
    step = asyncio.create_task(cli.Repl(session).run_request("Inspect the project", session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for command in commands:
            await cli.CommandRunner(session).handle(command)
        assert not session.workspace.access.danger
        if stop:
            await cli.CommandRunner(session).handle("/stop")
        release.set()
        assert await asyncio.wait_for(step, 5) is False
        assert session.workspace.access.danger == expected_danger
        assert len(client.calls) == expected_calls
        assert not session.extensions.get("deferred_commands")
        assert "Queued /" in session.renderer.stream.getvalue()
        if expected_calls == 2:
            system = content_text(client.calls[1]["messages"][0].content)
            assert ("Danger Mode ON" in system) == expected_danger
    finally:
        release.set()
        step.cancel()
        await asyncio.gather(step, return_exceptions=True)
        await cli.Repl(session).shutdown()


@pytest.mark.parametrize("outcome", ["final", "failure", "quit"])
async def test_deferred_settings_after_final_failure_or_exit(tmp_path, monkeypatch, metadata_server, outcome):
    session = await cli.Launcher(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)])).build_session()
    session.renderer.stream = io.StringIO()
    entered = asyncio.Event()
    release = asyncio.Event()

    class GatedClient(StubClient):
        async def chat(self, **kwargs):
            entered.set()
            await release.wait()
            if outcome == "failure":
                raise RuntimeError("Test request failed")
            return await super().chat(**kwargs)

    client = GatedClient([completion("All work is complete.")])
    session.agent.client = client

    async def read(*arguments):
        await asyncio.Event().wait()

    monkeypatch.setattr(cli.Repl, "read_line", read)
    step = asyncio.create_task(cli.Repl(session).run_request("Inspect the project", session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await cli.CommandRunner(session).handle("/danger")
        assert not session.workspace.access.danger
        if outcome == "quit":
            assert await cli.CommandRunner(session).handle("/quit")
            # The input loop passes the exit decision to the deferred boundary.
            assert await cli.CommandRunner(session).apply_deferred(exiting=True) == (False, True)
        release.set()
        if outcome == "failure":
            with pytest.raises(RuntimeError, match="Test request failed"):
                await asyncio.wait_for(step, 5)
        else:
            assert await asyncio.wait_for(step, 5) is False
        assert session.workspace.access.danger == (outcome != "quit")
        assert len(client.calls) == (0 if outcome == "failure" else 1)
        assert not session.extensions.get("deferred_commands")
    finally:
        release.set()
        step.cancel()
        await asyncio.gather(step, return_exceptions=True)
        await cli.Repl(session).shutdown()


@pytest.mark.parametrize("queued_setting", [False, True])
@pytest.mark.parametrize("new_input", [None, "Inspect the next file", "/rename New title"])
async def test_stop_holds_earlier_input_until_new_input(
    tmp_path, monkeypatch, metadata_server, queued_setting, new_input,
):
    session = await cli.Launcher(cli.build_parser().parse_args([
        "--no-mcp", "--no-reload", "-w", str(tmp_path),
    ])).build_session()
    session.renderer.stream = io.StringIO()
    entered, release = asyncio.Event(), asyncio.Event()

    class GatedClient(StubClient):
        async def chat(self, **kwargs):
            if not self.calls:
                entered.set()
                await release.wait()
            return await super().chat(**kwargs)

    client = GatedClient([completion("First request finished."), completion("New request finished.")])
    session.agent.client = client
    lines = asyncio.Queue()

    async def read(*args):
        return await lines.get()

    monkeypatch.setattr(cli.Repl, "read_line", read)
    task = asyncio.create_task(cli.Repl(session).run_request("Start work", session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if queued_setting:
            await cli.CommandRunner(session).handle("/rename Queued title")
            cli.CommandRunner(session).queue("Earlier queued prompt", "input")
        else:
            session.agent.enqueue("Earlier queued prompt")
        await cli.CommandRunner(session).handle("/stop")
        if new_input is not None:
            lines.put_nowait(new_input)
            async with asyncio.timeout(2):
                while new_input not in session.agent.pending and new_input not in (session.work.deferred_commands or []):
                    await asyncio.sleep(0)
        release.set()
        assert not await asyncio.wait_for(task, 5)
        if new_input is None:
            assert len(client.calls) == 1
            assert session.agent.stopped
            assert session.agent.pending == ["Earlier queued prompt"]
            assert not session.work.deferred_commands
            assert not await cli.Repl(session).run_request("Inspect the next file", session.renderer.style)
        elif new_input.startswith("/"):
            assert "title: New title" in session.renderer.stream.getvalue()
        assert len(client.calls) == 2
        messages = [m.content for m in unpack_context(client.calls[1]["messages"]) if m.role == "user"]
        assert "Earlier queued prompt" in messages
        if new_input is None or not new_input.startswith("/"):
            assert messages.index("Earlier queued prompt") < messages.index("Inspect the next file")
        assert not session.work.deferred_commands
        assert not session.agent.pending
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()
