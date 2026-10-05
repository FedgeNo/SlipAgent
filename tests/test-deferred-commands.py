"""State-changing commands apply only after the active tool batch has settled."""

import asyncio
import io

import pytest

from slipagent import cli
from slipagent.tools.base import Tool, ToolResult
from test_agent import StubClient, call, completion
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
    session = await cli.build_session(cli.build_parser().parse_args([
        "--no-mcp", "-w", str(tmp_path), "--max-steps", str(max_steps),
    ]))
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
    client = StubClient([
        completion(tool_calls=[call("gated_read")]),
        completion("Finished with the selected access mode."),
    ])
    session.agent.client = client

    async def read(*arguments):
        await asyncio.Event().wait()

    monkeypatch.setattr(cli, "_read_line", read)
    turn = asyncio.create_task(cli._run_turn(session, "Inspect the project", session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for command in commands:
            await cli._handle_command(session, command)
        assert not session.workspace.access.danger
        if stop:
            await cli._handle_command(session, "/stop")
        release.set()
        assert await asyncio.wait_for(turn, 5) is False
        assert session.workspace.access.danger == expected_danger
        assert len(client.calls) == expected_calls
        assert not session.extensions.get("deferred_commands")
        assert "Queued /" in session.renderer.stream.getvalue()
        if expected_calls == 2:
            system = client.calls[1]["messages"][0].content
            assert ("Danger mode ON" in system) == expected_danger
    finally:
        release.set()
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await cli._shutdown(session)


@pytest.mark.parametrize("outcome", ["final", "failure", "quit"])
async def test_deferred_settings_after_final_failure_or_exit(tmp_path, monkeypatch, metadata_server, outcome):
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
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

    monkeypatch.setattr(cli, "_read_line", read)
    turn = asyncio.create_task(cli._run_turn(session, "Inspect the project", session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await cli._handle_command(session, "/danger")
        assert not session.workspace.access.danger
        if outcome == "quit":
            assert await cli._handle_command(session, "/quit")
            # The input loop passes the exit decision to the deferred boundary.
            assert await cli._apply_deferred_commands(session, exiting=True) == (False, True)
        release.set()
        if outcome == "failure":
            with pytest.raises(RuntimeError, match="Test request failed"):
                await asyncio.wait_for(turn, 5)
        else:
            assert await asyncio.wait_for(turn, 5) is False
        assert session.workspace.access.danger == (outcome != "quit")
        assert len(client.calls) == (0 if outcome == "failure" else 1)
        assert not session.extensions.get("deferred_commands")
    finally:
        release.set()
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await cli._shutdown(session)
