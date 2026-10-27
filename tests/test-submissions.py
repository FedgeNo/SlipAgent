"""Immediate, cancellable previews and FIFO command/prompt boundaries."""

import asyncio
import io
from importlib import import_module
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType, MouseButton
from prompt_toolkit.data_structures import Point

from slipagent import cli
from slipagent.inputqueue import input_queue
from slipagent.submissions import Submission
from slipagent.terminal import TerminalUI
from test_agent import StubClient, completion, unpack_context
from test_cli_e2e import metadata_server

display = import_module("test-terminal").display
wait_until = import_module("test-terminal").wait_until


@pytest.mark.parametrize("initial_queued", [False, True])
@pytest.mark.parametrize("removal", [None, "command", "prompt", "limited"])
async def test_mixed_input_obeys_submission_order(tmp_path, monkeypatch, metadata_server, initial_queued, removal):
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

    client = GatedClient([completion("first"), completion("second"), completion("third")])
    session.agent.client = client
    from slipagent.capabilities import RequestProfile
    monkeypatch.setattr(session.client, "model_capabilities", AsyncMock(return_value=RequestProfile(
        parameters={"temperature"}, context_length=100000)))
    original_temperature = session.agent.temperature
    if removal == "limited":
        session.agent.max_steps = 1
    lines = asyncio.Queue()

    async def read(*args):
        return await lines.get()

    monkeypatch.setattr(cli.Repl, "read_line", read)
    if initial_queued:
        session.agent.enqueue(Submission("original"))
    task = asyncio.create_task(cli.Repl(session).run_request("" if initial_queued else "original", session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        queued = [Submission(text) for text in ["  earlier correction  ", "  /temperature 0.2  ", "later correction",
                                                "another later correction", "/temperature 0.3"]]
        for item in queued:
            lines.put_nowait(item)
        async with asyncio.timeout(2):
            while len(session.work.deferred_commands or []) < 4:
                await asyncio.sleep(0)
        if removal in {"command", "prompt"}:
            queued[1 if removal == "command" else 2].cancelled = True
        release.set()
        assert not await asyncio.wait_for(task, 5)
        if removal == "limited":
            assert len(client.calls) == 1
            assert session.agent.has_queued_input
            assert session.work.deferred_commands
            assert session.agent.temperature == original_temperature
            return
        assert [call["temperature"] for call in client.calls] == [
            original_temperature, original_temperature, original_temperature if removal == "command" else .2]
        assert session.agent.temperature == .3
        second = [m.content for m in unpack_context(client.calls[1]["messages"]) if m.role == "user"]
        third = [m.content for m in unpack_context(client.calls[2]["messages"]) if m.role == "user"]
        assert "earlier correction" in second and "later correction" not in second
        assert ("later correction" in third) == (removal != "prompt")
        assert "another later correction" in third
        assert not session.work.deferred_commands
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await cli.Repl(session).shutdown()


@pytest.mark.parametrize("keyboard", [False, True])
async def test_delete_specific_duplicate_and_command(display, keyboard):
    sink, _, output, _, snapshot = display
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "readout", sink, input=pipe, output=output)
        app = asyncio.create_task(ui.run())
        try:
            pipe.send_text("界 same\r界 same\r/temperature 0.2\r")
            await wait_until(lambda: len(ui.__dict__.get("_submissions", [])) == 3)
            first, second, command = ui.__dict__["_submissions"]
            await wait_until(lambda: "/temperature 0.2" in "\n".join(snapshot()))
            for target in [second, command]:
                stops = ui._focus.stops()
                stop = next(s for s in stops if s[1] == ("discard-input", target.identifier))
                window, key, point, handler = stop
                if keyboard:
                    ui._focus.focus(window)
                    window.content.__dict__["_keyboard_key"] = key
                    window.content.__dict__["_keyboard_point"] = point
                    pipe.send_text("\r")
                    await wait_until(lambda: target.cancelled)
                else:
                    y, x = window.render_info._rowcol_to_yx[point.y, point.x]
                    pipe.send_text(f"\x1b[<0;{x + 1};{y + 1}M\x1b[<0;{x + 1};{y + 1}m")
                    await wait_until(lambda: target.cancelled)
                assert target.cancelled
                await asyncio.sleep(.05)
            assert not first.cancelled
            assert await ui.read_line() is first
            ui._lines.put_nowait(None)
            assert await ui.read_line() is None
        finally:
            ui.close()
            await app


async def test_delete_agent_pending_and_hide_key(display, monkeypatch):
    sink, _, output, _, _ = display
    ui = TerminalUI(lambda width: "readout", sink, output=output)
    renderer = cli.Renderer(cli.Style(False), sink, False)
    renderer.terminal = ui
    from slipagent.agent import Agent
    from slipagent.tools.base import ToolRegistry
    agent = Agent(StubClient([]), ToolRegistry(), "stub", on_event=renderer.handle)
    ui.__dict__["_agent_input_queue"] = lambda: input_queue(agent)
    try:
        agent.enqueue("same")
        agent.enqueue("same")
        entries = input_queue(agent).snapshot()
        ui._submit("/key dummy-secret")
        rows = ui._queued_prompt_content(40)
        assert "dummy-secret" not in str([rows.get_line(i) for i in range(rows.line_count)])
        line = rows.get_line(1)
        assert line[-1][1] == "Cancel"
        narrow_line = ui._queued_prompt_content(12).get_line(1)
        assert "".join(fragment[1] for fragment in narrow_line) == "> sa… Cancel"
        line[-1][2](MouseEvent(Point(39, 1), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
        assert input_queue(agent).snapshot() == entries[:1]
        assert agent.pending == ["same"]
        # A reset replaces InputQueue; new Cancel actions must use the replacement.
        agent.reset()
        agent.enqueue("after reset")
        ui._queued_items()[0][2]()
        assert not agent.pending
        # The previously painted Cancel link follows a submission across its handoff.
        ui._submit("transferred")
        submitted = ui.__dict__["_submissions"][-1]
        remove = ui._queued_items()[-1][2]
        ui.consume_submission(submitted)
        agent.enqueue(submitted)
        remove()
        assert not agent.pending
        # A click against a previously rendered row must not remove the row
        # which moved into its place since that paint.
        ui.consume_submission(ui.__dict__["_submissions"][0])  # hidden key
        ui._submit("already sent")
        old = ui.__dict__["_submissions"][-1]
        ui._submit("keep next")
        ui._queued_prompt_content(40)
        ui.consume_submission(old)
        ui._queued_prompt_mouse(MouseEvent(Point(39, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
        assert [str(item) for item in ui.__dict__["_submissions"]] == ["keep next"]
        from slipagent.sessions import SessionError
        def fail(identifier):
            raise SessionError("Test save failure")
        notices = []
        agent.enqueue("keep on failure")
        monkeypatch.setattr(input_queue(agent), "discard", fail)
        monkeypatch.setattr(ui, "notify", notices.append)
        ui._queued_items()[0][2]()
        assert agent.pending == ["keep on failure"]
        assert notices == ["Could not remove queued input: Test save failure"]
    finally:
        ui.close()


async def test_question_reply_does_not_consume_queued_submission(display):
    sink, _, output, _, _ = display
    ui = TerminalUI(lambda width: "readout", sink, output=output)
    try:
        ui._submit("later prompt")
        question = asyncio.create_task(ui.ask("Confirm: "))
        await asyncio.sleep(0)
        ui.input.text = "yes"
        ui._accept(ui.input.buffer)
        assert await question == "yes"
        assert await ui.read_line() == "later prompt"
    finally:
        ui.close()


async def test_exit_preserves_prompts_waiting_after_commands(tmp_path, metadata_server):
    session = await cli.Launcher(cli.build_parser().parse_args([
        "--no-mcp", "--no-reload", "-w", str(tmp_path),
    ])).build_session()
    session.renderer.stream = io.StringIO()
    try:
        session.work.deferred_commands = [Submission("/temperature 0.2"), Submission("keep for later")]
        original = session.agent.temperature
        assert await cli.CommandRunner(session).apply_deferred(exiting=True) == (False, True)
        assert session.agent.temperature == original
        assert session.agent.pending == ["keep for later"]
        assert not session.work.deferred_commands
    finally:
        await cli.Repl(session).shutdown()


def test_suggested_prompt_stays_verbatim_and_a_suggestion_when_stripped():
    from slipagent.terminal import SuggestedPrompt
    line = SuggestedPrompt("/etc/hosts has a typo ")
    stripped = line.strip()
    assert stripped is line and isinstance(stripped, SuggestedPrompt)
    assert line.strip(" ") == "/etc/hosts has a typo"
