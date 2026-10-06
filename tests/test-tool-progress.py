"""Progress reaches the user during execution without requesting another model step."""

import asyncio
import io
import os
import shlex
import subprocess
import sys

import pytest

from slipagent.activity import OutputProgress, command_output
from slipagent.agent import Agent, AgentEvent
from slipagent.cli import Renderer, Style
from slipagent.progress import LoopGuard
from slipagent.tools.base import ToolRegistry, ToolResult
from slipagent.tools.shell import RunCommandTool
from slipagent.types import ToolCall
from test_agent import StubClient, RecordingTool, completion, context_records


async def test_shell_streams_before_exit_and_preserves_final_output(workspace):
    seen = asyncio.Event()
    chunks = []
    def publish(text):
        chunks.append(text)
        seen.set()
    quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
    command = quote([sys.executable, "-u", "-c", "import time; print('FIRST'); time.sleep(0.8); print('LAST')"])
    token = command_output.set(publish)
    task = asyncio.create_task(RunCommandTool(workspace).run(command))
    try:
        await asyncio.wait_for(seen.wait(), 3)
        assert not task.done()
        result = await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        command_output.reset(token)
    assert not result.is_error
    assert "FIRST" in "".join(chunks) and "LAST" in "".join(chunks)
    assert "FIRST" in result.content and "LAST" in result.content


async def test_output_throttle_bounds_pending_text_and_flushes_on_close():
    chunks = []
    progress = OutputProgress(chunks.append)
    for _ in range(100):
        progress.feed("x" * 1000)
    assert len(progress.pending) <= 32768
    assert chunks == []
    progress.close()
    assert len(chunks) == 1 and "Live output abbreviated" in chunks[0]
    assert progress.timer is None
    progress.feed("ignored after close")
    assert len(chunks) == 1


def test_renderer_streams_command_output_once_and_escapes_terminal_controls():
    output = io.StringIO()
    renderer = Renderer(Style(False), output, verbose=True)
    call = ToolCall("one", "run_command", {"command": "test"})
    renderer.handle(AgentEvent("tool_start", tool_call=call))
    renderer.handle(AgentEvent("tool_output", text="FIRST\n\x1b]52;danger\x07", tool_call=call))
    renderer.handle(AgentEvent("tool_output", text="LAST\n", tool_call=call))
    renderer.handle(AgentEvent("tool_end", result=ToolResult.ok("FIRST\nLAST\n")))
    value = output.getvalue()
    assert value.count("FIRST") == 1 and value.count("LAST") == 1
    assert "\x1b" not in value and "\x07" not in value
    assert "Command completed" in value


def test_loop_guard_ignores_ids_and_argument_key_order_but_tracks_results():
    guard, registry = LoopGuard(), ToolRegistry()
    for number in range(1, 5):
        args = {"a": 1, "b": 2} if number % 2 else {"b": 2, "a": 1}
        message, stop = guard.observe([(ToolCall(str(number), "inspect", args), ToolResult.ok("same"))], registry, number)
        assert stop == (number == 4)
        assert bool(message) == (number >= 3)
    assert guard.observe([(ToolCall("new", "inspect", args), ToolResult.ok("changed"))], registry, 5) == ("", False)


def test_explicit_shell_polling_does_not_trigger_loop_guard():
    guard, registry = LoopGuard(), ToolRegistry()
    for number in range(1, 8):
        batch = [(ToolCall(str(number), "run_command", {"command": "status", "poll": True}), ToolResult.ok("waiting"))]
        assert guard.observe(batch, registry, number) == ("", False)


async def test_repeated_batches_get_recovery_context_before_stopping():
    tool = RecordingTool("unchanged")
    client = StubClient([completion("", [ToolCall(str(i), "record", {"value": "same"})]) for i in range(8)])
    agent = Agent(client, ToolRegistry([tool]), "test")
    answer = await agent.run("inspect")
    assert "unchanged results again" in answer
    assert len(client.calls) == 4 and len(agent.history.steps) == 4
    assert "Change the approach" in client.calls[3]["messages"][0].content
    current = context_records(client.calls[3]["messages"])[-1]
    assert any("Change the approach" in prompt for prompt in current["user_prompt"])
    assert current["is_tool_result_response"] is True
    assert [message.content for message in agent.messages if message.role == "user"] == ["inspect"]
    assert agent.stopped
    await agent.wait_for_compaction()


async def test_changed_action_after_recovery_continues_normally():
    responses = [completion("", [ToolCall(str(i), "record", {"value": "same"})]) for i in range(3)]
    responses += [completion("", [ToolCall("different", "record", {"value": "different"})]), completion("Done")]
    client = StubClient(responses)
    agent = Agent(client, ToolRegistry([RecordingTool("unchanged")]), "test")
    assert await agent.run("inspect") == "Done"
    assert "progress" not in agent.registry.context_notes
    await agent.wait_for_compaction()
