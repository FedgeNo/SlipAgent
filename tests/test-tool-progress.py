"""Progress reaches the user during execution without requesting another model step."""

from slipagent.types import content_text

import asyncio
import io
import json
from data_text_reader import read_data
import os
import shlex
import subprocess
import sys

import pytest

from slipagent.activity import OutputProgress, command_output
from slipagent.agent import Agent, AgentEvent
from slipagent.cli import Renderer, Style
from slipagent.progress import LoopGuard, StreamLoopGuard, StreamLoopError
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
    assert "FIRST" in content_text(result.content) and "LAST" in content_text(result.content)


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
    events: list[AgentEvent] = []
    agent.on_event = events.append
    answer = await agent.run("inspect")
    assert "unchanged results again" in answer
    assert len(client.calls) == 4 and len(agent.history.steps) == 4
    assert "Change the approach" in content_text(client.calls[3]["messages"][0].content)
    delivered = content_text(client.calls[3]["messages"][0].content)
    assert 'value: text (4 characters):' in delivered and 'same' in delivered
    assert not any(event.kind == "warning" and "Change the approach" in event.text for event in events)
    assert any(event.kind == "warning" and "unchanged results again" in event.text for event in events)
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


@pytest.mark.parametrize("period", [2, 3, 6])
def test_loop_guard_detects_cycles_with_unchanged_results(period):
    guard, registry = LoopGuard(), ToolRegistry()
    for index in range(period * 4):
        batch = [(ToolCall(str(index), "inspect", {"file": index % period}), ToolResult.ok("same"))]
        message, stop = guard.observe(batch, registry, index + 1)
        assert bool(message) == (index + 1 >= period * 3)
        assert stop == (index + 1 >= period * 4)
        if index + 1 == period * 3:
            calls = read_data(message.split('```text\n', 1)[1].split('```', 1)[0])
            assert [item['step'] for item in calls] == list(range(period * 2 + 1, period * 3 + 1))
            assert [item['tool_calls'][0]['function']['arguments'] for item in calls] == [
                {'file': value} for value in range(period)]


def test_loop_guard_allows_cycles_whose_results_change():
    guard, registry = LoopGuard(), ToolRegistry()
    for index in range(30):
        batch = [(ToolCall(str(index), "inspect", {"file": index % 2}), ToolResult.ok(str(index // 2)))]
        assert guard.observe(batch, registry, index + 1) == ("", False)


PROSE = "I need to inspect the same information once more to decide whether this request is complete. "


@pytest.mark.parametrize("kind", ["reasoning", "content"])
@pytest.mark.parametrize("chunk_size", [1, 37, 10000])
def test_stream_guard_detects_repetition_independently_of_chunks(kind, chunk_size):
    guard = StreamLoopGuard()
    text = PROSE * 12
    with pytest.raises(StreamLoopError, match="sustained repetition"):
        for index in range(0, len(text), chunk_size):
            guard.feed(kind, text[index:index + chunk_size])
        guard.finish()


def test_stream_guard_preserves_code_tables_lists_and_short_repetition():
    samples = ["```python\n" + PROSE * 12 + "\n```", "~~~\n" + PROSE * 12,
               ("| " + PROSE + " |\n") * 12, ("- " + PROSE + "\n") * 12, PROSE * 3]
    for text in samples:
        guard = StreamLoopGuard()
        for chunk in text:
            guard.feed("content", chunk)
        guard.finish()


async def test_repetitive_completed_response_does_not_execute_its_tools():
    tool = RecordingTool()
    response = completion(PROSE * 12, [ToolCall("one", "record", {"value": "unsafe"})])
    agent = Agent(StubClient([response]), ToolRegistry([tool]), "test")
    assert "sustained repetition" in await agent.run("inspect")
    assert agent.stopped and tool.seen == []
    assert len(agent.messages) == 1
    await agent.registry.aclose()


async def test_repetitive_decoded_response_does_not_execute_its_tools():
    import json
    import httpx
    from slipagent.capabilities import ModelCapabilities
    from slipagent.openrouter import OpenRouterClient
    tool = RecordingTool()
    content = json.dumps({"response": PROSE * 12, "tool_calls": [{"id": "one", "name": "record", "arguments": '{"value":"unsafe"}'}]})
    def handle(request):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}],
                                        "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}})
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities("test", ModelCapabilities({}, [{"tag": "stub", "context_length": 1000000, "supported_parameters": ["response_format"]}]))
        agent = Agent(client, ToolRegistry([tool]), "test")
        assert "sustained repetition" in await agent.run("inspect")
        assert tool.seen == [] and agent.stopped
        assert agent.usage.total_tokens == 18
        await agent.registry.aclose()


def test_stream_guard_bounds_buffers_and_excludes_long_code_fences():
    guard = StreamLoopGuard()
    guard.feed("content", "```\n")
    for index in range(1000):
        guard.feed("content", str(index) + PROSE + "\n")
    guard.feed("content", "```\nDone.")
    guard.finish()
    assert all(len(value) <= 32768 for value in guard.buffers.values())
    assert len(guard.pending) <= 32768
