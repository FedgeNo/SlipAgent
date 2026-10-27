"""Models may label tool calls in plain language; tools never receive the label."""

import io
import json
from importlib import import_module

import httpx

from slipagent.agent import Agent, AgentEvent
from slipagent.calllabels import LABEL, call_label, labeled_spec, raw_call, tool_arguments
from slipagent.cli import Renderer, Style
from slipagent.openrouter import OpenRouterClient
from slipagent.progress import LoopGuard
from slipagent.tools.base import ToolRegistry, ToolResult
from slipagent.types import ToolCall, ToolSpec
from test_agent import RecordingTool

helpers = import_module("test-native-tools")
Router, wire = helpers.Router, helpers.wire


def labeled_call(value="A", label="Recording the value", call_id="read-1"):
    return {"id": call_id, "type": "function",
            "function": {"name": "record", "arguments": json.dumps({"value": value, LABEL: label})}}


async def test_label_is_advertised_shown_and_never_reaches_the_tool():
    router = Router({"test/native": ["tools"]}, [wire("Recording.", [labeled_call()]), wire()])
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    events = []
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(router.handle)) as client:
        agent = Agent(client, registry, "test/native", on_event=events.append)
        assert await agent.run("Record A") == "Done."
        await agent.wait_for_compaction()
    assert tool.seen == [{"value": "A"}]
    definition = next(item for item in router.requests[0]["tools"] if item["function"]["name"] == "record")
    assert LABEL in definition["function"]["parameters"]["properties"]
    assert definition["function"]["parameters"]["required"] == ["value"]
    started = next(event.tool_call for event in events if event.kind == "tool_start")
    assert call_label(started) == "Recording the value"
    await registry.aclose()


def test_specs_gain_an_optional_label_unless_the_tool_defines_one():
    spec = ToolSpec("record", "", {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]})
    labeled = labeled_spec(spec)
    assert set(labeled.parameters["properties"]) == {"value", LABEL}
    assert labeled.parameters["required"] == ["value"] and LABEL not in spec.parameters["properties"]
    own = ToolSpec("own", "", {"type": "object", "properties": {LABEL: {"type": "integer"}}})
    assert labeled_spec(own) is own
    answer = ToolSpec("answer", "", {"type": "object", "properties": {"text": {"type": "string"}}})
    assert labeled_spec(answer) is answer


def test_feed_shows_the_label_in_tool_color_with_the_exact_call_in_thought_grey():
    sink = io.StringIO()
    renderer = Renderer(Style(True), sink, verbose=False)
    renderer.handle(AgentEvent(kind="tool_start", step=1, tool_call=ToolCall(
        "one", "read_file", {"path": "index.dat", LABEL: "Reading index.dat"})))
    renderer.handle(AgentEvent(kind="tool_start", step=1, tool_call=ToolCall("two", "list_dir", {"path": "."})))
    assert "list_dir" not in sink.getvalue()  # Its heading waits for the call before it to finish.
    renderer.handle(AgentEvent(kind="tool_end", step=1, tool_call=ToolCall("one", "read_file"), result=ToolResult.ok("done")))
    text = sink.getvalue()
    tool, grey = "\x1b[38;2;196;167;239m", "\x1b[38;2;160;160;160m"
    assert f"{tool}  ⚙ Reading index.dat\x1b[0m" in text
    assert f'{grey}    read_file({{"path": "index.dat"}})\x1b[0m' in text
    assert f'{tool}  ⚙ list_dir({{"path": "."}})\x1b[0m' in text
    assert LABEL not in text


def test_rewording_a_label_does_not_hide_a_repeated_call():
    registry = ToolRegistry([RecordingTool()])
    guard = LoopGuard()
    for step, label in enumerate(["Recording A", "Recording A again", "One more time"], start=1):
        call = ToolCall(f"c{step}", "record", {"value": "A", LABEL: label})
        guard.observe([(call, ToolResult.ok("same"))], registry, step)
    assert guard.count == 3


def test_label_helpers():
    call = ToolCall("one", "record", {"value": "A", LABEL: "  Recording\n the value "})
    assert call_label(call) == "Recording the value"
    assert tool_arguments(call.arguments) == {"value": "A"}
    assert raw_call(call) == 'record({"value": "A"})'
    assert call_label(ToolCall("two", "record", {LABEL: 5})) == ""
