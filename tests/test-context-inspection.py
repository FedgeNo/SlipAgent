"""Context inspection captures the outgoing request rather than stored history."""

import io
import json

import httpx
import pytest

from test_agent import summary_response

from slipagent.agent import Agent
from slipagent.cli import Renderer, Style
from slipagent.openrouter import OpenRouterClient
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolSpec
from test_agent import RecordingTool, context_body, task_record


def response(text="Done.", calls=None, previous=""):
    return {"choices": [{"message": {"role": "assistant", "content": text,
        "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}} for call in calls or []]}, "finish_reason": "stop"}]}



@pytest.mark.parametrize("streaming", [False, True])
async def test_snapshot_matches_complete_wire_body_before_request(streaming):
    snapshots = []
    captured = []
    def transport(request):
        captured.append(json.loads(request.content))
        assert len(snapshots) == 1
        assert json.loads(snapshots[-1]) == captured[-1]
        return httpx.Response(200, json=response())
    async with OpenRouterClient("private-test-key", transport=httpx.MockTransport(transport)) as client:
        await client.chat(
            model="test/model", messages=[Message.system("System guidance."), Message.user("Inspect.")],
            tools=[ToolSpec("example", "A tool.", {"type": "object"})],
            temperature=.1, max_tokens=500, session_id="session",
            extra_body={"response_format": {"type": "json_object"}, "provider": {"require_parameters": True}},
            on_delta=(lambda *args: None) if streaming else None,
            on_request=snapshots.append,
        )
    assert captured[0]["messages"][0]["content"] == "System guidance."
    assert "tools" in captured[0] and "response_format" in captured[0]
    assert "private-test-key" not in snapshots[0]
    assert "Authorization" not in snapshots[0]
    if streaming:
        assert captured[0]["stream"] is True and "reasoning" not in captured[0]


async def test_agent_snapshots_include_retry_prompt_and_matching_tool_results():
    requests, events = [], []
    tool = RecordingTool("EXACT TOOL RESULT")
    def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test", "context_length": 1000000}]})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": '{"response":'}, "finish_reason": "stop"}]})
        if len(requests) == 2:
            return httpx.Response(200, json=response("Reading.", [{"id": "call-1", "name": "record", "arguments": '{"value":"A"}'}]))
        return httpx.Response(200, json=response(previous="record returned EXACT TOOL RESULT."))
    async with OpenRouterClient("test", transport=httpx.MockTransport(transport)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test", system_prompt="PINNED SYSTEM", on_event=events.append)
        assert await agent.run("Inspect.") == "Done."
    snapshots = [json.loads(event.text) for event in events if event.kind == "context"]
    assert snapshots == requests
    assert "last response was rejected" in snapshots[1]["messages"][0]["content"]
    assert "last response was rejected" not in snapshots[2]["messages"][0]["content"]
    record = json.loads(snapshots[2]["messages"][-2]["content"])
    assert record["tool_results"] == [{"call_id": "call-1", "tool_name": "record", "status": "success", "content": "EXACT TOOL RESULT"}]
    assert json.loads(snapshots[2]["messages"][-1]["content"])["record_type"] == "current_step"
    renderer = Renderer(Style(False), io.StringIO(), False)
    for event in events:
        if event.kind == "context":
            renderer.handle(event)
    assert renderer.stream.getvalue() == ""


async def test_snapshot_uses_selected_compacted_context_and_stays_immutable():
    requests, events = [], []
    original = "OLD ORIGINAL RESPONSE " * 100
    def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test", "context_length": 1000000}]})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        result = response(original if len(requests) == 1 else "Done.")
        return httpx.Response(200, json=result)
    async with OpenRouterClient("test", transport=httpx.MockTransport(transport)) as client:
        agent = Agent(client, ToolRegistry(), "test", context_steps=1, on_event=events.append)
        for prompt in ["First task.", *[f"Next task {index}." for index in range(6)]]:
            await agent.run(prompt)
        snapshots = [json.loads(event.text) for event in events if event.kind == "context"]
        assert snapshots == requests
        assert original not in json.dumps(snapshots[-1])
        assert agent.history.steps[0].summary in snapshots[-1]["messages"][1]["content"]
        agent.reset()
        assert snapshots[-1] == requests[-1]
