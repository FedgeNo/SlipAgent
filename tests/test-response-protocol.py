"""One structured response carries actions and three separately stored records."""

import asyncio
import json

import httpx
import pytest

from test_agent import summary_response

from slipagent.capabilities import ModelCapabilities
from slipagent.agent import Agent
from slipagent.tools.base import ToolRegistry
from slipagent.openrouter import OpenRouterClient
from slipagent.protocol import ResponseFormatError, parse_response
from test_agent import RecordingTool, task_record


def record(response="Done.", calls=None, previous="", user="User asked to inspect the project.", agent="Assistant reports completion."):
    return {"task": task_record(), "response": response, "tool_calls": calls or [],
            "previous_tool_responses_compressed": previous,
            "user_prompt_compressed": user, "agent_response_compressed": agent}


def completion(value):
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps({"response": value["response"], "tool_calls": value["tool_calls"]})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


async def test_json_response_executes_tools_and_archives_original_parts():
    requests, events = [], []
    tool = RecordingTool("ACTUAL RESULT")
    first = record("Reading.", [{"id": "c", "name": "record", "arguments": '{"value":"A"}'}],
                   agent="Assistant plans record(value=A); its result is pending.")
    second = record(previous="record returned ACTUAL RESULT.")
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test", "context_length": 1000000}]})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(first if len(requests) == 1 else second))
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities("test", ModelCapabilities({}, [{"tag": "stub", "supported_parameters": ["response_format"], "context_length": 1000000}]))
        agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
        assert await agent.run("inspect") == "Done."
    assert tool.seen == [{"value": "A"}]
    assert len(requests) == 2
    assert "tool_choice" not in requests[0]
    assert requests[0]["provider"]["require_parameters"] is True
    assert requests[0]["response_format"] == {"type": "json_object"}
    for step, expected in zip(agent.history.steps, [first, second]):
        assert step.agent_response == expected["response"]
        assert step.summary
        assert "_slipagent_context" not in step.full_text()
        assert "agent_response_compressed" not in step.full_text()
    assert [event.text for event in events if event.kind == "assistant_text"] == ["Reading.", "Done."]
    assert not any(event.kind == "assistant_delta" for event in events)


@pytest.mark.parametrize("field", ["previous_tool_responses_compressed", "user_prompt_compressed", "agent_response_compressed"])
async def test_missing_legacy_compressed_field_does_not_reject_valid_tools(field):
    requests, events = [], []
    tool = RecordingTool()
    bad = record("REJECTED", [{"id": "c", "name": "record", "arguments": '{"value":"BAD"}'}])
    del bad[field]
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(bad if len(requests) == 1 else record()))
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities("test", ModelCapabilities({}, [{"tag": "stub", "supported_parameters": ["response_format"], "context_length": 1000000}]))
        agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
        assert await agent.run("work") == "Done."
    assert len(requests) == 2 and tool.seen == [{"value": "BAD"}]
    assert [event.text for event in events if event.kind == "assistant_text"] == ["REJECTED", "Done."]
    assert not any(event.kind == "retry" for event in events)
    assert len(agent.history.steps) == 2


async def test_only_separate_reasoning_streams_while_json_and_tools_wait():
    started, release = asyncio.Event(), asyncio.Event()
    requests, events = [], []
    tool = RecordingTool()
    text = json.dumps({"response": "Reading.", "tool_calls": [{"id": "c", "name": "record", "arguments": '{"value":"A"}'}]})
    def frame(delta, finish=None):
        return ("data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()
    class Paused(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield frame({"reasoning": "Inspecting the project."})
            yield frame({"content": text[:50]})
            started.set()
            await release.wait()
            yield frame({"content": text[50:]}, "stop")
            yield b"data: [DONE]\n\n"
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Paused())
        return httpx.Response(200, json=completion(record(previous="record returned ok.")))
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities("test", ModelCapabilities({}, [{"tag": "stub", "supported_parameters": ["response_format"], "context_length": 1000000}]))
        agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
        task = asyncio.create_task(agent.run("work"))
        await asyncio.wait_for(started.wait(), 3)
        try:
            assert any(event.kind == "reasoning_delta" for event in events)
            assert not any(event.kind in {"assistant_text", "assistant_delta", "tool_start"} for event in events)
            assert tool.seen == [] and not task.done()
        finally:
            release.set()
        assert await task == "Done."
    assert tool.seen == [{"value": "A"}]


@pytest.mark.parametrize("change", [
    {"response": 7}, {"tool_calls": {}}, {"user_prompt_compressed": " "},
    {"agent_response_compressed": ""}, {"previous_tool_responses_compressed": "invented results"},
    {"extra": "field"},
    {"tool_calls": [{"id": "c", "name": "record", "arguments": "[]"}]},
    {"tool_calls": [{"id": "c", "name": "record", "arguments": "{broken}"}]},
    {"tool_calls": [{"id": "c", "name": "record", "arguments": "{}"}] * 2},
    {"response": ""}, {"agent_response_compressed": "x" * 6001},
    {"agent_response_compressed": "a" * 3001, "user_prompt_compressed": "b" * 3001},
])
def test_invalid_field_types_values_and_batches_are_rejected(change):
    with pytest.raises(ResponseFormatError):
        parse_response(json.dumps({**record(), **change}), False)


@pytest.mark.parametrize("text", [
    "prose " + json.dumps(record()),
    json.dumps(record()) + json.dumps(record()),
    '<slipagent_context>{"step":1,"summary":"old format"}</slipagent_context>',
    json.dumps(record()).replace('"response": "Done."', '"response": "Done.", "response": "duplicate"'),
    json.dumps(record()).replace('"response": "Done."', '"response": NaN'),
])
def test_non_object_formats_duplicate_keys_and_nonfinite_json_are_rejected(text):
    with pytest.raises(ResponseFormatError):
        parse_response(text, False)


def test_previous_tool_compression_is_required_when_results_exist():
    with pytest.raises(ResponseFormatError, match="previous_tool_responses_compressed"):
        parse_response(json.dumps(record()), True)
    parsed = parse_response(json.dumps(record(previous="Tool failed: permission denied.")), True)
    assert parsed.previous_tool_responses_compressed == "Tool failed: permission denied."


@pytest.mark.parametrize("arguments", ['{"value":"\\ud800"}', '{"value":1e999}'])
def test_invalid_unicode_and_overflow_cannot_enter_tool_history(arguments):
    reply = record(calls=[{"id": "one", "name": "record", "arguments": arguments}])
    with pytest.raises(ResponseFormatError) as error:
        parse_response(json.dumps(reply), False)
    assert "tool_calls[0].arguments" in str(error.value)
    error.value.excerpt.encode("utf-8")


def test_retry_diagnostic_has_json_position_and_bounded_rejected_excerpt():
    reply = record(calls=[{"id": "one", "name": "record", "arguments": '{"value": ' + '"' + 'x' * 4000 + '" BAD}'}])
    with pytest.raises(ResponseFormatError) as error:
        parse_response(json.dumps(reply), False)
    assert "tool_calls[0].arguments" in str(error.value)
    assert "line 1, column" in str(error.value)
    assert "BAD" in error.value.excerpt and len(error.value.excerpt) < 1000
