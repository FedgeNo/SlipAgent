"""Selected request contracts reject alternate carriers before any effects."""

import json

import httpx
import pytest

from slipagent.agent import Agent, build_system_prompt
from slipagent.capabilities import ModelCapabilities
from slipagent.openrouter import OpenRouterClient
from slipagent.protocol import ResponseFormatError, parse_agent_response
from slipagent.tools.base import ToolRegistry
from test_agent import RecordingTool, summary_response


CALL = {"id": "one", "name": "record", "arguments": '{"value":"A"}'}
API_CALL = {"id": "one", "type": "function", "function": {"name": "record", "arguments": CALL["arguments"]}}


@pytest.mark.parametrize("mode", ["native", "schema", "json"])
async def test_silent_tool_batch_executes_and_receives_results(mode):
    native = mode != "json"
    parameters = ["tools"] if native else ["response_format"]
    if mode == "schema":
        parameters.append("structured_outputs")
    profile = ModelCapabilities({}, [{"tag": "stub", "supported_parameters": parameters, "context_length": 1000000}])
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    requests = []

    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [API_CALL]} if native else {
                "role": "assistant", "content": json.dumps({"response": "", "tool_calls": [CALL]})}
        else:
            assert tool.seen == [{"value": "A"}]
            assert "last response was rejected" not in body["messages"][0]["content"]
            content = "Done" if mode == "native" else json.dumps({"response": "Done", **({"tool_calls": []} if not native else {})})
            message = {"role": "assistant", "content": content}
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})

    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities("test", profile)
        agent = Agent(client, registry, "test", system_prompt=build_system_prompt("."))
        assert await agent.run("Inspect") == "Done"
        await agent.wait_for_compaction()
    assert len(requests) == 2
    assert tool.seen == [{"value": "A"}]
    await registry.aclose()


@pytest.mark.parametrize("mode", ["native", "schema", "json"])
@pytest.mark.parametrize("invalid", ["other_carrier", "extra_field", "tagged", "fenced", "empty_reply"])
async def test_selected_contract_rejects_then_recovers_without_effects(mode, invalid):
    native = mode != "json"
    parameters = ["tools"] if native else ["response_format"]
    if mode == "schema":
        parameters.append("structured_outputs")
    profile = ModelCapabilities({}, [{"tag": "stub", "supported_parameters": parameters, "context_length": 1000000}])
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    requests, events = [], []

    def valid(reply, calls):
        content = reply if mode == "native" else json.dumps({"response": reply, **({"tool_calls": calls} if not native else {})})
        return {"role": "assistant", "content": content, **({"tool_calls": [API_CALL] if calls else []} if native else {})}

    rejected = valid("Rejected text", [CALL])
    if invalid == "other_carrier":
        if native:
            rejected["content"] = json.dumps({"response": "Rejected text", "tool_calls": [CALL]})
        else:
            rejected["tool_calls"] = [API_CALL]
    elif invalid == "extra_field":
        rejected["content"] = json.dumps({"response": "Rejected text", "tool_calls": [CALL], "task": {}})
    elif invalid == "tagged":
        rejected["content"] = "<tool_call>" + json.dumps(CALL) + "</tool_call>"
    elif invalid == "fenced":
        rejected["content"] = '```json\n' + json.dumps({"response": "Rejected text", "tool_calls": [CALL]}) + '\n```'
    else:
        rejected = valid("", [])

    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        if len(requests) <= 2:
            assert tool.seen == []
            assert not any(e.kind == "assistant_text" and e.text == "Rejected text" for e in events)
        message = [rejected, valid("Reading", [CALL]), valid("Done", [])][len(requests)-1]
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})

    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities("test", profile)
        agent = Agent(client, registry, "test", system_prompt=build_system_prompt("."), on_event=events.append)
        assert await agent.run("Inspect") == "Done"
        await agent.wait_for_compaction()
    assert tool.seen == [{"value": "A"}]
    assert len(requests) == 3
    correction = json.loads(requests[1]["messages"][-1]["content"])["user_prompt"]
    assert any("rejected" in prompt.lower() for prompt in correction)
    for body in requests:
        system = body["messages"][0]["content"]
        if native:
            assert "Replies and Tool Calls:" not in system
            assert "============================= BEGIN AVAILABLE TOOL DEFINITIONS ==============================" not in system
            assert "tagged calls" not in system and "when using JSON" not in system
            assert body["tools"]
        else:
            assert "native" not in system.lower()
            assert "tools" not in body
        assert "Alternatively" not in system
    assert [e.text for e in events if e.kind == "assistant_text"] == ["Reading", "Done"]
    await registry.aclose()


@pytest.mark.parametrize("change", [
    {"arguments": {}}, {"arguments": "[]"}, {"arguments": ""}, {"id": ""},
    {"input": "{}"}, {"type": "function"}, {"name": ""},
])
def test_json_calls_require_exact_fields_and_encoded_objects(change):
    with pytest.raises(ResponseFormatError):
        parse_agent_response(json.dumps({"response": "Reading", "tool_calls": [{**CALL, **change}]}), [], native_tools=False)


@pytest.mark.parametrize("text", [
    "Done", '{"response":"Done"}', '{"response":"Done","tool_calls":[],"extra":1}',
    '{"response":"Done","response":"Again","tool_calls":[]}',
    '{"response":"Done","tool_calls":[' + json.dumps(CALL) + ',' + json.dumps(CALL) + ']}',
])
def test_json_final_and_batch_structure_is_strict(text):
    with pytest.raises(ResponseFormatError):
        parse_agent_response(text, [], native_tools=False)


def test_plain_reply_examples_do_not_execute_calls():
    text = 'Example:\n```json\n' + json.dumps({"tool_calls": [CALL]}) + '\n```'
    result = parse_agent_response(text, [])
    assert result.text == text and result.calls == []


@pytest.mark.parametrize("text", [
    '{"extra":1,"tool_calls":[]}', '{\n"extra":1,\n"response":"Done"\n}',
    '```\n{"tool_calls":[]}\n```', '<function_call>{}</function_call>',
])
def test_native_plain_replies_reject_response_containers(text):
    with pytest.raises(ResponseFormatError):
        parse_agent_response(text, [])


@pytest.mark.parametrize("change", [
    {"extra": "field"}, {"type": "tool_use"},
    {"function": {"name": "record", "arguments": CALL["arguments"], "input": {"value": "A"}}},
])
async def test_streamed_call_fields_are_validated_without_running_tools(change):
    requests = []
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    fragment = {"index": 0, **API_CALL, **change}

    class Frames(httpx.AsyncByteStream):
        async def __aiter__(self):
            for delta, finish in [({"content": "Rejected"}, None), ({"tool_calls": [fragment]}, "tool_calls")]:
                yield ("data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Frames())
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}]})

    profile = ModelCapabilities({}, [{"tag": "stub", "supported_parameters": ["tools"], "context_length": 1000000}])
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities("test", profile)
        agent = Agent(client, registry, "test")
        assert await agent.run("Inspect") == "Done"
        await agent.wait_for_compaction()
    assert tool.seen == [] and len(requests) == 2
    assert "last response was rejected" in requests[1]["messages"][0]["content"]
    await registry.aclose()
