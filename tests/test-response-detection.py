"""Response carriers are detected per message, independently of request mode."""

import json
from importlib import import_module

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.api import _parse_completion, _StreamCompletion
from slipagent.openrouter import OpenRouterClient
from slipagent.protocol import ResponseFormatError, parse_agent_response
from slipagent.tools.base import ToolRegistry
from slipagent.types import ToolCall, ToolSpec
from test_agent import RecordingTool, unpack_api_context

native_helpers = import_module("test-native-tools")
Router, wire, native_call = native_helpers.Router, native_helpers.wire, native_helpers.native_call


XML = '<tool_call>\n<function=record>\n<parameter=value>\nA\n</parameter>\n</function>\n</tool_call>'
FLAT = {"id": "one", "name": "record", "arguments": {"value": "A"}}
SPEC = ToolSpec("record", "Record", {"properties": {"value": {"type": "string"}}})


@pytest.mark.parametrize("native_tools,json_response", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("content", [
    XML, "Reading.\n" + XML,
    json.dumps({"response": "Reading.", "tool_calls": [FLAT]}),
    '```json\n' + json.dumps({"response": "Reading.", "tool_calls": [FLAT]}) + '\n```',
    '<tool_call>' + json.dumps(FLAT) + '</tool_call>',
    '<function_call>' + json.dumps(FLAT) + '</function_call>',
    '<tool_call>```json\n' + json.dumps(FLAT) + '\n```</tool_call>',
    '  ' + XML,
    json.dumps({"response": "Reading.", "function_call": {"name": "record", "arguments": {"value": "A"}}}),
    json.dumps({"response": "Reading."}) + '\n' + XML,
    '```json\n' + json.dumps({"response": "Reading."}) + '\n```\n' + XML,
])
def test_all_text_formats_work_with_or_without_native_calls(content, native_tools, json_response):
    for native in ([], [ToolCall("native-id", "record", {"value": "A"})]):
        result = parse_agent_response(content, native, native_tools=native_tools,
                                      json_response=json_response, tools=[SPEC])
        assert len(result.calls) == 1
        assert result.calls[0].arguments == {"value": "A"}
        if native:
            assert result.calls[0].id == "native-id"


def test_mixed_distinct_calls_and_intentional_repetitions():
    native = [ToolCall("n1", "record", {"value": "A"}), ToolCall("n2", "record", {"value": "A"})]
    result = parse_agent_response(XML + '\n' + XML.replace('>\nA\n<', '>\nB\n<'), native, tools=[SPEC])
    assert [call.arguments for call in result.calls] == [{"value": "A"}, {"value": "A"}, {"value": "B"}]
    assert [call.id for call in result.calls[:2]] == ["n1", "n2"]
    assert len(parse_agent_response(XML + '\n' + XML, [], tools=[SPEC]).calls) == 2


def test_conflicting_ids_reject_entire_batch():
    with pytest.raises(ResponseFormatError, match="Conflicting"):
        parse_agent_response(json.dumps({"tool_calls": [FLAT]}), [ToolCall("one", "record", {"value": "B"})])


def test_qwen_parameters_use_tool_schema_and_preserve_code():
    values = {"code": '  if a < b:\n    print("&amp; </tool_call>")\n', "number": "42", "flag": "false", "items": '[1,"x"]'}
    types = {"code": "string", "number": "integer", "flag": "boolean", "items": "array"}
    spec = ToolSpec("edit", "", {"properties": {k: {"type": v} for k, v in types.items()}})
    text = '<tool_call><function=edit>' + ''.join(f'<parameter={k}>\n{v}\n</parameter>' for k, v in values.items()) + '</function></tool_call>'
    result = parse_agent_response(text, [], tools=[spec])
    assert result.calls[0].arguments == {"code": values["code"], "number": 42, "flag": False, "items": [1, "x"]}


@pytest.mark.parametrize("text", [XML[:-1], XML + '\nOops',
    XML.replace('</function>', '<parameter=value>B</parameter></function>'),
    XML + '\n<tool_call>{"name":"record","arguments":'] )
def test_incomplete_or_ambiguous_text_batches_reject_native_calls_too(text):
    with pytest.raises(ResponseFormatError):
        parse_agent_response(text, [ToolCall("native", "record", {"value": "A"})], tools=[SPEC])


@pytest.mark.parametrize("text", ['Example:\n```xml\n' + XML + '\n```', 'Use `<tool_call>` tags.',
                                 '> ' + XML.replace('\n', '\n> '), '    ' + XML.replace('\n', '\n    ')])
def test_quoted_examples_do_not_become_calls(text):
    result = parse_agent_response(text, [], tools=[SPEC])
    assert result.text == text and result.calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parameters", [["tools"], ["structured_outputs", "tools"], ["response_format"]])
async def test_mixed_native_and_xml_round_trip(stream, parameters):
    router = Router({"test/model": parameters}, [wire()])
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    first = {"content": 'Reading.\n' + XML + '\n' + XML.replace('>\nA\n<', '>\nB\n<'), "tool_calls": [native_call("A")]}
    def handle(request):
        if request.method == 'POST' and not router.requests:
            router.requests.append(json.loads(request.content))
            if stream:
                frames = [{"choices": [{"delta": {"content": first["content"][:25]}}]},
                          {"choices": [{"delta": {"content": first["content"][25:], "tool_calls": [{"index": 0, **native_call("A")}]}, "finish_reason": "stop"}]}]
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      text=''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames) + 'data: [DONE]\n\n')
            return httpx.Response(200, json={"choices": [{"message": first, "finish_reason": "stop"}]})
        return router.handle(request)
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, registry, "test/model")
        assert await agent.run("Inspect") == "Done."
        await agent.wait_for_compaction()
    assert tool.seen == [{"value": "A"}, {"value": "B"}]
    assert len(router.requests) == 2
    assert agent.history.steps[0].agent_response == "Reading."
    messages = unpack_api_context(router.requests[1]["messages"])
    results = [message for message in messages if message['role'] == 'tool']
    assert len(results) == 2
    assert results[0]['tool_call_id'] == 'read-1'
    await registry.aclose()


def test_native_legacy_blocks_and_embedded_calls_can_coexist():
    message = {"tool_calls": [native_call("A")], "function_call": {"name": "record", "arguments": '{"value":"B"}'},
               "content": [{"type": "text", "text": XML}, {"type": "tool_use", "id": "block", "name": "record", "input": {"value": "C"}}]}
    completion = _parse_completion({"choices": [{"message": message, "finish_reason": "stop"}]})
    assert completion.response_error is None
    result = parse_agent_response(completion.text, completion.tool_calls, tools=[SPEC])
    assert [call.arguments for call in result.calls] == [{"value": "A"}, {"value": "B"}, {"value": "C"}]


@pytest.mark.parametrize("mask", range(8))
@pytest.mark.parametrize("content", ["Reading.", XML, json.dumps({"response": "Reading.", "tool_calls": [FLAT]})])
def test_every_combination_of_api_carriers_and_answer_calls(mask, content):
    message = {"content": content}
    expected = []
    if mask & 1:
        message["tool_calls"] = [native_call("N")]
        expected.append({"value": "N"})
    if mask & 2:
        message["function_call"] = {"name": "record", "arguments": '{"value":"L"}'}
        expected.append({"value": "L"})
    if mask & 4:
        message["content"] = [{"type": "text", "text": content}, {"type": "tool_use", "id": "block", "name": "record", "input": {"value": "B"}}]
        expected.append({"value": "B"})
    if content != "Reading.":
        expected.append({"value": "A"})
    completion = _parse_completion({"choices": [{"message": message, "finish_reason": "stop"}]})
    assert completion.response_error is None
    result = parse_agent_response(completion.text, completion.tool_calls, tools=[SPEC])
    assert [call.arguments for call in result.calls] == expected


def test_streamed_legacy_and_content_blocks_mix_with_text_calls():
    stream = _StreamCompletion(lambda *args: None)
    deltas = [
        {"function_call": {"name": "rec", "arguments": '{"value":'}},
        {"function_call": {"name": "ord", "arguments": '"L"}'}, "content": XML[:35]},
        {"content": [{"type": "text", "text": XML[35:]}, {"type": "tool_use", "id": "block", "name": "record", "input": {"value": "B"}}]},
    ]
    for delta in deltas:
        stream.add(json.dumps({"choices": [{"delta": delta}]}))
    stream.add(json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    completion = stream.completion()
    assert completion.response_error is None
    result = parse_agent_response(completion.text, completion.tool_calls, tools=[SPEC])
    assert [call.arguments for call in result.calls] == [{"value": "L"}, {"value": "B"}, {"value": "A"}]


def test_legacy_summary_record_is_detected_without_model_metadata():
    value = native_helpers.record("Reading.")
    value["tool_calls"] = [FLAT]
    result = parse_agent_response(json.dumps(value), [], tools=[SPEC])
    assert result.text == "Reading." and result.calls[0].arguments == {"value": "A"}
