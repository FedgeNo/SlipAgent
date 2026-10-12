"""Wrapped invoke XML from Dots uses the normal all-or-nothing call boundary."""

import json
from importlib import import_module

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.openrouter import OpenRouterClient
from slipagent.protocol import ResponseFormatError, parse_agent_response
from slipagent.streaming import ResponseStream
from slipagent.tools.base import ToolRegistry
from slipagent.types import ToolCall, ToolSpec
from test_agent import RecordingTool


native_helpers = import_module("test-native-tools")
Router, wire, native_call = native_helpers.Router, native_helpers.wire, native_helpers.native_call

SPEC = ToolSpec("record", "Record", {"properties": {"value": {"type": "string"}}})
INVOKE = '<invoke name="record"><parameter name="value">A</parameter></invoke>'
WRAPPED = '<tool_calls><call>' + INVOKE + '</call></tool_calls>'
MODELS = ["dots-studio/dots-3-note-preview", "dots-studio/dots-3-note-preview:free"]

# The failed response's structure, with disposable example arguments.
MALFORMED = '''<tool_calls>
<call>
<invoke name="record">
<parameter name="value">A</parameter>
</invoke>
</invoke>
<call>
<parameter name="value">B</parameter>
</invoke>
</invoke>
<call>
<parameter name="value">C</parameter>
</invoke>
</invoke>
<call>
<parameter name="value">D</parameter>
</parameter>
</invoke>
</tool_calls>'''


@pytest.mark.parametrize("text", [
    WRAPPED, "\ufeff" + WRAPPED, "Reading.\n" + WRAPPED, "```xml\n" + WRAPPED + "\n```",
    json.dumps({"response": "Reading.", "tool_calls": []}) + "\n" + WRAPPED,
    json.dumps({"response": "Reading.\n" + WRAPPED, "tool_calls": []}),
])
def test_wrapped_xml_normalizes_and_deduplicates_native_calls(text):
    for native in ([], [ToolCall("native", "record", {"value": "A"})]):
        result = parse_agent_response(text, native, tools=[SPEC], xml_tool_calls=True)
        assert [(c.name, c.arguments) for c in result.calls] == [("record", {"value": "A"})]
        if native:
            assert result.calls[0].id == "native"


def test_wrapped_xml_keeps_order_repetitions_ids_and_typed_values():
    spec = ToolSpec("record", "", {"properties": {
        "value": {"type": "string"}, "offset": {"type": "integer"},
    }})
    invoke = '<invoke name="record" id="first"><parameter name="value">  a &lt; b\n</parameter><parameter name="offset">\n121</parameter></invoke>'
    text = '<tool_calls><call>' + invoke + '</call><call>' + INVOKE + '</call><call>' + INVOKE + '</call></tool_calls>'
    result = parse_agent_response(text, [], tools=[spec], xml_tool_calls=True)
    assert result.calls[0].id == "first"
    assert [c.arguments for c in result.calls] == [{"value": "  a < b\n", "offset": 121}, {"value": "A"}, {"value": "A"}]


@pytest.mark.parametrize("text", [
    MALFORMED,
    WRAPPED.replace('</call>', ''),
    WRAPPED.replace('</invoke>', '</invoke></invoke>'),
    WRAPPED.replace(' name="record"', ''),
    WRAPPED.replace(INVOKE, '<parameter name="value">A</parameter>'),
    WRAPPED.replace(INVOKE, INVOKE + INVOKE),
    WRAPPED.replace('</invoke>', '<parameter name="value">B</parameter></invoke>'),
    WRAPPED.replace('</tool_calls>', ''),
    WRAPPED + '\nUnexpected prose.',
])
def test_malformed_wrapped_xml_rejects_the_entire_batch_including_native(text):
    with pytest.raises(ResponseFormatError):
        parse_agent_response('Reading.\n' + text, [ToolCall("native", "record", {"value": "N"})],
                             tools=[SPEC], xml_tool_calls=True)


@pytest.mark.parametrize("text", [
    'Example:\n```xml\n' + WRAPPED + '\n```',
    '> ' + WRAPPED, '    ' + WRAPPED, 'Use `<tool_calls>` tags.',
])
def test_wrapped_xml_examples_remain_literal(text):
    result = parse_agent_response(text, [], tools=[SPEC], xml_tool_calls=True)
    assert result.text == text and result.calls == []


def test_wrapped_xml_requires_the_xml_profile():
    result = parse_agent_response(WRAPPED, [], tools=[SPEC])
    assert result.text == WRAPPED and result.calls == []


@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_wrapped_xml_arguments_stay_out_of_streamed_reply(envelope, chunk_size):
    text = 'Reading.\n' + WRAPPED
    if envelope:
        text = json.dumps({"response": text})
    stream = ResponseStream(xml_tool_calls=True)
    visible = ''.join(stream.push(text[i:i + chunk_size]) for i in range(0, len(text), chunk_size))
    assert visible == 'Reading.'


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streamed", [False, True])
async def test_dots_retries_malformed_xml_before_executing_any_calls(model, streamed):
    router = Router({model: ["tools"]}, [wire()])
    tool, events = RecordingTool(), []
    registry = ToolRegistry([tool])
    responses = [
        {"content": MALFORMED, "tool_calls": [native_call("N")]},
        {"content": 'Reading.\n' + WRAPPED},
    ]

    def handle(request):
        if request.method == 'POST' and len(router.requests) < len(responses):
            assert tool.seen == []
            router.requests.append(json.loads(request.content))
            message = responses[len(router.requests) - 1]
            if not streamed:
                return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
            content = message['content']
            frames = [{"choices": [{"delta": {"content": content[i:i + 7]}}]} for i in range(0, len(content), 7)]
            frames.append({"choices": [{"delta": {"tool_calls": [{"index": 0, **c} for c in message.get('tool_calls', [])]}, "finish_reason": "stop"}]})
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames) + 'data: [DONE]\n\n')
        return router.handle(request)

    try:
        async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
            agent = Agent(client, registry, model, on_event=events.append)
            assert await agent.run("Record A") == "Done."
            await agent.wait_for_compaction()
        assert tool.seen == [{"value": "A"}]
        assert len(router.requests) == 3
        correction = router.requests[1]["messages"][0]["content"]
        assert 'Invalid tool-call XML' in correction
        assert 'No tools from it were executed' in correction
        assert any(e.kind == 'retry' for e in events)
        assert all('<tool_calls>' not in e.text for e in events if e.kind in {'assistant_text', 'assistant_delta'})
    finally:
        await registry.aclose()
