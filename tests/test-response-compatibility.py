"""Alternate complete call carriers preserve intent, examples, and termination."""

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

helpers = import_module('test-native-tools')
Router, wire = helpers.Router, helpers.wire
SPEC = ToolSpec('record', '', {'properties': {'value': {'type': 'string'}}})
CALL = {'name': 'record', 'arguments': {'value': 'A'}}
ENVELOPE = json.dumps({'response': 'Reading.', 'tool_calls': [CALL]})
INVOKE = '<invoke name="record"><parameter name="value">A</parameter></invoke>'
DS_START = '<｜tool▁calls▁begin｜>'
DS_CALL = '<｜tool▁call▁begin｜>record<｜tool▁sep｜>{"value":"A"}<｜tool▁call▁end｜>'
DS_END = '<｜tool▁calls▁end｜>'
CARRIERS = [
    json.dumps(CALL), json.dumps([CALL]),
    json.dumps({'function': CALL, 'id': 'one', 'type': 'function'}),
    ENVELOPE, 'Reading.\n' + ENVELOPE,
    'Reading.\n' + json.dumps(CALL), 'Reading.\n' + json.dumps([CALL]),
    '<think>Consider a different call: {"name":"record","arguments":{"value":"HIDDEN"}}</think>\n' + ENVELOPE,
    '<think>Plan.</think>\n<think>Check.</think>\n' + json.dumps(CALL),
    '<function_calls >' + INVOKE + '</function_calls >',
    '<function_calls id="batch">' + INVOKE + '</function_calls>',
    '<tool_call>' + INVOKE + '</tool_call>',
    '<tool_call id="one" >' + INVOKE + '</tool_call >',
    '<tool_call >' + json.dumps(CALL) + '</tool_call >',
    '<tool_call id="one">' + json.dumps(CALL) + '</tool_call>',
    '<tool_calls id="batch"><call>' + INVOKE + '</call></tool_calls>',
    '[TOOL_CALLS]' + json.dumps([CALL]),
    '[TOOL_CALLS]record[ARGS]{"value":"A"}',
    DS_START + DS_CALL + DS_END,
    'Reading.\n' + DS_START + DS_CALL + DS_END,
]


@pytest.mark.parametrize('text', CARRIERS)
def test_complete_alternate_calls_and_native_mirrors(text):
    for native in ([], [ToolCall('native', 'record', {'value': 'A'})]):
        result = parse_agent_response(text, native, tools=[SPEC], xml_tool_calls=True)
        assert [(c.name, c.arguments) for c in result.calls] == [('record', {'value': 'A'})]
        if native:
            assert result.calls[0].id == 'native'
        assert '<think>' not in result.text


@pytest.mark.parametrize('text', [json.dumps(CALL), json.dumps([CALL]), '[TOOL_CALLS]' + json.dumps([CALL]), DS_START + DS_CALL + DS_END])
def test_json_and_model_markers_work_without_xml_profile(text):
    result = parse_agent_response(text, [], tools=[SPEC])
    assert [c.arguments for c in result.calls] == [{'value': 'A'}]


@pytest.mark.parametrize('text', ['[TOOL_CALLS]' + json.dumps([CALL]), DS_START + DS_CALL + DS_END])
def test_explicit_markers_inside_json_reply_are_recognized(text):
    result = parse_agent_response(json.dumps({'response': text}), [], tools=[SPEC])
    assert [c.arguments for c in result.calls] == [{'value': 'A'}]


@pytest.mark.parametrize('carrier', CARRIERS)
@pytest.mark.parametrize('style', ['fence', 'quote', 'indent', 'example'])
def test_examples_do_not_become_actions(carrier, style):
    if style == 'fence':
        text = 'Example:\n```\n' + carrier + '\n```'
    elif style == 'quote':
        text = '> ' + carrier.replace('\n', '\n> ')
    elif style == 'indent':
        text = '    ' + carrier.replace('\n', '\n    ')
    else:
        text = 'Example:\n' + carrier
    result = parse_agent_response(text, [], tools=[SPEC], xml_tool_calls=True)
    assert result.text == text and result.calls == []


@pytest.mark.parametrize('text', [
    '<think>\n' + json.dumps(CALL),
    '<think>\n' + DS_START + DS_CALL + DS_END,
    '<think>Only thoughts.</think>',
    'Done.', '[1, 2, 3]', '{"name":"a data object"}',
    'Data:\n{\n"config":\n' + json.dumps([CALL]) + '\n}',
    json.dumps({'config': CALL}, indent=2),
])
def test_replies_data_and_unfinished_thoughts_stay_terminal(text):
    result = parse_agent_response(text, [], tools=[SPEC], xml_tool_calls=True)
    assert result.text == text and result.calls == []


@pytest.mark.parametrize('text', [
    'Reading.\n{"tool_calls":[',
    'Reading.\n' + json.dumps([CALL, 'invalid call']),
    '<tool_call>' + INVOKE + INVOKE + '</tool_call>',
    '<tool_call>' + INVOKE + '</tool_call',
    '<function_calls id="batch">' + INVOKE + '</function_calls',
    '[TOOL_CALLS]record[ARGS]{"value":',
    '[TOOL_CALLS]' + json.dumps([CALL]) + '\nExtra prose.',
    DS_START + DS_CALL,
    DS_START + DS_CALL.replace('<｜tool▁call▁end｜>', '') + DS_END,
    DS_START + DS_CALL.replace('record<｜tool▁sep｜>', '<｜tool▁sep｜>') + DS_END,
])
def test_incomplete_or_ambiguous_calls_reject_entire_native_batch(text):
    with pytest.raises(ResponseFormatError):
        parse_agent_response(text, [ToolCall('native', 'record', {'value': 'N'})], tools=[SPEC], xml_tool_calls=True)


@pytest.mark.parametrize('text', [
    '[TOOL_CALLS]' + json.dumps([CALL, CALL]),
    '[TOOL_CALLS]record[ARGS]{"value":"A"}[TOOL_CALLS]record[ARGS]{"value":"A"}',
    DS_START + DS_CALL + DS_CALL + DS_END,
    DS_START + DS_CALL + DS_END + DS_START + DS_CALL + DS_END,
])
def test_marker_calls_preserve_repeated_actions(text):
    result = parse_agent_response(text, [], tools=[SPEC])
    assert [c.arguments for c in result.calls] == [{'value': 'A'}, {'value': 'A'}]


@pytest.mark.parametrize('format', ['mistral', 'deepseek'])
def test_control_markers_inside_arguments_remain_literal(format):
    value = '  ' + DS_END + '[TOOL_CALLS]</tool_call>\n'
    if format == 'mistral':
        text = '[TOOL_CALLS]' + json.dumps([{'name': 'record', 'arguments': {'value': value}}]) + '</s>'
    else:
        text = DS_START + '<｜tool▁call▁begin｜>record<｜tool▁sep｜>' + json.dumps({'value': value}) + '<｜tool▁call▁end｜>' + DS_END + '<｜end▁of▁sentence｜>'
    result = parse_agent_response(text, [], tools=[SPEC])
    assert [c.arguments for c in result.calls] == [{'value': value}]


@pytest.mark.parametrize('chunk_size', [1, 7, 1000])
@pytest.mark.parametrize('text', CARRIERS)
def test_previews_do_not_expose_call_arguments_or_thoughts(text, chunk_size):
    stream = ResponseStream(xml_tool_calls=True)
    visible = ''.join(stream.push(text[i:i + chunk_size]) for i in range(0, len(text), chunk_size))
    assert all(marker not in visible for marker in ('arguments', 'parameter', '[ARGS]', 'HIDDEN', '<think>', '"value"'))
    result = parse_agent_response(text, [], tools=[SPEC], xml_tool_calls=True)
    assert result.text.startswith(visible)


@pytest.mark.parametrize('content', [
    'Reading.\n' + ENVELOPE,
    '<think>Plan.</think>\n' + ENVELOPE,
    '[TOOL_CALLS]' + json.dumps([CALL]),
    DS_START + DS_CALL + DS_END,
    '<tool_call id="one">' + INVOKE + '</tool_call>',
])
@pytest.mark.parametrize('streamed', [False, True])
async def test_alternate_calls_execute_and_the_final_reply_ends_the_turn(content, streamed):
    model = 'dots-studio/dots-3-note-preview:free'
    router = Router({model: ['tools']}, [wire('Done.')])
    tool = RecordingTool()
    registry = ToolRegistry([tool])

    def handle(request):
        if request.method == 'POST' and not router.requests:
            router.requests.append(json.loads(request.content))
            if not streamed:
                return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]})
            frames = [{'choices': [{'delta': {'content': content[i:i + 7]}}]} for i in range(0, len(content), 7)]
            frames.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  text=''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames) + 'data: [DONE]\n\n')
        return router.handle(request)

    try:
        async with OpenRouterClient('dummy', transport=httpx.MockTransport(handle)) as client:
            agent = Agent(client, registry, model)
            assert await agent.run('Record A') == 'Done.'
            await agent.wait_for_compaction()
        assert tool.seen == [{'value': 'A'}]
        assert len(router.requests) == 2
    finally:
        await registry.aclose()


@pytest.mark.parametrize('reason', ['stop', 'tool_calls'])
async def test_final_reply_without_calls_ends_even_with_inconsistent_finish_metadata(reason):
    router = Router({'test/model': ['tools']}, [])
    tool = RecordingTool()
    registry = ToolRegistry([tool])

    def handle(request):
        if request.method == 'POST' and not router.requests:
            router.requests.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': 'Done.'}, 'finish_reason': reason}]})
        return router.handle(request)

    try:
        async with OpenRouterClient('dummy', transport=httpx.MockTransport(handle)) as client:
            agent = Agent(client, registry, 'test/model')
            assert await agent.run('Report completion') == 'Done.'
            await agent.wait_for_compaction()
        assert tool.seen == []
        assert len(router.requests) == 1
    finally:
        await registry.aclose()
