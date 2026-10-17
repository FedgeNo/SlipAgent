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
@pytest.mark.parametrize('style', ['fence', 'quote', 'indent'])
def test_examples_do_not_become_actions(carrier, style):
    if style == 'fence':
        text = 'Example:\n```\n' + carrier + '\n```\nThat is the format.'
    elif style == 'quote':
        text = '> ' + carrier.replace('\n', '\n> ')
    else:
        text = '    ' + carrier.replace('\n', '\n    ')
    result = parse_agent_response(text, [], tools=[SPEC], xml_tool_calls=True)
    assert result.text == text and result.calls == []


@pytest.mark.parametrize('carrier', [json.dumps(CALL), json.dumps([CALL]), ENVELOPE])
@pytest.mark.parametrize('language', ['', 'json'])
def test_call_block_ending_the_reply_executes(carrier, language):
    text = 'Reading the file.\n\n```' + language + '\n' + carrier + '\n```\n'
    result = parse_agent_response(text, [], tools=[SPEC])
    assert [(c.name, c.arguments) for c in result.calls] == [('record', {'value': 'A'})]
    assert result.text.startswith('Reading the file.')


def test_call_block_naming_no_registered_tool_stays_text():
    text = 'Example:\n```json\n' + json.dumps({'name': 'other', 'arguments': {}}) + '\n```'
    result = parse_agent_response(text, [], tools=[SPEC])
    assert result.text == text and result.calls == []


@pytest.mark.parametrize('text', [
    'Reading now.<tool_call>response_text>:"x","tool_calls":[{"id":"a","name":"record"}]',
    'Calling it: "tool_calls": [{"name": "record"',
    'Done.\n```json\n{"tool_calls":[{"name":"record","arguments":\n```',
])
def test_broken_calls_in_prose_are_rejected(text):
    with pytest.raises(ResponseFormatError):
        parse_agent_response(text, [], tools=[SPEC])


@pytest.mark.parametrize('text', [
    'Use `<tool_call>` tags or `"tool_calls": [{` fields.',
    '> <tool_call>' + json.dumps(CALL),
    '    "tool_calls": [{"name": "record"',
    '```\n"tool_calls": [{"name": "record"\n```\nThat is broken.',
])
def test_quoted_call_fragments_stay_text(text):
    result = parse_agent_response(text, [], tools=[SPEC])
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


GLM_SPEC = ToolSpec('read', '', {'properties': {'path': {'type': 'string'}, 'limit': {'type': 'integer'},
                                                'flags': {'type': 'array'}}})


@pytest.mark.parametrize('text', [
    'Reading.\n<tool_call>read<arg_key>path</arg_key><arg_value>a b.md</arg_value>'
    '<arg_key>limit</arg_key><arg_value>5</arg_value><arg_key>flags</arg_key><arg_value>["x"]</arg_value></tool_call>',
    'Reading.<tool_call>read<arg_key>path</arg_key><arg_value>a b.md</arg_value>\n'
    '<arg_key>limit</arg_key><arg_value>5</arg_value><arg_key>flags</arg_key><arg_value>["x"]</arg_value>\n</tool_call>',
])
def test_glm_calls_decode_strings_literally_and_other_values_as_json(text):
    result = parse_agent_response(text, [], tools=[GLM_SPEC])
    assert result.text == 'Reading.'
    assert [(c.name, c.arguments) for c in result.calls] == [('read', {'path': 'a b.md', 'limit': 5, 'flags': ['x']})]


def test_glm_call_after_prose_with_inline_code_executes():
    # GLM's own reply: inline code earlier in the sentence must not hide the call.
    text = ('I am loading the conventions `list_rules` tool first, along with `git_status`.'
            '<tool_call>load_tools<arg_key>names</arg_key><arg_value>["a", "b"]</arg_value></tool_call>')
    spec = ToolSpec('load_tools', '', {'properties': {'names': {'type': 'array'}}})
    result = parse_agent_response(text, [], tools=[spec])
    assert [(c.name, c.arguments) for c in result.calls] == [('load_tools', {'names': ['a', 'b']})]
    assert result.text.endswith('`git_status`.')


def test_glm_calls_batch_and_allow_no_arguments():
    text = '<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value></tool_call>\n<tool_call>record</tool_call>'
    result = parse_agent_response(text, [], tools=[GLM_SPEC, SPEC])
    assert [(c.name, c.arguments) for c in result.calls] == [('read', {'path': 'a'}), ('record', {})]


@pytest.mark.parametrize('text', [
    '<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value>',
    '<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value><arg_key>path</arg_key><arg_value>b</arg_value></tool_call>',
])
def test_incomplete_or_duplicated_glm_calls_are_rejected(text):
    with pytest.raises(ResponseFormatError):
        parse_agent_response(text, [], tools=[GLM_SPEC])


def test_glm_history_shows_earlier_calls_in_its_own_format():
    from slipagent.context import tool_history_as_text
    from slipagent.types import Message, content_text
    message = Message(role='assistant', content='Reading.',
                      tool_calls=[ToolCall('one', 'read', {'path': 'a.md', 'limit': 5})])
    text = content_text(tool_history_as_text([message], glm_xml=True)[0].content)
    assert text == ('Reading.\n<tool_call>read<arg_key>path</arg_key><arg_value>a.md</arg_value>'
                    '<arg_key>limit</arg_key><arg_value>5</arg_value></tool_call>')
