"""Upstream compatibility cases retain batch boundaries and literal payloads."""

import json
from importlib import import_module

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.api import _StreamCompletion
from slipagent.openrouter import OpenRouterClient
from slipagent.protocol import ResponseFormatError, parse_agent_response
from slipagent.streaming import ResponseStream
from slipagent.tools.base import Tool, ToolRegistry, ToolResult
from slipagent.types import ToolCall, ToolSpec
from test_agent import RecordingTool, StubClient, completion

helpers = import_module('test-native-tools')
Router, wire = helpers.Router, helpers.wire
SPEC = ToolSpec('record', '', {'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value']})
KIMI_BEGIN = '<|tool_calls_section_begin|>'
KIMI_CALL = '<|tool_call_begin|>functions.record:0<|tool_call_argument_begin|>{"value":"A"}<|tool_call_end|>'
KIMI_END = '<|tool_calls_section_end|>'
FUNCTION = '<function=record><parameter=value>A</parameter></function>'
CARRIERS = [KIMI_BEGIN + KIMI_CALL + KIMI_END, FUNCTION,
            '</think>\n' + FUNCTION, '<tool_call>' + FUNCTION.replace('<parameter=value>', '<parameter=value=') + '</tool_call>']


@pytest.mark.parametrize('text', CARRIERS)
@pytest.mark.parametrize('native', [False, True])
def test_additional_complete_carriers_preserve_native_mirrors(text, native):
    result = parse_agent_response(text, [ToolCall('native', 'record', {'value': 'A'})] if native else [], tools=[SPEC])
    assert [(c.name, c.arguments) for c in result.calls] == [('record', {'value': 'A'})]
    if native:
        assert result.calls[0].id == 'native'
    assert result.text == ''


@pytest.mark.parametrize('carrier', CARRIERS)
@pytest.mark.parametrize('style', ['quote', 'indent', 'fence', 'thought'])
def test_new_carriers_remain_inert_in_examples_and_reasoning(carrier, style):
    text = {'quote': '> ' + carrier.replace('\n', '\n> '), 'indent': '    ' + carrier.replace('\n', '\n    '),
            'fence': 'Example:\n```\n' + carrier + '\n```',
            'thought': '<think>' + carrier.removeprefix('</think>\n') + '</think>'}[style]
    result = parse_agent_response(text, [], tools=[SPEC])
    assert result.calls == []


@pytest.mark.parametrize('size', [1, 5, 1000])
@pytest.mark.parametrize('carrier', CARRIERS)
def test_new_carriers_do_not_leak_arguments_in_stream_previews(carrier, size):
    stream = ResponseStream()
    shown = ''.join(stream.push(carrier[i:i + size]) for i in range(0, len(carrier), size))
    assert shown == ''


@pytest.mark.parametrize('text', [KIMI_BEGIN + KIMI_CALL, KIMI_BEGIN + KIMI_CALL.replace('{"value":"A"}', '{"value":"A"') + KIMI_END,
                                FUNCTION.replace('</parameter>', ''), FUNCTION + ' extra',
                                KIMI_BEGIN + KIMI_CALL + KIMI_END + ' extra'])
def test_incomplete_or_ambiguous_carriers_reject_the_entire_batch(text):
    with pytest.raises(ResponseFormatError):
        parse_agent_response(text, [ToolCall('native', 'record', {'value': 'N'})], tools=[SPEC])


@pytest.mark.parametrize('arguments,expected', [
    ('{"value":"line one\nline two\tend"}', 'line one\nline two\tend'),
    (r'{"value":"C:\project\q.txt"}', r'C:\project\q.txt'),
    ('{"value":"A",}', 'A'),
    (json.dumps(json.dumps({'value': 'A'})), 'A'),
])
@pytest.mark.parametrize('carrier', ['native', 'bare', 'tagged', 'kimi'])
def test_complete_argument_encoding_variants(arguments, expected, carrier):
    native = []
    if carrier == 'native':
        text = ''
        native = [ToolCall('native', 'record', arguments)]
    elif carrier == 'bare':
        text = '{"name":"record","arguments":' + arguments + '}'
    elif carrier == 'tagged':
        text = '<tool_call>{"name":"record","arguments":' + arguments + '}</tool_call>'
    else:
        text = KIMI_BEGIN + '<|tool_call_begin|>functions.record:0<|tool_call_argument_begin|>' + arguments + '<|tool_call_end|>' + KIMI_END
    result = parse_agent_response(text, native, tools=[SPEC])
    assert [c.arguments for c in result.calls] == [{'value': expected}]


@pytest.mark.parametrize('arguments', ['{"value":"cut', '{"value":"A"', '{"value":}', '{"value":"A","value":"B"}', '{"value":NaN}', '{,}', '{"value":,}'])
def test_compatibility_does_not_complete_truncated_or_ambiguous_arguments(arguments):
    with pytest.raises(ResponseFormatError):
        parse_agent_response('', [ToolCall('native', 'record', arguments)], tools=[SPEC])


@pytest.mark.parametrize('name', ['functions.record', 'tools.record', 'RECORD', ' record '])
def test_only_unique_registered_tool_names_are_normalized(name):
    result = parse_agent_response('', [ToolCall('native', name, {'value': 'A'})], tools=[SPEC])
    assert result.calls[0].name == 'record'
    assert result.calls[0].id == 'native'


def test_name_matching_does_not_resolve_ambiguous_mcp_suffixes():
    tools = [ToolSpec('first__record', '', {}), ToolSpec('second__record', '', {})]
    result = parse_agent_response('', [ToolCall('native', 'functions.record', {})], tools=tools)
    assert result.calls[0].name == 'functions.record'


def test_namespaces_do_not_override_an_exact_registered_name():
    tools = [SPEC, ToolSpec('functions.record', '', {})]
    result = parse_agent_response('', [ToolCall('native', 'functions.record', {})], tools=tools)
    assert result.calls[0].name == 'functions.record'


def test_native_numeric_call_ids_are_ignored():
    from slipagent.api import _parse_completion
    result = _parse_completion({'choices': [{'message': {'tool_calls': [
        {'id': 7, 'function': {'name': 'record', 'arguments': '{"value":"A"}'}}
    ]}}]})
    assert result.tool_calls[0].id.startswith('call_')


def test_control_markers_inside_kimi_arguments_remain_literal_and_repetitions_survive():
    value = KIMI_BEGIN + KIMI_END + '\n' + KIMI_CALL
    body = '<|tool_call_begin|>functions.record:0<|tool_call_argument_begin|>' + json.dumps({'value': value}) + '<|tool_call_end|>'
    result = parse_agent_response(KIMI_BEGIN + body + body + KIMI_END, [], tools=[SPEC])
    assert [c.arguments for c in result.calls] == [{'value': value}] * 2


@pytest.mark.parametrize('prefix', ['', 'Data:\n'])
def test_tolerated_json_data_does_not_expose_nested_function_tags(prefix):
    text = prefix + '{"code":"line one\n' + FUNCTION + '\nline three"}'
    result = parse_agent_response(text, [], tools=[SPEC])
    assert result.text == text and result.calls == []


@pytest.mark.parametrize('schema,value,expected', [
    ({'type': ['integer', 'null']}, '3', 3),
    ({'anyOf': [{'type': 'integer'}, {'type': 'boolean'}]}, 'false', False),
    ({'anyOf': [{'type': 'integer'}, {'type': 'string'}]}, '3', '3'),
    ({'$ref': '#/$defs/count', '$defs': {'count': {'type': 'integer'}}}, '3', 3),
])
def test_schema_coercion_handles_unambiguous_unions_and_local_references(schema, value, expected):
    result = parse_agent_response('', [ToolCall('native', 'typed', {'value': value})],
                                  tools=[ToolSpec('typed', '', {'type': 'object', 'properties': {'value': schema}, '$defs': schema.get('$defs', {})})])
    assert result.calls[0].arguments == {'value': expected}
    assert type(result.calls[0].arguments['value']) is type(expected)


@pytest.mark.parametrize('text', ['Done.', '</think>', '<think>Still considering.</think>'])
def test_plain_and_thought_only_replies_still_end_without_calls(text):
    result = parse_agent_response(text, [], tools=[SPEC])
    assert result.calls == [] and result.text == text


def test_schema_coercion_preserves_strings_unknown_fields_and_ambiguous_unions():
    schema = {'type': 'object', 'properties': {'count': {'type': 'integer'}, 'flag': {'type': 'boolean'},
              'text': {'type': 'string'}, 'options': {'type': 'object', 'properties': {'limit': {'type': 'integer'}}},
              'counts': {'type': 'array', 'items': {'type': 'integer'}}, 'union': {'type': ['string', 'integer']}}}
    args = {'count': '12', 'flag': 'false', 'text': '0012', 'options': '{"limit":"3"}',
            'counts': '["1","2"]', 'union': '7', 'extra': 'keep'}
    original = json.dumps(args)
    result = parse_agent_response('', [ToolCall('native', 'typed', args)], tools=[ToolSpec('typed', '', schema)])
    assert result.calls[0].arguments == {'count': 12, 'flag': False, 'text': '0012', 'options': {'limit': 3},
                                          'counts': [1, 2], 'union': '7', 'extra': 'keep'}
    assert json.dumps(args) == original


def test_nested_stringified_objects_with_duplicate_fields_are_not_repaired():
    schema = {'type': 'object', 'properties': {'options': {'type': 'object'}}}
    arguments = {'options': '{"limit":1,"limit":2}'}
    result = parse_agent_response('', [ToolCall('native', 'typed', arguments)], tools=[ToolSpec('typed', '', schema)])
    assert result.calls[0].arguments == arguments


@pytest.mark.parametrize('value', ['a,b,}', 'a,b,]', '"quoted"', '\\', '\\u1234', '\n\t\r\x00', '<function=record>'])
def test_valid_string_arguments_retain_exact_content(value):
    result = parse_agent_response(json.dumps({'name': 'record', 'arguments': {'value': value}}), [], tools=[SPEC])
    assert result.calls[0].arguments == {'value': value}


def test_xml_namespaced_tool_names_preserve_string_parameter_types():
    text = '<tool_call><invoke name="functions.record"><parameter name="value">true</parameter></invoke></tool_call>'
    result = parse_agent_response(text, [], tools=[SPEC])
    assert result.calls[0].name == 'record'
    assert result.calls[0].arguments == {'value': 'true'}


@pytest.mark.parametrize('mode', ['delta', 'snapshot'])
def test_stream_identity_policy_preserves_fragmented_and_repeated_names(mode):
    stream = _StreamCompletion(lambda *_: None, identity_mode=mode)
    for name, args in ([('rec', '{"value":'), ('ord', '"A"}')] if mode == 'delta' else [('record', '{"value":'), ('record', '"A"}')]):
        stream.add(json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'one', 'function': {'name': name, 'arguments': args}}]}}]}))
    stream.add(json.dumps({'choices': [{'finish_reason': 'tool_calls'}]}))
    result = stream.completion()
    assert [(c.name, c.arguments) for c in result.tool_calls] == [('record', {'value': 'A'})]
    assert result.tool_calls[0].id.startswith('call_')


def test_stream_indexes_keep_parallel_calls_with_the_same_incoming_id():
    stream = _StreamCompletion(lambda *_: None)
    for index, args in [(0, '{"value":'), (1, '{"value":'), (0, '"A"}'), (1, '"B"}')]:
        stream.add(json.dumps({'choices': [{'delta': {'tool_calls': [{'index': index, 'id': 'same', 'function': {'name': 'record' if args.startswith('{') else None, 'arguments': args}}]}}]}))
    stream.add(json.dumps({'choices': [{'finish_reason': 'tool_calls'}]}))
    calls = stream.completion().tool_calls
    assert [c.arguments for c in calls] == [{'value': 'A'}, {'value': 'B'}]
    assert len({c.id for c in calls}) == 2 and all(c.id.startswith('call_') for c in calls)


@pytest.mark.parametrize('conflict', [{'function': {'name': 'other'}}])
def test_snapshot_stream_conflicts_are_errors_instead_of_retargeting_calls(conflict):
    from slipagent.api import APIResponseError
    stream = _StreamCompletion(lambda *_: None, identity_mode='snapshot')
    stream.add(json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'one', 'function': {'name': 'record', 'arguments': '{}'}}]}}]}))
    with pytest.raises(APIResponseError, match='Conflicting streamed'):
        stream.add(json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, **conflict}]}}]}))


def test_snapshot_stream_empty_identity_continuations_do_not_erase_the_call():
    stream = _StreamCompletion(lambda *_: None, identity_mode='snapshot')
    for identifier, name, args in [('one', 'record', '{"value":'), ('', None, '"A"'), (None, '', '}')]:
        stream.add(json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': identifier, 'function': {'name': name, 'arguments': args}}]}}]}))
    stream.add(json.dumps({'choices': [{'finish_reason': 'tool_calls'}]}))
    result = stream.completion()
    assert [(c.name, c.arguments) for c in result.tool_calls] == [('record', {'value': 'A'})]
    assert result.tool_calls[0].id.startswith('call_')


def test_finished_stream_choices_cannot_be_reopened_by_late_tool_deltas():
    stream = _StreamCompletion(lambda *_: None)
    stream.add(json.dumps({'choices': [{'delta': {'content': 'Done.'}, 'finish_reason': 'stop'}]}))
    stream.add(json.dumps({'choices': [{'delta': {'content': 'More work', 'tool_calls': [{'id': 'late', 'name': 'record', 'arguments': {}}]}, 'finish_reason': 'tool_calls'}]}))
    stream.add(json.dumps({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}))
    result = stream.completion()
    assert result.text == 'Done.' and result.tool_calls == []
    assert result.finish_reason == 'stop' and result.usage.total_tokens == 12


async def test_nvidia_minimax_uses_the_provider_specific_snapshot_policy():
    from slipagent.nvidia import NvidiaClient
    from slipagent.types import Message
    frames = [{'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'one', 'function': {'name': 'record', 'arguments': args}}]}}]}
              for args in ['{"value":', '"A"}']]
    frames.append({'choices': [{'finish_reason': 'tool_calls'}]})
    response = ''.join('data: ' + json.dumps(f) + '\n\n' for f in frames) + 'data: [DONE]\n\n'
    async with NvidiaClient('dummy', transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=response))) as client:
        result = await client.chat(model='minimaxai/minimax-m3', messages=[Message.user('work')], on_delta=lambda *_: None)
    assert [(c.name, c.arguments) for c in result.tool_calls] == [('record', {'value': 'A'})]
    assert result.tool_calls[0].id.startswith('call_')


async def test_stream_profile_applies_to_transport_and_can_change_without_restart(tmp_path, monkeypatch):
    import shutil
    import slipagent.prompts as prompts
    directory = tmp_path / 'prompts'
    shutil.copytree(prompts.prompt_directory(), directory)
    monkeypatch.setattr(prompts, 'prompt_directory', lambda: directory)
    mapping = directory / 'model-profiles.json'
    frames = [{'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'one', 'function': {'name': 'record', 'arguments': args}}]}}]}
              for args in ['{"value":', '"A"}']]
    frames.append({'choices': [{'finish_reason': 'tool_calls'}]})
    response = ''.join('data: ' + json.dumps(f) + '\n\n' for f in frames) + 'data: [DONE]\n\n'
    async with OpenRouterClient('dummy', transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=response))) as client:
        from slipagent.types import Message
        for mode, name in [('snapshot', 'record'), ('delta', 'recordrecord')]:
            mapping.write_text(json.dumps({'test/model': {'stream_tool_identity': mode}}))
            result = await client.chat(model='test/model', messages=[Message.user('work')], on_delta=lambda *_: None)
            assert result.tool_calls[0].name == name


@pytest.mark.parametrize('field,value', [('stream_tool_identity', 'guess'), ('stream_tool_indices', 'guess'), ('stream_tool_indices', 'id'),
                                        ('stream_tool_identity', True), ('stream_tool_indices', None)])
def test_invalid_stream_profiles_are_reported(tmp_path, monkeypatch, field, value):
    import shutil
    import slipagent.prompts as prompts
    directory = tmp_path / 'prompts'
    shutil.copytree(prompts.prompt_directory(), directory)
    monkeypatch.setattr(prompts, 'prompt_directory', lambda: directory)
    (directory / 'model-profiles.json').write_text(json.dumps({'test/model': {field: value}}))
    with pytest.raises(prompts.PromptError, match=field):
        prompts.model_prompt_profile('test/model')


class TypedTool(Tool):
    name = 'typed'
    description = 'Record a number.'
    parameters = {'type': 'object', 'properties': {'count': {'type': 'integer'}}, 'required': ['count']}

    def __init__(self):
        self.seen = []

    async def run(self, count):
        self.seen.append(count)
        return ToolResult.ok('recorded')


@pytest.mark.parametrize('bad', [{'count': 'oops'}, {'count': 2, 'extra': True}, {}])
async def test_schema_invalid_call_prevents_every_tool_in_the_batch(bad):
    first, second = RecordingTool(), TypedTool()
    registry = ToolRegistry([first, second])
    client = StubClient([completion('Recording.', [ToolCall('one', 'record', {'value': 'A'}), ToolCall('two', 'typed', bad)]), completion('Done.')])
    try:
        agent = Agent(client, registry, 'test')
        assert await agent.run('Record values') == 'Done.'
        await agent.wait_for_compaction()
        assert first.seen == second.seen == []
        errors = [m for m in agent.messages if m.role == 'tool']
        assert len(errors) == 2
        assert all(m.content['status'] == 'error' and 'No tools were executed' in m.content['content'] for m in errors)
    finally:
        await registry.aclose()


@pytest.mark.parametrize('carrier', CARRIERS)
@pytest.mark.parametrize('streamed', [False, True])
async def test_new_carriers_execute_through_the_real_client_and_end_normally(carrier, streamed):
    router = Router({'test/model': ['tools']}, [wire('Done.')])
    tool = RecordingTool()
    registry = ToolRegistry([tool])

    def handle(request):
        if request.method == 'POST' and not router.requests:
            router.requests.append(json.loads(request.content))
            if not streamed:
                return httpx.Response(200, json={'choices': [{'message': {'content': carrier}, 'finish_reason': 'stop'}]})
            frames = [{'choices': [{'delta': {'content': carrier[i:i + 3]}}]} for i in range(0, len(carrier), 3)]
            frames.append({'choices': [{'finish_reason': 'stop'}]})
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=''.join('data: ' + json.dumps(f) + '\n\n' for f in frames) + 'data: [DONE]\n\n')
        return router.handle(request)

    try:
        async with OpenRouterClient('dummy', transport=httpx.MockTransport(handle)) as client:
            agent = Agent(client, registry, 'test/model')
            assert await agent.run('Record A') == 'Done.'
            await agent.wait_for_compaction()
        assert tool.seen == [{'value': 'A'}]
        assert len(router.requests) == 2
    finally:
        await registry.aclose()
