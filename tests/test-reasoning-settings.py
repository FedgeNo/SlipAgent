"""Published reasoning limits override gateway acceptance and reach actual requests."""

import json

import httpx
import pytest

from slipagent.api import APITransportError, RetryPolicy
from slipagent.anthropic import AnthropicClient
from slipagent.capabilities import ModelCapabilities
from slipagent.providers import create_client
from slipagent.reasoning import gateway_reasoning
from slipagent.types import Message


@pytest.mark.parametrize('provider,model,expected', [
    ('nvidia', 'minimaxai/minimax-m3.1-flash', {'reasoning_effort': 'max'}),
    ('nvidia', 'meta/muse-glimmer-30b', {'reasoning_effort': 'max'}),
    ('nvidia', 'deepseek-ai/deepseek-v4.1-flash', {'reasoning_effort': 100}),
    ('nvidia', 'openai/gpt-oss-20b', {'reasoning_effort': 'high'}),
    ('nvidia', 'nvidia/nemotron-3-ultra-550b-a55b', {'reasoning_effort': 'high', 'reasoning_budget': -1}),
    ('nvidia', 'nvidia/nemotron-3.5-lightning-30b-a3b', {'reasoning_budget': -1, 'chat_template_kwargs': {'enable_thinking': True}}),
    ('groq', 'openai/gpt-oss-120b', {'reasoning_effort': 'high'}),
    ('groq', 'qwen/qwen3.8-27b', {'reasoning_effort': 'high'}),
    ('mistral', 'mistral-large-4-0', {'reasoning_effort': 'high'}),
    ('mistral', 'zai-glm-5-3-flash', {'reasoning_effort': 'max'}),
    ('google', 'gemini-3.8-flash', {'extra_body': {'google': {'thinking_config': {'include_thoughts': True, 'thinking_level': 'high'}}}}),
    ('google', 'gemini-2.5-pro', {'extra_body': {'google': {'thinking_config': {'include_thoughts': True, 'thinking_budget': 32768}}}}),
    ('google', 'gemini-2.5-flash', {'extra_body': {'google': {'thinking_config': {'include_thoughts': True, 'thinking_budget': 24576}}}}),
])
async def test_highest_settings_are_sent(provider, model, expected):
    bodies = []
    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '42'}, 'finish_reason': 'stop'}]})
    async with create_client(provider, api_key='dummy', transport=httpx.MockTransport(respond)) as client:
        await client.chat(model=model, messages=[Message.user('Compute')])
    for key, value in expected.items():
        assert bodies[0][key] == value
    if provider == 'google':
        assert 'reasoning_effort' not in bodies[0]


@pytest.mark.parametrize('provider,model', [('groq', 'plain-model'), ('mistral', 'codestral-2508'), ('nvidia', 'unknown')])
async def test_unknown_and_unsupported_models_do_not_receive_guessed_effort(provider, model):
    bodies = []
    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '42'}}]})
    async with create_client(provider, api_key='dummy', transport=httpx.MockTransport(respond)) as client:
        await client.chat(model=model, messages=[Message.user('Compute')])
    assert 'reasoning_effort' not in bodies[0]


async def test_claude_native_overload_retries_before_output_and_preserves_raw_events():
    calls = []
    captured = []
    def respond(request):
        calls.append(request)
        events = ([{'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Busy'}}] if len(calls) == 1 else [
            {'type': 'message_start', 'message': {'model': 'claude-test', 'usage': {'input_tokens': 2}}},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': '42'}},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
            {'type': 'message_stop'}])
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
    async with AnthropicClient('dummy', retry=RetryPolicy(max_retries=1, initial_backoff=0), transport=httpx.MockTransport(respond)) as client:
        result = await client.chat(model='claude-test', messages=[Message.user('Compute')],
                                   on_delta=lambda *d: None, on_response=captured.append)
    assert len(calls) == 2 and result.message.content == '42'
    assert captured[0]['events'][0]['error']['type'] == 'overloaded_error'
    assert captured[1]['events'][-1]['type'] == 'message_stop'


@pytest.mark.parametrize('model,old', [('gemini-3.8-flash', {'thinking_budget': 1000}),
                                     ('gemini-2.5-pro', {'thinking_level': 'low'})])
async def test_google_highest_config_removes_conflicting_lower_controls(model, old):
    bodies = []
    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '42'}}]})
    async with create_client('google', api_key='dummy', transport=httpx.MockTransport(respond)) as client:
        await client.chat(model=model, messages=[Message.user('Compute')],
                          extra_body={'reasoning_effort': 'low', 'extra_body': {'google': {'thinking_config': old}}})
    body = bodies[0]
    assert 'reasoning_effort' not in body
    config = body['extra_body']['google']['thinking_config']
    assert not set(old).intersection(config)
    assert config['include_thoughts'] is True


@pytest.mark.parametrize('choices,expected', [(['high'], 'high'), (['xhigh', 'high'], 'xhigh'), (None, 'max'), ([], None)])
def test_gateway_published_choices_override_tested_max(choices, expected):
    model = {'context_length': 8192}
    if choices is not None:
        model['reasoning'] = {'supported_efforts': choices}
    profile = ModelCapabilities(model, [{'tag': 'route', 'supported_parameters': ['tools', 'reasoning']}])
    gateway_reasoning('openrouter', 'cohere/north-mini-code:free', profile)
    assert (profile.reasoning or {}).get('effort') == expected


def claude_catalog(*, adaptive=True, efforts=('low', 'high', 'max')):
    return {'data': [{'id': 'claude-test', 'max_input_tokens': 200000, 'max_tokens': 64000,
                     'capabilities': {'effort': {e: {'supported': True} for e in efforts},
                                      'thinking': {'types': {'adaptive': {'supported': adaptive},
                                                             'enabled': {'supported': not adaptive}}}}}]}


@pytest.mark.parametrize('adaptive,efforts', [(True, ('high', 'max')), (False, ('low', 'high')), (False, ())])
async def test_claude_keeps_default_reasoning_regardless_of_published_choices(adaptive, efforts):
    bodies = []
    def respond(request):
        if request.method == 'GET':
            return httpx.Response(200, json=claude_catalog(adaptive=adaptive, efforts=efforts))
        assert request.url.path.endswith('/messages')
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'model': 'claude-test', 'content': [{'type': 'text', 'text': '42'}],
                                        'stop_reason': 'end_turn', 'usage': {'input_tokens': 10, 'output_tokens': 2}})
    async with AnthropicClient('dummy', transport=httpx.MockTransport(respond)) as client:
        await client.model_capabilities('claude-test')
        result = await client.chat(model='claude-test', messages=[Message.system('Rules'), Message.user('Compute')], max_tokens=2048)
        assert result.message.content == '42' and result.usage.prompt_tokens == 10
        await client.chat(model='claude-test', messages=[Message.user('Compute')], max_tokens=1024)
    body = bodies[0]
    assert 'output_config' not in body
    assert 'thinking' not in body
    assert body['system'] == 'Rules'
    assert 'reasoning_effort' not in body


@pytest.mark.parametrize('provider', ['openrouter', 'kilo'])
async def test_gateway_claude_keeps_default_effort_and_routes(provider):
    model = 'anthropic/claude-sonnet-5.5'
    entry = {'id': model, 'context_length': 200000, 'supported_parameters': ['tools', 'reasoning'],
             'reasoning': {'supported_efforts': ['low', 'max']}}
    endpoints = [{'tag': 'low', 'context_length': 200000, 'supported_parameters': ['tools', 'reasoning'],
                  'reasoning': {'supported_efforts': ['low']}},
                 {'tag': 'max', 'context_length': 200000, 'supported_parameters': ['tools', 'reasoning'],
                  'reasoning': {'supported_efforts': ['max']}}]
    bodies = []
    def respond(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'data': {'endpoints': endpoints}} if request.url.path.endswith('/endpoints') else {'data': [entry]})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '42'}}]})
    async with create_client(provider, api_key='dummy', transport=httpx.MockTransport(respond)) as client:
        profile = await client.model_capabilities(model)
        await client.chat(model=model, messages=[Message.user('Compute')])
    assert not {'reasoning', 'reasoning_effort', 'include_reasoning'}.intersection(bodies[0])
    if provider == 'openrouter':
        assert profile.providers == ['low', 'max']


@pytest.mark.parametrize('complete', [True, False])
async def test_claude_native_reasoning_stream_and_truncated_response(complete):
    events = [
        {'type': 'message_start', 'message': {'model': 'claude-test', 'usage': {'input_tokens': 12, 'cache_read_input_tokens': 5}}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'Checking the sum.'}},
        {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': '42'}},
        {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 9}},
    ]
    if complete:
        events.append({'type': 'message_stop'})
    def respond(request):
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
    deltas = []
    async with AnthropicClient('dummy', transport=httpx.MockTransport(respond)) as client:
        if not complete:
            with pytest.raises(APITransportError, match='message_stop'):
                await client.chat(model='claude-test', messages=[Message.user('Compute')], on_delta=lambda *d: deltas.append(d))
        else:
            result = await client.chat(model='claude-test', messages=[Message.user('Compute')], on_delta=lambda *d: deltas.append(d))
            assert result.message.content == '42' and result.message.reasoning == 'Checking the sum.'
            assert result.usage.prompt_tokens == 17 and result.usage.completion_tokens == 9
    assert ('reasoning', 'Checking the sum.') in deltas


@pytest.mark.parametrize('published,expected', [({'mandatory': True}, 'xhigh'), ({'supported_efforts': ['high']}, 'high')])
async def test_kilo_variants_supply_only_missing_published_choices(published, expected):
    entry = {'id': 'stealth/glyph-cluster', 'context_length': 262144,
             'supported_parameters': ['tools', 'reasoning'], 'reasoning': published,
             'opencode': {'variants': {'high': {'reasoning': {'effort': 'high'}},
                                      'xhigh': {'reasoning': {'effort': 'xhigh'}}}}}
    def respond(request):
        return httpx.Response(200, json={'data': [entry]})
    async with create_client('kilo', api_key='dummy', transport=httpx.MockTransport(respond)) as client:
        profile = await client.model_capabilities(entry['id'])
    assert profile.reasoning['effort'] == expected


@pytest.mark.parametrize('provider,model', [('google', 'gemma-4-31b-it'), ('nvidia', 'google/gemma-4-31b-it'),
                                           ('nvidia', 'google/diffusiongemma-26b-a4b-it')])
async def test_gemma_thinking_switch_is_only_in_outgoing_system_text(provider, model):
    messages = [Message.system('Rules'), Message.user('Compute')]
    bodies = []
    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': '42'}}]})
    async with create_client(provider, api_key='dummy', transport=httpx.MockTransport(respond)) as client:
        await client.chat(model=model, messages=messages)
    assert bodies[0]['messages'][0]['content'].startswith('<|think|>Rules')
    assert messages[0].content == 'Rules'
    assert 'reasoning_effort' not in bodies[0]
