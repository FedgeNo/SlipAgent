"""Wall-clock startup deadlines cancel silent requests and preserve active streams."""

import asyncio
import json
from importlib import import_module
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent.api import APIResponseError
from slipagent.capabilities import RequestProfile
from slipagent.diagnostics import RequestDiagnostics
from slipagent.openrouter import OpenRouterClient
from slipagent.types import Message, ModelInfo

session_for = import_module('test-providers').session_for


@pytest.fixture
async def advance(monkeypatch):
    loop = asyncio.get_running_loop()
    time = loop.time
    elapsed = 0.0
    async def advance(seconds):
        nonlocal elapsed
        elapsed += seconds
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    with monkeypatch.context() as clock:
        clock.setattr(loop, 'time', lambda: time() + elapsed)
        yield advance


def packet(delta, finish=None):
    return ('data: ' + json.dumps({'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}) + '\n\n').encode()


@pytest.mark.parametrize('phase', ['headers', 'heartbeat', 'tool_marker', 'reasoning_marker'])
@pytest.mark.parametrize('optimal', [False, True])
async def test_silent_request_deadline_cancels_and_falls_back(tmp_path, monkeypatch, advance, phase, optimal):
    started = asyncio.Event()
    heartbeat = asyncio.Event()
    closed = asyncio.Event()
    calls = []

    class Silent(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await heartbeat.wait()
            yield b': keepalive\n\n'
            yield packet({'role': 'assistant', 'content': ''})
            if phase == 'tool_marker':
                yield packet({'tool_calls': [{'index': 0, 'id': 'pending-call'}]})
            if phase == 'reasoning_marker':
                yield packet({'reasoning_details': [{'index': 0, 'type': 'reasoning.text', 'text': ''}]})
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async def handle(request):
        body = json.loads(request.content)
        if 'Summarizing the Previous Turn' in body['messages'][0]['content']:
            return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
                'summary': 'The silent request failed and a different model completed the task.',
                'reasoning_summary': '',
            })}, 'finish_reason': 'stop'}]})
        calls.append(body['model'])
        if body['model'] == 'first':
            if phase == 'headers':
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Silent())
        assert closed.is_set()
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Recovered.'}, 'finish_reason': 'stop'}]})

    async with OpenRouterClient('dummy', transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(client, 'model_capabilities', AsyncMock(return_value=RequestProfile(
            parameters={'tools'}, context_length=1_000_000)))
        retry = AsyncMock(return_value=True)
        monkeypatch.setattr(client, 'wait_retry', retry)
        session = session_for(client, tmp_path)
        session._catalog = [ModelInfo(name, provider='openrouter', score=None if rank is None else 100.0 - rank,
                                     pricing={'prompt': '0', 'completion': '0'})
                            for rank, name in enumerate(('first', 'second'), 1)]
        diagnostics = RequestDiagnostics()
        diagnostics.use_directory(tmp_path / 'requests')
        session.registry.services['request_diagnostics'] = diagnostics
        if optimal:
            await session.model_routing().enable()
        else:
            session.agent.model = 'first'
        task = asyncio.create_task(session.agent.run('Complete the original task'))
        try:
            await asyncio.wait_for(started.wait(), 1)
            await advance(119)
            heartbeat.set()
            await advance(0)
            assert calls == ['first'] and not task.done()
            await advance(2)
            if optimal:
                assert await asyncio.wait_for(task, 1) == 'Recovered.'
                assert calls == ['first', 'second']
                assert 'Optimal fallback' in session.renderer.stream.getvalue()
            else:
                with pytest.raises(APIResponseError, match='120 seconds'):
                    await asyncio.wait_for(task, 1)
                assert calls == ['first']
                assert session.agent.model == 'first'
            retry.assert_not_awaited()
            assert closed.is_set()
            first = diagnostics.listing()[0]
            assert first['outcome'] == 'request_error'
            assert '120 seconds' in first['detail']
            assert '120 seconds' in session.agent.history.steps[0].request_failure['error']
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.agent._compactor().aclose()
            await session.registry.aclose()


@pytest.mark.parametrize('delta', [
    {'content': 'Working. '},
    {'reasoning_content': 'Considering the code.'},
    {'tool_calls': [{'index': 0, 'id': 'call', 'function': {'name': 'probe', 'arguments': '{}'}}]},
])
async def test_started_output_can_stream_past_startup_deadline(advance, delta):
    class Active(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield packet(delta)
            await advance(121)
            yield packet({'content': 'Done.'}, 'stop')
            yield b'data: [DONE]\n\n'

    async with OpenRouterClient('dummy', transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Active())
    )) as client:
        result = await client.chat(model='test', messages=[Message.user('work')],
                                   on_delta=lambda *args: None, first_output_timeout=120)
    assert result.text.endswith('Done.')
    if 'tool_calls' in delta:
        assert result.tool_calls[0].name == 'probe'


async def test_background_call_without_startup_deadline_can_wait_longer(advance):
    async def handle(request):
        await advance(121)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Summary.'}, 'finish_reason': 'stop'}]})
    async with OpenRouterClient('dummy', transport=httpx.MockTransport(handle)) as client:
        result = await client.chat(model='test', messages=[Message.user('summarize')],
                                   on_delta=lambda *args: None, disable_timeout=True)
    assert result.text == 'Summary.'


async def test_rate_limit_retries_do_not_extend_startup_deadline(advance):
    calls = 0

    async def handle(request):
        nonlocal calls
        calls += 1
        await advance(50)
        return httpx.Response(429, headers={'retry-after': '0'}, json={'error': {'message': 'Resource exhausted'}})

    async with OpenRouterClient('dummy', transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(APIResponseError, match='No model output within 120 seconds'):
            await client.chat(model='test', messages=[Message.user('work')],
                              on_delta=lambda *args: None, first_output_timeout=120)
    assert calls == 3
