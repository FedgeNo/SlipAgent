"""Ready old aggregates precede step summaries; lag gates the next working request."""

import asyncio
import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent.agent import Agent, STOP_NOTICE
from slipagent.capabilities import RequestProfile
from slipagent.compaction import StepCompactor
from slipagent.compressionproviders import CompressionProviders
from slipagent.context import ContextError, ContextStopped, ConversationHistory, message_tokens
from slipagent.modelrouting import ModelRouting
from slipagent.openrouter import OpenRouterClient
from slipagent.providers import create_client
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ModelInfo
from data_text_reader import read_data, request_input
from test_agent import StubClient

helpers = import_module('test-history-merge')


async def test_ready_aeons_eras_and_narratives_precede_individual_steps(monkeypatch):
    history, _ = helpers.history_with_posts(625, size=70)
    helpers.attach_layers(history)
    del history.steps[599].era_summary
    del history.steps[619].historical_summary
    history.steps[624].summary = None
    owner = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    monkeypatch.setattr(owner, 'dispatch', lambda: None)
    client = StubClient([])
    owner.submit(history.steps[624], client, 'test', None, None,
                 history=history.steps[599:624], conversation=history.steps)
    for expected in [('aeon', 500), ('era', 600), ('narrative', 620), ('step', 625)]:
        key, work = owner._next_job(set(), owner.generation)
        try:
            assert key == expected
        finally:
            await work


async def test_oldest_pending_individual_precedes_newer_queued_work(monkeypatch):
    history, _ = helpers.history_with_posts(5, size=70)
    owner = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    monkeypatch.setattr(owner, 'dispatch', lambda: None)
    for index in (4, 1):
        history.steps[index].summary = None
        owner.submit(history.steps[index], StubClient([]), 'test', None, None, conversation=history.steps)
    for expected in (2, 5):
        key, work = owner._next_job(set(), owner.generation)
        try:
            assert key == ('step', expected)
        finally:
            await work


async def test_small_context_deficit_uses_oldest_individual_summaries_before_the_aeon():
    history, messages = helpers.history_with_posts(750, size=70)
    helpers.attach_layers(history)
    history.steps[499].aeon_summary = {'record_type': 'history_step', 'representation': 'aeon_summary',
        'step_id': 500, 'step_ids': list(range(1, 501)), 'compressed_summary': 'Older durable findings.', 'created_at_step': 750}
    messages.append(Message.user('Continue'))
    full = await history.view(messages, [], keep_steps=5, context_length=4_000_000, max_output=1000)
    length = int(message_tokens(full) / .85)
    view = await history.view(messages, [], keep_steps=5, context_length=length, max_output=1000)
    records = request_input(view)['history']
    # Finer forms first: a small deficit costs a few of the oldest turns their
    # originals, while the aeon and every narrative stay unused.
    representations = [record['representation'] for record in records]
    assert representations[0] == 'compressed' and 'full' in representations
    assert set(representations) == {'compressed', 'full'}
    assert representations == sorted(representations)  # Every summary precedes every original.
    assert [record['step_id'] for record in records] == list(range(1, 751))
    assert message_tokens(view) <= int(length * .85) - 1000
    assert message_tokens(view) <= int(length * .85) - 1000


@pytest.fixture
async def catchup(monkeypatch):
    requests, errors = [], []
    control = SimpleNamespace(pause=False, started=asyncio.Event(), release=asyncio.Event())

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        control.started.set()
        if control.pause:
            await control.release.wait()
        result = {'response': 'Done', 'tool_calls': []}
        if body['messages'][0]['content'].lstrip().startswith('# Summarizing the Previous Turn'):
            result = {'summary': 'The request and recorded observations were retained.', 'reasoning_summary': ''}
        return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': json.dumps(result)},
            'finish_reason': 'stop'}]})

    async with OpenRouterClient('dummy', transport=httpx.MockTransport(handle)) as client:
        profile = RequestProfile(parameters={'max_tokens'}, context_length=100000, max_completion_tokens=1024)
        monkeypatch.setattr(client, 'model_capabilities', AsyncMock(return_value=profile))
        model = ModelInfo('test/primary', provider=client.provider, score=95,
                          pricing={'prompt': '0', 'completion': '0'})
        session = SimpleNamespace(client=client, agent=SimpleNamespace(model=model.id),
            compression_options=lambda: {client.provider: {'api_key': 'dummy'}},
            catalog=AsyncMock(return_value=[model]), notice=lambda text: None)
        routing = CompressionProviders(ModelRouting(session))
        history = ConversationHistory()
        messages = [message for index in range(1, 6)
                    for message in (Message.user(f'Request {index}'), Message.assistant(f'Answer {index}'))]
        history.sync(messages)
        for step in history.steps:
            step.compaction_status = 'failed'
        owner = StepCompactor(lambda usage: None, errors.append)
        owner.routing = routing
        try:
            yield SimpleNamespace(owner=owner, routing=routing, history=history, messages=messages,
                                  model=model, client=client, requests=requests, errors=errors, control=control,
                                  session=session, profile=profile, handle=handle)
        finally:
            control.release.set()
            owner.reset()
            await owner.wait()
            await routing.aclose()


async def test_free_primary_compresses_the_fifth_step_without_executing_tools(catchup):
    phases = []
    originals = [step.full_text() for step in catchup.history.steps]
    await catchup.owner.catch_up(catchup.history.steps, phases.append)
    assert catchup.history.steps[0].summary
    assert all(step.summary is None for step in catchup.history.steps[1:])
    assert len(catchup.requests) == 1
    request = catchup.requests[0]
    assert request['model'] == catchup.model.id
    assert not request.get('tools') and 'temperature' not in request
    assert read_data(request['messages'][1]['content'])['user_prompt'] == 'Request 1'
    assert any('Compressing step 1 with the free primary' in phase for phase in phases)
    assert [step.full_text() for step in catchup.history.steps] == originals


async def test_retained_full_history_does_not_trigger_catchup(catchup):
    catchup.history.steps[0].summary = 'Existing summary.'
    await catchup.owner.catch_up(catchup.history.steps, lambda text: pytest.fail(text))
    assert not catchup.requests
    catchup.history.steps.pop(0)
    assert not catchup.owner.behind(catchup.history.steps)


async def test_paid_primary_waits_for_background_progress(catchup):
    catchup.model.pricing = {'prompt': '1', 'completion': '1'}
    release, waiting, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def background():
        await release.wait()
        catchup.owner._publish_summary(catchup.history.steps[0], 'Background summary.', '')
        await finish.wait()

    job = asyncio.create_task(background())
    catchup.owner.jobs.add(job)
    job.add_done_callback(catchup.owner.jobs.discard)
    task = asyncio.create_task(catchup.owner.catch_up(catchup.history.steps,
        lambda text: waiting.set() if 'Waiting for' in text else None))
    await asyncio.wait_for(waiting.wait(), 5)
    assert not task.done() and not catchup.requests
    release.set()
    await asyncio.wait_for(task, 5)
    assert not catchup.requests
    assert not job.done()
    finish.set()
    await job


async def test_catchup_stop_keeps_background_work_owned_and_sends_no_working_request(catchup):
    catchup.model.pricing = {'prompt': '1', 'completion': '1'}
    release, waiting = asyncio.Event(), asyncio.Event()
    job = asyncio.create_task(release.wait())
    catchup.owner.jobs.add(job)
    job.add_done_callback(catchup.owner.jobs.discard)
    client = StubClient([])
    agent = Agent(client, ToolRegistry(), 'test', on_event=lambda event:
                  waiting.set() if event.kind == 'phase' and 'Waiting for free background' in event.text else None)
    agent.messages, agent.history = catchup.messages, catchup.history
    agent.registry.resources.compactor = catchup.owner
    task = asyncio.create_task(agent.run('Continue'))
    await asyncio.wait_for(waiting.wait(), 5)
    assert agent.request_stop()
    assert await asyncio.wait_for(task, 5) == STOP_NOTICE
    assert not client.calls and not catchup.requests and not job.cancelled()
    release.set()
    await job


async def test_primary_does_not_duplicate_an_inflight_step(catchup):
    catchup.owner.__dict__['_active_steps'] = {(catchup.owner.generation, 1)}
    await catchup.owner.catch_up(catchup.history.steps, lambda text: None)
    assert catchup.history.steps[0].summary is None
    assert catchup.history.steps[1].summary
    assert len(catchup.requests) == 1
    assert read_data(catchup.requests[0]['messages'][1]['content'])['user_prompt'] == 'Request 2'


async def test_paid_primary_without_background_jobs_warns_and_continues(catchup):
    catchup.model.pricing = {'prompt': '1', 'completion': '1'}
    for _ in range(2):
        await asyncio.wait_for(catchup.owner.catch_up(catchup.history.steps, lambda text: None), 5)
    assert not catchup.requests
    # One persistent warning, not one per request; originals stay in place.
    assert [error for error in catchup.errors if 'No free compression job is making progress' in error] == [
        'Compression unavailable: ' + __import__('slipagent.compaction', fromlist=['x']).CATCH_UP_UNAVAILABLE]
    assert all(step.summary is None for step in catchup.history.steps)


async def test_stalled_compression_lets_the_working_request_proceed(catchup):
    catchup.model.pricing = {'prompt': '1', 'completion': '1'}
    agent = Agent(catchup.client, ToolRegistry(), catchup.model.id)
    agent.messages, agent.history = catchup.messages, catchup.history
    agent.registry.resources.compactor = catchup.owner
    assert await asyncio.wait_for(agent.run('Continue'), 10) == 'Done'
    assert [request for request in catchup.requests
            if 'Summarizing the Previous Turn' not in request['messages'][0]['content']]


async def test_cancelling_primary_assistance_preserves_original_and_releases_claim(catchup):
    catchup.control.pause = True
    original = catchup.history.steps[0].full_text()
    task = asyncio.create_task(catchup.owner.catch_up(catchup.history.steps, lambda text: None))
    await asyncio.wait_for(catchup.control.started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert catchup.history.steps[0].summary is None
    assert catchup.history.steps[0].full_text() == original
    assert not catchup.owner.__dict__['_active_steps']


async def test_paid_primary_requeues_missing_resume_work_on_a_free_provider(catchup, monkeypatch):
    catchup.model.pricing = {'prompt': '1', 'completion': '1'}
    background = ModelInfo('test/background', provider='nvidia', score=95,
                           pricing={'prompt': '0', 'completion': '0'})
    catchup.session.catalog.return_value = [catchup.model, background]
    catchup.session.compression_options = lambda: {'openrouter': {'api_key': 'dummy'}, 'nvidia': {'api_key': 'dummy'}}

    def factory(provider, **kwargs):
        client = create_client(provider, transport=httpx.MockTransport(catchup.handle), **kwargs)
        monkeypatch.setattr(client, 'model_capabilities', AsyncMock(return_value=catchup.profile))
        return client

    monkeypatch.setattr('slipagent.compressionproviders.create_client', factory)
    await catchup.owner.catch_up(catchup.history.steps, lambda text: None)
    assert catchup.requests and all(request['model'] == background.id for request in catchup.requests)
    assert not catchup.owner.behind(catchup.history.steps)


async def test_single_free_provider_alternates_assistance_and_normal_requests(catchup):
    agent = Agent(catchup.client, ToolRegistry(), catchup.model.id)
    agent.messages, agent.history = catchup.messages, catchup.history
    agent.registry.resources.compactor = catchup.owner
    for _ in range(2):
        assert await agent.run('Continue') == 'Done'
        await catchup.owner.wait()
    assert len(catchup.requests) == 4
    assert ['Summarizing the Previous Turn' in request['messages'][0]['content'] for request in catchup.requests] == [
        True, False, True, False]
    assert read_data(catchup.requests[0]['messages'][1]['content'])['user_prompt'] == 'Request 1'
    assert read_data(catchup.requests[2]['messages'][1]['content'])['user_prompt'] == 'Request 2'


async def test_reset_during_route_lookup_does_not_queue_old_history(catchup, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    async def lookup():
        started.set()
        await release.wait()
        return [catchup.model]

    monkeypatch.setattr(catchup.routing, 'background', lookup)
    task = asyncio.create_task(catchup.owner.catch_up(catchup.history.steps, lambda text: None))
    await asyncio.wait_for(started.wait(), 5)
    catchup.owner.reset()
    catchup.history.steps.clear()
    release.set()
    await asyncio.wait_for(task, 5)
    assert not catchup.requests and not catchup.owner.jobs


async def test_primary_repriced_as_paid_is_not_called(catchup, monkeypatch):
    catchup.model.pricing = {'prompt': '1', 'completion': '1'}
    monkeypatch.setattr(catchup.routing, 'primary', AsyncMock(return_value=(catchup.client, catchup.model)))
    await asyncio.wait_for(catchup.owner.catch_up(catchup.history.steps, lambda text: None), 5)
    assert not catchup.requests
    assert any('No free compression job is making progress' in error for error in catchup.errors)


async def test_backlogged_window_summaries_precede_ready_grouped_work(monkeypatch):
    history, _ = helpers.history_with_posts(45, size=70)
    helpers.attach_layers(history)
    del history.steps[39].historical_summary  # Block 21–40 has a ready narrative job.
    owner = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    monkeypatch.setattr(owner, 'dispatch', lambda: None)
    client = StubClient([])
    for step in history.steps[-5:]:
        step.summary = None
    for step in history.steps[-5:]:
        owner.submit(step, client, 'test', None, None, history=history.steps[:step.id - 1], conversation=history.steps)
    key, work = owner._next_job(set(), owner.generation)
    work.close()
    assert key == ('step', 41)
    for step in history.steps[-5:]:
        step.summary = 'Done.'
        step.__dict__.pop('_compaction_request', None)
    key, work = owner._next_job(set(), owner.generation)
    work.close()
    assert key == ('narrative', 40)


async def test_context_preview_does_not_start_catchup(catchup):
    agent = Agent(catchup.client, ToolRegistry(), catchup.model.id)
    agent.messages, agent.history = catchup.messages, catchup.history
    agent.registry.resources.compactor = catchup.owner
    await agent._context_view([], 1, preview=True)
    assert not catchup.requests
    assert all(step.summary is None for step in catchup.history.steps)


async def test_already_requested_stop_does_not_start_primary_assistance(catchup):
    agent = Agent(catchup.client, ToolRegistry(), catchup.model.id)
    agent.messages, agent.history = catchup.messages, catchup.history
    agent.registry.resources.compactor = catchup.owner
    agent._stop_event().set()
    with pytest.raises(ContextStopped):
        await agent._catch_up_compression(6)
    assert not catchup.requests


async def test_canned_failure_notices_do_not_count_as_completed_ai_summaries(catchup):
    for step in catchup.history.steps:
        step.summary = 'Harness failure notice; no AI summary yet.'
        step.reasoning_summary = None
        step.request_failure = {'outcome': 'request_error', 'error': 'Provider failed.',
            'response_excerpt': 'Partial thoughts.', 'diagnostic_attempt': None,
            'request_number': step.id, 'consecutive_failures': step.id,
            'total_failed_attempts': step.id, 'tools_executed': False}
    assert catchup.owner.behind(catchup.history.steps)
    await catchup.owner.catch_up(catchup.history.steps, lambda text: None)
    assert catchup.history.steps[0].reasoning_summary == ''
    assert len(catchup.requests) == 1
    assert not catchup.owner.behind(catchup.history.steps)


async def test_input_queued_during_paid_catchup_reaches_the_next_context(catchup):
    catchup.model.pricing = {'prompt': '1', 'completion': '1'}
    release, waiting = asyncio.Event(), asyncio.Event()
    job = asyncio.create_task(release.wait())
    catchup.owner.jobs.add(job)
    job.add_done_callback(catchup.owner._job_finished)
    agent = Agent(catchup.client, ToolRegistry(), catchup.model.id, on_event=lambda event:
                  waiting.set() if event.kind == 'phase' and 'Waiting for free background' in event.text else None)
    agent.messages, agent.history = catchup.messages, catchup.history
    agent.registry.resources.compactor = catchup.owner
    task = asyncio.create_task(agent._queued_context([], 6))
    await asyncio.wait_for(waiting.wait(), 5)
    agent.enqueue('Use this newest correction.')
    catchup.owner._publish_summary(catchup.history.steps[0], 'Background summary.', '')
    view = await asyncio.wait_for(task, 5)
    assert request_input(view)['user_prompts'] == ['Use this newest correction.']
    assert not catchup.requests
    release.set()
    await job
