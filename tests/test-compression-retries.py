"""Failed summaries retry without spinning, blocking shutdown or shrinking working context."""

import asyncio
import json
from importlib import import_module
from unittest.mock import AsyncMock
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from slipagent.api import APIContextError
from slipagent.capabilities import RequestProfile
from slipagent.compaction import StepCompactor, FAILED_SUMMARY_RETRY_STEPS
from slipagent.context import CompletedStep, message_tokens
from slipagent.types import Completion, Message, ModelInfo

helpers = import_module('test-history-merge')


class RetryClient:
    def __init__(self):
        self.calls = []
        self.fail = True

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise OSError(28, 'No space left on device')
        result = {'summary': 'Recovered factual memory.'}
        if 'user_prompt' in kwargs['messages'][1].content:
            result['reasoning_summary'] = ''
        return Completion(Message.assistant(json.dumps(result)), 'test')


async def failed_job(monkeypatch):
    history, messages = helpers.history_with_posts(3, size=70)
    step = history.steps[0]
    step.summary = None
    step.compaction_status = 'failed'
    errors = []
    owner = StepCompactor(lambda _: None, errors.append)
    monkeypatch.setattr(owner, 'wait_request_retry', AsyncMock(return_value=False))
    client = RetryClient()
    owner.track(history.steps, client, 'test', None, None)
    await asyncio.wait_for(owner.wait(), 2)
    assert step.compaction_status == 'failed'
    assert len(client.calls) == 1
    return history, messages, step, owner, client, errors


@pytest.mark.parametrize('trigger', ['turns', 'timer'])
async def test_disk_failure_retries_plain_request_after_cooldown(monkeypatch, trigger):
    history, messages, step, owner, client, errors = await failed_job(monkeypatch)
    try:
        owner.track(history.steps, client, 'test', None, None)
        await owner.wait()
        assert len(client.calls) == 1
        client.fail = False
        if trigger == 'turns':
            helpers.append_posts(history, messages, FAILED_SUMMARY_RETRY_STEPS, size=70)
            owner.track(history.steps, client, 'test', None, None)
        else:
            owner.__dict__['_retry_timer'].cancel()
            step.__dict__['_summary_retry_at'] = asyncio.get_running_loop().time() - 1
            owner._retry_ready(owner.generation)
        await asyncio.wait_for(owner.wait(), 2)
        requests = [call for call in client.calls if 'user_prompt' in call['messages'][1].content]
        assert len(requests) == 2
        assert all('phase' not in call['messages'][1].content for call in requests)
        assert step.summary == 'Recovered factual memory.'
        assert step.compaction_status == 'complete' and step.compaction_error is None
    finally:
        await owner.aclose()


async def test_reset_cancels_retry_timer_and_stale_callbacks_do_not_clear_new_timer(monkeypatch):
    history, messages, step, owner, client, errors = await failed_job(monkeypatch)
    previous = owner.generation
    old_timer = owner.__dict__['_retry_timer']
    owner.reset()
    assert old_timer.cancelled()
    owner.track(history.steps, client, 'test', None, None)
    owner._schedule_retry(step)
    fresh = owner.__dict__['_retry_timer']
    owner._retry_ready(previous)
    assert owner.__dict__['_retry_timer'] is fresh and not fresh.cancelled()
    await owner.aclose()
    assert fresh.cancelled()


async def test_local_disk_error_does_not_blacklist_or_rotate_free_providers():
    owner = StepCompactor(lambda _: None, lambda _: None)
    profile = RequestProfile(parameters={'max_tokens'}, context_length=32768)
    client = SimpleNamespace(model_capabilities=AsyncMock(return_value=profile))
    model = ModelInfo('test/free', provider='nvidia', pricing={'prompt': '0', 'completion': '0'})
    routing = SimpleNamespace(models=SimpleNamespace(revision=0, reject=Mock()),
        select=AsyncMock(return_value=(client, model)), release=Mock(), aclose=AsyncMock())
    owner.routing = routing
    async def fail(*args):
        raise OSError(28, 'No space left on device')
    with pytest.raises(OSError, match='No space left'):
        await owner.request(fail, client, model.id, profile, 8192, owner.generation)
    routing.models.reject.assert_not_called()
    routing.select.assert_awaited_once()
    routing.release.assert_called_once()
    await owner.aclose()


async def test_provider_context_error_replans_smaller_lossless_chunks(monkeypatch):
    source_text = 'Exact original source with facts. ' * 1000
    step = CompletedStep(1, 'Retain the facts', [Message.user('Retain the facts'), Message.assistant(source_text)])
    source = step.compaction_input()
    source.update(history=[], history_omitted=0)
    profile = RequestProfile(parameters={'max_tokens'}, context_length=32768, max_completion_tokens=8192)
    calls = []
    sizes = []
    class Client:
        async def chat(self, **kwargs):
            calls.append(kwargs)
            sizes.append(message_tokens(kwargs['messages']))
            if len(calls) == 1:
                raise APIContextError('Provider says the request exceeds its actual context')
            return Completion(Message.assistant(json.dumps({'summary': 'Exact factual notes.', 'reasoning_summary': ''})), 'test')
    owner = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    await owner._compact_once(step, source, Client(), 'test', profile, 8192, owner.generation)
    assert step.compaction_status == 'complete' and step.summary == 'Exact factual notes.'
    assert 'phase' not in calls[0]['messages'][1].content
    assert any(call['messages'][1].content.get('phase') == 'chunk' for call in calls[1:])
    smaller = (int(profile.context_length * .85) - 8192) // 2
    assert all(size <= smaller for size in sizes[1:])
    assert step.agent_response == source_text
    await owner.aclose()
