"""Async thought filtering preserves originals and scopes contextual evidence."""

import asyncio
import json

import pytest

from slipagent.agent import Agent
from slipagent.capabilities import RequestProfile
from slipagent.compaction import StepCompactor
from slipagent.context import ConversationHistory
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry
from slipagent.types import Completion, Message
from test_agent import StubClient, context_records


@pytest.mark.parametrize('filtered', ['Useful hypothesis, still unverified.', ''])
async def test_pending_original_then_filtered_and_journal_roundtrip(filtered, tmp_path):
    ready, release = asyncio.Event(), asyncio.Event()
    class Client:
        async def chat(self, **options):
            ready.set()
            await release.wait()
            return Completion(Message.assistant(json.dumps({'summary':'Action observed.', 'reasoning_summary':filtered})), 'test')
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, 'test')
    response = Message.assistant('Read complete')
    response.reasoning = 'Repetition. Repetition. Useful hypothesis?'
    agent.extend([Message.user('Inspect'), response, Message.user('Continue')])
    step = agent.history.steps[0]
    journal = SessionJournal(str(tmp_path), tmp_path/'state')
    registry.services['session_journal'] = journal
    journal.begin(agent)
    compactor = StepCompactor(lambda usage:None, lambda error:None)
    compactor.on_step = journal.step
    try:
        compactor.submit(step, Client(), 'test', None, None)
        await ready.wait()
        view = await agent._context_view(registry.specs(), 1)
        assert context_records(view)[0]['reasoning'] == response.reasoning
        release.set()
        await compactor.wait()
        view = await agent._context_view(registry.specs(), 1)
        record = context_records(view)[0]
        if filtered:
            assert record['reasoning'] == filtered
            assert record['reasoning_representation'] == 'filtered'
        else:
            assert 'reasoning' not in record
        assert step.parts()['reasoning'] == response.reasoning
        restored = journal.load(journal.session_id)['history'].steps[0]
        assert restored.reasoning_summary == filtered
        assert restored.reasoning == response.reasoning
    finally:
        await compactor.aclose()
        await registry.aclose()


async def test_history_is_frozen_limited_and_not_the_compression_target():
    history = ConversationHistory()
    history.sync([m for i in range(30) for m in (Message.user(f'Question {i}'), Message.assistant(f'Answer {i}'))])
    captured = []
    class Client:
        async def chat(self, **options):
            captured.append(options['messages'][1].content)
            return Completion(Message.assistant(json.dumps({'summary':'Current only.', 'reasoning_summary':''})), 'test')
    compactor = StepCompactor(lambda usage:None, lambda error:None)
    compactor.submit(history.steps[-1], Client(), 'test', None, None, history=history.steps[:-1])
    history.steps[-2].messages[-1].content = 'LATER MUTATION'
    await compactor.wait()
    source = captured[0]
    assert source['user_prompt'] == 'Question 29'
    assert len(source['history']) == 25
    assert source['history'][0]['step_id'] == 5
    assert source['history'][-1]['step_id'] == 29
    assert 'LATER MUTATION' not in json.dumps(source)
    assert history.steps[-1].summary == 'Current only.'


async def test_budget_omits_context_without_truncating_current_step():
    history = ConversationHistory()
    history.sync([Message.user('Old'), Message.assistant('x'*100000), Message.user('Current'), Message.assistant('Current answer')])
    captured = []
    class Client:
        async def chat(self, **options):
            captured.append(options['messages'][1].content)
            return Completion(Message.assistant(json.dumps({'summary':'Current.', 'reasoning_summary':''})), 'test')
    compactor = StepCompactor(lambda usage:None, lambda error:None)
    compactor.submit(history.steps[-1], Client(), 'test', RequestProfile(parameters=set(), context_length=16000), 1000,
                     history=history.steps[:-1])
    await compactor.wait()
    assert captured[0]['history'] == []
    assert captured[0]['history_omitted'] == 1
    assert captured[0]['agent_response'] == 'Current answer'


@pytest.mark.parametrize('output', ['not JSON', '{"summary":"ok"}', '{"summary":"ok","reasoning_summary":null}',
                                  '{"summary":"ok","reasoning_summary":"x","extra":true}'])
async def test_malformed_filter_does_not_publish_partial_state(output, monkeypatch):
    monkeypatch.setattr('slipagent.compaction.COMPACTION_RETRIES', 0)
    response = Message.assistant('Reply')
    response.reasoning = 'Original thoughts'
    history = ConversationHistory()
    history.sync([Message.user('Question'), response])
    class Client:
        async def chat(self, **options):
            return Completion(Message.assistant(output), 'test')
    compactor = StepCompactor(lambda usage:None, lambda error:None)
    step = history.steps[0]
    compactor.submit(step, Client(), 'test', None, None)
    await compactor.wait()
    assert step.compaction_status == 'failed'
    assert step.summary is None and step.reasoning_summary is None
    assert step.reasoning == 'Original thoughts'
