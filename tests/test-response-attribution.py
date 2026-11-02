"""Response origins survive model switches, archival and context compression."""

import pytest

from data_text_reader import request_input
from slipagent.agent import Agent
from slipagent.api import APITransportError
from slipagent.context import CompletedStep, ConversationHistory
from slipagent.historyindex import HistorySizeIndex
from slipagent.historymerge import historical_records, era_records
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry
from slipagent.turnstore import TurnStore
from slipagent.types import Message
from test_agent import StubClient, completion


async def test_model_switch_attribution_reaches_next_request_and_survives_resume(tmp_path):
    client = StubClient([completion('First answer'), completion('Second answer')])
    client.provider = 'openrouter'
    agent = Agent(client, ToolRegistry(), 'model-one')
    journal = SessionJournal(str(tmp_path), directory=tmp_path / 'saved')
    agent.registry.resources.journal = journal
    try:
        await agent.run('First question')
        await agent.wait_for_compaction()
        agent.model = 'model-two'
        client.provider = 'nvidia'
        await agent.run('Second question')
        await agent.wait_for_compaction()
        assert request_input(client.calls[1]['messages'])['history'][0]['response_source'] == {
            'model': 'model-one', 'provider': 'openrouter'}
        loaded = journal.load(journal.session_id)
        try:
            replies = [message for message in loaded['messages'] if message.role == 'assistant']
            assert [(message.reasoning_model, message.response_provider) for message in replies] == [
                ('model-one', 'openrouter'), ('model-two', 'nvidia')]
            assert [step.parts()['response_source'] for step in loaded['history'].steps] == [
                {'model': 'model-one', 'provider': 'openrouter'},
                {'model': 'model-two', 'provider': 'nvidia'}]
        finally:
            loaded['history'].clear()
    finally:
        await agent.registry.aclose()


async def test_failed_attempt_without_reasoning_keeps_its_origin():
    class FailingClient(StubClient):
        provider = 'openrouter'

        async def chat(self, **kwargs):
            raise APITransportError('Provider disconnected')

        async def wait_retry(self, *args, **kwargs):
            return False

    agent = Agent(FailingClient([]), ToolRegistry(), 'failed-model')
    try:
        with pytest.raises(APITransportError):
            await agent.run('Try once')
        record = agent.history.steps[0].context_messages()[0].content
        assert record['response_source'] == {'model': 'failed-model', 'provider': 'openrouter'}
        assert record['request_failure']['tools_executed'] is False
    finally:
        await agent.registry.aclose()


@pytest.mark.parametrize('archived', [False, True])
def test_attribution_survives_full_compressed_excerpt_and_exact_budgeting(tmp_path, archived):
    reply = Message.assistant('Original answer ' * 2000)
    reply.reasoning_model = 'model-one'
    reply.response_provider = 'openrouter'
    messages = [Message.user('Question'), reply]
    step = CompletedStep(1, 'Question', messages)
    store = TurnStore(tmp_path / 'turns')
    try:
        if archived:
            store.archive(step, messages)
        step.summary = 'A short summary'
        if archived:
            store.summarized(step)
        records = [step.context_messages()[0].content, step.context_messages(compressed=True)[0].content,
                   step.excerpt_context_messages(400)[0].content]
        for record in records:
            assert record['response_source'] == {'model': 'model-one', 'provider': 'openrouter'}
        # The production byte index includes attribution as ordinary metadata.
        from slipagent.context import TokenEstimates
        sizing = HistorySizeIndex(TokenEstimates().history_bytes)
        costs = [sizing.measure_record(record) for record in records]
        assert sizing.empty + sizing.total_bytes(sum(c.content_bytes for c in costs),
            sum(c.blocks for c in costs), len(costs)) == TokenEstimates().history_bytes(records)
    finally:
        store.close()


def test_legacy_response_does_not_invent_a_provider():
    reply = Message.assistant('Old answer')
    reply.reasoning_model = 'old-model'
    step = CompletedStep(1, 'Question', [Message.user('Question'), reply])
    assert step.context_messages()[0].content['response_source'] == {'model': 'old-model'}
    reply.reasoning_model = None
    assert 'response_source' not in step.context_messages()[0].content


@pytest.mark.parametrize('kind,read', [('historical_summary', historical_records), ('era_summary', era_records)])
def test_grouped_history_preserves_model_change_ranges(kind, read):
    history = ConversationHistory()
    messages = []
    for index in range(100):
        reply = Message.assistant('Answer')
        reply.reasoning_model = 'model-one' if index < 40 else 'model-two'
        reply.response_provider = 'openrouter' if index < 40 else 'nvidia'
        messages.extend([Message.user('Question'), reply])
    history.sync(messages)
    count = 20 if kind == 'historical_summary' else 100
    # Put a transition inside either summary's covered range.
    first = 31 if count == 20 else 1
    last = first + count - 1
    anchor = history.steps[last - 1]
    stored = {'record_type': 'history_step', 'representation': kind, 'step_id': last,
              'step_ids': list(range(first, last + 1)), 'runs': [1], 'compressed_summary': 'Earlier work',
              'created_at_step': 100}
    setattr(anchor, 'historical_summary' if count == 20 else 'era_summary', stored)
    record = read(history.steps)[0]
    assert record['response_sources'] == [
        {'first_step': first, 'last_step': 40, 'model': 'model-one', 'provider': 'openrouter'},
        {'first_step': 41, 'last_step': last, 'model': 'model-two', 'provider': 'nvidia'}]
    assert 'response_sources' not in stored
    assert read(history.steps)[0] is record
