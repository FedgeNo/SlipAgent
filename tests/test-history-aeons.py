"""Aeons reduce five eras before older evidence is removed from model context."""

import copy
import asyncio
import json
from importlib import import_module

import pytest

from data_text_reader import request_input
from slipagent.compaction import StepCompactor
from slipagent.context import RecallHistoryTool, message_tokens
from slipagent.historymerge import aeon_records, era_records, ready_job, validate_aeon_record
from slipagent.sessions import SessionJournal
from slipagent.turnstore import TurnStore
from slipagent.types import Completion, Message, content_text

helpers = import_module('test-history-merge')


def prepared_history(total=500):
    history, messages = helpers.history_with_posts(total, size=70)
    helpers.attach_layers(history)
    return history, messages


async def test_aeon_is_generated_from_five_eras_and_retains_original_ids():
    history, _ = prepared_history()
    client = helpers.MergeClient()
    errors = []
    owner = StepCompactor(lambda _: None, errors.append)
    eras = copy.deepcopy(era_records(history.steps))
    helpers.schedule(owner, history, client)
    await owner.wait()
    assert not errors
    assert len(client.calls) == 1
    source = client.calls[0]['messages'][1].content
    assert len(source['eras']) == 5
    assert [entry['step_ids'] for entry in source['eras']] == [era['step_ids'] for era in eras]
    assert [entry['compressed_summary'] for entry in source['eras']] == [era['compressed_summary'] for era in eras]
    record = aeon_records(history.steps)[0]
    assert record['step_ids'] == list(range(1, 501))
    assert record['step_id'] == 500 and record['created_at_step'] == 500
    assert history.steps[-1].aeon_attempt == 500
    assert era_records(history.steps) == eras
    assert message_tokens([Message.user(record)]) < message_tokens([Message.user(era) for era in eras])
    original = await RecallHistoryTool(history).invoke({'step_id': 1, 'call_id': 'read-1'})
    assert 'ORIGINAL ' * 1000 in content_text(original.content)


async def test_aeons_follow_variable_run_blocks_through_the_entire_pipeline():
    history, _ = helpers.history_with_runs([11] * 25)
    errors = []
    owner = StepCompactor(lambda _: None, errors.append)
    client = helpers.MergeClient()
    helpers.schedule(owner, history, client)
    await owner.wait()
    assert not errors
    assert len(client.calls) == 31  # 25 narratives, five eras, one aeon.
    record = aeon_records(history.steps)[0]
    assert record['step_id'] == 275
    assert record['step_ids'] == list(range(1, 276))
    assert record['runs'] == list(range(1, 26))
    assert len(client.calls[-1]['messages'][1].content['eras']) == 5


async def test_reset_cancels_aeon_without_publishing_late_results():
    history, _ = prepared_history()
    expected = copy.deepcopy(era_records(history.steps))
    owner = StepCompactor(lambda _: None, lambda _: None)
    client = helpers.MergeClient(pause=True)
    helpers.schedule(owner, history, client)
    await asyncio.wait_for(client.started.wait(), 5)
    owner.reset()
    client.release.set()
    await owner.wait()
    assert not aeon_records(history.steps)
    assert era_records(history.steps) == expected


def test_aeon_waits_for_every_era_in_its_closed_block():
    history, _ = prepared_history()
    del history.steps[-1].era_summary
    owner = StepCompactor(lambda _: None, lambda _: None)
    assert ready_job(owner, history.steps, {('era', 500)}, (helpers.MergeClient(), 'test', None, None)) is None


@pytest.mark.parametrize('archived', [False, True])
async def test_aeon_is_selected_before_omission_and_keeps_recent_third_full(tmp_path, archived):
    history, messages = prepared_history(750)
    for step in history.steps:
        reply = next(message for message in step.messages if message.role == 'assistant')
        reply.reasoning_model = 'first/model' if step.id <= 250 else 'second/model'
        reply.response_provider = 'test-provider'
    owner = StepCompactor(lambda _: None, lambda _: None)
    helpers.schedule(owner, history, helpers.MergeClient())
    await owner.wait()
    assert len(aeon_records(history.steps)) == 1
    store = TurnStore(tmp_path / 'turns')
    try:
        if archived:
            for step in history.steps:
                store.archive(step, messages)
        messages.append(Message.user('Continue'))
        # Measure the all-era form, then ask for slightly less than it needs.
        saved = history.steps[499].aeon_summary
        del history.steps[499].aeon_summary
        history.compact_through = 500
        without = await history.view(messages, [], keep_steps=5, context_length=4_000_000, max_output=1000)
        history.compact_through = 0
        history.steps[499].aeon_summary = saved
        length = int(message_tokens(without) / .85)
        view = await history.view(messages, [], keep_steps=5, context_length=length, max_output=1000)
        records = request_input(view)['history']
        aeon = next(record for record in records if record['representation'] == 'aeon_summary')
        assert aeon['step_ids'] == list(range(1, 501))
        assert [(entry['first_step'], entry['last_step'], entry['model']) for entry in aeon['response_sources']] == [
            (1, 250, 'first/model'), (251, 500, 'second/model')]
        assert [record['step_id'] for record in records if record['representation'] == 'full'] == list(range(501, 751))
        covered = [step_id for record in records for step_id in record.get('step_ids', [record['step_id']])]
        assert covered == list(range(1, 751))
        assert message_tokens(view) <= int(length * .85) - 1000
        assert 'aeon_summary' in content_text(view[0].content)
    finally:
        store.close()


async def test_aeon_record_and_attempt_survive_resume_and_fork(tmp_path):
    history, messages = prepared_history()
    agent, _ = helpers.make_agent([])
    agent.messages, agent.history = messages, history
    journal = SessionJournal(str(tmp_path), directory=tmp_path / 'state')
    journal.begin(agent)
    owner = StepCompactor(lambda _: None, lambda _: None)
    owner.on_step = journal.step
    helpers.schedule(owner, history, helpers.MergeClient())
    await owner.wait()
    expected = copy.deepcopy(aeon_records(history.steps))
    assert len(expected) == 1
    data = journal.load(journal.session_id)
    assert aeon_records(data['history'].steps) == expected
    assert data['history'].steps[499].aeon_attempt == 500
    assert list(data['history'].store.directory.glob('*.aeon.json'))
    restored, _ = helpers.make_agent([])
    journal.restore(restored, data, fork=True)
    child = journal.load(journal.session_id)
    assert aeon_records(child['history'].steps) == expected
    assert child['history'].steps[499].aeon_attempt == 500
    await agent.registry.aclose()
    await restored.registry.aclose()


@pytest.mark.parametrize('malformed', [False, True])
async def test_failed_or_nonshrinking_aeon_keeps_eras_and_waits_before_retry(monkeypatch, malformed):
    class BadClient:
        def __init__(self):
            self.calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            return Completion(Message.assistant('invalid' if malformed else json.dumps({'summary': 'x' * 12000})), 'test')

    async def no_retry(*args):
        return False

    history, messages = prepared_history()
    expected = copy.deepcopy(era_records(history.steps))
    errors = []
    owner = StepCompactor(lambda _: None, errors.append)
    monkeypatch.setattr(owner, 'wait_request_retry', no_retry)
    client = BadClient()
    helpers.schedule(owner, history, client)
    await owner.wait()
    assert errors and 'aeon' in errors[0]
    assert not aeon_records(history.steps)
    assert era_records(history.steps) == expected
    helpers.schedule(owner, history, client)
    await owner.wait()
    assert client.calls == 1 and history.steps[-1].aeon_attempt == 500
    helpers.append_posts(history, messages, 20, size=70)
    helpers.attach_layers(history)
    helpers.schedule(owner, history, client)
    await owner.wait()
    assert client.calls == 2 and history.steps[499].aeon_attempt == 520


def test_invalid_aeon_source_ids_are_rejected():
    record = {'record_type': 'history_step', 'representation': 'aeon_summary', 'step_id': 500,
              'step_ids': [1, 1, 500], 'runs': [1], 'compressed_summary': 'Evidence.', 'created_at_step': 500}
    with pytest.raises(ValueError):
        validate_aeon_record(record, 500, 500)
