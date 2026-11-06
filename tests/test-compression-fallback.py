"""Failed compression falls back to bounded evidence without displacing recent history."""

from importlib import import_module

import pytest

from data_text_reader import request_input
from slipagent.context import message_tokens
from slipagent.types import Message

helpers = import_module('test-history-merge')


@pytest.mark.parametrize('status', ['failed', 'deferred', 'pending'])
async def test_failed_gap_uses_excerpt_before_dropping_aeon_or_compressing_recent_history(status):
    history, messages = helpers.history_with_posts(400, size=70)
    for step in history.steps:
        step.messages[-1].content['content'] = f'Original result {step.id}. ' * 30
    failed = history.steps[34]
    original = 'Large original tool result. ' * 50000
    failed.messages[-1].content['content'] = original
    failed.summary = None
    failed.compaction_status = status
    failed.compaction_error = 'OSError: No space left on device'
    history.steps[249].aeon_summary = {
        'record_type': 'history_step', 'representation': 'aeon_summary', 'step_id': 250,
        'step_ids': [i for i in range(1, 251) if i != 35], 'runs': list(range(1, 14)),
        'compressed_summary': 'Earlier durable work; step 35 has no AI summary.', 'created_at_step': 400}
    messages.append(Message.user('Continue the current work'))
    view = await history.view(messages, [], keep_steps=5, context_length=120000, max_output=1000)
    records = request_input(view)['history']
    assert any(record['representation'] == 'aeon_summary' for record in records)
    gap = next(record for record in records if record['step_id'] == 35)
    assert gap['representation'] == 'excerpt'
    assert all(record['representation'] == 'full' for record in records if record['step_id'] > 250)
    covered = [step for record in records for step in record.get('step_ids', [record['step_id']])]
    assert sorted(covered) == list(range(1, 401))
    assert message_tokens(view) <= int(120000 * .85) - 1000
    assert failed.messages[-1].content['content'] == original


async def test_failed_summary_does_not_reduce_history_when_originals_fit():
    history, messages = helpers.history_with_posts(10, size=70)
    history.steps[0].summary = None
    history.steps[0].compaction_status = 'failed'
    messages.append(Message.user('Continue'))
    view = await history.view(messages, [], keep_steps=5, context_length=1000000, max_output=1000)
    assert all(record['representation'] == 'full' for record in request_input(view)['history'])


def test_current_valid_summary_survives_a_failed_persistence_callback(tmp_path):
    from slipagent.turnstore import TurnStore
    from slipagent.historymerge import _narrative_source, run_numbers
    from slipagent.sources import materialize
    history, messages = helpers.history_with_posts(20, size=70)
    store = TurnStore(tmp_path / 'turns')
    try:
        for step in history.steps:
            store.archive(step, messages)
        first = history.steps[0]
        first.summary = 'New validated factual memory, not yet written to disk.'
        first.compaction_status = 'failed'
        first.compaction_error = 'OSError: No space left on device'
        batch, source = _narrative_source(history.steps, run_numbers(history.steps), 0, 19)
        assert materialize(source['steps'][0]['compressed_summary']) == first.summary
    finally:
        store.close()


async def test_unavailable_aeon_uses_its_five_available_eras():
    history, messages = helpers.history_with_posts(750, size=70)
    helpers.attach_layers(history)
    history.steps[499].aeon_attempt = 750  # Aeon generation failed; its source eras survive.
    messages.append(Message.user('Continue'))
    view = await history.view(messages, [], keep_steps=5, context_length=1100000, max_output=1000)
    records = request_input(view)['history']
    assert [record['representation'] for record in records[:5]] == ['era_summary'] * 5
    assert [record['step_ids'] for record in records[:5]] == [list(range(i, i + 100)) for i in range(1, 501, 100)]
    assert all(record['representation'] == 'full' for record in records[5:])
    assert [record['step_id'] for record in records[5:]] == list(range(501, 751))


@pytest.mark.parametrize('level,total', [('historical_summary', 20), ('era_summary', 100), ('aeon_summary', 500)])
async def test_available_aggregate_is_excerpted_before_its_source_steps_are_dropped(level, total):
    history, messages = helpers.history_with_posts(total, size=70)
    helpers.attach_layers(history)
    anchor = history.steps[-1]
    if level == 'aeon_summary':
        anchor.aeon_summary = {'record_type': 'history_step', 'representation': level,
            'step_id': total, 'step_ids': list(range(1, total + 1)),
            'compressed_summary': '', 'created_at_step': total}
    record = getattr(anchor, {'historical_summary': 'historical_summary',
                             'era_summary': 'era_summary', 'aeon_summary': 'aeon_summary'}[level])
    original = 'Stored compressed facts. ' * 400
    record['compressed_summary'] = original
    history.compact_through = total
    messages.append(Message.user('Continue'))
    full = await history.view(messages, [], keep_steps=5, context_length=1000000, max_output=512)
    length = int((message_tokens(full) - 1000 + 512) / .85)
    view = await history.view(messages, [], keep_steps=5, context_length=length, max_output=512)
    selected = request_input(view)['history']
    assert len(selected) == 1
    assert selected[0]['representation'] == level
    assert selected[0]['step_ids'] == list(range(1, total + 1))
    assert ['compressed_summary'] in selected[0]['excerpted_fields']
    assert isinstance(selected[0]['compressed_summary'], dict)
    assert record['compressed_summary'] == original
    assert message_tokens(view) <= int(length * .85) - 512
