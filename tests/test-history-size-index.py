"""Indexed history costs compose exactly without rereading selected sources."""

import pytest

from slipagent.context import ConversationHistory, TokenEstimates, message_tokens
from slipagent.historyindex import HistorySizeIndex
from slipagent.records import step_record
from slipagent.types import Message, ToolCall, ToolSpec


@pytest.mark.parametrize('count', [0, 1, 2, 9, 10, 11, 99, 100, 101])
def test_index_composes_exact_wire_bytes_across_block_number_widths(count):
    records = []
    for index in range(count):
        calls = [ToolCall(f'{index}-{child}', 'read_file', {'path': '界\\"\n.py'}) for child in range(index % 3 + 1)]
        results = [Message.tool_result(call.id, {'tool': call.name, 'call_id': call.id, 'status': 'success',
                    'content': '界\\"\n\t👩‍💻' if child % 2 else {'rows': ['界\\"\n\t👩‍💻', child]}})
                   for child, call in enumerate(calls)]
        records.append(step_record([Message.user('Read\n界'), Message.assistant('Done', calls), *results], step_id=index + 1))
    measure = TokenEstimates().history_bytes
    sizing = HistorySizeIndex(measure)
    sizes = [sizing.measure_record(record) for record in records]
    predicted = sizing.empty + sizing.total_bytes(sum(size.content_bytes for size in sizes),
                                                   sum(size.blocks for size in sizes), len(sizes))
    assert predicted == measure(records)
    assert all(len(size.result_bytes) == size.blocks for size in sizes)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('scale', [0.7, 1.0, 2.7])
async def test_selection_uses_index_and_always_budgets_definitions(monkeypatch, native, scale):
    import slipagent.context as context
    original = context.message_tokens

    def no_full_history_measurement(messages):
        from slipagent.data_text import JSONInput
        assert not any(isinstance(message.content, JSONInput) and message.content.data.get('history') for message in messages)
        return original(messages)

    monkeypatch.setattr(context, 'message_tokens', no_full_history_measurement)
    messages = [Message.system('Required instructions. ' * 100)]
    for index in range(12):
        call = ToolCall(str(index), 'read_file', {})
        messages += [Message.user(f'Read {index}'), Message.assistant('Reading', [call]),
                     Message.tool_result(call.id, {'tool': call.name, 'call_id': call.id,
                         'status': 'success', 'content': '界\\"\n' * 2000})]
    specs = [ToolSpec('read_file', 'Required definition. ' * 300, {'type': 'object', 'properties': {}})]
    history = ConversationHistory()
    history.sync(messages)
    for step in history.steps[:-1]:
        step.summary = f'Earlier result {step.id}'
        step.observed = True
    view = await history.view(messages, specs, keep_steps=10, context_length=36000, max_output=1024,
                              native_tools=native, token_scale=scale)
    import json
    import math
    definitions = math.ceil(context.estimate_tokens(json.dumps([spec.to_api() for spec in specs])) * scale)
    assert math.ceil(message_tokens(view) * scale) + definitions <= int(36000 * .85) - 1024


@pytest.mark.parametrize('change', ['summary', 'reasoning', 'artifact', 'input'])
async def test_archived_index_tracks_updated_representation(tmp_path, change):
    import json
    from data_text_reader import read_json_input
    from slipagent.turnstore import TurnStore

    call = ToolCall('read', 'read_file', {})
    messages = [Message.user('Original request'), Message(role='assistant', content='Read source', tool_calls=[call], reasoning='Earlier thoughts ' * 100),
                Message.tool_result(call.id, {'tool': call.name, 'call_id': call.id, 'status': 'success',
                                              'content': 'Original result ' * 100})]
    history = ConversationHistory()
    history.sync(messages)
    history.store = TurnStore(tmp_path / 'turns')
    step = history.steps[0]
    history.store.archive(step, messages)
    try:
        first = await history.view(messages, [], keep_steps=5, context_length=30000, max_output=512)
        first_wire = first[-1].to_api()['content']
        if change == 'summary':
            step.summary = 'New compacted evidence'
            history.store.summarized(step)
        elif change == 'reasoning':
            step.reasoning_summary = 'New filtered reasoning'
            history.store.summarized(step)
        elif change == 'artifact':
            history.store.attach_artifact(step, call.id, {'record_id': 'new-record', 'kind': 'output'})
        else:
            step.request = 'Changed request'
            step.messages[0] = Message.user(step.request)
            history.store.refresh_input(step)
        second = await history.view(messages, [], keep_steps=5, context_length=30000, max_output=512)
        assert first[-1].to_api()['content'] == first_wire
        if change == 'summary':
            assert second[-1].to_api()['content'] == first_wire
        else:
            assert second[-1].to_api()['content'] != first_wire
        record = read_json_input(second[-1].to_api()['content'])['history'][0]
        snapshot = json.loads((history.store.directory / '00000001.sizes.json').read_text())
        entry = next(entry for entry in snapshot['representations'] if entry['form'][0] is False)
        from slipagent.historyindex import number_bytes
        predicted = entry['content_bytes'] + snapshot['join_bytes'] + 2 * number_bytes(entry['blocks'])
        raw = second[-1].content.data['history'][0]
        assert predicted == TokenEstimates().history_bytes([raw]) - TokenEstimates().history_bytes([])
        assert record['tool_calls'][0]['result'] == 'Original result ' * 100
    finally:
        history.clear()
