"""Oversized results shrink without erasing their turn or unrelated history."""

import pytest

from data_text_reader import request_input
from slipagent.context import ConversationHistory, RecallHistoryTool, message_tokens
from slipagent.turnstore import TurnStore
from slipagent.sources import materialize
from slipagent.types import Message, ToolCall


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_older_steps_give_way_before_the_newest_large_result_is_clipped(tmp_path, archived, native):
    history = ConversationHistory()
    messages = []
    for index in range(8):
        messages += [Message.user(f"Earlier instruction {index}"), Message.assistant(f"Earlier finding {index}")]
    calls = [ToolCall("local-large", "run_command", {"command": "read-file"}),
             ToolCall("local-small", "read_file", {"path": "small.txt"})]
    reply = Message.assistant("Checking both results.", calls)
    reply.reasoning = "Compare the observations before concluding."
    text = "START\n" + "λ numerical output with exact values.\n" * 10000 + "\nEND"
    messages += [Message.user("Inspect the results; keep the permissions unchanged."), reply,
        Message.tool_result(calls[0].id, {"tool": "run_command", "call_id": calls[0].id,
            "status": "success", "content": {"returncode": 0, "stdout": text, "stderr": "Small warning."}}),
        Message.tool_result(calls[1].id, {"tool": "read_file", "call_id": calls[1].id,
            "status": "success", "content": "Complete small result."})]
    history.sync(messages)
    for step in history.steps[:-1]:
        step.summary = f"Earlier finding {step.id}."
    store = TurnStore(tmp_path / "turns")
    try:
        if archived:
            for step in history.steps:
                store.archive(step, messages)
        originals = [step.full_text() for step in history.steps]
        view = await history.view(messages, [], keep_steps=50, context_length=30000, max_output=1000, native_tools=native)
        payload = request_input(view)
        assert [record['step_id'] for record in payload['history']] == [9]
        result = payload['history'][-1]
        assert result['representation'] == 'excerpt'
        assert result['excerpted_fields'] == [['tool_results', 0, 'content', 'stdout']]
        assert result['agent_response'] == reply.content
        assert result['reasoning'] == reply.reasoning
        assert result['user_prompts'] == ["Inspect the results; keep the permissions unchanged."]
        assert [call['arguments'] for call in result['tool_calls']] == [call.arguments for call in calls]
        clipped = result['tool_calls'][0]['result_excerpt']
        assert clipped['returncode'] == 0 and clipped['stderr'] == "Small warning."
        stdout = clipped['stdout']
        assert stdout['prefix'].startswith('START') and stdout['suffix'].endswith('END')
        assert stdout['omitted_characters'] == len(text) - len(stdout['prefix']) - len(stdout['suffix'])
        assert stdout['original']['step_id'] == 9
        reference = stdout['original']
        retrieved = await RecallHistoryTool(history).run(**{key: value for key, value in reference.items() if key != 'field_path'})
        assert not retrieved.is_error
        recovered = retrieved.content['content']
        for key in reference['field_path']:
            recovered = recovered[key]
        assert recovered == text
        assert result['tool_calls'][1]['result'] == "Complete small result."
        assert message_tokens(view) <= int(30000 * .85) - 1000
        assert [step.full_text() for step in history.steps] == originals
        assert history.compact_through == 0
    finally:
        store.close()


async def test_oversized_structured_array_is_clipped_without_losing_call_or_metadata():
    history = ConversationHistory()
    call = ToolCall("local", "read_file", {"path": "values.json"})
    messages = [Message.user("Read the values."), Message.assistant("Reading values.", [call]),
                Message.tool_result(call.id, {"values": list(range(50000)), "units": "meters"})]
    view = await history.view(messages, [], keep_steps=50, context_length=20000, max_output=1000)
    record = request_input(view)['history'][0]
    call_record = record['tool_calls'][0]
    assert call_record['arguments'] == call.arguments
    result = call_record['unclassified_result_excerpt']
    assert result['units'] == 'meters'
    assert result['values']['prefix_items'][0] == 0
    assert result['values']['suffix_items'][-1] == 49999
    assert result['values']['omitted_items'] > 0
    assert history.steps[0].tool_results[0].content['values'] == list(range(50000))


async def test_oversized_results_reduce_older_turns_before_the_latest_turn():
    history, messages = ConversationHistory(), []
    for index in range(4):
        call = ToolCall(f"local-{index}", "run_command", {"command": f"read {index}"})
        reply = Message.assistant(f"Verified fact {index}. " * 20, [call])
        reply.reasoning = f"Important uncertainty {index}. " * 20
        messages += [Message.user(f"Permission for task {index}."), reply,
                     Message.tool_result(call.id, f"Result {index}. " * 20000)]
    history.sync(messages)
    view = await history.view(messages, [], keep_steps=50, context_length=30000, max_output=1000)
    records = request_input(view)['history']
    assert [record['step_id'] for record in records] == [4]
    for record in records:
        index = record['step_id'] - 1
        assert record['agent_response'] == f"Verified fact {index}. " * 20
        assert record['reasoning']['omitted_characters'] == len(f"Important uncertainty {index}. " * 20)
        assert record['user_prompts'] == [f"Permission for task {index}."]
        assert record['tool_calls'][0]['arguments'] == {"command": f"read {index}"}
        assert record['excerpted_fields'][0] == ['reasoning']
        assert all(path[0] in {'reasoning', 'tool_results'} for path in record['excerpted_fields'])


async def test_older_result_is_removed_before_clipping_an_oversized_new_result():
    history = ConversationHistory()
    first, second = ToolCall('first', 'read_file', {}), ToolCall('second', 'read_file', {})
    small = 'Keep this fitting earlier result. ' * 100
    messages = [Message.user('Inspect.'), Message.assistant('First.', [first]),
                Message.tool_result(first.id, small), Message.assistant('Second.', [second]),
                Message.tool_result(second.id, 'Huge new result. ' * 30000)]
    history.sync(messages)
    history.steps[0].observed = True
    view = await history.view(messages, [], keep_steps=50, context_length=30000, max_output=1000)
    records = request_input(view)['history']
    assert [record['step_id'] for record in records] == [2]
    assert records[0]['representation'] == 'excerpt'
    assert history.steps[0].tool_results[0].content == small


async def test_fitting_newest_tool_results_take_priority_over_oversized_reasoning():
    history = ConversationHistory()
    call = ToolCall('rules', 'conventions__list_rules', {})
    reply = Message.assistant('Reading the current rules.', [call])
    reply.reasoning = 'Extended analysis of earlier work. ' * 15000
    rules = 'COMPLETE RULES\n' + 'R' * 17186
    messages = [Message.user('Read the rules.'), reply,
                Message.tool_result(call.id, {'tool': call.name, 'call_id': call.id,
                                             'status': 'success', 'content': rules})]
    view = await history.view(messages, [], keep_steps=50, context_length=20000, max_output=1000)
    record = request_input(view)['history'][-1]
    assert record['tool_calls'][0]['result'] == rules
    assert record['excerpted_fields'] == [['reasoning']]
    assert history.steps[0].messages[1].reasoning == reply.reasoning
    assert message_tokens(view) <= int(20000 * .85) - 1000


@pytest.mark.parametrize('age', [0, 1, 4, 10])
@pytest.mark.parametrize('archived', [False, True])
async def test_reasoning_gives_way_before_smaller_tool_results(tmp_path, age, archived):
    history = ConversationHistory()
    call = ToolCall('read', 'read_file', {})
    reply = Message.assistant('Reading.', [call])
    reply.reasoning = 'Reasoning ' * 3000
    result = 'EXACT RESULT ' * 700
    messages = [Message.user('Inspect.'), reply,
                Message.tool_result(call.id, {'tool': call.name, 'call_id': call.id,
                                             'status': 'success', 'content': result})]
    for index in range(age):
        messages += [Message.user(f'Follow-up {index}'), Message.assistant('Noted.')]
    history.sync(messages)
    for step in history.steps:
        step.observed = True
    store = TurnStore(tmp_path / 'turns')
    try:
        if archived:
            for step in history.steps:
                store.archive(step, messages)
        messages.append(Message.user('Continue.'))
        view = await history.view(messages, [], keep_steps=5, context_length=18000, max_output=1000)
        record = materialize(next(record for record in request_input(view)['history'] if record['step_id'] == 1))
        assert record['tool_calls'][0]['result'] == result
        assert record['excerpted_fields'] == [['reasoning']]
        assert message_tokens(view) <= int(18000 * .85) - 1000
        assert materialize(history.steps[0].parts()['reasoning']) == 'Reasoning ' * 3000
    finally:
        store.close()


@pytest.mark.parametrize('age', [0, 4, 10])
async def test_monolithic_result_retains_short_reasoning_only_when_old(age):
    history = ConversationHistory()
    call = ToolCall('read', 'read_file', {})
    reply = Message.assistant('Reading.', [call])
    reply.reasoning = 'Useful interpretation. ' * 100
    messages = [Message.user('Inspect.'), reply,
                Message.tool_result(call.id, {'tool': call.name, 'call_id': call.id,
                                             'status': 'success', 'content': 'Huge result ' * 10000})]
    for index in range(age):
        messages += [Message.user(f'Follow-up {index}'), Message.assistant('Noted.')]
    history.sync(messages)
    for step in history.steps:
        step.observed = True
    messages.append(Message.user('Continue.'))
    view = await history.view(messages, [], keep_steps=5, context_length=18000, max_output=1000)
    record = next(record for record in request_input(view)['history'] if record['step_id'] == 1)
    if age >= 5:
        assert record['reasoning'] == reply.reasoning
        assert record['excerpted_fields'] == [['tool_results', 0, 'content']]
    else:
        assert record['excerpted_fields'] == [['reasoning'], ['tool_results', 0, 'content']]
        assert record['reasoning']['omitted_characters'] == len(reply.reasoning)
    assert record['tool_calls'][0]['result_excerpt']['omitted_characters'] > 0
