"""Oversized results shrink without erasing their turn or unrelated history."""

import pytest

from data_text_reader import request_input
from slipagent.context import ConversationHistory, RecallHistoryTool, message_tokens
from slipagent.turnstore import TurnStore
from slipagent.types import Message, ToolCall


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_only_large_result_field_is_clipped_and_all_steps_survive(tmp_path, archived, native):
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
        assert {record['step_id'] for record in payload['history']} == set(range(1, 10))
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


async def test_several_oversized_results_do_not_force_clipping_fitting_replies_or_reasoning():
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
    assert len(records) == 4
    for index, record in enumerate(records):
        assert record['agent_response'] == f"Verified fact {index}. " * 20
        assert record['reasoning'] == f"Important uncertainty {index}. " * 20
        assert record['user_prompts'] == [f"Permission for task {index}."]
        assert record['tool_calls'][0]['arguments'] == {"command": f"read {index}"}
        assert all(path[0] == 'tool_results' for path in record.get('excerpted_fields', []))


async def test_individually_oversized_new_result_is_clipped_before_fitting_older_result():
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
    assert len(records) == 2
    assert records[0]['tool_calls'][0]['unclassified_result'] == small
    assert records[0]['representation'] == 'full'
    assert records[1]['representation'] == 'excerpt'
