"""Prefer recent originals and never spend more context on an older summary."""

from slipagent.types import content_text

import json
from data_text_reader import read_data, read_json_input
import pytest

from test_agent import unpack_context, context_records

from slipagent.context import (
    ConversationHistory, message_tokens, tool_history_as_text,
)
from slipagent.types import Message, ToolCall
from test_agent import context_body


def conversation(responses):
    history = ConversationHistory()
    messages = []
    for index, response in enumerate(responses, 1):
        messages.extend([Message.user(f"Question {index}"), Message.assistant(response)])
    history.sync(messages)
    for step in history.steps:
        step.summary = f"SUMMARY {step.id}"
    messages.append(Message.user("Current question"))
    return history, messages


async def test_only_explicit_compaction_has_a_conversation_compaction_notice():
    history, messages = conversation([f'Answer {index}' * 100 for index in range(6)])
    options = dict(keep_steps=5, context_length=100000, max_output=1000)
    ordinary = await history.view(messages, [], **options)
    assert '\n# Explicit Conversation Compaction Notice\n' not in content_text(ordinary[0].content)
    assert 'Without that notice, do not infer such an event' in content_text(ordinary[0].content)
    history.compact_through = 6
    for _ in range(2):
        compacted = await history.view(messages, [], **options)
        guidance = content_text(compacted[0].content)
        assert '\n# Explicit Conversation Compaction Notice\n' in guidance
        assert 'explicitly requested conversation compaction through step 6' in guidance
        assert 'does not announce a new compaction on each request' in guidance
        assert history.compact_through == 6


async def test_current_run_command_survives_tool_results_without_entering_history():
    history = ConversationHistory()
    messages = [Message.user("Old task"), Message.assistant("Old task finished"),
                Message.user("Current request"),
                Message.assistant("Checking", [ToolCall("check", "list_dir", {})]),
                Message.tool_result("check", "Directory is empty")]
    history.sync(messages)
    originals = [step.full_text() for step in history.steps]
    view = await history.view(messages, [], keep_steps=50,
                              context_length=1_000_000, max_output=8192)
    current = context_records(view)[-1]
    assert current["record_type"] == "current_step"
    from data_text_reader import read_json_input
    payload = read_json_input(view[-1].to_api()['content'])
    assert payload['retained_user_prompts'] == ["Current request"]
    assert current["is_tool_result_response"] is True
    assert "step_id 2" in content_text(view[0].content)
    assert 'Current request' not in content_text(view[0].content)
    assert "No new user message was supplied" not in content_text(view[0].content)
    assert payload['user_prompts'] == []
    assert "This turn includes new user input" not in content_text(view[0].content)
    assert [step.full_text() for step in history.steps] == originals
    assert all("User Request for This Run" not in step.compaction_input() for step in history.steps)
    messages.extend([Message.assistant("Done"), Message.user("New request")])
    view = await history.view(messages, [], keep_steps=50,
                              context_length=1_000_000, max_output=8192)
    assert context_records(view)[-1]["user_prompts"] == ["New request"]
    assert read_json_input(view[-1].to_api()['content'])['retained_user_prompts'] == []
    assert "step_id 4" in content_text(view[0].content)
    assert "This turn includes new user input" in content_text(view[0].content)
    assert 'additional_user_messages' not in content_text(view[0].content)
    assert "without new user input" not in content_text(view[0].content)


async def test_new_messages_are_retained_only_after_their_tool_batch():
    history = ConversationHistory()
    messages = [Message.user("Inspect the file"), Message.user("Use the literal path a'b.py")]
    options = dict(keep_steps=50, context_length=100000, max_output=1000)
    fresh = read_json_input((await history.view(messages, [], **options))[-1].to_api()['content'])
    assert fresh['user_prompts'] == ["Inspect the file", "Use the literal path a'b.py"]
    assert fresh['retained_user_prompts'] == []
    fresh_system = content_text((await history.view(messages, [], **options))[0].content)
    assert 'later messages supersede conflicting earlier instructions' in fresh_system
    messages.extend([Message.assistant("Reading", [ToolCall('r', 'read_file', {'path': "a'b.py"})]),
                     Message.tool_result('r', 'File contents')])
    continuation = read_json_input((await history.view(messages, [], **options))[-1].to_api()['content'])
    assert continuation['user_prompts'] == []
    assert continuation['retained_user_prompts'] == fresh['user_prompts']
    continued_system = content_text((await history.view(messages, [], **options))[0].content)
    assert "`retained_user_prompts` contains the last request. Use that request and the previous turn's tool calls" in continued_system
    assert 'This turn includes new user input' not in continued_system
    assert '`additional_user_messages`' not in continued_system


async def test_continuation_after_an_answer_does_not_claim_tool_results():
    history = ConversationHistory()
    messages = [Message.user('Explain'), Message.assistant('Explanation')]
    view = await history.view(messages, [], keep_steps=50, context_length=100000, max_output=1000)
    from data_text_reader import read_json_input
    payload = read_json_input(view[-1].to_api()['content'])
    assert payload['user_prompts'] == []
    assert payload['retained_user_prompts'] == ['Explain']
    assert '`retained_user_prompts` contains the last request. Use that request and the supplied history' in content_text(view[0].content)
    assert 'Continuation After Tools' not in content_text(view[0].content)


async def test_large_recent_originals_use_the_model_allowance_above_200000_tokens():
    answers = [f"Answer {index} " * 40_000 for index in range(5)]
    history, messages = conversation(answers)
    view = await history.view(messages, [], keep_steps=50,
                              context_length=1_000_000, max_output=8192)
    supplied = [context_body(message.content) for message in unpack_context(view) if message.role == "assistant"]
    assert supplied == answers
    assert 200_000 < message_tokens(view) < 850_000 - 8192


@pytest.mark.parametrize('archived', [False, True])
@pytest.mark.parametrize('native', [False, True])
async def test_last_five_tool_turns_displace_old_summaries_before_losing_results(tmp_path, archived, native):
    from data_text_reader import request_input
    from slipagent.turnstore import TurnStore
    history, messages = ConversationHistory(), []
    outputs = []
    for index in range(25):
        call = ToolCall(f'call-{index}', 'conventions__list_rules' if index == 24 else 'read_file', {})
        output = f'COMPLETE RESULT {index}\n' + 'R' * (17186 if index == 24 else 4000)
        outputs.append(output)
        messages += [Message.user(f'Inspect item {index}'),
                     Message.assistant(f'Reading item {index}.', [call]),
                     Message.tool_result(call.id, {'tool': call.name, 'call_id': call.id,
                                                  'status': 'success', 'content': output})]
    history.sync(messages)
    for step in history.steps[:-1]:
        step.observed = True
        step.summary = f'Earlier summary {step.id}. ' * (100 if step.id <= 20 else 1)
    store = TurnStore(tmp_path / 'turns')
    try:
        if archived:
            for step in history.steps:
                store.archive(step, messages)
        originals = [step.full_text() for step in history.steps]
        view = await history.view(messages, [], keep_steps=50, context_length=30000,
                                  max_output=1000, native_tools=native)
        records = read_json_input(view[-1].to_api()['content'])['history']
        recent = [record for record in records if record['step_id'] >= 21]
        assert [record['step_id'] for record in recent] == list(range(21, 26))
        assert all(record['representation'] == 'full' for record in recent)
        assert [record['tool_calls'][0]['result'] for record in recent] == outputs[-5:]
        assert len(records) < len(history.steps)
        assert message_tokens(view) <= int(30000 * .85) - 1000
        assert [step.full_text() for step in history.steps] == originals
    finally:
        store.close()


async def test_five_turn_window_reduces_older_results_before_the_latest_one():
    from data_text_reader import request_input
    history, messages = ConversationHistory(), []
    newest = 'COMPLETE NEWEST RESULT ' * 600
    for index in range(5):
        call = ToolCall(str(index), 'read_file', {})
        output = newest if index == 4 else f'OLDER RESULT {index} ' * 1200
        messages += [Message.user(f'Read {index}'), Message.assistant('Reading.', [call]),
                     Message.tool_result(call.id, {'tool': call.name, 'call_id': call.id,
                                                  'status': 'success', 'content': output})]
    history.sync(messages)
    for step in history.steps[:-1]:
        step.observed = True
        step.summary = f'Older durable summary {step.id}. ' * 300
    view = await history.view(messages, [], keep_steps=50, context_length=20000, max_output=1000)
    records = request_input(view)['history']
    assert records[-1]['step_id'] == 5
    assert records[-1]['representation'] == 'full'
    assert records[-1]['tool_calls'][0]['result'] == newest
    assert all(record['representation'] == 'compressed' for record in records[:-1])
    # An older summary that cannot close the shortfall alone is omitted intact,
    # never clipped to a stub; the remaining older summaries stay whole.
    assert not any(record.get('excerpted_fields') for record in records[:-1])
    assert 0 < len(records) - 1 < 4
    assert [record['step_id'] for record in records] == list(range(6 - len(records), 6))
    assert message_tokens(view) <= int(20000 * .85) - 1000


@pytest.mark.parametrize('archived', [False, True])
@pytest.mark.parametrize('native', [False, True])
async def test_recent_edit_diffs_remain_complete_when_older_history_is_reduced(workspace, tmp_path, archived, native):
    from slipagent.tools.files import EditFileTool
    from slipagent.turnstore import TurnStore

    history, messages = ConversationHistory(), []
    outputs = []
    for index in range(25):
        name = 'read_file'
        arguments = {'path': 'example.txt'}
        output = f'Older observation {index}: ' + 'x' * 5000
        if index >= 20:
            name = 'edit_file'
            before = ''.join(f'old {index} {line}: ' + 'a' * 60 + '\n' for line in range(80))
            after = ''.join(f'new {index} {line}: ' + 'b' * 60 + '\n' for line in range(80))
            (workspace.root / 'example.txt').write_text(before)
            arguments.update(old_string=before, new_string=after)
            result = await EditFileTool(workspace).invoke(arguments)
            assert not result.is_error
            output = result.text()
            assert '-' + before.splitlines()[0] in output
            assert '+' + after.splitlines()[-1] in output
            assert 'Diff truncated' not in output
            outputs.append(output)
        call = ToolCall(f'call-{index}', name, arguments)
        messages += [Message.user(f'Inspect or edit {index}'), Message.assistant('Working.', [call]),
                     Message.tool_result(call.id, {'tool': name, 'call_id': call.id,
                                                  'status': 'success', 'content': output})]
    history.sync(messages)
    for step in history.steps:
        step.observed = True
        step.summary = f'Short summary {step.id}'
    store = TurnStore(tmp_path / 'edit-turns')
    try:
        if archived:
            for step in history.steps:
                store.archive(step, messages)
        originals = [step.full_text() for step in history.steps]
        view = await history.view(messages, [], keep_steps=50, context_length=65000,
                                  max_output=1000, native_tools=native)
        records = read_json_input(view[-1].to_api()['content'])['history']
        recent = [record for record in records if record['step_id'] >= 21]
        assert [record['step_id'] for record in recent] == list(range(21, 26))
        assert all(record['representation'] == 'full' for record in recent)
        assert [record['tool_calls'][0]['result'] for record in recent] == outputs
        assert any(record['representation'] != 'full' for record in records if record['step_id'] < 21)
        assert message_tokens(view) <= int(65000 * .85) - 1000
        assert [step.full_text() for step in history.steps] == originals
    finally:
        store.close()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("boundary", [0, 8])
async def test_oversized_recent_step_displaces_older_history_before_it_is_excerpted(native, boundary):
    history, messages = conversation([f"Useful earlier fact {index}. " * 20 for index in range(8)]
                                     + ["OVERSIZED UNSUMMARIZED RESULT " * 10000]
                                     + ["Recent fact one.", "Recent fact two."])
    history.steps[8].summary = None
    history.compact_through = boundary
    originals = [step.full_text() for step in history.steps]
    view = await history.view(messages, [], keep_steps=50, context_length=24000,
                              max_output=1000, native_tools=native)
    records = [record for record in context_records(view) if record['record_type'] == 'history_step']
    assert {record['step_id'] for record in records} == {9, 10, 11}
    oversized = next(record for record in records if record['step_id'] == 9)
    assert oversized['representation'] == 'excerpt'
    assert oversized['excerpted_fields'] == [['agent_response']]
    assert message_tokens(view) <= 24000 * .85 - 1000
    assert history.compact_through == boundary
    if boundary:
        assert all(record['representation'] == 'compressed' for record in records if record['step_id'] <= boundary)
    assert [step.full_text() for step in history.steps] == originals


async def test_ready_summary_does_not_change_a_fitting_original_record():
    history = ConversationHistory()
    messages = [Message.user('Inspect'), Message.assistant('Reading', [ToolCall('r', 'read_file', {'path': 'a.py'})]),
                Message.tool_result('r', 'EXACT ORIGINAL'), Message.user('Continue')]
    options = dict(keep_steps=50, context_length=100000, max_output=1000)
    pending = await history.view(messages, [], **options)
    original = history.steps[0].full_text()
    assert 'compressed_summary' not in context_records(pending)[0]
    history.steps[0].summary = 'Observations: The file contained EXACT ORIGINAL.'
    ready = await history.view(messages, [], **options)
    records = context_records(ready)
    assert len(records) == 2
    assert records[0] == context_records(pending)[0]
    assert records[1]['record_type'] == 'current_step'
    assert 'compressed_summary' not in records[1]
    assert message_tokens(ready) == message_tokens(pending)
    assert history.steps[0].full_text() == original


async def test_selected_history_is_budgeted_and_does_not_duplicate_steps():
    history, messages = conversation(['Original ' * 400 for _ in range(12)])
    for step in history.steps:
        step.summary = 'Observations: ' + 'Analysis ' * 500
    view = await history.view(messages, [], keep_steps=50, context_length=20000, max_output=1000)
    records = context_records(view)[:-1]
    assert len({record['step_id'] for record in records}) == len(records)
    assert 0 < len(records) < 12
    assert records[-1]['step_id'] == 12
    assert records[-1]['representation'] == 'full'
    assert any(record['representation'] == 'full' for record in records)
    for record in records:
        if record['representation'] == 'full':
            assert 'compressed_summary' not in record
            assert record['agent_response'] == 'Original ' * 400
    assert message_tokens(view) <= int(20000 * .85) - 1000
    assert len(history.steps) == 12


async def test_history_that_fits_is_never_summarized():
    answers = [f"ANSWER {index} " * 200 for index in range(8)]
    history, messages = conversation(answers)
    view = await history.view(messages, [], keep_steps=1,
                              context_length=1_000_000, max_output=8192)
    supplied = [context_body(message.content) for message in unpack_context(view) if message.role == "assistant"]
    assert supplied == answers
    assert all('compressed_summary' not in record for record in context_records(view)
               if record['record_type'] == 'history_step')


@pytest.mark.parametrize('archived', [False, True])
@pytest.mark.parametrize('native', [False, True])
async def test_unused_large_summaries_do_not_consume_the_original_history_budget(tmp_path, archived, native):
    from data_text_reader import request_input
    from slipagent.turnstore import TurnStore

    answers = [f'EXACT ORIGINAL {index}' for index in range(8)]
    history, messages = conversation(answers)
    for step in history.steps:
        step.summary = 'UNUSED SUMMARY ' * 10000
    store = TurnStore(tmp_path / 'turns')
    try:
        if archived:
            for step in history.steps:
                store.archive(step, messages)
        view = await history.view(messages, [], keep_steps=5, context_length=16000,
                                  max_output=1000, native_tools=native, text_tool_history=not native)
        records = request_input(view)['history']
        assert [record['agent_response'] for record in records] == answers
        assert all(record['representation'] == 'full' and 'compressed_summary' not in record for record in records)
        assert 'UNUSED SUMMARY' not in content_text(view[-1].content)
        assert message_tokens(view) <= int(16000 * .85) - 1000
        assert all(step.summary == 'UNUSED SUMMARY ' * 10000 for step in history.steps)
    finally:
        store.close()


async def test_over_budget_summarizes_only_the_oldest_steps_needed():
    answers = [f"ANSWER {index} " * 2000 for index in range(9)]
    history, messages = conversation(answers)
    full = await history.view(messages, [], keep_steps=50, context_length=1_000_000, max_output=1000)
    # Leave room for roughly six of the nine originals.
    length = int((message_tokens(full) - 2 * message_tokens([Message.assistant(answers[0])]) + 1000) / .85)
    view = await history.view(messages, [], keep_steps=50, context_length=length, max_output=1000)
    supplied = [context_body(message.content) for message in unpack_context(view) if message.role == "assistant"]
    summarized = [text for text in supplied if text.startswith("SUMMARY")]
    assert 0 < len(summarized) <= 3
    assert supplied == [f"SUMMARY {index}" for index in range(1, len(summarized) + 1)] + answers[len(summarized):]
    assert message_tokens(view) <= int(length * .85) - 1000


@pytest.mark.parametrize('representation,span,attribute', [
    ('historical_summary', 20, 'historical_summary'),
    ('era_summary', 100, 'era_summary'),
])
async def test_older_groups_preserve_the_newest_third(representation, span, attribute):
    history, messages = conversation([f'Original {index}: ' + 'x' * 4000 for index in range(180)])
    for step in history.steps:
        step.summary = f'Summary {step.id}: ' + 's' * 2000
        step.observed = True
    history.compact_through = 120
    individual = await history.view(messages, [], keep_steps=5,
                                    context_length=1_000_000, max_output=1000)
    history.compact_through = 0
    record = {'record_type': 'history_step', 'representation': representation,
              'step_id': span, 'step_ids': list(range(1, span + 1)),
              'compressed_summary': 'Older durable findings.', 'created_at_step': 180}
    setattr(history.steps[span - 1], attribute, record)
    # The input allowance is 1,000 tokens below the individually summarized view.
    length = int(message_tokens(individual) / .85)
    view = await history.view(messages, [], keep_steps=5,
                              context_length=length, max_output=1000)
    records = [record for record in context_records(view) if record['record_type'] == 'history_step']
    assert any(record['representation'] == representation for record in records)
    recent = [record for record in records if record['step_id'] > 120]
    assert len(recent) == 60
    assert all(record['representation'] == 'full' for record in recent)
    assert message_tokens(view) <= int(length * .85) - 1000


@pytest.mark.parametrize('archived', [False, True])
@pytest.mark.parametrize('native', [False, True])
async def test_budget_selection_sends_exactly_one_oldest_summary(tmp_path, archived, native):
    from data_text_reader import request_input
    from slipagent.turnstore import TurnStore

    answers = [f'ORIGINAL ANSWER {index} ' * 3000 for index in range(8)]
    history, messages = conversation(answers)
    store = TurnStore(tmp_path / 'turns')
    try:
        if archived:
            for step in history.steps:
                store.archive(step, messages)
        options = dict(keep_steps=5, max_output=1000, native_tools=native, text_tool_history=not native)
        full = await history.view(messages, [], context_length=1_000_000, **options)
        first = history.steps[0]
        saving = message_tokens(first.context_messages()) - message_tokens(first.context_messages(compressed=True))
        length = int((message_tokens(full) - saving // 2 + 1000) / .85)
        view = await history.view(messages, [], context_length=length, **options)
        records = request_input([message.to_api() for message in view])['history']
        assert [record['representation'] for record in records] == ['compressed', *['full'] * 7]
        assert records[0]['compressed_summary'] == 'SUMMARY 1'
        assert 'agent_response' not in records[0]
        assert [record['agent_response'] for record in records[1:]] == answers[1:]
        assert all('compressed_summary' not in record for record in records[1:])
        assert message_tokens(view) <= int(length * .85) - 1000
        expanded = await history.view(messages, [], context_length=1_000_000, **options)
        assert all(record['representation'] == 'full' and 'compressed_summary' not in record
                   for record in request_input([message.to_api() for message in expanded])['history'])
    finally:
        store.close()


@pytest.mark.parametrize("native_tools", [False, True])
async def test_larger_summary_falls_back_to_original_roles_and_tool_pairing(native_tools):
    history = ConversationHistory()
    messages = [Message.user("Read it"), Message.assistant("Reading", [ToolCall("read", "read_file", {"path": "a.py"})]),
                Message.tool_result("read", "EXACT TOOL RESULT")]
    messages += [m for index in range(5) for m in (Message.user(f"Next {index}"), Message.assistant(f"Done {index}"))]
    history.sync(messages)
    history.steps[0].summary = "INFLATED SUMMARY " * 1000
    messages.append(Message.user("Current question"))
    originals = [step.full_text() for step in history.steps]
    view = await history.view(messages, [], keep_steps=5,
                              context_length=1_000_000, max_output=8192,
                              native_tools=native_tools, text_tool_history=not native_tools)
    restored = unpack_context(view)
    assert [message.role for message in restored[1:4]] == ["user", "assistant", "tool"]
    assert restored[2].tool_calls == messages[1].tool_calls
    assert restored[3].tool_call_id == "read" and context_body(restored[3].content) == "EXACT TOOL RESULT"
    assert 'compressed_summary' not in context_records(view)[0]
    assert [step.full_text() for step in history.steps] == originals


async def test_hard_allowance_can_reduce_the_full_window_below_five():
    answers = [f"ANSWER {index} " * 1500 for index in range(5)]
    history, messages = conversation(answers)
    view = await history.view(messages, [], keep_steps=50,
                              context_length=14_000, max_output=1000)
    supplied = [context_body(message.content) for message in unpack_context(view) if message.role == "assistant"]
    assert answers[-1] in supplied
    assert 0 < sum(answer in supplied for answer in answers) < 5
    assert message_tokens(view) <= 14_000 * .85 - 1000


async def test_large_original_is_summarized_only_once_the_budget_requires_it():
    answers = ["LARGE FIRST ANSWER " * 40_000]
    history, messages = conversation(answers)
    first = await history.view(messages, [], keep_steps=50,
                               context_length=400_000, max_output=8192)
    assert any(context_body(message.content) == answers[0] for message in unpack_context(first))
    messages.extend([Message.assistant("Small answer"), Message.user("New question")])
    second = await history.view(messages, [], keep_steps=50,
                                context_length=400_000, max_output=8192)
    assert any(context_body(message.content) == answers[0] for message in unpack_context(second))
    messages.extend([Message.assistant("LARGE SECOND ANSWER " * 20_000), Message.user("Third question")])
    third = await history.view(messages, [], keep_steps=50,
                               context_length=400_000, max_output=8192)
    wire = "\n".join(content_text(message.content) for message in unpack_context(third))
    assert "LARGE FIRST ANSWER" not in wire and "SUMMARY 1" in wire
    assert "Small answer" in wire and "LARGE SECOND ANSWER" in wire
    assert answers[0] in history.steps[0].full_text()


async def test_newest_full_turn_displaces_old_history_until_it_ages_out():
    answers = [f"EARLY ANSWER {index} " * 5000 for index in range(10)]
    answers += [f"Recent answer {index}" for index in range(49)]
    answers.append("LARGE RECENT ANSWER " * 10_000)
    history, messages = conversation(answers)
    for step in history.steps[:10]:
        step.summary = f"EARLY SUMMARY {step.id}: " + "detail " * 500
    first = await history.view(messages, [], keep_steps=50, context_length=90_000, max_output=1000)
    wire = "\n".join(content_text(message.content) for message in unpack_context(first))
    assert "EARLY SUMMARY 1:" not in wire
    assert answers[-1] in wire
    assert context_records(first)[-2]['representation'] == 'full'
    for index in range(50):
        messages.extend([Message.assistant(f"Later answer {index}"), Message.user(f"Later question {index}")])
    second = await history.view(messages, [], keep_steps=50, context_length=90_000, max_output=1000)
    restored = "\n".join(content_text(message.content) for message in unpack_context(second))
    assert "EARLY SUMMARY 1:" in restored
    assert "LARGE RECENT ANSWER" not in restored
    assert "SUMMARY 60" in restored
    assert message_tokens(first) <= 90_000 * .85 - 1000
    assert message_tokens(second) <= 90_000 * .85 - 1000


@pytest.mark.parametrize("token_scale", [1.0, 2.7])
async def test_equal_cost_summary_keeps_the_original(token_scale):
    history, messages = conversation(["ORIGINAL " * 100] + [f"Recent {index}" for index in range(5)])
    step = history.steps[0]
    original_cost = message_tokens(step.context_messages())
    for length in range(2000):
        step.summary = "S" * length
        compressed = step.context_messages(compressed=True)[0]
        if message_tokens([compressed]) == original_cost:
            break
    else:
        pytest.fail("fixture must have equal full and compressed token costs")
    view = await history.view(messages, [], keep_steps=5, context_length=1_000_000,
                              max_output=8192, token_scale=token_scale)
    assert context_records(view)[0]["agent_response"] == "ORIGINAL " * 100
    assert 'compressed_summary' not in context_records(view)[0]


async def test_mixed_older_originals_and_summaries_keep_chronological_order():
    history, messages = conversation(["FIRST " * 300_000, "Second original", "THIRD " * 300_000]
                                     + [f"Recent {index}" for index in range(5)])
    history.steps[1].summary = "EXPANDED SECOND SUMMARY " * 300
    view = await history.view(messages, [], keep_steps=5, context_length=1_000_000, max_output=8192)
    wire = "\n".join(content_text(message.content) for message in unpack_context(view))
    # The larger summary keeps step 2's original; step 3 stays full once step 1 fits.
    assert wire.index("SUMMARY 1") < wire.index("Second original") < wire.index("THIRD ") < wire.index("Recent 2")
    assert "FIRST " not in wire
    assert 'compressed_summary' not in context_records(view)[1]
    assert [context_body(message.content) for message in unpack_context(view) if message.role == "user"] == [
        *[f"Question {index}" for index in range(2, 9)], "Current question",
    ]


async def test_summary_comparison_uses_the_same_json_records_for_all_tool_profiles():
    history = ConversationHistory()
    messages = [Message.user("Read"), Message.assistant("Reading", [
        ToolCall("a", "record", {"value": '"nested"\\value' * 3000}),
    ]), Message.tool_result("a", "Result")]
    messages += [Message.assistant(f"Recent {index}") for index in range(5)]
    history.sync(messages)
    messages.append(Message.user("Next"))
    step = history.steps[0]
    small = 16_000  # Too small for the original step, large enough for its summary.
    original = step.context_messages()
    native_cost = message_tokens(original)
    text_cost = message_tokens(tool_history_as_text(original))
    assert native_cost == text_cost
    step.summary = "The recorded tool returned Result."
    assert message_tokens(step.context_messages(compressed=True)) < native_cost
    for native in (True, False):
        view = await history.view(messages, [], keep_steps=5, context_length=small, max_output=256,
                                  native_tools=native, text_tool_history=not native)
        assert context_records(view)[0] == {"record_type": "history_step", "representation": "compressed", "step_id": step.id,
                                            "run": 1, "compressed_summary": step.summary}


def archived_code_history(lines):
    from slipagent.toolbatch import tool_message
    from slipagent.tools.base import ToolResult
    from slipagent.turnstore import TurnStore

    code = "".join(f"    def f{i}(x):\n        return x + {i}\n" for i in range(lines))
    call = ToolCall("c1", "read_file", {"path": "a.py"})
    messages = [Message.system("System."), Message.user("Read it"), Message.assistant("Reading.", [call]),
                tool_message(call, ToolResult.ok(code))]
    history = ConversationHistory()
    history.sync(messages)
    history.store = TurnStore()
    history.store.archive(history.steps[0], messages)
    history.steps[0].observed = True
    messages.append(Message.user("Next question"))
    return history, messages, code


async def test_history_is_costed_in_its_sent_form():
    from slipagent.records import record_message

    history, messages, code = archived_code_history(3000)
    try:
        labelled = message_tokens([record_message(history.steps[0].context_messages()[0].content)])

        async def included(length):
            view = await history.view(messages, [], keep_steps=10, context_length=length, max_output=256)
            return code[-200:] in content_text(view[-1].content)

        low, high = 4096, 400_000
        while low < high:
            middle = (low + high) // 2
            low, high = (low, middle) if await included(middle) else (middle + 1, high)
        # The step is kept once its sent size fits, below its labelled-text size.
        assert int(low * .85) - 256 < labelled
    finally:
        history.clear()


async def test_archived_history_costs_are_measured_once(monkeypatch):
    from slipagent.sources import FileText

    history, messages, code = archived_code_history(300)
    reads = []
    original = FileText.chunks
    monkeypatch.setattr(FileText, "chunks", lambda self: (reads.append(self.characters), original(self))[1])
    try:
        await history.view(messages, [], keep_steps=10, context_length=200_000, max_output=256)
        first = reads.count(len(code))
        reads.clear()
        await history.view(messages, [], keep_steps=10, context_length=200_000, max_output=256)
        # Building the view no longer rereads selected history for a final check.
        assert first == 2 and reads.count(len(code)) == 0
    finally:
        history.clear()


async def test_surplus_from_the_last_reduction_restores_the_newest_summarized_originals():
    # The walk must summarize the large turn 15 to fit, which frees far more than
    # the shortfall; that surplus returns originals to the newest small turns.
    answers = [f"Older detail {index} " * 160 for index in range(14)]
    answers += ["LARGE RESULT " * 6000]
    answers += [f"Recent answer {index}" for index in range(5)]
    history, messages = conversation(answers)
    full = await history.view(messages, [], keep_steps=5, context_length=4_000_000, max_output=1000)
    # The shortfall exceeds what the small turns' summaries save, so turn 15 must
    # be summarized; its saving then leaves room for only some small originals.
    length = int((message_tokens(full) - 32_000 + 1000) / .85)
    view = await history.view(messages, [], keep_steps=5, context_length=length, max_output=1000)
    records = {record['step_id']: record['representation'] for record in context_records(view)
               if record['record_type'] == 'history_step'}
    assert records[15] == 'compressed'
    assert all(records[step] == 'full' for step in range(16, 21))
    restored = [step for step in range(1, 15) if records[step] == 'full']
    # Newest first: the restored originals are the newest of the small turns.
    assert restored and restored == list(range(15 - len(restored), 15))
    assert records[1] == 'compressed'
    assert message_tokens(view) <= int(length * .85) - 1000
