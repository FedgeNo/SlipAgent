"""Prefer recent originals and never spend more context on an older summary."""

from slipagent.types import content_text

import json
from data_text_reader import read_data
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
    retained = content_text(view[0].content).split('## Original Messages\n\n', 1)[1]
    request_text = retained.split('```text\n', 1)[1].split('\n```', 1)[0]
    assert read_data(request_text) == ["Current request"]
    assert current["is_tool_result_response"] is True
    assert "step_id 2" in content_text(view[0].content)
    assert "overall goal for this run" in content_text(view[0].content)
    assert "multiple steps ago" in content_text(view[0].content)
    assert "No new user message was supplied" in content_text(view[0].content)
    assert view[-1].content == ""
    assert "This turn includes new user input" not in content_text(view[0].content)
    assert [step.full_text() for step in history.steps] == originals
    assert all("User Request for This Run" not in step.compaction_input() for step in history.steps)
    messages.extend([Message.assistant("Done"), Message.user("New request")])
    view = await history.view(messages, [], keep_steps=50,
                              context_length=1_000_000, max_output=8192)
    assert context_records(view)[-1]["user_prompt"] == ["New request"]
    assert "step_id 4" in content_text(view[0].content)
    assert "This turn includes new user input" in content_text(view[0].content)
    assert "without new user input" not in content_text(view[0].content)


async def test_large_recent_originals_use_the_model_allowance_above_200000_tokens():
    answers = [f"Answer {index} " * 20_000 for index in range(5)]
    history, messages = conversation(answers)
    view = await history.view(messages, [], keep_steps=50,
                              context_length=1_000_000, max_output=8192)
    supplied = [context_body(message.content) for message in unpack_context(view) if message.role == "assistant"]
    assert supplied == answers
    assert 200_000 < message_tokens(view) < 850_000 - 8192


async def test_ready_summary_joins_original_in_one_record_without_rewriting_history():
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
    assert records[0] == {**context_records(pending)[0], 'compressed_summary': history.steps[0].summary}
    assert records[1]['record_type'] == 'current_step'
    assert 'compressed_summary' not in records[1]
    assert message_tokens(ready) > message_tokens(pending)
    assert history.steps[0].full_text() == original


async def test_combined_history_is_budgeted_and_does_not_duplicate_steps():
    history, messages = conversation(['Original ' * 400 for _ in range(12)])
    for step in history.steps:
        step.summary = 'Observations: ' + 'Analysis ' * 500
    view = await history.view(messages, [], keep_steps=50, context_length=20000, max_output=1000)
    records = context_records(view)[:-1]
    assert len({record['step_id'] for record in records}) == len(records)
    assert len(records) < 12
    assert any(record['representation'] == 'full' for record in records)
    for record in records:
        if record['representation'] == 'full':
            assert record['compressed_summary'] == history.steps[record['step_id'] - 1].summary
            assert record['agent_response'] == 'Original ' * 400
    assert message_tokens(view) <= int(20000 * .85) - 1000
    assert len(history.steps) == 12


async def test_five_recent_originals_take_priority_over_older_history():
    answers = [f"ANSWER {index} " * 200 for index in range(8)]
    history, messages = conversation(answers)
    view = await history.view(messages, [], keep_steps=1,
                              context_length=1_000_000, max_output=8192)
    supplied = [context_body(message.content) for message in unpack_context(view) if message.role == "assistant"]
    assert supplied[-5:] == answers[-5:]
    assert all(answer not in supplied for answer in answers[:-5])


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
    assert context_records(view)[0]['compressed_summary'] == history.steps[0].summary
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


async def test_ageing_out_large_original_recovers_context_on_the_next_request():
    answers = ["LARGE FIRST ANSWER " * 40_000] + [f"Answer {index}" for index in range(49)]
    history, messages = conversation(answers)
    first = await history.view(messages, [], keep_steps=50,
                               context_length=1_000_000, max_output=8192)
    assert any(context_body(message.content) == answers[0] for message in unpack_context(first))
    messages.extend([Message.assistant("Another answer"), Message.user("New question")])
    second = await history.view(messages, [], keep_steps=50,
                                context_length=1_000_000, max_output=8192)
    wire = "\n".join(content_text(message.content) for message in unpack_context(second))
    assert "LARGE FIRST ANSWER" not in wire and "SUMMARY 1" in wire
    assert message_tokens(second) < message_tokens(first) / 10
    assert answers[0] in history.steps[0].full_text()


async def test_omitted_older_history_returns_after_large_turn_ages_out():
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
    assert context_records(view)[0]['compressed_summary'] == step.summary


async def test_mixed_older_originals_and_summaries_keep_chronological_order():
    history, messages = conversation(["FIRST " * 500, "Second original", "THIRD " * 500]
                                     + [f"Recent {index}" for index in range(5)])
    history.steps[1].summary = "EXPANDED SECOND SUMMARY " * 300
    view = await history.view(messages, [], keep_steps=5, context_length=1_000_000, max_output=8192)
    wire = "\n".join(content_text(message.content) for message in unpack_context(view))
    assert wire.index("SUMMARY 1") < wire.index("Second original") < wire.index("SUMMARY 3") < wire.index("Recent 0")
    assert "FIRST " not in wire and "THIRD " not in wire
    assert context_records(view)[1]['compressed_summary'] == history.steps[1].summary
    assert [context_body(message.content) for message in unpack_context(view) if message.role == "user"] == [
        "Question 2", *[f"Question {index}" for index in range(4, 9)], "Current question",
    ]


async def test_summary_comparison_uses_the_same_json_records_for_all_tool_profiles():
    history = ConversationHistory()
    messages = [Message.user("Read"), Message.assistant("Reading", [
        ToolCall("a", "record", {"value": '"nested"\\value' * 100}),
    ]), Message.tool_result("a", "Result")]
    messages += [Message.assistant(f"Recent {index}") for index in range(5)]
    history.sync(messages)
    messages.append(Message.user("Next"))
    step = history.steps[0]
    original = step.context_messages()
    native_cost = message_tokens(original)
    text_cost = message_tokens(tool_history_as_text(original))
    assert native_cost == text_cost
    step.summary = "The recorded tool returned Result."
    assert message_tokens(step.context_messages(compressed=True)) < native_cost
    for native in (True, False):
        view = await history.view(messages, [], keep_steps=5, context_length=1_000_000, max_output=8192,
                                  native_tools=native, text_tool_history=not native)
    assert context_records(view)[0] == {"record_type": "history_step", "representation": "compressed", "step_id": step.id, "compressed_summary": step.summary}
