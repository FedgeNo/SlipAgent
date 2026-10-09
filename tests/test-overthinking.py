"""Optional thoughts belong to their step's input record, within the context budget."""

import pytest

from slipagent.config import Config
from slipagent.context import ConversationHistory, message_tokens
from slipagent.types import Message, ToolCall
from test_agent import completion, context_records, build_agent


@pytest.mark.parametrize("native_tools", [False, True])
async def test_default_includes_only_last_twenty_five_turns_thoughts(native_tools):
    messages = [Message.system("Guidance")]
    for index in range(27):
        answer = Message.assistant(f"Answer {index}")
        answer.reasoning = f"Unexpressed thought {index}"
        answer.reasoning_details = [{"type": "reasoning.encrypted", "data": "OPAQUE"}]
        messages += [Message.user(f"Question {index}"), answer]
    messages.append(Message.user("Continue"))
    originals = [message.to_api() for message in messages]
    history = ConversationHistory()
    view = await history.view(messages, [], keep_steps=50, context_length=100000,
                              max_output=1000, native_tools=native_tools)
    records = context_records(view)
    assert all("reasoning" not in record for record in records[:2] + records[-1:])
    assert [record["reasoning"] for record in records[2:-1]] == [f"Unexpressed thought {i}" for i in range(2, 27)]
    assert "OPAQUE" not in str(records)
    assert [message.to_api() for message in messages] == originals
    assert all(message.reasoning is None for message in view)


async def test_no_thoughts_is_idempotent_and_mode_can_be_disabled():
    history = ConversationHistory()
    messages = [Message.user("Question"), Message.assistant("Answer"), Message.user("Continue")]
    options = dict(keep_steps=50, context_length=100000, max_output=1000)
    enabled = await history.view(messages, [], **options)
    disabled = await history.view(messages, [], overthinking=False, **options)
    repeated = await history.view(messages, [], **options)
    assert context_records(enabled) == context_records(disabled) == context_records(repeated)
    messages[1].reasoning = "Thought"
    enabled = await history.view(messages, [], **options)
    disabled = await history.view(messages, [], overthinking=False, **options)
    assert context_records(enabled)[0]["reasoning"] == "Thought"
    assert "reasoning" not in context_records(disabled)[0]


async def test_thoughts_are_budgeted_and_oversized_turn_is_excerpted():
    answer = Message.assistant("Answer", [ToolCall("call", "read_file", {})])
    answer.reasoning = "Long thought " * 100000
    messages = [Message.user("Question"), answer, Message.tool_result("call", "Result"), Message.user("Continue")]
    history = ConversationHistory()
    view = await history.view(messages, [], keep_steps=50, context_length=16000, max_output=1000)
    records = context_records(view)
    assert records[0]["representation"] == "excerpt"
    assert "reasoning" not in records[0]
    assert message_tokens(view) < int(16000 * .85) - 1000
    assert history.steps[0].reasoning == answer.reasoning


async def test_recent_thoughts_survive_summary_selection_but_never_enter_compaction():
    history = ConversationHistory()
    messages = [Message.system("Guidance")]
    for index in range(5):
        answer = Message.assistant("Long answer " * 3000)
        answer.reasoning = f"Thought {index}"
        messages += [Message.user(f"Question {index}"), answer]
        history.sync(messages)
        history.steps[-1].summary = f"Summary {index}"
    messages.append(Message.user("Continue"))
    view = await history.view(messages, [], keep_steps=5, context_length=16000, max_output=1000)
    records = context_records(view)[:-1]
    assert any(record["representation"] == "compressed" for record in records)
    assert all(record["reasoning"] == f"Thought {record['step_id'] - 1}" for record in records)
    assert all("Thought" not in step.compaction_input() for step in history.steps)


async def test_agent_passes_mode_to_working_requests():
    first = completion("First")
    first.message.reasoning = "Provider thought"
    agent, client = build_agent([first, completion("Second")])
    await agent.run("Question")
    await agent.run("Continue")
    assert context_records(client.calls[-1]["messages"])[0]["reasoning"] == "Provider thought"


def test_mode_configuration_defaults():
    assert Config.from_env(environ={"OPENROUTER_API_KEY": "dummy"}).overthinking is True
    assert Config.from_env(environ={"OPENROUTER_API_KEY": "dummy"}, overthinking=False).overthinking is False
