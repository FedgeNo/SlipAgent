"""Prefer recent originals and never spend more context on an older summary."""

import pytest

from slipagent.context import (
    COMPRESSED_HISTORY_HEADING, ConversationHistory, message_tokens, tool_history_as_text,
)
from slipagent.types import Message, ToolCall
from test_agent import context_body


def conversation(responses):
    history = ConversationHistory()
    messages = []
    for index, response in enumerate(responses, 1):
        messages.extend([Message.user(f"Question {index}"), Message.assistant(response)])
    history.sync(messages)
    for post in history.posts:
        post.summary = f"SUMMARY {post.id}"
    messages.append(Message.user("Current question"))
    return history, messages


async def test_large_recent_originals_use_the_model_allowance_above_200000_tokens():
    answers = [f"Answer {index} " * 20_000 for index in range(5)]
    history, messages = conversation(answers)
    view = await history.view(messages, [], keep_posts=50,
                              context_length=1_000_000, max_output=8192)
    supplied = [context_body(message.content) for message in view if message.role == "assistant"]
    assert supplied == answers
    assert 200_000 < message_tokens(view) < 850_000 - 8192


async def test_five_recent_originals_take_priority_over_older_history():
    answers = [f"ANSWER {index} " * 200 for index in range(8)]
    history, messages = conversation(answers)
    view = await history.view(messages, [], keep_posts=1,
                              context_length=1_000_000, max_output=8192)
    supplied = [context_body(message.content) for message in view if message.role == "assistant"]
    assert supplied[-5:] == answers[-5:]
    assert all(answer not in supplied for answer in answers[:-5])


@pytest.mark.parametrize("native_tools", [False, True])
async def test_larger_summary_falls_back_to_original_roles_and_tool_pairing(native_tools):
    history = ConversationHistory()
    messages = [Message.user("Read it"), Message.assistant("Reading", [ToolCall("read", "read_file", {"path": "a.py"})]),
                Message.tool_result("read", "EXACT TOOL RESULT")]
    messages += [m for index in range(5) for m in (Message.user(f"Next {index}"), Message.assistant(f"Done {index}"))]
    history.sync(messages)
    history.posts[0].summary = "INFLATED SUMMARY " * 1000
    messages.append(Message.user("Current question"))
    originals = [post.full_text() for post in history.posts]
    view = await history.view(messages, [], keep_posts=5,
                              context_length=1_000_000, max_output=8192,
                              native_tools=native_tools, text_tool_history=not native_tools)
    assert [message.role for message in view[1:4]] == ["user", "assistant", "tool"]
    assert view[2].tool_calls == messages[1].tool_calls
    assert view[3].tool_call_id == "read" and context_body(view[3].content) == "EXACT TOOL RESULT"
    assert "INFLATED SUMMARY" not in "\n".join(message.content or "" for message in view)
    assert [post.full_text() for post in history.posts] == originals


async def test_hard_allowance_can_reduce_the_full_window_below_five():
    answers = [f"ANSWER {index} " * 1500 for index in range(5)]
    history, messages = conversation(answers)
    view = await history.view(messages, [], keep_posts=50,
                              context_length=14_000, max_output=1000)
    supplied = [context_body(message.content) for message in view if message.role == "assistant"]
    assert answers[-1] in supplied
    assert 0 < sum(answer in supplied for answer in answers) < 5
    assert message_tokens(view) <= 14_000 * .85 - 1000


async def test_ageing_out_large_original_recovers_context_on_the_next_request():
    answers = ["LARGE FIRST ANSWER " * 40_000] + [f"Answer {index}" for index in range(49)]
    history, messages = conversation(answers)
    first = await history.view(messages, [], keep_posts=50,
                               context_length=1_000_000, max_output=8192)
    assert any(context_body(message.content) == answers[0] for message in first)
    messages.extend([Message.assistant("Another answer"), Message.user("New question")])
    second = await history.view(messages, [], keep_posts=50,
                                context_length=1_000_000, max_output=8192)
    wire = "\n".join(message.content or "" for message in second)
    assert "LARGE FIRST ANSWER" not in wire and "SUMMARY 1" in wire
    assert message_tokens(second) < message_tokens(first) / 10
    assert answers[0] in history.posts[0].full_text()


async def test_omitted_older_history_returns_after_large_turn_ages_out():
    answers = [f"EARLY ANSWER {index} " * 5000 for index in range(10)]
    answers += [f"Recent answer {index}" for index in range(49)]
    answers.append("LARGE RECENT ANSWER " * 10_000)
    history, messages = conversation(answers)
    for post in history.posts[:10]:
        post.summary = f"EARLY SUMMARY {post.id}: " + "detail " * 500
    first = await history.view(messages, [], keep_posts=50, context_length=90_000, max_output=1000)
    wire = "\n".join(message.content or "" for message in first)
    assert "EARLY SUMMARY 1:" not in wire
    assert answers[-1] in wire
    for index in range(50):
        messages.extend([Message.assistant(f"Later answer {index}"), Message.user(f"Later question {index}")])
    second = await history.view(messages, [], keep_posts=50, context_length=90_000, max_output=1000)
    restored = "\n".join(message.content or "" for message in second)
    assert "EARLY SUMMARY 1:" in restored
    assert "LARGE RECENT ANSWER" not in restored
    assert "SUMMARY 60" in restored
    assert message_tokens(first) <= 90_000 * .85 - 1000
    assert message_tokens(second) <= 90_000 * .85 - 1000


@pytest.mark.parametrize("token_scale", [1.0, 2.7])
async def test_equal_cost_summary_keeps_the_original(token_scale):
    history, messages = conversation(["ORIGINAL " * 100] + [f"Recent {index}" for index in range(5)])
    post = history.posts[0]
    original_cost = message_tokens(post.context_messages())
    for length in range(2000):
        post.summary = "S" * length
        compressed = post.context_messages(compressed=True)[0]
        if message_tokens([Message.assistant(COMPRESSED_HISTORY_HEADING + compressed.content)]) == original_cost:
            break
    else:
        pytest.fail("fixture must have equal full and compressed token costs")
    view = await history.view(messages, [], keep_posts=5, context_length=1_000_000,
                              max_output=8192, token_scale=token_scale)
    assert context_body(view[2].content) == "ORIGINAL " * 100
    assert post.summary not in "\n".join(message.content or "" for message in view)


async def test_mixed_older_originals_and_summaries_keep_chronological_order():
    history, messages = conversation(["FIRST " * 500, "Second original", "THIRD " * 500]
                                     + [f"Recent {index}" for index in range(5)])
    history.posts[1].summary = "EXPANDED SECOND SUMMARY " * 300
    view = await history.view(messages, [], keep_posts=5, context_length=1_000_000, max_output=8192)
    wire = "\n".join(message.content or "" for message in view)
    assert wire.index("SUMMARY 1") < wire.index("Second original") < wire.index("SUMMARY 3") < wire.index("Recent 0")
    assert "FIRST " not in wire and "THIRD " not in wire and "EXPANDED SECOND SUMMARY" not in wire
    assert [context_body(message.content) for message in view if message.role == "user"] == [
        "Question 2", *[f"Question {index}" for index in range(4, 9)], "Current question",
    ]


async def test_summary_comparison_uses_the_selected_tool_projection():
    history = ConversationHistory()
    messages = [Message.user("Read"), Message.assistant("Reading", [
        ToolCall("a", "record", {"value": '"nested"\\value' * 100}),
    ]), Message.tool_result("a", "Result")]
    messages += [Message.assistant(f"Recent {index}") for index in range(5)]
    history.sync(messages)
    messages.append(Message.user("Next"))
    post = history.posts[0]
    original = post.context_messages()
    native_cost = message_tokens(original)
    text_cost = message_tokens(tool_history_as_text(original))
    assert native_cost < text_cost
    for length in range(10_000):
        post.summary = "S" * length
        summary = post.context_messages(compressed=True)[0]
        cost = message_tokens([Message.assistant(COMPRESSED_HISTORY_HEADING + summary.content)])
        if native_cost < cost < text_cost:
            break
    else:
        pytest.fail("fixture must fall between native and text tool costs")
    for native in (True, False):
        view = await history.view(messages, [], keep_posts=5, context_length=1_000_000, max_output=8192,
                                  native_tools=native, text_tool_history=not native)
        assert any(message.tool_calls for message in view) is native
        assert (post.summary in "\n".join(message.content or "" for message in view)) is not native
