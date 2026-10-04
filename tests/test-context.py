"""Rolling context preserves complete exchanges and exact retrievable originals."""

from __future__ import annotations

import asyncio
import io
import json
import re

import httpx
import pytest

from test_agent import summary_response

from slipagent.agent import Agent, STOP_NOTICE, STEP_LIMIT_NOTICE
from slipagent.cli import Renderer, Style, build_parser
from slipagent.config import Config, ConfigError
from slipagent.context import (
    ContextError, ConversationHistory, RecallHistoryTool, SUMMARY_START,
    SUMMARY_END, TOOL_SUMMARY_KEY, response_memory, tool_response_memory,
)
from slipagent.tools.base import ToolRegistry
from slipagent.openrouter import OpenRouterClient
from slipagent.types import Message, ToolCall
from slipagent.protocol import parse_response
from test_agent import StubClient, RecordingTool, completion, structured_message, context_body
from test_cli_e2e import StubOpenRouter, run_cli, text_step, tool_step


def reply(post, text, summary, previous=None):
    record = {"post": post, "summary": summary}
    if previous:
        record["previous"] = {"post": post - 1, "summary": previous}
    return text + "\n" + SUMMARY_START + json.dumps(record) + SUMMARY_END


def make_agent(responses, *, include_memory=True, **kwargs):
    client = StubClient(responses, include_memory=include_memory)
    agent = Agent(client=client, registry=ToolRegistry([RecordingTool("EXACT TOOL RESULT")]),
                  model="test/model", system_prompt="Project instructions stay pinned.", **kwargs)
    return agent, client


def add_intermediate_posts(agent):
    """Age a record past the five-call minimum without extra inference fixtures."""
    agent.extend([Message.assistant("Intermediate response") for _ in range(4)])


async def test_keeps_recent_full_and_both_sides_in_older_summary():
    first_answer = "FIRST FULL ANSWER " * 100
    agent, client = make_agent([
        completion(first_answer),
        completion(reply(2, "SECOND FULL ANSWER", "User asked second; model answered second.")),
        completion(reply(3, "THIRD FULL ANSWER", "User asked third; model answered third.")),
    ], context_posts=1)
    events = []
    agent.on_event = events.append
    assert (await agent.run("FIRST FULL QUESTION")).strip() == first_answer.strip()
    add_intermediate_posts(agent)
    await agent.run("SECOND FULL QUESTION")
    await agent.run("THIRD FULL QUESTION")
    messages = client.calls[-1]["messages"]
    assert context_body(messages[0].content).startswith("Project instructions stay pinned.")
    assert agent.history.posts[0].summary in messages[1].content
    assert not any(context_body(m.content).strip() == first_answer.strip() for m in messages)
    assert any(context_body(m.content) == "SECOND FULL QUESTION" for m in messages)
    assert any(context_body(m.content).startswith("SECOND FULL ANSWER") for m in messages)
    assert context_body(messages[-1].content) == "THIRD FULL QUESTION"
    assert len(client.calls) == 3
    assert not any(SUMMARY_START in e.text for e in events)
    assert agent.messages[2].content.strip() == first_answer.strip()
    original = json.loads((await agent.registry.invoke("recall_history", {"post_id": 1})).content)
    assert "FIRST FULL QUESTION" in original["content"]
    assert "FIRST FULL ANSWER" in original["content"]


async def test_background_summary_is_stored_with_the_first_post():
    summary = "User asked the first question; model gave the first answer."
    agent, client = make_agent([completion(reply(1, "The exact first answer.", summary))])
    await agent.run("The exact first question?")
    post = agent.history.posts[0]
    assert post.agent_response == "The exact first answer."
    assert post.summary and len(client.summary_calls) == 1
    assert post.results_summarized
    assert "The exact first answer." in post.full_text()
    assert len(client.calls) == 1


async def test_stopped_tool_batch_stores_both_versions_immediately():
    agent, client = make_agent([
        completion(reply(1, "Reading.", "User asked to read; model requested record."),
                   [ToolCall("a", "record", {"value": "first"}),
                    ToolCall("b", "record", {"value": "second"})]),
    ])
    def stop_after_tool(event):
        if event.kind == "tool_end":
            agent.request_stop()
    agent.on_event = stop_after_tool
    assert await agent.run("read both") == STOP_NOTICE
    post = agent.history.posts[0]
    assert post.summary is not None
    assert post.agent_response == "Reading."
    assert post.results_summarized and not post.observed
    assert len([m for m in post.messages if m.role == "tool"]) == 2
    assert len(client.calls) == 1


async def test_stopped_tool_post_rolls_with_tool_summary_without_summary_repair():
    agent, client = make_agent([
        completion(reply(1, "Reading.", "User asked read; model requested record."),
                   [ToolCall("c", "record", {"value": "v"})]),
        completion(reply(2, "Second.", "User asked next; model answered second.")),
        completion(reply(3, "Third.", "User asked next again; model answered third.")),
    ], context_posts=1)
    def stop_after_tool(event):
        if event.kind == "tool_end":
            agent.request_stop()
    agent.on_event = stop_after_tool
    assert await agent.run("read") == STOP_NOTICE
    agent.on_event = None
    add_intermediate_posts(agent)
    await agent.run("next")
    await agent.run("next again")
    wire = "\n".join(m.content or "" for m in client.calls[-1]["messages"])
    assert "Conversation Record (Compressed):" in wire and agent.history.posts[0].summary in wire
    assert "retrieve post 1 with recall_history" not in wire
    assert "EXACT TOOL RESULT" in wire
    assert "EXACT TOOL RESULT" in (await agent.registry.invoke("recall_history", {"post_id": 1})).content
    assert len(client.calls) == 3


@pytest.mark.parametrize("total", [1, 50, 51, 100, 150, 151, 160])
async def test_only_50_full_posts_and_100_older_summaries_are_supplied(total):
    history = ConversationHistory()
    messages = [Message.system("pinned guidance")]
    for post_id in range(1, total + 1):
        messages.extend([Message.user(f"FULL_QUESTION_{post_id:03d}"),
                         Message.assistant(f"FULL_ANSWER_{post_id:03d}")])
    history.sync(messages)
    for post in history.posts:
        post.summary = f"STORED_SUMMARY_{post.id:03d}"
        post.results_summarized = True
    messages.append(Message.user("current question"))
    view = await history.view(messages, [], keep_posts=50,
                              context_length=1000000, max_output=8192)
    wire = "\n".join(message.content or "" for message in view)
    boundary = max(0, total - 50)
    oldest_summary = max(0, boundary - 100)
    for post_id in range(1, total + 1):
        assert (f"FULL_QUESTION_{post_id:03d}" in wire) == (post_id > boundary)
        assert (f"FULL_ANSWER_{post_id:03d}" in wire) == (post_id > boundary)
        assert (f"STORED_SUMMARY_{post_id:03d}" in wire) == (oldest_summary < post_id <= boundary)
    assert len(history.posts) == total
    original = await RecallHistoryTool(history).invoke({"post_id": 1})
    assert "FULL_QUESTION_001" in original.content
    assert "FULL_ANSWER_001" in original.content
    assert history.posts[0].summary == "STORED_SUMMARY_001"


async def test_background_summary_contains_tool_findings_and_user_prompt():
    agent, client = make_agent([
        completion(reply(1, "Reading.", "User asked read file; model requested record."), [ToolCall("c", "record", {"value": "v"})]),
        completion(reply(2, "Read done.", "User wanted file read; model confirmed completion.",
                         "User asked read file; model read it and found EXACT TOOL RESULT.")),
        completion(reply(3, "New answer.", "User asked next; model answered next.")),
    ], context_posts=1)
    await agent.run("read file")
    assert context_body(client.calls[1]["messages"][-1].content) == "EXACT TOOL RESULT"
    add_intermediate_posts(agent)
    await agent.run("next")
    archived = client.calls[2]["messages"][1]
    assert "User: read file" in archived.content
    assert "EXACT TOOL RESULT" in archived.content
    assert not any(m.role == "tool" for m in client.calls[2]["messages"])
    assert agent.history.posts[0].results_summarized
    full = await agent.registry.invoke("recall_history", {"post_id": 1})
    assert "EXACT TOOL RESULT" in full.content


async def test_over_budget_preserves_recent_tool_batch_without_extra_calls():
    payload = "full large payload " * 1500
    agent, client = make_agent([
        completion(reply(1, "Read two.", "User asked read; two tools requested."), [
            ToolCall("a", "record", {"value": "one"}), ToolCall("b", "record", {"value": "two"})]),
        completion(reply(2, "Received excerpts.", "Model received bounded results.", "Two tools returned large payloads; only excerpts were visible.")),
    ])
    agent._context_lengths[agent.model] = 9000
    agent.registry.get("record").result = payload
    assert await agent.run("read") == "Received excerpts."
    assert len(client.calls) == 2
    assert "Excerpt" in "\n".join(m.content or "" for m in client.calls[1]["messages"])
    assert agent.history.posts[0].agent_response == "Read two."
    assert len(client.summary_calls) == 2
    assert len([m for m in agent.messages if m.role == "tool"]) == 2
    assert json.loads(agent.history.posts[0].messages[-1].content)["content"] == payload


async def test_full_window_shrinks_by_whole_posts_at_model_limit():
    history = ConversationHistory()
    messages = [Message.system("pinned")]
    for index in range(51):
        messages.append(Message.user(f"FULL USER REQUEST {index}: preserve this exact wording λ"))
        if index == 1:
            messages.extend([
                Message.assistant("OLD TOOL RESPONSE", [ToolCall("a", "record", {"value": "one"}),
                                                        ToolCall("b", "record", {"value": "two"})]),
                Message.tool_result("a", "LARGE ORIGINAL TOOL OUTPUT " * 18000),
                Message.tool_result("b", "LARGE ORIGINAL TOOL OUTPUT " * 18000),
            ])
        else:
            messages.append(Message.assistant(f"FULL RESPONSE {index}"))
    history.sync(messages)
    for post in history.posts:
        post.summary = f"User asked request {post.id - 1}; model completed response {post.id - 1}."
        post.results_summarized = True
    messages.append(Message.user("current request"))
    async def unused(source):
        pytest.fail("valid stored summaries should not require another call")
    view = await history.view(messages, [], keep_posts=50,
                              context_length=200000, max_output=8192, summarize=unused)
    wire = "\n".join(message.content or "" for message in view)
    assert [context_body(message.content) for message in view if message.role == "user"] == [
        *(f"FULL USER REQUEST {index}: preserve this exact wording λ" for index in range(2, 51)),
        "current request",
    ]
    assert "OLD TOOL RESPONSE" not in wire
    assert "LARGE ORIGINAL TOOL OUTPUT" not in wire
    assert history.posts[1].summary in wire
    assert all(f"FULL RESPONSE {index}" in wire for index in range(2, 51))
    assert not any(message.role == "tool" or message.tool_calls for message in view)
    assert len(history.posts) == 51
    for index, post in enumerate(history.posts):
        assert post.messages[0].content == f"FULL USER REQUEST {index}: preserve this exact wording λ"
        assert post.summary is not None
    assert "LARGE ORIGINAL TOOL OUTPUT" in history.posts[1].full_text()


async def test_small_model_does_not_arbitrarily_halve_the_recent_budget():
    history = ConversationHistory()
    answer = "FULL RESPONSE " * 30000
    messages = [Message.user("original request"), Message.assistant(answer), Message.user("next")]
    history.sync(messages)
    history.posts[0].summary = "Short stored summary."
    view = await history.view(messages, [], keep_posts=50,
                              context_length=262144, max_output=8192)
    assert any(context_body(message.content) == answer for message in view)
    assert not any("Short stored summary." in (message.content or "") for message in view)


async def test_older_window_uses_original_when_summary_is_larger():
    history = ConversationHistory()
    messages = [Message.user("first request"), Message.assistant("Short exact answer."),
                Message.user("second request"), Message.assistant("LARGE SECOND ANSWER " * 1000),
                Message.user("third request"), Message.assistant("Latest exact answer."),
                *[Message.assistant("Intermediate response") for _ in range(4)],
                Message.user("current request")]
    history.sync(messages)
    history.posts[0].summary = "INFLATED SUMMARY " * 300
    history.posts[1].summary = "Second answer compressed."
    history.posts[2].summary = "Latest answer compressed."
    view = await history.view(messages, [], keep_posts=5,
                              context_length=1000000, max_output=8192)
    wire = "\n".join(message.content or "" for message in view)
    assert "Short exact answer." in wire and "Latest exact answer." in wire
    assert "INFLATED SUMMARY" not in wire
    assert "LARGE SECOND ANSWER" not in wire and "Second answer compressed." in wire
    assert "Latest answer compressed." not in wire
    assert [context_body(message.content) for message in view if message.role == "user"] == [
        "first request", *("third request" for _ in range(5)), "current request",
    ]


async def test_shrunk_window_has_whole_turn_summary_and_full_latest_turn():
    agent, client = make_agent([
        completion(reply(1, "FULL TOOL RESPONSE", "Assistant requested record."),
                   [ToolCall("a", "record", {"value": "one"}), ToolCall("b", "record", {"value": "two"})]),
        completion(reply(2, "LATEST FULL ANSWER", "Assistant completed the request.",
                         "Both tools returned their actual results successfully.")),
        completion(reply(3, "NEXT ANSWER", "Assistant answered the follow-up.")),
    ])
    agent.registry.get("record").result = "FULL LARGE TOOL OUTPUT " * 1000
    await agent.run("Original exact request")
    agent._context_lengths[agent.model] = 9000
    await agent.run("Follow-up exact request")
    view = client.calls[-1]["messages"]
    wire = "\n".join(message.content or "" for message in view)
    assert [context_body(message.content) for message in view if message.role == "user"] == [
        "Original exact request", "Follow-up exact request",
    ]
    assert agent.history.posts[0].summary in wire
    assert not any(context_body(m.content) == "FULL TOOL RESPONSE" for m in view)
    assert "LATEST FULL ANSWER" in wire
    assert "User requested Original exact request" not in wire
    assert "Assistant completed the request." not in wire
    assert not any(message.role == "tool" or message.tool_calls for message in view)
    assert len(client.calls) == 3
    assert "FULL LARGE TOOL OUTPUT" in (await agent.registry.invoke("recall_history", {"post_id": 1})).content


async def test_large_archived_reasoning_does_not_consume_context_headroom():
    answer = Message.assistant("ACTUAL ANSWER", [ToolCall("a", "record", {})])
    answer.reasoning = "THOUGHTS " * 100000
    answer.reasoning_details = [{"type": "reasoning.encrypted", "data": "OPAQUE " * 100000}]
    messages = [Message.user("Work"), answer, Message.tool_result("a", "ACTUAL RESULT"), Message.user("Continue")]
    history = ConversationHistory()
    view = await history.view(messages, [], keep_posts=50,
                              context_length=16000, max_output=1000)
    assert any(context_body(m.content) == "ACTUAL ANSWER" and m.tool_calls for m in view)
    assert any(context_body(m.content) == "ACTUAL RESULT" for m in view)
    assert all(m.reasoning is None and m.reasoning_details is None for m in view)
    assert answer.reasoning == "THOUGHTS " * 100000
    assert answer.reasoning_details == [{"type": "reasoning.encrypted", "data": "OPAQUE " * 100000}]
    assert history.posts[0].reasoning == answer.reasoning


async def test_under_budget_keeps_entire_recent_window_in_full():
    history = ConversationHistory()
    messages = [Message.user("first question"), Message.assistant("first answer"),
                Message.user("second question"), Message.assistant("second answer", [ToolCall("c", "record", {})]),
                Message.tool_result("c", "full tool result"), Message.user("next")]
    async def unused(source):
        pytest.fail("no compression should run below the budget")
    view = await history.view(messages, [], keep_posts=50,
                              context_length=1000000, max_output=8192, summarize=unused)
    assert [context_body(m.content) for m in view if m.role == "user"] == ["first question", "second question", "next"]
    assert any(context_body(m.content).startswith("first answer") for m in view)
    assert any(m.tool_calls and m.tool_calls[0].id == "c" for m in view)
    assert any(m.role == "tool" and m.tool_call_id == "c" and context_body(m.content) == "full tool result" for m in view)


async def test_only_posts_outside_window_use_summaries():
    history = ConversationHistory()
    messages = [Message.user("oldest question"), Message.assistant("OLDEST FULL RESPONSE " * 1000),
                Message.user("middle question"), Message.assistant("MIDDLE FULL RESPONSE " * 50),
                Message.user("latest question"), Message.assistant("LATEST FULL RESPONSE " * 50),
                *[Message.assistant("Intermediate response") for _ in range(3)],
                Message.user("next question")]
    history.sync(messages)
    for post in history.posts:
        post.summary = f"User asked question {post.id}; model answered question {post.id}."
        post.results_summarized = True
    async def unused(source):
        pytest.fail("the oldest stored summary is available")
    view = await history.view(messages, [], keep_posts=2,
                              context_length=1000000, max_output=8192, summarize=unused)
    wire = "\n".join(m.content or "" for m in view)
    assert "OLDEST FULL RESPONSE" not in wire
    assert "Conversation Record (Compressed):" in wire
    assert "MIDDLE FULL RESPONSE " * 50 in wire
    assert "LATEST FULL RESPONSE " * 50 in wire
    assert [context_body(m.content) for m in view if m.role == "user"] == [
        "middle question", *("latest question" for _ in range(4)), "next question",
    ]
    assert history.posts[0].summary in wire
    assert history.posts[1].summary not in wire
    assert history.posts[2].summary not in wire


async def test_missing_archived_summary_is_omitted_without_a_repair_call():
    history = ConversationHistory()
    messages = [Message.user("original request"), Message.user("queued correction"),
                Message.assistant("FULL RESPONSE", [ToolCall("c", "record", {"value": "v"})]),
                Message.tool_result("c", "HUGE TOOL OUTPUT " * 1000), Message.user("current request")]
    messages.insert(-1, Message.assistant("next response"))
    messages[-1:-1] = [Message.assistant("Intermediate response") for _ in range(4)]
    async def unused(source):
        pytest.fail("missing stored summaries must not trigger additional calls")
    view = await history.view(messages, [], keep_posts=1,
                              context_length=1000000, max_output=8192, summarize=unused)
    assert any(context_body(message.content) == "current request" for message in view)
    assert not any("HUGE TOOL OUTPUT" in (message.content or "") for message in view)
    assert history.posts[0].messages[0].content == "original request"
    assert history.posts[0].messages[1].content == "queued correction"
    assert "HUGE TOOL OUTPUT" in history.posts[0].full_text()


async def test_large_old_user_requests_reduce_the_recent_window_without_compression_calls():
    history = ConversationHistory()
    first, second = "FIRST USER REQUEST " * 1000, "SECOND USER REQUEST " * 100
    messages = [Message.user(first), Message.assistant("first answer"),
                Message.user(second), Message.assistant("second answer")]
    async def unused(source):
        pytest.fail("compressing responses cannot make the user requests fit")
    view = await history.view(messages, [], keep_posts=50,
                              context_length=9000, max_output=1000, summarize=unused)
    assert [context_body(message.content) for message in view if message.role == "user"] == [second]
    assert any(context_body(message.content) == "second answer" for message in view)
    assert history.posts[0].messages[0].content == first
    assert history.posts[1].messages[0].content == second


async def test_reduced_history_keeps_active_request_and_complete_latest_tool_batch():
    history = ConversationHistory()
    request, correction = "Inspect the project environment.", "Use the existing .venv."
    messages = [Message.user("OLD USER REQUEST " * 1000), Message.assistant("Old answer."),
                Message.user(request), Message.user(correction), Message.assistant("OLD RESPONSE " * 3000),
                Message.assistant("INTERMEDIATE RESPONSE " * 3000),
                Message.assistant("Checking.", [ToolCall("a", "record", {"value": "one"}),
                                               ToolCall("b", "record", {"value": "two"})]),
                Message.tool_result("a", "EXACT FIRST RESULT"),
                Message.tool_result("b", "EXACT SECOND RESULT")]
    history.sync(messages)
    for post in history.posts:
        post.user_prompt_compressed = "USER SUMMARY MUST NOT DUPLICATE THE ACTIVE REQUEST"
        post.agent_response_compressed = "INFLATED RESPONSE SUMMARY " * 500
        post.previous_tool_responses_compressed = ""
        post.results_summarized = not post.has_results
    originals = [post.full_text() for post in history.posts]
    view = await history.view(messages, [], keep_posts=50,
                              context_length=9000, max_output=1000)
    assert [context_body(message.content) for message in view if message.role == "user"] == [request + "\n" + correction]
    assert [(message.tool_call_id, context_body(message.content)) for message in view if message.role == "tool"] == [
        ("a", "EXACT FIRST RESULT"), ("b", "EXACT SECOND RESULT"),
    ]
    assert [call.id for message in view for call in message.tool_calls or []] == ["a", "b"]
    wire = "\n".join(message.content or "" for message in view)
    assert "OLD USER REQUEST" not in wire
    assert "USER SUMMARY MUST NOT DUPLICATE THE ACTIVE REQUEST" not in wire
    assert [post.full_text() for post in history.posts] == originals


async def test_history_can_be_omitted_entirely_if_even_its_summaries_are_too_large():
    history = ConversationHistory()
    messages = [Message.user("old request"), Message.assistant("OLD RESPONSE " * 3000),
                Message.user("latest exact request")]
    history.sync(messages)
    history.posts[0].summary = "INFLATED SUMMARY " * 2000
    view = await history.view(messages, [], keep_posts=50,
                              context_length=8000, max_output=1000)
    assert [context_body(message.content) for message in view if message.role != "system"] == ["latest exact request"]
    assert "OLD RESPONSE" in (await RecallHistoryTool(history).invoke({"post_id": 1})).content


async def test_background_summaries_have_separate_requests_and_accounting():
    agent, client = make_agent([
        completion("Original answer. " * 100),
        completion(reply(2, "Second answer.", "User asked second; model replied.")),
        completion(reply(3, "Third answer.", "User asked third; model replied.")),
    ], context_posts=1)
    await agent.run("original question")
    add_intermediate_posts(agent)
    await agent.run("second question")
    assert len(client.calls) == 2
    await agent.run("third question")
    assert len(client.calls) == 3
    assert agent.history.posts[0].summary in client.calls[2]["messages"][1].content
    assert not any(context_body(m.content).strip() == ("Original answer. " * 100).strip() for m in client.calls[2]["messages"])
    assert len(client.summary_calls) == 3
    assert agent.usage.total_tokens == 45
    assert agent.total_cost == pytest.approx(.03)


@pytest.mark.parametrize("verbose", [False, True])
async def test_history_compression_does_not_print_a_maintenance_notice(verbose):
    output = io.StringIO()
    renderer = Renderer(Style(enabled=True), output, verbose=verbose)
    agent, client = make_agent([
        completion(reply(1, "Original answer.", "User asked original question; model gave original answer.")),
        completion(reply(2, "Second answer.", "User asked second; model replied.")),
        completion(reply(3, "Third answer.", "User asked third; model replied.")),
    ], context_posts=1, on_event=renderer.handle)
    await agent.run("original question")
    await agent.run("second question")
    await agent.run("third question")

    assert "Compressing conversation history" not in output.getvalue()
    assert "Third answer." in output.getvalue()
    assert len(client.calls) == 3
    assert agent.usage.total_tokens == 45
    original = await agent.registry.invoke("recall_history", {"post_id": 1})
    assert "original question" in original.content and "Original answer." in original.content


@pytest.mark.parametrize("suffix", [
    "", "<slipagent_context>{broken}</slipagent_context>",
    "<slipagent_context>{\"post\":1", reply(9, "", "wrong ID"),
    reply(1, "", ""), reply(1, "", "oversized" * 1000),
], ids=["missing", "malformed", "truncated", "wrong-id", "empty", "oversized"])
async def test_optional_legacy_summary_does_not_gate_the_response(suffix):
    events = []
    agent, client = make_agent([completion("Accepted answer." + suffix)], include_memory=False, on_event=events.append)
    assert await agent.run("Original question?") == "Accepted answer."
    await agent.wait_for_compaction()
    assert len(client.calls) == 1 and len(client.summary_calls) == 1
    assert not any(event.kind == "retry" for event in events)
    assert agent.history.posts[0].agent_response == "Accepted answer."
    assert agent.history.posts[0].summary


async def test_invalid_tool_json_retries_before_output_and_actions():
    tool = RecordingTool("PREVIOUS TOOL RESULT")
    events = []
    class CheckingClient(StubClient):
        async def chat(self, **kwargs):
            if len(self.calls) == 1:
                assert tool.seen == []
                assert not any(event.kind in {"assistant_text", "tool_start", "tool_end"} for event in events)
            return await super().chat(**kwargs)
    client = CheckingClient([
        completion(json.dumps({"response": "Rejected.", "tool_calls": [{"name": "record", "arguments": "["}]})),
        completion(reply(1, "Accepted tool plan.", "User asked inspect; model requested record."),
                   [ToolCall("accepted", "record", {"value": "accepted"})]),
        completion(reply(2, "Done.", "User asked inspect; previous tools returned PREVIOUS TOOL RESULT; model confirmed completion.",
                         "User asked inspect; model ran record and received PREVIOUS TOOL RESULT.")),
    ], include_memory=False)
    agent = Agent(client=client, registry=ToolRegistry([tool]), model="test", on_event=events.append)
    assert await agent.run("inspect") == "Done."
    assert tool.seen == [{"value": "accepted"}]
    assert [event.text for event in events if event.kind == "assistant_text"] == ["Accepted tool plan.", "Done."]
    assert len(client.calls) == 3
    assert len(agent.history.posts) == 2
    assert "PREVIOUS TOOL RESULT" in agent.history.posts[0].summary
    assert "rejected" not in "\n".join(post.full_text() for post in agent.history.posts)
    assert client.calls[1]["tools"] == client.calls[0]["tools"]
    assert "CURRENT_POST_ID: 1" in client.calls[1]["messages"][0].content


async def test_repeated_invalid_actions_respect_request_limit():
    events = []
    agent, client = make_agent([
        completion(json.dumps({"response": "Rejected.", "tool_calls": [{"name": "record", "arguments": "["}]})) for _ in range(3)
    ], include_memory=False, max_steps=3, on_event=events.append)
    assert await agent.run("inspect") == STEP_LIMIT_NOTICE
    assert len(client.calls) == 3
    assert agent.history.posts == []
    assert agent.registry.get("record").seen == []
    assert not any(event.kind in {"assistant_text", "tool_start", "tool_end"} for event in events)
    assert agent.messages[-1].content == "inspect"
    assert agent.usage.total_tokens == 45
    client.responses.append(completion(reply(1, "Resumed.", "User asked inspect then continue; model resumed.")))
    assert await agent.run("continue") == "Resumed."
    assert len(client.calls) == 4
    assert agent.history.posts[0].request == "inspect\ncontinue"


async def test_retry_keeps_previous_tool_results_and_current_user_correction():
    agent, client = make_agent([
        completion(reply(1, "Reading.", "User asked read; model requested record."),
                   [ToolCall("c", "record", {"value": "v"})]),
        completion(json.dumps({"response": "Rejected.", "tool_calls": [{"name": "record", "arguments": "["}]})),
        completion(reply(2, "Corrected.", "User asked read then corrected the task; previous tools returned EXACT TOOL RESULT; model answered the correction.",
                         "User asked read; model received EXACT TOOL RESULT.")),
    ], include_memory=False)
    def correct(event):
        if event.kind == "tool_end":
            agent.enqueue("current correction")
    agent.on_event = correct
    assert await agent.run("read") == "Corrected."
    assert len(client.calls) == 3
    for request in client.calls[1:]:
        assert any(m.role == "tool" and context_body(m.content) == "EXACT TOOL RESULT" for m in request["messages"])
        assert context_body(request["messages"][-1].content) == "current correction"
        assert "CURRENT_POST_ID: 2" in request["messages"][0].content
    assert agent.registry.get("record").seen == [{"value": "v"}]
    assert agent.history.posts[1].agent_response == "Corrected."


async def test_stop_during_rejected_response_prevents_retry_and_tool_calls():
    events = []
    agent, client = make_agent([
        completion(json.dumps({"response": "Rejected.", "tool_calls": [{"name": "record", "arguments": "["}]})),
    ], include_memory=False)
    def stop(event):
        events.append(event)
        if event.kind == "step_start":
            agent.request_stop()
    agent.on_event = stop
    assert await agent.run("inspect") == STOP_NOTICE
    assert len(client.calls) == 1
    assert agent.history.posts == []
    assert agent.registry.get("record").seen == []
    assert not any(event.kind in {"assistant_text", "tool_start", "tool_end"} for event in events)


async def test_stop_during_response_stores_its_background_summary_and_original():
    started, release = asyncio.Event(), asyncio.Event()
    class WaitingClient(StubClient):
        async def chat(self, **kwargs):
            started.set()
            await release.wait()
            return await super().chat(**kwargs)
    client = WaitingClient([
        completion(reply(1, "Done.", "User asked work; model completed work.")),
        completion(reply(2, "resumed", "User asked resume; model resumed.")),
    ])
    agent = Agent(client=client, registry=ToolRegistry(), model="test", context_posts=1)
    task = asyncio.create_task(agent.run("work"))
    await started.wait()
    assert agent.request_stop()
    release.set()
    assert await task == "Done."
    assert agent.stopped
    assert len(client.calls) == 1
    assert agent.history.posts[0].agent_response == 'Done.'
    assert "Done." in agent.history.posts[0].full_text()
    assert await agent.run("continue") == "resumed"
    assert len(client.calls) == 2


async def test_background_summary_is_saved_at_primary_request_limit():
    agent, client = make_agent([
        completion(reply(1, "Reading.", "User asked read; model requested record."),
                   [ToolCall("c", "record", {"value": "v"})]),
        completion(reply(2, "continued", "User asked continue; model continued.",
                         "User asked read; model received EXACT TOOL RESULT.")),
    ], context_posts=1, max_steps=1)
    assert await agent.run("read") == STEP_LIMIT_NOTICE
    assert len(client.calls) == 1
    assert agent.history.posts[0].agent_response == 'Reading.'
    assert "EXACT TOOL RESULT" in agent.history.posts[0].full_text()
    assert await agent.run("continue") == "continued"
    assert len(client.calls) == 2
    assert "EXACT TOOL RESULT" in agent.history.posts[0].summary


async def test_normal_response_respects_configured_completion_cap():
    agent, client = make_agent([completion(reply(1, "Done.", "User asked work; model completed."))], max_tokens=64)
    assert await agent.run("work") == "Done."
    assert len(client.calls) == 1
    assert client.calls[0]["max_tokens"] == 64


async def test_cancel_during_response_preserves_existing_summaries_and_originals():
    started = asyncio.Event()
    class BlockingClient(StubClient):
        async def chat(self, **kwargs):
            if len(self.calls) == 2 and not kwargs["messages"][0].content.startswith("Summarize one completed SlipAgent turn"):
                started.set()
                await asyncio.Event().wait()
            return await super().chat(**kwargs)
    client = BlockingClient([completion("one"), completion(reply(2, "two", "User asked two; model replied two."))])
    agent = Agent(client=client, registry=ToolRegistry(), model="test", context_posts=1)
    await agent.run("one?")
    await agent.run("two?")
    task = asyncio.create_task(agent.run("three?"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not agent.running
    assert agent.history.posts[0].summary is not None
    assert "one?" in agent.history.posts[0].full_text()
    assert agent.messages[-1].content == "three?"


async def test_cancelled_tool_batch_preserves_valid_summary_and_original():
    started = asyncio.Event()
    class BlockingTool(RecordingTool):
        async def run(self, **kwargs):
            started.set()
            await asyncio.Event().wait()
    client = StubClient([completion(reply(1, "Reading.", "User asked read; model requested record."),
                                    [ToolCall("c", "record", {"value": "v"})])],
                        include_memory=False)
    agent = Agent(client=client, registry=ToolRegistry([BlockingTool()]), model="test")
    task = asyncio.create_task(agent.run("read"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(client.calls) == 1
    assert agent.history.posts[0].agent_response == 'Reading.'
    assert "Reading." in agent.history.posts[0].full_text()
    assert "Tool interrupted" in agent.history.posts[0].full_text()


async def test_recall_searches_full_original_and_pages_without_truncation_loss():
    history = ConversationHistory()
    history.sync([Message.user("user needle λ"), Message.assistant("answer " * 500)])
    tool = RecallHistoryTool(history)
    assert "Post 1" in (await tool.invoke({"query": "NEEDLE"})).content
    assert "Post 1" in (await tool.invoke({"query": "answer"})).content
    offset, chunks = 0, []
    while True:
        page = json.loads((await tool.invoke({"post_id": 1, "offset": offset, "limit": 97})).content)
        chunks.append(page["content"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    full = json.loads("".join(chunks))
    assert full["prompt"] == "user needle λ"
    assert full["response"] == "answer " * 500
    assert (await tool.invoke({"post_id": 999})).is_error
    assert (await tool.invoke({"limit": 16001})).is_error
    assert (await tool.invoke({"offset": -1})).is_error


@pytest.mark.parametrize("query, expected_ids", [
    ("", [1, 2, 3]), ("needle", [1, 3]), ("SUMMARY", [2]), ("missing", []),
])
async def test_recall_listing_reports_total_matches_on_every_character_page(query, expected_ids):
    history = ConversationHistory()
    history.sync([
        Message.user("needle λ"), Message.assistant("first answer"),
        Message.user("unrelated"), Message.assistant("second answer"),
        Message.user("third request"), Message.assistant("NEEDLE in original response"),
    ])
    history.posts[1].summary = "summary marker"
    tool = RecallHistoryTool(history)
    offset, chunks = 0, []
    while True:
        result = await tool.invoke({"query": query, "offset": offset, "limit": 7})
        assert not result.is_error
        page = json.loads(result.content)
        assert page["total_matches"] == len(expected_ids)
        chunks.append(page["content"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] == offset + len(page["content"])
        offset = page["next_offset"]
    listing = "".join(chunks)
    assert len(listing) == page["total_characters"]
    assert [int(value) for value in re.findall(r"^Post (\d+):", listing, re.MULTILINE)] == expected_ids
    beyond = json.loads((await tool.invoke({"query": query, "offset": len(listing) + 1})).content)
    assert beyond["total_matches"] == len(expected_ids)
    assert beyond["content"] == "" and beyond["next_offset"] is None


async def test_recall_empty_history_reports_zero_matches():
    result = await RecallHistoryTool(ConversationHistory()).invoke({})
    page = json.loads(result.content)
    assert page["total_matches"] == 0
    assert page["next_offset"] is None
    assert page["content"] == "No matching history posts."


@pytest.mark.parametrize("arguments", [
    {"sections": ["bogus"]}, {"sections": ["tool_results", "bogus"]},
    {"sections": ["all"]}, {"sections": ["user"]},
    {"section": "bogus"}, {"section": ""},
])
@pytest.mark.parametrize("direct", [True, False])
async def test_recall_unknown_parts_return_errors_even_without_schema_validation(arguments, direct):
    history = ConversationHistory()
    history.sync([Message.user("request"), Message.assistant("response")])
    tool = RecallHistoryTool(history)
    supplied = {"post_id": 1, **arguments}
    result = await tool.run(**supplied) if direct else await tool.invoke(supplied)
    assert result.is_error
    assert "prompt" in result.content and "tool_results" in result.content


async def test_reset_clears_both_versions_and_restarts_ids():
    agent, _ = make_agent([completion(reply(1, "one", "User asked one; model answered one.")),
                           completion(reply(1, "new", "User asked new; model answered new."))])
    await agent.run("one?")
    agent.reset()
    assert agent.history.posts == []
    assert "No matching" in (await agent.registry.invoke("recall_history", {})).content
    assert agent.messages[0].content == "Project instructions stay pinned."
    await agent.run("new?")
    assert agent.history.posts[0].id == 1
    assert agent.history.posts[0].agent_response == 'new'


async def test_imported_conversation_is_archived_with_tool_protocol_intact():
    agent, _ = make_agent([])
    agent.extend([Message.user("restore?"), Message.assistant(None, [ToolCall("c", "record", {})]),
                  Message.tool_result("c", "restored result"), Message.assistant("restored answer")])
    assert len(agent.history.posts) == 2
    assert "restored result" in agent.history.posts[0].full_text()
    assert agent.history.posts[1].request == "restore?"


async def test_stored_summaries_over_budget_preserve_originals_without_extra_calls():
    history = ConversationHistory()
    messages = [Message.system("pinned instructions")]
    for i in range(20):
        messages.extend([Message.user(f"question {i}"), Message.assistant(f"answer {i} " * 300)])
    history.sync(messages)
    for post in history.posts:
        post.summary = f"User asked question {post.id}; model answered. " + "detail " * 150
        post.results_summarized = True
    messages.append(Message.user("next"))
    summaries = [post.summary for post in history.posts]
    async def unused(source):
        pytest.fail("stored summaries must not trigger an overview request")
    view = await history.view(messages, [], keep_posts=1,
                              context_length=14000, max_output=1000, summarize=unused)
    wire = "\n".join(message.content or "" for message in view)
    assert history.posts[0].summary not in wire
    assert history.posts[14].summary in wire
    assert any(context_body(message.content) == "answer 19 " * 300 for message in view)
    assert any(context_body(message.content) == "next" for message in view)
    assert [post.summary for post in history.posts] == summaries
    assert len(history.posts) == 20
    assert "question 0" in history.posts[0].full_text()
    assert "answer 19" in history.posts[-1].full_text()


async def test_current_input_never_silently_truncates():
    history = ConversationHistory()
    question = "large user input " * 10000
    async def unused(source):
        pytest.fail("should not spend calls summarizing when the current input cannot fit")
    with pytest.raises(ContextError, match="Shorten the input"):
        await history.view([Message.user(question)], [], keep_posts=50,
                           context_length=9000, max_output=1000, summarize=unused)
    assert not history.posts


@pytest.mark.parametrize("suffix", ["<slipagent_context>{broken}</slipagent_context>",
                                    "<slipagent_context>{\"post\":1", reply(9, "", "wrong ID")])
def test_invalid_or_truncated_envelope_is_not_used_as_memory(suffix):
    memory = response_memory("Visible answer." + suffix, 1, None)
    assert memory.text == "Visible answer."
    assert memory.summary is None


def test_memory_envelope_checks_previous_id():
    text = reply(3, "Done", "User asked done; model finished.", "User asked previous; tools returned results.")
    assert response_memory(text, 3, 2).previous is not None
    assert response_memory(text, 3, 1).previous is None


@pytest.mark.parametrize("text", [
    "Example:\n```json\n<slipagent_context>{\"post\":1,\"summary\":\"example\"}</slipagent_context>\n```",
    "The <slipagent_context> marker carries metadata.",
    "An example <slipagent_context>{}</slipagent_context> followed by ordinary prose.",
])
def test_literal_marker_examples_remain_visible(text):
    assert response_memory(text, 1, None).text == text


def test_summary_can_quote_its_own_markers_without_leaking_metadata():
    summary = f"User asked about {SUMMARY_START}; model explained {SUMMARY_END}."
    memory = response_memory(reply(1, "Visible answer.", summary), 1, None)
    assert memory.text == "Visible answer."
    assert memory.summary == summary


@pytest.mark.parametrize("visible", [
    "Fixed the bug.\n\n[Harness context metadata]\nCURRENT_POST_ID: 1\nPREVIOUS_POST_ID: none",
    "[Harness context metadata]\n1\nnone\n\nFixed the bug.",
    "[Harness context metadata]\nCURRENT_POST_ID: 1\nPREVIOUS_POST_ID: none\nPrevious post's current summary: internal bookkeeping\n\nFixed the bug.",
    "Fixed the bug.\nCURRENT_POST_ID: 1\nPREVIOUS_POST_ID: none",
])
async def test_echoed_bookkeeping_never_reaches_output_or_saved_reply(visible):
    agent, _ = make_agent([completion(reply(1, visible, "User asked for a fix; model fixed the bug."))])
    events = []
    agent.on_event = events.append
    assert await agent.run("fix the bug") == "Fixed the bug."
    assert [event.text for event in events if event.kind == "assistant_text"] == ["Fixed the bug."]
    assert agent.messages[-1].content == "Fixed the bug."
    assert agent.history.posts[0].agent_response == 'Fixed the bug.'


def test_bookkeeping_examples_in_code_and_ordinary_numbers_remain_visible():
    text = "The result is 42.\n\nExample:\n```text\n[Harness context metadata]\nCURRENT_POST_ID: 7\nPREVIOUS_POST_ID: 6\n```\n\n1\n2\n3"
    assert response_memory(text, 1, None).text == text


def test_real_cli_filters_echoed_bookkeeping_from_the_final_answer(tmp_path):
    text = reply(1, "Fix complete.\n\n[Harness context metadata]\nCURRENT_POST_ID: 1\nPREVIOUS_POST_ID: none",
                 "User asked for a fix; model confirmed completion.")
    with StubOpenRouter([text_step(text)]) as stub:
        result = run_cli("-p", "fix the bug", "--base-url", stub.base_url, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Fix complete."
    assert "[Harness context metadata]" not in result.stderr
    assert "CURRENT_POST_ID" not in result.stderr


def test_real_cli_retries_before_emitting_rejected_text_or_writing_rejected_files(tmp_path):
    rejected = tool_step("write_file", {"path": "rejected.txt", "content": "rejected"})
    rejected["choices"][0]["message"] = {"role": "assistant", "content": json.dumps({"response": "Rejected response that must stay hidden.", "tool_calls": [{"name": "write_file", "arguments": "["}]})}
    accepted = tool_step("write_file", {"path": "accepted.txt", "content": "accepted"})
    accepted["choices"][0]["message"]["content"] = reply(1, "Writing accepted file.", "User asked write; model requested accepted.txt.")
    final = text_step(reply(2, "Done.", "User asked write; previous tool wrote accepted.txt; model confirmed completion.",
                            "User asked write; model wrote accepted.txt successfully."))
    with StubOpenRouter([rejected, accepted, final], include_memory=False) as stub:
        result = run_cli("-p", "write the file", "--base-url", stub.base_url, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Done."
    assert not (tmp_path / "rejected.txt").exists()
    assert (tmp_path / "accepted.txt").read_text() == "accepted"
    assert "Rejected response" not in result.stdout + result.stderr
    assert "rejected.txt" not in result.stdout + result.stderr
    assert "compressed record" not in result.stdout + result.stderr
    assert len(stub.requests) == 3
    assert context_body(stub.requests[1]["messages"][-1]["content"]) == "write the file"
    assert not any(m.get("tool_calls") for m in stub.requests[1]["messages"])


@pytest.mark.parametrize("keep_posts", [1, 50])
async def test_post_ids_are_internal_and_recent_summaries_are_not_duplicated(keep_posts):
    agent, client = make_agent([
        completion(reply(1, "First answer.", "User asked first; model answered first.")),
        completion(reply(2, "Reading.", "User asked second; model requested record."),
                   [ToolCall("c", "record", {"value": "v"})]),
        completion(reply(3, "Done.", "User asked second; model confirmed completion.",
                         "User asked second; model read EXACT TOOL RESULT.")),
    ], context_posts=keep_posts)
    await agent.run("first")
    await agent.run("second")
    view = client.calls[-1]["messages"]
    system = "\n".join(message.content or "" for message in view if message.role == "system")
    assert "CURRENT_POST_ID: 3" in system
    assert "PREVIOUS_POST_ID: 2" in system
    assert "User asked second; model requested record." not in "\n".join(m.content or "" for m in view)
    for message in view:
        if message.role != "system":
            assert "[Harness context metadata]" not in (message.content or "")
            assert "CURRENT_POST_ID" not in (message.content or "")
            assert "PREVIOUS_POST_ID" not in (message.content or "")


@pytest.mark.parametrize("has_previous_results", [False, True])
async def test_memory_prompt_explains_background_summaries_and_retrieval(has_previous_results):
    messages = [Message.user("inspect the project")]
    if has_previous_results:
        messages.extend([Message.assistant("Reading.", [ToolCall("c", "read_file", {"path": "README.md"})]),
                         Message.tool_result("c", "File contents.")])
    view = await ConversationHistory().view(messages, [], keep_posts=50,
                                            context_length=1000000, max_output=8192)
    system = view[0].content
    assert "ordinary plain text is allowed" in system
    assert "separately in the background" in system
    assert "sections=[...]" in system and "update_task" in system
    assert "ONE JSON object in EVERY response" not in system
    assert "previous_tool_responses_compressed" not in system


async def test_catalog_context_length_limits_full_history_and_is_cached():
    requests, catalog_calls = [], []
    responses = [
        text_step(reply(1, "UNCOMPRESSED ORIGINAL " * 700, "User asked original question; model gave a long answer.")),
        text_step(reply(2, "next", "User asked next; model replied next.")),
    ]
    def respond(request):
        if request.method == "GET":
            catalog_calls.append(request)
            return httpx.Response(200, json={"data": [{"id": "small/model", "context_length": 9000}]})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        response = responses[len(requests) - 1]
        response["choices"][0]["message"] = structured_message(response["choices"][0]["message"], requests[-1]["messages"], include_memory=False)
        return httpx.Response(200, json=response)
    async with OpenRouterClient(api_key="test", transport=httpx.MockTransport(respond)) as client:
        agent = Agent(client=client, registry=ToolRegistry(), model="small/model")
        await agent.run("original question")
        assert await agent.run("next question") == "next"
    assert len(catalog_calls) == 1
    assert len(requests) == 2
    wire = "\n".join(message.get("content") or "" for message in requests[-1]["messages"])
    assert not any(context_body(m.get("content")) == "UNCOMPRESSED ORIGINAL " * 700 for m in requests[-1]["messages"])
    assert agent.history.posts[0].summary in wire
    assert "UNCOMPRESSED ORIGINAL" in agent.history.posts[0].full_text()


def test_real_cli_hides_memory_rolls_window_and_recalls_originals(tmp_path):
    (tmp_path / "app.py").write_text("EXACT ORIGINAL FILE CONTENT\n", encoding="utf-8")
    first = tool_step("read_file", {"path": "app.py"})
    first["choices"][0]["message"]["content"] = reply(1, "Reading file.", "User asked inspect and report; model requested app.py.")
    second = tool_step("list_dir", {})
    second["choices"][0]["message"]["content"] = reply(2, "Checking layout.", "User asked inspect and report; model requested directory layout.",
                                                         "User asked inspect and report; model read app.py and found original file content.")
    third = tool_step("recall_history", {"post_id": 1})
    third["choices"][0]["message"]["content"] = reply(3, "Checking the original record.", "User asked inspect and report; model requested history post 1.",
                                                        "User asked inspect and report; model listed the directory and found app.py.")
    for index in range(4):
        (tmp_path / f"directory-{index}").mkdir()
    intermediate = [tool_step("list_dir", {"path": f"directory-{index}"}) for index in range(4)]
    with StubOpenRouter([first, *intermediate, second, third, text_step(reply(8, "Inspection complete.", "User asked inspect and report; model reported completion.",
                                                               "User asked inspect and report; model recalled exact original input, response and file content."))]) as stub:
        result = run_cli("-p", "inspect and report", "--context-posts", "5",
                         "--base-url", stub.base_url, "--model", "stub/one", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Inspection complete."
    assert SUMMARY_START not in result.stdout + result.stderr
    assert "read_file(" in result.stderr
    assert len(stub.requests) == 8
    third_messages = stub.requests[6]["messages"]
    assert "User: inspect and report" in third_messages[1]["content"]
    assert not any(call["function"]["name"] == "read_file" for message in third_messages for call in message.get("tool_calls", []))
    recalled = stub.requests[7]["messages"][-1]
    assert recalled["role"] == "tool"
    record = json.loads(json.loads(context_body(recalled["content"]))["content"])
    assert record["prompt"] == "inspect and report"
    assert record["response"] == "Reading file."
    assert "EXACT ORIGINAL FILE CONTENT" in record["tool_results"][-1]["content"]


def test_config_and_cli_expose_post_window_without_separate_token_cap():
    config = Config.from_env(environ={"OPENROUTER_API_KEY": "test"})
    assert config.context_posts == 50
    assert not hasattr(config, "context_tokens")
    args = build_parser().parse_args(["--context-posts", "12"])
    config = Config.from_env(environ={"OPENROUTER_API_KEY": "test"},
                             context_posts=args.context_posts)
    assert config.context_posts == 12
    with pytest.raises(ConfigError):
        Config.from_env(environ={"OPENROUTER_API_KEY": "test"}, context_posts=0)
    assert "--context-tokens" not in build_parser().format_help()
