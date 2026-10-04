"""Normal replies and isolated, asynchronous whole-turn compaction."""

import asyncio
import json

import pytest

from test_agent import unpack_context, context_records

from slipagent.agent import Agent
from slipagent.context import ConversationHistory, RecallHistoryTool
from slipagent.capabilities import ModelCapabilities
from slipagent.openrouter import OpenRouterClient
from slipagent.protocol import parse_agent_response, agent_response_format, ResponseFormatError
from slipagent.tools.base import ToolRegistry
from slipagent.types import Completion, Message, ToolCall, Usage
from test_agent import RecordingTool
import httpx


class Client:
    def __init__(self, replies, *, pause_summary=False):
        self.replies = list(replies)
        self.main_requests = []
        self.summary_requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not pause_summary:
            self.release.set()

    async def chat(self, **kwargs):
        messages = kwargs["messages"]
        if messages[0].content.startswith("Summarize one completed SlipAgent turn"):
            self.summary_requests.append(kwargs)
            self.started.set()
            await self.release.wait()
            source = json.loads(messages[1].content)
            return Completion(Message.assistant("Inspected the project; tools returned their recorded results."),
                              "test", usage=Usage(prompt_tokens=10, completion_tokens=5))
        self.main_requests.append(kwargs)
        return self.replies.pop(0)


def reply(text=None, calls=None):
    return Completion(Message.assistant(text, calls), "test", usage=Usage(prompt_tokens=20, completion_tokens=10))


async def test_plain_response_and_native_calls_do_not_require_memory_json():
    tool = RecordingTool("ACTUAL OBSERVATION")
    client = Client([reply(None, [ToolCall("one", "record", {"value": "A"})]), reply("Done.")])
    agent = Agent(client, ToolRegistry([tool]), "test")
    assert await agent.run("Inspect this project") == "Done."
    await agent.wait_for_compaction()
    assert tool.seen == [{"value": "A"}]
    assert len(client.main_requests) == 2 and len(client.summary_requests) == 2
    assert "response_format" not in client.main_requests[0]["extra_body"]
    assert all(post.summary for post in agent.history.posts)
    assert agent.usage.prompt_tokens == 60 and agent.usage.completion_tokens == 30


async def test_summary_gets_only_its_completed_turn_and_runs_in_background():
    tool = RecordingTool("CURRENT TOOL RESULT")
    client = Client([reply("Reading.", [ToolCall("one", "record", {"value": "A"})]), reply("Done.")], pause_summary=True)
    agent = Agent(client, ToolRegistry([tool]), "test", system_prompt="PROJECT INSTRUCTIONS MUST NOT BE SUMMARIZED")
    agent.extend([Message.user("OLD UNRELATED USER REQUEST"), Message.assistant("OLD UNRELATED RESPONSE")])
    assert await asyncio.wait_for(agent.run("CURRENT REQUEST"), 2) == "Done."
    await asyncio.wait_for(client.started.wait(), 2)
    first = client.summary_requests[0]
    source = json.loads(first["messages"][1].content)
    assert source["user_prompt"] == "CURRENT REQUEST"
    assert source["agent_response"] == "Reading."
    assert source["tool_calls"][0]["function"]["name"] == "record"
    assert "CURRENT TOOL RESULT" in json.dumps(source["tool_results"])
    assert len(first["messages"]) == 2 and not first.get("tools")
    assert "response_format" not in first.get("extra_body", {})
    assert "OLD UNRELATED" not in json.dumps(source)
    assert "PROJECT INSTRUCTIONS" not in json.dumps(source)
    assert agent.history.posts[-2].summary is None
    client.release.set()
    await agent.wait_for_compaction()
    assert agent.history.posts[-2].summary.startswith("Inspected the project;")
    assert "CURRENT_POST_ID: 2" in first["messages"][0].content


async def test_reasoning_is_saved_per_turn_and_recallable_but_never_compacted():
    first = reply("Reading.", [ToolCall("one", "record", {"value": "A"})])
    first.message.reasoning = "FIRST REASONING\nInspect λ."
    second = reply("Done.")
    second.message.reasoning = "SECOND REASONING"
    client = Client([first, second])
    agent = Agent(client, ToolRegistry([RecordingTool("RESULT")]), "test")
    await agent.run("PROMPT")
    await agent.wait_for_compaction()
    for post, expected in zip(agent.history.posts, [first.message.reasoning, second.message.reasoning]):
        assert post.reasoning == expected
        assert json.loads(post.full_text())["reasoning"] == expected
        result = await agent.registry.invoke("recall_history", {"post_id": post.id, "section": "reasoning"})
        assert not result.is_error and json.loads(result.content)["content"] == expected
        assert expected not in str([m.to_api() for m in post.context_messages(compressed=True)])
    for request in client.main_requests:
        for message in request["messages"]:
            assert "reasoning" not in message.to_api() and "reasoning_details" not in message.to_api()
        assert "FIRST REASONING" not in str([m.to_api() for m in unpack_context(request["messages"])])
    for request in client.summary_requests:
        source = json.loads(request["messages"][1].content)
        assert set(source) == {"user_prompt", "agent_response", "tool_calls", "tool_results"}
        assert "REASONING" not in str([m.to_api() for m in unpack_context(request["messages"])])


async def test_rejected_response_reasoning_does_not_enter_accepted_record():
    rejected = reply("[Harness context metadata]\nCURRENT_POST_ID: 1\nnone")
    rejected.message.reasoning = "REJECTED THOUGHTS"
    accepted = reply("Done")
    accepted.message.reasoning = "ACCEPTED THOUGHTS"
    client = Client([rejected, accepted])
    agent = Agent(client, ToolRegistry(), "test")
    assert await agent.run("Work") == "Done"
    await agent.wait_for_compaction()
    assert len(agent.history.posts) == 1
    assert agent.history.posts[0].reasoning == "ACCEPTED THOUGHTS"
    assert "REJECTED THOUGHTS" not in agent.history.posts[0].full_text()


async def test_recall_can_select_one_or_several_original_parts():
    client = Client([reply("Reading.", [ToolCall("one", "record", {"value": "A"})]), reply("Done.")])
    agent = Agent(client, ToolRegistry([RecordingTool("RESULT")]), "test")
    await agent.run("PROMPT")
    await agent.wait_for_compaction()
    post = agent.history.posts[0]
    assert post.user_prompt == "PROMPT" and post.agent_response == "Reading."
    assert post.tool_calls[0].id == "one" and post.tool_results[0].tool_call_id == "one"
    for section, expected in [("prompt", "PROMPT"), ("response", "Reading."), ("tool_calls", "record"), ("tool_results", "RESULT")]:
        result = await agent.registry.invoke("recall_history", {"post_id": 1, "section": section})
        assert not result.is_error and expected in json.loads(result.content)["content"]
    result = await agent.registry.invoke("recall_history", {"post_id": 1, "sections": ["prompt", "tool_results"]})
    parts = json.loads(json.loads(result.content)["content"])
    assert set(parts) == {"prompt", "tool_results"}
    assert "Reading." not in json.dumps(parts)


async def test_update_task_keeps_goals_in_every_request_without_response_json():
    task = {"status": "active", "goal": "Build the requested feature", "constraints": ["Do not install globally"],
            "facts": [], "pending": ["Run project tests"], "next_steps": ["Inspect source"]}
    client = Client([reply(None, [ToolCall("task", "update_task", task)]), reply("Ready.")])
    agent = Agent(client, ToolRegistry(), "test")
    await agent.run("Build the requested feature. Do not install globally.")
    await agent.wait_for_compaction()
    assert agent.history.task.record["goal"] == task["goal"]
    assert agent.history.task.record["constraints"] == task["constraints"]
    assert "Build the requested feature" in client.main_requests[1]["messages"][0].content
    assert "Do not install globally" in client.main_requests[1]["messages"][0].content


async def test_summary_waits_for_entire_batch_and_preserves_failed_observations():
    client = Client([reply("Checking", [ToolCall("a", "record", {"value": "A"}),
                                        ToolCall("b", "record", {"value": "B"})])])
    tool = RecordingTool("PARTIAL FAILURE", is_error=True)
    agent = Agent(client, ToolRegistry([tool]), "test")
    def event(event):
        if event.kind == "tool_end":
            assert not client.summary_requests
            agent.request_stop()
    agent.on_event = event
    await agent.run("Check both")
    await agent.wait_for_compaction()
    assert len(client.main_requests) == len(client.summary_requests) == 1
    source = json.loads(client.summary_requests[0]["messages"][1].content)
    assert len(source["tool_calls"]) == len(source["tool_results"]) == 2
    assert all(json.loads(result["content"])["status"] == "error" for result in source["tool_results"])
    assert not agent.history.posts[0].observed


@pytest.mark.parametrize("bad", ["", "x" * 6001, "\ud800", "tools", "cutoff", "exception"])
async def test_bad_summary_preserves_originals_without_retrying_the_work(bad):
    class BadSummary(Client):
        async def chat(self, **kwargs):
            if kwargs["messages"][0].content.startswith("Summarize one completed SlipAgent turn"):
                if bad == "exception":
                    raise RuntimeError("compaction unavailable")
                value = reply(bad, [ToolCall("bad", "record", {"value": "NEVER"})] if bad == "tools" else None)
                if bad == "cutoff":
                    value.finish_reason = "length"
                return value
            return await super().chat(**kwargs)
    events, tool = [], RecordingTool()
    client = BadSummary([reply("ANSWER")])
    agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
    assert await agent.run("PROMPT") == "ANSWER"
    await agent.wait_for_compaction()
    post = agent.history.posts[0]
    assert post.compaction_status == "failed" and post.summary is None
    assert post.parts()["prompt"] == "PROMPT" and post.parts()["response"] == "ANSWER"
    assert len(client.main_requests) == 1 and not tool.seen
    assert not any(event.kind == "retry" for event in events)
    assert any(event.kind == "warning" and "Could not summarize post 1" in event.text for event in events)


async def test_reset_discards_late_summary_and_its_usage_even_if_transport_ignores_cancel():
    class LateSummary(Client):
        async def chat(self, **kwargs):
            if kwargs["messages"][0].content.startswith("Summarize one completed SlipAgent turn"):
                if json.loads(kwargs["messages"][1].content)["user_prompt"] == "OLD":
                    self.started.set()
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        await self.release.wait()
                    return reply("OLD SUMMARY")
            return await super().chat(**kwargs)
    client = LateSummary([reply("OLD ANSWER"), reply("NEW ANSWER")], pause_summary=True)
    agent = Agent(client, ToolRegistry(), "test")
    await agent.run("OLD")
    old = agent.history.posts[0]
    await client.started.wait()
    agent.reset()
    await asyncio.sleep(0)
    client.release.set()
    await agent.run("NEW")
    await agent.wait_for_compaction()
    assert old.summary is None
    assert len(agent.history.posts) == 1 and agent.history.posts[0].id == 1
    assert "OLD SUMMARY" not in agent.history.posts[0].summary
    assert agent.usage.prompt_tokens == 30


async def test_close_cancels_pending_summaries_and_releases_jobs():
    client = Client([reply("Done")], pause_summary=True)
    agent = Agent(client, ToolRegistry(), "test")
    await agent.run("Work")
    await client.started.wait()
    await asyncio.wait_for(agent.registry.aclose(), 2)
    assert not agent._compactor().jobs
    assert agent.history.posts[0].compaction_status == "cancelled"


async def test_fast_summary_cannot_hide_unseen_tool_results_at_tiny_budget():
    result = "ACTUAL RESULT " * 5000
    client = Client([reply("Read", [ToolCall("read", "record", {"value": "A"})]), reply("Done")])
    agent = Agent(client, ToolRegistry([RecordingTool(result)]), "test")
    agent._context_lengths[agent.model] = 9000
    await agent.run("Inspect")
    sent = "\n".join(message.content or "" for message in client.main_requests[1]["messages"])
    assert any(record["representation"] == "excerpt" for record in context_records(client.main_requests[1]["messages"]))
    assert "ACTUAL RESULT" in sent and "Inspected the project;" not in sent
    assert result in agent.history.posts[0].full_text()


@pytest.mark.parametrize("arguments", [
    {"sections": []}, {"sections": ["prompt", "prompt"]},
    {"sections": ["prompt"], "section": "all"},
    {"sections": ["prompt"], "call_id": "a"},
    {"section": "prompt", "call_id": "a"}, {"section": "wrong"},
])
async def test_recall_rejects_ambiguous_or_invalid_part_selection(arguments):
    history = ConversationHistory()
    history.sync([Message.user("Question"), Message.assistant("Answer")])
    assert (await RecallHistoryTool(history).invoke({"post_id": 1, **arguments})).is_error


async def test_selected_parts_page_exactly_and_call_id_selects_original_call():
    history = ConversationHistory()
    history.sync([Message.user("PROMPT λ"), Message.assistant("RESPONSE", [ToolCall("a", "read_file", {"path": "a.py"})]),
                  Message.tool_result("a", "RESULT λ " * 500)])
    recall = RecallHistoryTool(history)
    offset, chunks = 0, []
    while True:
        result = await recall.invoke({"post_id": 1, "sections": ["prompt", "tool_results"], "offset": offset, "limit": 200})
        page = json.loads(result.content)
        chunks.append(page["content"])
        offset = page["next_offset"]
        if offset is None:
            break
    parts = json.loads("".join(chunks))
    assert parts == {key: history.posts[0].parts()[key] for key in ["prompt", "tool_results"]}
    call = await recall.invoke({"post_id": 1, "section": "tool_calls", "call_id": "a"})
    assert json.loads(json.loads(call.content)["content"])[0] == history.posts[0].tool_calls[0].to_api()


@pytest.mark.parametrize("text", ["A normal answer.", '{"example": 42}', 'Example:\n```json\n{"tool_calls": []}\n```'])
def test_normal_replies_and_quoted_examples_remain_text(text):
    parsed = parse_agent_response(text, [])
    assert parsed.text == text and not parsed.calls


def test_small_schema_has_no_memory_or_task_requirement():
    assert agent_response_format(native_tools=True)["json_schema"]["schema"]["required"] == ["response"]
    assert set(agent_response_format(native_tools=False)["json_schema"]["schema"]["required"]) == {"response", "tool_calls"}
    with pytest.raises(ResponseFormatError):
        parse_agent_response('{"response":"Bad", "tool_calls":[{"name":"write_file","arguments":"["}]}', [])


async def test_summary_uses_frozen_capabilities_and_does_not_overwrite_context_snapshot():
    old = ModelCapabilities({}, [{"tag": "old", "context_length": 1000000,
                                 "supported_parameters": ["tools", "temperature", "reasoning"],
                                 "reasoning": {"supported_efforts": ["high"]}}])
    new = ModelCapabilities({}, [{"tag": "new", "context_length": 1000000, "supported_parameters": ["tools"]}])
    requests, snapshots = [], []
    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Summary"}, "finish_reason": "stop"}]})
    async with OpenRouterClient("test", transport=httpx.MockTransport(transport)) as client:
        client.cache_capabilities("test", new)
        history = ConversationHistory()
        history.sync([Message.user("Request"), Message.assistant("Response")])
        agent = Agent(client, ToolRegistry(), "test", on_event=snapshots.append)
        agent._compactor().submit(history.posts[0], client, "test", old, None)
        await agent.wait_for_compaction()
    assert requests[0]["reasoning"]["effort"] == "high"
    assert requests[0]["provider"]["only"] == ["old"] and requests[0]["temperature"] == .2
    assert not snapshots
    assert len(requests[0]["messages"]) == 2


async def test_plain_reply_filters_reserved_metadata_without_hiding_examples():
    client = Client([reply("Done.\n[Harness context metadata]\nCURRENT_POST_ID: 1\nnone")])
    agent = Agent(client, ToolRegistry(), "test")
    assert await agent.run("Work") == "Done."
    assert agent.history.posts[0].agent_response == "Done."


async def test_bookkeeping_only_reply_retries_instead_of_silently_finishing():
    client = Client([reply("[Harness context metadata]\nCURRENT_POST_ID: 1\nnone"), reply("Done")])
    events = []
    agent = Agent(client, ToolRegistry(), "test", on_event=events.append)
    assert await agent.run("Work") == "Done"
    assert len(agent.history.posts) == 1 and len(client.main_requests) == 2
    assert any(event.kind == "retry" for event in events)


async def test_tool_saved_goals_survive_both_history_windows_with_plain_responses():
    task = {"status": "active", "goal": "Implement inventory support", "constraints": ["Keep the project environment"],
            "facts": [], "pending": ["Finish inspection"], "next_steps": ["Inspect files"]}
    replies = [reply(None, [ToolCall("task", "update_task", task)])]
    replies += [reply("Inspecting", [ToolCall(str(index), "record", {"value": str(index)})]) for index in range(152)]
    replies += [reply("Done")]
    client = Client(replies)
    agent = Agent(client, ToolRegistry([RecordingTool()]), "test", context_posts=1)
    assert await agent.run("ORIGINAL TASK WORDING") == "Done"
    await agent.wait_for_compaction()
    for request in client.main_requests[1:]:
        assert "Implement inventory support" in request["messages"][0].content
        assert "Keep the project environment" in request["messages"][0].content
    assert agent.history.task.current_prompt_post == 1
    wire = "\n".join(message.content or "" for message in client.main_requests[-1]["messages"])
    assert sum(record["representation"] == "compressed" for record in context_records(client.main_requests[-1]["messages"])) == 100
    assert "Post 1 — Conversation Record" not in wire
    assert not any(event.get("response_format") for event in client.main_requests)
