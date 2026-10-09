"""Normal replies and isolated, asynchronous whole-step compaction."""

from slipagent.types import content_text

from slipagent.types import decode_json_content

import asyncio
import json
from data_text_reader import read_data

import pytest

from test_agent import unpack_context, context_records

from slipagent.agent import Agent
from slipagent.context import ConversationHistory, RecallHistoryTool
from slipagent.capabilities import ModelCapabilities
from slipagent.openrouter import OpenRouterClient, RetryPolicy
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
        if content_text(messages[0].content).lstrip().startswith("# Summarizing the Previous Turn"):
            self.summary_requests.append(kwargs)
            self.started.set()
            await self.release.wait()
            source = decode_json_content(messages[1].content)
            return Completion(Message.assistant(json.dumps({'summary': "Inspected the project; tools returned their recorded results.", 'reasoning_summary': source.get('reasoning', '')})),
                              "test", usage=Usage(prompt_tokens=10, completion_tokens=5))
        self.main_requests.append(kwargs)
        return self.replies.pop(0)


def reply(text=None, calls=None):
    return Completion(Message.assistant(text, calls), "test", usage=Usage(prompt_tokens=20, completion_tokens=10))


@pytest.mark.parametrize("recover", [True, False])
@pytest.mark.parametrize("failure", ["overload", "bad-gzip"])
async def test_summary_retries_transient_failures_with_bounded_provider_policy(monkeypatch, recover, failure):
    from slipagent.compaction import StepCompactor
    attempts, delays, errors = [], [], []
    def handle(request):
        attempts.append(json.loads(request.content))
        if len(attempts) < 6 or not recover:
            if failure == "bad-gzip":
                return httpx.Response(200, headers={"content-encoding": "gzip"}, content=b"not a gzip stream")
            return httpx.Response(200, json={"error": {"message": "Service temporarily overloaded", "code": 503}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps({'summary': 'Completed the work.', 'reasoning_summary': ''})}, "finish_reason": "stop"}]})
    history = ConversationHistory()
    history.sync([Message.user("Request"), Message.assistant("Response")])
    step = history.steps[0]
    original = step.full_text()
    compactor = StepCompactor(lambda usage: None, errors.append)
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle), retry=RetryPolicy(extended=True)) as client:
        async def sleep(delay):
            delays.append(delay)
            assert errors == []
        monkeypatch.setattr("slipagent.api.asyncio.sleep", sleep)
        monkeypatch.setattr("slipagent.api.random.uniform", lambda lower, upper: upper)
        compactor.submit(step, client, "test", None, None)
        await compactor.wait()
    assert len(attempts) == (6 if recover else 7)
    assert delays == ([1, 2, 4, 8, 16] if recover else [1, 2, 4, 8, 16, 32])
    assert all(request == attempts[0] for request in attempts)
    assert step.full_text() == original
    assert len(errors) == (0 if recover else 1)
    assert step.compaction_status == ("complete" if recover else "failed")


async def test_plain_response_and_native_calls_do_not_require_memory_json():
    tool = RecordingTool("ACTUAL OBSERVATION")
    client = Client([reply("Reading.", [ToolCall("one", "record", {"value": "A"})]), reply("Done.")])
    agent = Agent(client, ToolRegistry([tool]), "test")
    assert await agent.run("Inspect this project") == "Done."
    await agent.wait_for_compaction()
    assert tool.seen == [{"value": "A"}]
    assert len(client.main_requests) == 2 and len(client.summary_requests) == 2
    assert "response_format" not in client.main_requests[0]["extra_body"]
    assert all(step.summary for step in agent.history.steps)
    assert agent.usage.prompt_tokens == 60 and agent.usage.completion_tokens == 30


async def test_summary_gets_current_turn_and_separate_history_and_runs_in_background():
    tool = RecordingTool("CURRENT TOOL RESULT")
    client = Client([reply("Reading.", [ToolCall("one", "record", {"value": "A"})]), reply("Done.")], pause_summary=True)
    agent = Agent(client, ToolRegistry([tool]), "test", system_prompt="PROJECT INSTRUCTIONS MUST NOT BE SUMMARIZED")
    agent.extend([Message.user("OLD UNRELATED USER REQUEST"), Message.assistant("OLD UNRELATED RESPONSE")])
    assert await asyncio.wait_for(agent.run("CURRENT REQUEST"), 2) == "Done."
    await asyncio.wait_for(client.started.wait(), 2)
    first = client.summary_requests[0]
    source = decode_json_content(first["messages"][1].content)
    assert source["user_prompt"] == "CURRENT REQUEST"
    assert source["agent_response"] == "Reading."
    assert source["tool_calls"][0]["function"]["name"] == "record"
    assert "CURRENT TOOL RESULT" in json.dumps(source["tool_results"])
    assert len(first["messages"]) == 2 and not first.get("tools")
    assert "response_format" not in first.get("extra_body", {})
    assert 'OLD UNRELATED' in json.dumps(source['history'])
    assert 'OLD UNRELATED' not in json.dumps({k:v for k,v in source.items() if k != 'history'})
    assert "PROJECT INSTRUCTIONS" not in json.dumps(source)
    assert agent.history.steps[-2].summary is None
    client.release.set()
    await agent.wait_for_compaction()
    assert agent.history.steps[-2].summary.startswith("Inspected the project;")


async def test_reasoning_is_saved_recallable_and_compacted_separately():
    first = reply("Reading.", [ToolCall("one", "record", {"value": "A"})])
    first.message.reasoning = "FIRST REASONING\nInspect λ."
    second = reply("Done.")
    second.message.reasoning = "SECOND REASONING"
    client = Client([first, second])
    agent = Agent(client, ToolRegistry([RecordingTool("RESULT")]), "test")
    await agent.run("PROMPT")
    await agent.wait_for_compaction()
    for step, expected in zip(agent.history.steps, [first.message.reasoning, second.message.reasoning]):
        assert step.reasoning == expected
        assert read_data(step.full_text())["reasoning"] == expected
        result = await agent.registry.invoke("recall_history", {"step_id": step.id, "section": "reasoning"})
        assert not result.is_error and decode_json_content(result.content)["content"] == expected
        assert expected not in str([m.to_api() for m in step.context_messages(compressed=True)])
    for index, request in enumerate(client.main_requests):
        for message in request["messages"]:
            assert "reasoning" not in message.to_api() and "reasoning_details" not in message.to_api()
        thoughts = [record["reasoning"] for record in context_records(request["messages"]) if "reasoning" in record]
        assert thoughts == ([first.message.reasoning] if index else [])
    for request in client.summary_requests:
        source = decode_json_content(request["messages"][1].content)
        assert set(source) == {"user_prompt", "agent_response", "tool_calls", "tool_results", "reasoning", "history", "history_omitted"}
        assert 'REASONING' in source['reasoning']


async def test_rejected_response_reasoning_does_not_enter_accepted_record():
    rejected = reply("[Harness context metadata]\nnone")
    rejected.message.reasoning = "REJECTED THOUGHTS"
    accepted = reply("Done")
    accepted.message.reasoning = "ACCEPTED THOUGHTS"
    client = Client([rejected, accepted])
    agent = Agent(client, ToolRegistry(), "test")
    assert await agent.run("Work") == "Done"
    await agent.wait_for_compaction()
    assert len(agent.history.steps) == 1
    assert agent.history.steps[0].reasoning == "ACCEPTED THOUGHTS"
    assert "REJECTED THOUGHTS" not in agent.history.steps[0].full_text()


async def test_recall_can_select_one_or_several_original_parts():
    client = Client([reply("Reading.", [ToolCall("one", "record", {"value": "A"})]), reply("Done.")])
    agent = Agent(client, ToolRegistry([RecordingTool("RESULT")]), "test")
    await agent.run("PROMPT")
    await agent.wait_for_compaction()
    step = agent.history.steps[0]
    assert step.user_prompt == "PROMPT" and step.agent_response == "Reading."
    assert step.tool_calls[0].id == "one" and step.tool_results[0].tool_call_id == "one"
    for section, expected in [("prompt", "PROMPT"), ("response", "Reading."), ("tool_calls", "record"), ("tool_results", "RESULT")]:
        result = await agent.registry.invoke("recall_history", {"step_id": 1, "section": section})
        assert not result.is_error and expected in content_text(result.content["content"])
    result = await agent.registry.invoke("recall_history", {"step_id": 1, "sections": ["prompt", "tool_results"]})
    parts = decode_json_content(decode_json_content(result.content)["content"])
    assert set(parts) == {"prompt", "tool_results"}
    assert "Reading." not in json.dumps(parts)


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
    source = decode_json_content(client.summary_requests[0]["messages"][1].content)
    assert len(source["tool_calls"]) == len(source["tool_results"]) == 2
    assert all(result["content"]["status"] == "error" for result in source["tool_results"])
    assert not agent.history.steps[0].observed


@pytest.mark.parametrize("bad", ["", "x" * 6001, "\ud800", "tools", "cutoff", "exception"])
async def test_bad_summary_preserves_originals_without_retrying_the_work(bad, monkeypatch):
    original_sleep = asyncio.sleep
    async def immediate_backoff(delay):
        await original_sleep(0)
    monkeypatch.setattr("slipagent.compaction.asyncio.sleep", immediate_backoff)
    class BadSummary(Client):
        async def chat(self, **kwargs):
            if content_text(kwargs["messages"][0].content).lstrip().startswith("# Summarizing the Previous Turn"):
                self.summary_requests.append(kwargs)
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
    step = agent.history.steps[0]
    assert step.compaction_status == "failed" and step.summary is None
    assert step.parts()["prompt"] == "PROMPT" and step.parts()["response"] == "ANSWER"
    assert len(client.main_requests) == 1 and not tool.seen
    assert len(client.summary_requests) == 6
    assert not any(event.kind == "retry" for event in events)
    assert any(event.kind == "warning" and "Could not summarize step 1" in event.text for event in events)


async def test_reset_discards_late_summary_and_its_usage_even_if_transport_ignores_cancel():
    class LateSummary(Client):
        async def chat(self, **kwargs):
            if content_text(kwargs["messages"][0].content).lstrip().startswith("# Summarizing the Previous Turn"):
                if decode_json_content(kwargs["messages"][1].content)["user_prompt"] == "OLD":
                    self.started.set()
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        await self.release.wait()
                    return reply(json.dumps({'summary': 'OLD SUMMARY', 'reasoning_summary': ''}))
            return await super().chat(**kwargs)
    client = LateSummary([reply("OLD ANSWER"), reply("NEW ANSWER")], pause_summary=True)
    agent = Agent(client, ToolRegistry(), "test")
    await agent.run("OLD")
    old = agent.history.steps[0]
    await client.started.wait()
    agent.reset()
    await asyncio.sleep(0)
    client.release.set()
    await agent.run("NEW")
    await agent.wait_for_compaction()
    assert old.summary is None
    assert len(agent.history.steps) == 1 and agent.history.steps[0].id == 1
    assert "OLD SUMMARY" not in agent.history.steps[0].summary
    assert agent.usage.prompt_tokens == 30


async def test_close_cancels_pending_summaries_and_releases_jobs():
    client = Client([reply("Done")], pause_summary=True)
    agent = Agent(client, ToolRegistry(), "test")
    await agent.run("Work")
    await client.started.wait()
    await asyncio.wait_for(agent.registry.aclose(), 2)
    assert not agent._compactor().jobs
    assert agent.history.steps[0].compaction_status == "cancelled"


async def test_fast_summary_cannot_hide_unseen_tool_results_at_tiny_budget():
    result = "ACTUAL RESULT " * 5000
    client = Client([reply("Read", [ToolCall("read", "record", {"value": "A"})]), reply("Done")])
    agent = Agent(client, ToolRegistry([RecordingTool(result)]), "test")
    agent._context_lengths[agent.model] = 12000
    await agent.run("Inspect")
    sent = "\n".join(content_text(message.content) for message in client.main_requests[1]["messages"])
    assert any(record["representation"] == "excerpt" for record in context_records(client.main_requests[1]["messages"]))
    assert "ACTUAL RESULT" in sent and "Inspected the project;" not in sent
    assert result in agent.history.steps[0].full_text()


@pytest.mark.parametrize("arguments", [
    {"sections": []}, {"sections": ["prompt", "prompt"]},
    {"sections": ["prompt"], "section": "all"},
    {"sections": ["prompt"], "call_id": "a"},
    {"section": "prompt", "call_id": "a"}, {"section": "wrong"},
])
async def test_recall_rejects_ambiguous_or_invalid_part_selection(arguments):
    history = ConversationHistory()
    history.sync([Message.user("Question"), Message.assistant("Answer")])
    assert (await RecallHistoryTool(history).invoke({"step_id": 1, **arguments})).is_error


async def test_selected_parts_page_exactly_and_call_id_selects_original_call():
    history = ConversationHistory()
    history.sync([Message.user("PROMPT λ"), Message.assistant("RESPONSE", [ToolCall("a", "read_file", {"path": "a.py"})]),
                  Message.tool_result("a", "RESULT λ " * 500)])
    recall = RecallHistoryTool(history)
    offset, chunks = 0, []
    while True:
        result = await recall.invoke({"step_id": 1, "sections": ["prompt", "tool_results"], "offset": offset, "limit": 200})
        page = decode_json_content(result.content)
        chunks.append(page["content"])
        offset = page["next_offset"]
        if offset is None:
            break
    parts = read_data("".join(chunks))
    assert parts == {key: history.steps[0].parts()[key] for key in ["prompt", "tool_results"]}
    call = await recall.invoke({"step_id": 1, "section": "tool_calls", "call_id": "a"})
    assert decode_json_content(decode_json_content(call.content)["content"])[0] == history.steps[0].tool_calls[0].to_record()


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
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps({'summary': 'Summary', 'reasoning_summary': ''})}, "finish_reason": "stop"}]})
    async with OpenRouterClient("test", transport=httpx.MockTransport(transport)) as client:
        client.cache_capabilities("test", new)
        history = ConversationHistory()
        history.sync([Message.user("Request"), Message.assistant("Response")])
        agent = Agent(client, ToolRegistry(), "test", on_event=snapshots.append)
        agent._compactor().submit(history.steps[0], client, "test", old, None)
        await agent.wait_for_compaction()
    assert requests[0]["reasoning"]["effort"] == "high"
    assert requests[0]["provider"]["only"] == ["old"] and requests[0]["temperature"] == .2
    assert not snapshots
    assert len(requests[0]["messages"]) == 2


async def test_plain_reply_filters_reserved_metadata_without_hiding_examples():
    client = Client([reply("Done.\n[Harness context metadata]\nnone")])
    agent = Agent(client, ToolRegistry(), "test")
    assert await agent.run("Work") == "Done."
    assert agent.history.steps[0].agent_response == "Done."


async def test_compaction_disables_transport_timeout_without_changing_normal_requests():
    timeouts = []

    def transport(request):
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps({'summary': 'Summary', 'reasoning_summary': ''})}, "finish_reason": "stop"}]})

    async with OpenRouterClient("test", timeout=42, transport=httpx.MockTransport(transport)) as client:
        history = ConversationHistory()
        history.sync([Message.user("Request"), Message.assistant("Response")])
        registry = ToolRegistry()
        agent = Agent(client, registry, "test")
        await client.chat(model="test", messages=[Message.user("Before")])
        agent._compactor().submit(history.steps[0], client, "test", None, None)
        await agent.wait_for_compaction()
        await client.chat(model="test", messages=[Message.user("After")])
        await registry.aclose()

    assert history.steps[0].compaction_status == "complete"
    assert timeouts[0] == timeouts[2] == {"connect": 42, "read": 42, "write": 42, "pool": 42}
    assert timeouts[1] == {"connect": None, "read": None, "write": None, "pool": None}


async def test_compaction_timeout_preserves_original_and_reports_failure(monkeypatch):
    monkeypatch.setattr("slipagent.compaction.COMPACTION_TIMEOUT", .01)
    monkeypatch.setattr("slipagent.compaction.COMPACTION_RETRIES", 0)
    client = Client([], pause_summary=True)
    history = ConversationHistory()
    history.sync([Message.user("Request"), Message.assistant("Response")])
    original = history.steps[0].full_text()
    events = []
    registry = ToolRegistry()
    agent = Agent(client, registry, "test", on_event=events.append)
    agent._compactor().submit(history.steps[0], client, "test", None, None)
    await agent.wait_for_compaction()
    assert history.steps[0].compaction_status == "failed"
    assert "TimeoutError" in history.steps[0].compaction_error
    assert history.steps[0].summary is None
    assert history.steps[0].full_text() == original
    assert any("Could not summarize" in event.text for event in events)
    assert not agent._compactor().jobs
    await registry.aclose()


async def test_bookkeeping_only_reply_retries_instead_of_silently_finishing():
    client = Client([reply("[Harness context metadata]\nnone"), reply("Done")])
    events = []
    agent = Agent(client, ToolRegistry(), "test", on_event=events.append)
    assert await agent.run("Work") == "Done"
    assert len(agent.history.steps) == 1 and len(client.main_requests) == 2
    assert any(event.kind == "retry" for event in events)
