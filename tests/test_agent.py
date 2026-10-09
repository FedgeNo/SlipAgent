"""The agent loop, driven by a stub client so no network is involved."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from slipagent.agent import Agent, AgentEvent, STEP_LIMIT_NOTICE, build_system_prompt
from slipagent.tools.base import Tool, ToolRegistry, ToolResult
from slipagent.types import Completion, Message, ToolCall, Usage, decode_json_content, content_text
from slipagent.context import response_memory, tool_response_memory
from slipagent.prompts import load_prompt
from data_text_reader import input_data, read_data


def context_messages(messages):
    """Expose system-embedded history records to behavioral fixture readers."""
    for source in messages:
        message = source if isinstance(source, Message) else Message.from_api(source)
        yield message
        if message.role == "system":
            _, marker, body = content_text(message.content).partition("\n" + load_prompt("history-opening.md").strip() + "\n")
            if marker:
                records = read_data(body.lstrip())
                yield from (Message.user(record) for record in records)


def context_records(messages):
    """Read the actual one-object-per-step request format for assertions."""
    records = []
    for message in context_messages(messages):
        raw = message.to_api() if isinstance(message, Message) else message
        if raw["role"] != "system":
            record = input_data(raw["content"])
            assert record["record_type"] in {"history_step", "current_step"}
            records.append(record)
    return records


def unpack_context(messages):
    """Compare original parts in literal input records with older behavioral fixtures.

    This is only an assertion/fixture reader. Clients retain the real JSON
    requests, which format-specific tests inspect using context_records.
    """
    result = []
    for source in context_messages(messages):
        message = source if isinstance(source, Message) else Message.from_api(source)
        try:
            record = input_data(message.content)
        except ValueError:
            record = None
        if not isinstance(record, dict) or record.get("record_type") not in {"history_step", "current_step"}:
            result.append(message)
            continue
        result.extend(Message.user(text) for text in record.get("user_prompt", []))
        if record["representation"] == "compressed":
            result.append(Message.assistant(record["compressed_summary"]))
        elif record["representation"] == "excerpt":
            result.append(Message.assistant(json.dumps(record, ensure_ascii=False)))
        else:
            calls = [ToolCall(call["call_id"], call["tool_name"], call["arguments"]) for call in record.get("tool_calls", [])]
            if record["record_type"] == "history_step" or record.get("agent_response") is not None or calls:
                result.append(Message.assistant(record.get("agent_response"), calls))
            for item in record.get("tool_results", []):
                content = item["content"]
                if item["status"] != "unknown":
                    content = {"tool": item["tool_name"], "call_id": item["call_id"],
                               "status": item["status"], "content": content}
                result.append(Message.tool_result(item["call_id"], content))
    return result


def unpack_api_context(messages):
    return [message.to_api() for message in unpack_context(messages)]


def task_record(messages=None, *, revision=1, **changes):
    system = "\n".join(m.get("content") or "" for m in messages or [] if m["role"] == "system")
    match = re.search(r"TASK_SOURCE_REVISION: (\d+)", system)
    return {"source_revision": int(match[1]) if match else revision, "status": "active",
            "goal": "Complete the user's coding task", "constraints": [], "facts": [],
            "pending": [], "next_steps": [], **changes}


def context_body(content: str | None) -> str:
    """Read original content inside the harness heading/observation envelope."""
    if isinstance(content, dict):
        return content.get("content", content_text(content))
    text = content_text(content)
    if text.lstrip().startswith(("## System Instructions", "## Current User Request", "### Step ",
                        "============================= BEGIN SYSTEM INSTRUCTIONS (FULL) ==============================", "System Instructions (Full):", "Current User Request (Full):",
                        "Conversation Record (Full):", "Conversation Record (Excerpt):",
                        "User Request (Full):", "Agent Response (Full):",
                        "Agent Response and Tool Calls (Full):", "Tool Result:")):
        text = text.lstrip().partition("\n\n")[2]
    try:
        observation = json.loads(text)
    except ValueError:
        observation = None
    if isinstance(observation, dict) and observation.keys() == {"tool", "call_id", "status", "content"}:
        return observation["content"]
    return text


def summary_response(body):
    """Serve isolated compaction without consuming the scripted working steps."""
    messages = body.get("messages", [])
    if not messages or not (messages[0].get("content") or "").lstrip().startswith("# Summarizing the Previous Turn"):
        return None
    source = input_data(messages[1]["content"])
    text = f"User: {source['user_prompt'][:300]}; agent: {source['agent_response'][:300]}; results: {str(source['tool_results'])[:300]}"
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps({'summary': text, 'reasoning_summary': ''})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}}


def context_step_ids(messages):
    current, previous = None, None
    for source in context_messages(messages):
        message = source.to_api()
        if message["role"] != "user":
            continue
        try:
            record = input_data(message.get("content") or "")
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("record_type") == "current_step":
            current = record["step_id"]
        elif record.get("record_type") == "history_step" and record.get("tool_results"):
            previous = record["step_id"]
    return current, previous


def inline_memory(text: str | None, messages: list[dict[str, Any]]) -> str:
    """Supply legacy envelopes to fixtures exercising older response formats."""
    current_id, previous_id = context_step_ids(messages)
    messages = unpack_api_context(messages)
    content = text or ""
    if "<slipagent_context>" in content:
        return content
    system = "\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
    if current_id is None:
        return content
    request = next((context_body(m.get("content")) for m in reversed(messages) if m["role"] == "user"), "continuation")
    record: dict[str, Any] = {
        "step": current_id,
        "summary": f"User requested {request[:300]}; model responded {content[:300] or 'with tool calls'}.",
    }
    if previous_id is not None:
        results = " ".join(context_body(m.get("content")) for m in messages if m["role"] == "tool")
        record["previous"] = {"step": previous_id, "summary": f"User requested {request[:300]}; model ran tools and found {results[:300]}."}
    return content + "\n<slipagent_context>" + json.dumps(record) + "</slipagent_context>"


def structured_message(message: dict[str, Any], messages: list[dict[str, Any]], *, include_memory: bool = True) -> dict[str, Any]:
    """Supply canned replies in the response contract requested by the harness."""
    system = "\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
    if "# Replies and Native Tool Calls" in system:
        text = message.get("content") or "Requesting tools."
        if text.lstrip().startswith('{"response"'):
            return message
        if not text.lstrip().startswith('{"response"'):
            memory, _ = tool_response_memory(text, [], 1, None)
            text = memory.text or "Requesting tools."
        if "Return exactly one JSON object:" in system:
            return {**message, "content": json.dumps({"response": text})}
        return {**message, "content": text}
    current_id, previous_id = context_step_ids(messages)
    messages = unpack_api_context(messages)
    text = message.get("content") or ""
    try:
        existing = json.loads(text)
    except ValueError:
        existing = None
    if isinstance(existing, dict) and any(key in existing for key in ("response", "agent_response_compressed", "tool_calls")):
        return message
    system = "\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
    step_id = current_id or 1
    if include_memory:
        text = inline_memory(text, messages)
    memory, calls = tool_response_memory(text, [ToolCall.from_api(call) for call in message.get("tool_calls") or []], step_id, previous_id)
    request = next((context_body(m.get("content")) for m in reversed(messages) if m["role"] == "user"), "continuation")
    results = " ".join(context_body(m.get("content")) for m in messages if m["role"] == "tool")
    record = {
        "task": task_record(messages),
        "response": memory.text,
        "tool_calls": [{"id": call.id, "name": call.name, "arguments": json.dumps(call.arguments)} for call in calls],
        "previous_tool_responses_compressed": (memory.previous or results[:1000] or "Previous tools returned their results.") if previous else "",
        "user_prompt_compressed": "User requested " + request[:1000],
        "agent_response_compressed": memory.summary,
    }
    return {"role": "assistant", "content": json.dumps(record)}


class StubClient:
    """Returns canned completions and records the requests it received.

    Copy each message list so later mutations cannot rewrite recorded requests.
    """

    def __init__(self, responses: list[Completion], *, include_memory: bool = True) -> None:
        self.responses = list(responses)
        self.include_memory = include_memory
        self.calls: list[dict[str, Any]] = []
        self.summary_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Completion:
        if content_text(kwargs["messages"][0].content).lstrip().startswith("# Summarizing the Previous Turn"):
            self.summary_calls.append(kwargs)
            source = decode_json_content(kwargs["messages"][1].content)
            text = f"User: {source['user_prompt'][:300]}; agent: {source['agent_response'][:300]}; results: {str(source['tool_results'])[:300]}"
            return Completion(Message.assistant(json.dumps({'summary': text, 'reasoning_summary': source.get('reasoning', '')})), "stub", usage=Usage(cost=0))
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.responses:
            raise AssertionError("stub client ran out of responses")
        response = self.responses.pop(0)
        wire = structured_message(response.message.to_api(), [m.to_api() for m in kwargs["messages"]], include_memory=self.include_memory)
        response = replace(response, message=Message.from_api(wire))
        return response


class RecordingTool(Tool):
    name = "record"
    description = "Records the arguments it was called with."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def __init__(self, result: str = "ok", is_error: bool = False) -> None:
        self.result = result
        self.is_error = is_error
        self.seen: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> ToolResult:
        self.seen.append(kwargs)
        return ToolResult(self.result, self.is_error)


def completion(
    text: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    *,
    finish_reason: str = "stop",
    cost: float | None = 0.01,
) -> Completion:
    return Completion(
        message=Message.assistant(text, tool_calls),
        model="test/model",
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=cost),
    )


def call(name: str = "record", **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


def build_agent(
    responses: list[Completion],
    tools: list[Tool] | None = None,
    **kwargs: Any,
) -> tuple[Agent, StubClient]:
    client = StubClient(responses)
    registry = ToolRegistry(tools if tools is not None else [RecordingTool()])
    agent = Agent(
        client=client,  # type: ignore[arg-type]
        registry=registry,
        model="test/model",
        system_prompt="you are a test agent",
        **kwargs,
    )
    return agent, client


# --------------------------------------------------------------------------- #
# Loop mechanics
# --------------------------------------------------------------------------- #


async def test_pending_correction_is_delivered_before_resumed_tools():
    from slipagent.openrouter import OpenRouterError
    tool = RecordingTool()
    agent, client = build_agent([completion(tool_calls=[call(value="act")]), completion("done")], tools=[tool])
    original = client.chat
    async def fail(**kwargs):
        agent.enqueue("Never change settings")
        raise OpenRouterError("offline")
    client.chat = fail
    import pytest
    with pytest.raises(OpenRouterError):
        await agent.run("Initial task")
    client.chat = original
    await agent.run("Continue")
    requests = [context_body(m.content) for m in unpack_context(client.calls[0]["messages"]) if m.role == "user"]
    assert requests == ["Initial task", "Never change settings", "Continue"]


async def test_error_status_reaches_model_and_archive():
    agent, client = build_agent([completion(tool_calls=[call(value="act")]), completion("done")],
                                tools=[RecordingTool("same body", is_error=True)])
    await agent.run("Task")
    result = next(m for m in unpack_context(client.calls[1]["messages"]) if m.role == "tool")
    assert result.content["status"] == "error"
    assert result.content["tool"] == "record"
    assert "same body" in content_text(result.content)
    assert next(m for m in agent.messages if m.role == "tool").content["status"] == "error"


async def test_broken_dispatch_cannot_leave_an_unanswered_tool_batch():
    agent, client = build_agent([completion(tool_calls=[call(value="act")]), completion("done")])
    async def broken(*args):
        raise AttributeError("broken registry")
    agent.registry.invoke = broken
    assert await agent.run("Task") == "done"
    results = [m for m in unpack_context(client.calls[1]["messages"]) if m.role == "tool"]
    assert len(results) == 1
    assert "broken registry" in content_text(results[0].content)


async def test_returns_final_text_without_tools() -> None:
    agent, client = build_agent([completion(text="All done.")])

    assert await agent.run("do the thing") == "All done."
    assert len(client.calls) == 1


async def test_tool_call_cycle_executes_then_answers() -> None:
    tool = RecordingTool(result="file contents")
    agent, client = build_agent(
        [
            completion(tool_calls=[call(value="a")]),
            completion(text="Read it."),
        ],
        tools=[tool],
    )

    answer = await agent.run("read it")

    assert answer == "Read it."
    assert tool.seen == [{"value": "a"}]
    assert len(client.calls) == 2


async def test_tool_result_is_appended_with_matching_id() -> None:
    tool = RecordingTool(result="payload")
    agent, _ = build_agent(
        [completion(tool_calls=[call(value="x")]), completion(text="done")],
        tools=[tool],
    )

    await agent.run("go")

    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_record"
    assert decode_json_content(tool_messages[0].content)["content"] == "payload"


async def test_assistant_tool_turn_preserves_reply_text() -> None:
    """Tool-call steps retain any accompanying reply text in history."""
    agent, _ = build_agent(
        [
            completion(text="thinking out loud", tool_calls=[call(value="x")]),
            completion(text="done"),
        ]
    )

    await agent.run("go")

    tool_turns = [m for m in agent.messages if m.tool_calls]
    assert len(tool_turns) == 1
    assert tool_turns[0].content == "thinking out loud"


async def test_parallel_tool_calls_all_run() -> None:
    tool = RecordingTool(result="r")
    agent, client = build_agent(
        [
            completion(
                tool_calls=[
                    ToolCall(id="c1", name="record", arguments={"value": "one"}),
                    ToolCall(id="c2", name="record", arguments={"value": "two"}),
                ]
            ),
            completion(text="done"),
        ],
        tools=[tool],
    )

    await agent.run("go")

    assert [entry["value"] for entry in tool.seen] == ["one", "two"]
    assert len([m for m in agent.messages if m.role == "tool"]) == 2


async def test_unknown_tool_error_is_fed_back_to_model() -> None:
    agent, client = build_agent(
        [
            completion(tool_calls=[call(name="does_not_exist")]),
            completion(text="recovered"),
        ]
    )

    answer = await agent.run("go")

    assert answer == "recovered"
    tool_message = [m for m in agent.messages if m.role == "tool"][0]
    assert "Unknown tool" in content_text(tool_message.content)
    # The model must receive the tool error before its next step.
    assert unpack_context(client.calls[1]["messages"])[-1].role == "tool"


async def test_tool_error_is_fed_back_to_model() -> None:
    tool = RecordingTool(result="permission denied", is_error=True)
    agent, client = build_agent(
        [
            completion(tool_calls=[call(value="x")]),
            completion(text="I see, denied."),
        ],
        tools=[tool],
    )

    assert await agent.run("go") == "I see, denied."
    assert context_body(unpack_context(client.calls[1]["messages"])[-1].content) == "permission denied"


async def test_conversation_history_accumulates_across_turns() -> None:
    agent, _ = build_agent([completion(text="one"), completion(text="two")])

    await agent.run("first")
    await agent.run("second")

    roles = [m.role for m in agent.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]


# --------------------------------------------------------------------------- #
# Limits, usage, events
# --------------------------------------------------------------------------- #


async def test_step_limit_stops_the_loop() -> None:
    responses = [completion(tool_calls=[call(value=str(i))]) for i in range(5)]
    agent, client = build_agent(responses, max_steps=3)

    answer = await agent.run("go")

    assert answer == STEP_LIMIT_NOTICE
    assert len(client.calls) == 3


async def test_default_step_budget_is_generous() -> None:
    """Long work must not be cut off just for taking many steps."""
    agent, _ = build_agent([completion(text="hi")])

    assert agent.max_steps >= 100


# --------------------------------------------------------------------------- #
# Mid-step input
# --------------------------------------------------------------------------- #


async def test_queued_message_reaches_the_model_as_user_input() -> None:
    """Text typed mid-step joins the conversation, clearly separated."""
    agent, client = build_agent(
        [completion(tool_calls=[call(value="x")]), completion(text="done")]
    )
    agent.on_event = lambda event: agent.enqueue("actually, use pytest") if event.kind == "step_start" and event.step == 1 else None

    await agent.run("go")

    queued = [m for m in unpack_context(client.calls[1]["messages"]) if m.role == "user"]
    assert [context_body(m.content) for m in queued] == ["go", "actually, use pytest"]


async def test_queued_message_lands_after_the_tool_result() -> None:
    """A user message must not split an assistant tool-call from its results."""
    agent, client = build_agent(
        [completion(tool_calls=[call(value="x")]), completion(text="done")]
    )
    agent.on_event = lambda event: agent.enqueue("wait, do this instead") if event.kind == "tool_start" and event.step == 1 else None

    await agent.run("go")

    roles = [m.role for m in unpack_context(client.calls[1]["messages"])]
    assert roles == ["system", "user", "assistant", "tool", "user"]


async def test_queue_is_drained_once() -> None:
    agent, _ = build_agent(
        [completion(tool_calls=[call(value="x")]), completion(text="done")]
    )
    agent.enqueue("only once")

    await agent.run("go")

    assert agent.pending == []


async def test_blank_input_is_not_queued() -> None:
    agent, _ = build_agent([completion(text="hi")])

    agent.enqueue("   ")

    assert agent.pending == []


async def test_message_typed_during_a_finishing_turn_survives() -> None:
    """Text arriving on the last step stays queued for the next step."""
    agent, _ = build_agent([completion(text="done")])
    agent.on_event = lambda event: agent.enqueue("one more thing") if event.kind == "assistant_text" and event.step == 1 else None

    await agent.run("go")

    assert agent.pending == ["one more thing"]


async def test_queued_input_emits_an_event() -> None:
    agent, _ = build_agent([completion(text="hi")])
    events: list[AgentEvent] = []
    agent.on_event = events.append

    agent.enqueue("a note")

    assert any(e.kind == "user_message" and e.text == "a note" for e in events)


# --------------------------------------------------------------------------- #
# Batched tool calls
# --------------------------------------------------------------------------- #


async def test_a_whole_batch_runs_in_one_turn() -> None:
    """Several calls in one message must all execute before the next request."""
    tool = RecordingTool(result="r")
    batch = [
        ToolCall(id="1", name="record", arguments={"value": "a"}),
        ToolCall(id="2", name="record", arguments={"value": "b"}),
        ToolCall(id="3", name="record", arguments={"value": "c"}),
    ]
    agent, client = build_agent(
        [completion(tool_calls=batch), completion(text="done")], tools=[tool]
    )

    await agent.run("go")

    assert [entry["value"] for entry in tool.seen] == ["a", "b", "c"]
    # One model request covered all three: no round trip between them.
    assert len(client.calls) == 2
    roles = [m.role for m in unpack_context(client.calls[1]["messages"])]
    assert roles == ["system", "user", "assistant", "tool", "tool", "tool"]


async def test_a_failing_call_does_not_abort_the_rest_of_the_batch() -> None:
    """One error in a batch must not cost the model the other results."""
    class Flaky(Tool):
        name = "flaky"
        description = "Fails sometimes."
        parameters = {"type": "object", "properties": {}}

        async def run(self, **kwargs: Any) -> ToolResult:
            return ToolResult.error("nope")

    good = RecordingTool(result="fine")
    agent, _ = build_agent(
        [
            completion(
                tool_calls=[
                    ToolCall(id="1", name="flaky", arguments={}),
                    ToolCall(id="2", name="record", arguments={"value": "b"}),
                ]
            ),
            completion(text="done"),
        ],
        tools=[Flaky(), good],
    )

    await agent.run("go")

    results = [m.content for m in agent.messages if m.role == "tool"]
    assert len(results) == 2
    assert results[0]["content"] == "nope"
    assert results[1]["content"] == "fine"


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #


def test_system_prompt_asks_for_batched_tool_calls() -> None:
    """The prompt encourages batching independent calls."""
    prompt = build_system_prompt("/tmp/ws")

    assert "Batch all predictable independent calls in one step" in prompt
    assert "group independent checks" in prompt


def test_system_prompt_allows_splitting_on_real_dependencies() -> None:
    """Batching must not become a rule that breaks read-then-edit work."""
    prompt = build_system_prompt("/tmp/ws")

    assert "Reserve sequencing for real dependencies" in prompt
    assert "reading source before editing it" in prompt


def test_system_prompt_allows_silent_tool_batches_and_requires_final_output() -> None:
    prompt = build_system_prompt("/tmp/ws")

    assert "Empty text is still allowed when requesting tools" in load_prompt("native-tools.md")
    assert "End the run with a useful, nonempty answer and no tool calls" in prompt


def test_system_prompt_states_parallel_calls_are_available() -> None:
    prompt = build_system_prompt("/tmp/ws")

    assert "Batch results arrive together in the supplied call order" in prompt


async def test_usage_and_cost_accumulate() -> None:
    agent, _ = build_agent(
        [
            completion(tool_calls=[call(value="x")], cost=0.01),
            completion(text="done", cost=0.02),
        ]
    )

    await agent.run("go")

    assert agent.usage.prompt_tokens == 20
    assert agent.usage.completion_tokens == 10
    assert agent.total_cost == 0.03


async def test_events_cover_the_full_cycle() -> None:
    agent, _ = build_agent(
        [
            completion(text="let me look", tool_calls=[call(value="x")]),
            completion(text="found it"),
        ]
    )
    events: list[AgentEvent] = []
    agent.on_event = events.append

    await agent.run("go")
    kinds = [event.kind for event in events]

    assert kinds[0] == "step_start"
    assert "tool_start" in kinds
    assert "tool_end" in kinds
    assert kinds.count("step_end") == 2


async def test_length_finish_reason_emits_warning() -> None:
    agent, _ = build_agent([completion(text="cut off", finish_reason="length")])
    events: list[AgentEvent] = []
    agent.on_event = events.append

    await agent.run("go")

    assert any(event.kind == "warning" for event in events)


async def test_tools_are_advertised_when_present() -> None:
    agent, client = build_agent([completion(text="hi")])

    await agent.run("go")

    tools = client.calls[0]["tools"]
    assert [spec.name for spec in tools] == ["recall_history", "record", "update_plan"]


async def test_session_id_is_stable_across_steps() -> None:
    agent, client = build_agent(
        [completion(tool_calls=[call(value="x")]), completion(text="done")]
    )

    await agent.run("go")

    first = client.calls[0]["session_id"]
    assert client.calls[1]["session_id"] == first == agent.session_id


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #


async def test_reset_keeps_system_prompt_and_clears_history() -> None:
    agent, _ = build_agent([completion(text="hi")])
    await agent.run("go")

    agent.reset()

    assert [m.role for m in agent.messages] == ["system"]
    assert agent.usage.total_tokens == 0


async def test_cancelled_batch_completes_tool_result_protocol() -> None:
    import asyncio
    import pytest

    started = asyncio.Event()
    class BlockingTool(RecordingTool):
        async def run(self, **kwargs: Any) -> ToolResult:
            started.set()
            await asyncio.Event().wait()
            return ToolResult.ok("unreachable")

    agent, _ = build_agent([completion(tool_calls=[
        ToolCall(id="first", name="record", arguments={"value": "one"}),
        ToolCall(id="second", name="record", arguments={"value": "two"}),
    ])], tools=[BlockingTool()])
    task = asyncio.create_task(agent.run("go"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [m.tool_call_id for m in agent.messages if m.role == "tool"] == ["first", "second"]


def test_reset_clears_pending_input() -> None:
    agent, _ = build_agent([])
    agent.enqueue("stale input")
    agent.reset()
    assert agent.pending == []


async def test_stop_finishes_whole_batch_and_prevents_next_request() -> None:
    from slipagent.agent import STOP_NOTICE
    tool = RecordingTool()
    agent, client = build_agent([
        completion(text="Reading both files.", tool_calls=[
            ToolCall(id="one", name="record", arguments={"value": "first"}),
            ToolCall(id="two", name="record", arguments={"value": "second"}),
        ]), completion(text="resumed"),
    ], tools=[tool])
    events = []
    def handle(event):
        events.append(event)
        if event.kind == "step_start":
            agent.request_stop()
    agent.on_event = handle
    assert await agent.run("go") == STOP_NOTICE
    assert len(client.calls) == 1
    assert [entry["value"] for entry in tool.seen] == ["first", "second"]
    assert [m.tool_call_id for m in agent.messages if m.role == "tool"] == ["one", "two"]
    assert agent.stopped
    assert any(e.kind == "assistant_text" and e.text == "Reading both files." for e in events)
    agent.on_event = None
    assert await agent.run("continue") == "resumed"
    assert not agent.stopped
    assert len(tool.seen) == 2
    assert [m.tool_call_id for m in unpack_context(client.calls[1]["messages"]) if m.role == "tool"] == ["one", "two"]


async def test_stop_during_a_tool_preserves_its_result() -> None:
    import asyncio
    started, release = asyncio.Event(), asyncio.Event()
    class WaitingTool(RecordingTool):
        async def run(self, **kwargs):
            started.set()
            await release.wait()
            return ToolResult.ok("completed tool")
    agent, client = build_agent([completion(tool_calls=[call(value="first")])], tools=[WaitingTool()])
    task = asyncio.create_task(agent.run("go"))
    await started.wait()
    assert agent.request_stop()
    release.set()
    await task
    assert decode_json_content(agent.messages[-1].content)["content"] == "completed tool"
    assert len(client.calls) == 1


async def test_stop_on_final_answer_keeps_pending_input_for_explicit_resume() -> None:
    agent, client = build_agent([completion(text="done"), completion(text="continued")])
    def stop(event):
        if event.kind == "step_start":
            agent.enqueue("followup")
            agent.request_stop()
    agent.on_event = stop
    assert await agent.run("go") == "done"
    assert agent.stopped
    agent.on_event = None
    await agent.run("continue")
    users = [context_body(m.content) for m in unpack_context(client.calls[-1]["messages"]) if m.role == "user"]
    assert users == ["go", "followup", "continue"]


def test_stop_while_idle_is_a_noop() -> None:
    agent, _ = build_agent([])
    assert not agent.request_stop()
    assert not agent.stop_requested
