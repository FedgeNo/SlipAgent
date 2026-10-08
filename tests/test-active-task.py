
from slipagent.types import content_text

from slipagent.types import decode_json_content
import io
import json
import re

import pytest

from test_agent import unpack_context, context_records

from slipagent.agent import Agent
from slipagent.cli import Renderer, Session, Style, _handle_command
from slipagent.context import ConversationHistory, HistoryStep, message_tokens
from slipagent.protocol import ResponseFormatError, parse_response
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolCall
from slipagent.workspace import Workspace
from test_agent import RecordingTool, StubClient, completion, task_record


def record(*, revision=1, calls=None, previous="", **task_changes):
    return json.dumps({"response": "Working" if calls else "Done", "tool_calls": calls or [],
        "previous_tool_responses_compressed": previous, "user_prompt_compressed": "Continue the coding task",
        "agent_response_compressed": "Work continues" if calls else "Reported results",
        "task": task_record(revision=revision, **{"goal": "Implement inventory support",
                            "constraints": ["Use only the project virtual environment"], **task_changes})})


def action(name="record", **arguments):
    return {"id": "call", "name": name, "arguments": json.dumps(arguments)}


class PlainClient(StubClient):
    """Keep working replies plain so task memory cannot depend on an echo."""

    async def chat(self, **kwargs):
        if kwargs["messages"][0].content.lstrip().startswith("Summarize one completed SlipAgent step"):
            return await super().chat(**kwargs)
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


async def test_new_prompt_replaces_active_prompt_without_forcing_old_source_recall():
    tool = RecordingTool()
    events = []
    client = PlainClient([
        completion("Inspecting", [ToolCall("call", "record", {"value": "allowed"})]),
        completion("Done"),
    ])
    agent = Agent(client, ToolRegistry([tool]), "test", context_steps=1, on_event=events.append)
    agent.extend([Message.user("Original task: preserve the database"), Message.assistant("Old answer"),
                  Message.user("Also keep the existing environment"), Message.assistant("Newer answer")])
    assert await agent.run("Current request") == "Done"
    assert tool.seen == [{"value": "allowed"}]
    assert not any(event.kind == "retry" for event in events)
    assert len(client.calls) == 2
    for request in client.calls:
        wire = "\n".join(content_text(message.content) for message in unpack_context(request["messages"]))
        assert "Current request" in wire
        assert 'originated at step_id 3' in wire
        assert "Request only recall_history" not in wire


@pytest.mark.parametrize("window", [5, 50])
async def test_current_prompt_and_its_source_survive_compression_without_task_updates(window):
    text = "ORIGINAL USER PROMPT λ\nCONSTRAINT " * 600
    tool = RecordingTool()
    client = PlainClient([completion("Inspecting", [ToolCall(str(index), "record", {"value": str(index)})])
                          for index in range(window + 2)] + [completion("Done")])
    events = []
    agent = Agent(client, ToolRegistry([tool]), "test", context_steps=window, on_event=events.append)
    assert await agent.run(text) == "Done"
    await agent.wait_for_compaction()
    assert len(tool.seen) == window + 2
    assert len(client.calls) == window + 3
    for request in client.calls:
        wire = "\n".join(content_text(message.content) for message in unpack_context(request["messages"]))
        assert text in wire
        assert 'originated at step_id 1' in wire
    assert any(record["representation"] == "compressed" for record in context_records(client.calls[-1]["messages"]))
    assert not any(event.kind == "retry" for event in events)


@pytest.mark.parametrize("include_revision", [False, True])
async def test_inline_task_records_are_rejected(include_revision):
    value = json.loads(record(revision=0))
    if not include_revision:
        del value["task"]["source_revision"]
    client = StubClient([completion(json.dumps(value)), completion("Done")])
    agent = Agent(client, ToolRegistry(), "test")
    assert await agent.run("Current request") == "Done"
    assert len(client.calls) == 2
    assert "last response was rejected" in content_text(client.calls[1]["messages"][0].content)


async def test_context_supplies_retained_prompt_when_legacy_full_post_has_no_copy():
    prompt = "ORIGINAL PROMPT\nUse the local environment."
    messages = [Message.user(prompt), Message.assistant("First answer " * 1000),
                *[Message.assistant("Later answer") for _ in range(5)]]
    history = ConversationHistory()
    history.sync(messages)
    history.steps[0].summary = "Old call summary without original wording."
    # Existing sessions can contain older records without CompletedStep.user_prompt.
    for index, step in enumerate(history.steps[1:], 1):
        history.steps[index] = HistoryStep(step.id, step.request, step.messages)
    originals = [message.to_api() for message in messages]
    view = await history.view(messages, [], keep_steps=1, context_length=9000, max_output=1000)
    supplied = [message for message in unpack_context(view) if message.role == "user"]
    assert len(supplied) == 5
    assert all(message.content == prompt for message in supplied)
    assert context_records(view)[-1]["user_prompt"] == []
    assert message_tokens(view) + 1000 < 9000 * .85
    assert [message.to_api() for message in messages] == originals


async def test_queued_prompt_replaces_retained_prompt_after_current_batch():
    tool = RecordingTool()
    client = PlainClient([completion("Inspecting", [ToolCall(str(i), "record", {"value": str(i)})])
                          for i in range(4)] + [completion("Done")])
    agent = Agent(client, ToolRegistry([tool]), "test", context_steps=1)
    queued = False
    def event(value):
        nonlocal queued
        if value.kind == "tool_start" and not queued:
            queued = True
            agent.enqueue("NEW PROMPT: run the targeted tests")
    agent.on_event = event
    assert await agent.run("FIRST PROMPT: inspect the project") == "Done"
    await agent.wait_for_compaction()
    assert 'originated at step_id 1' in content_text(client.calls[0]["messages"][0].content)
    for request in client.calls[1:]:
        assert 'originated at step_id 2' in content_text(request["messages"][0].content)
        assert any((m.content or "").endswith("NEW PROMPT: run the targeted tests") for m in unpack_context(request["messages"]) if m.role == "user")
    result = await agent.registry.invoke("recall_history", {"step_id": 2, "section": "user"})
    assert not result.is_error
    assert decode_json_content(decode_json_content(result.content)["content"]) == ["NEW PROMPT: run the targeted tests"]
