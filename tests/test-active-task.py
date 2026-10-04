import io
import json
import re

import pytest

from slipagent.agent import Agent
from slipagent.cli import Renderer, Session, Style, _handle_command
from slipagent.context import ConversationHistory, HistoryPost, message_tokens
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
        if kwargs["messages"][0].content.startswith("Summarize one completed SlipAgent turn"):
            return await super().chat(**kwargs)
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


async def test_task_contract_is_supplied_before_the_first_update_and_examples_are_valid():
    client = PlainClient([completion("Ready")])
    agent = Agent(client, ToolRegistry(), "test")
    await agent.run("Inspect the project")
    system = client.calls[0]["messages"][0].content
    spec = next(spec for spec in client.calls[0]["tools"] if spec.name == "update_task")
    for text in (system, spec.description):
        assert "Batching is encouraged" in text
        assert "pending=[]" in text and "next_steps=[]" in text
        assert "all six fields" in text
        assert "8000" in text and "2000" in text and "24" in text
        assert "source_revision" in text and "Do not pass" in text
    examples = [json.loads(line) for line in system.splitlines() if line.startswith('{"status":')]
    assert {example["status"] for example in examples} == {"active", "complete"}
    for example in examples:
        result = await agent.registry.invoke("update_task", example)
        assert not result.is_error, result.content


async def test_completion_updates_can_be_batched_with_any_other_tools():
    active = {"status": "active", "goal": "Check the project", "constraints": [], "facts": [],
              "pending": ["Receive verification result"], "next_steps": ["Inspect verification result"]}
    complete = {**active, "status": "complete", "facts": ["Verification returned ok"], "pending": [], "next_steps": []}
    tool, events = RecordingTool(), []
    client = PlainClient([
        completion(None, [ToolCall("start", "update_task", active), ToolCall("verify", "record", {"value": "test"})]),
        completion(None, [ToolCall("before", "record", {"value": "before"}),
                          ToolCall("finish", "update_task", complete),
                          ToolCall("finish-again", "update_task", complete),
                          ToolCall("after", "record", {"value": "after"})]),
        completion("Done"),
    ])
    agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
    assert await agent.run("Check the project") == "Done"
    await agent.wait_for_compaction()
    assert len(client.calls) == 3 and tool.seen == [{"value": "test"}, {"value": "before"}, {"value": "after"}]
    assert agent.history.task.record["status"] == "complete"
    assert not any(event.kind == "retry" for event in events)
    assert all(not json.loads(message.content)["status"] == "error" for message in agent.messages if message.role == "tool")


async def test_inline_completed_task_can_accompany_tool_calls():
    tool, events = RecordingTool(), []
    client = PlainClient([completion(record(status="complete", pending=[], next_steps=[], calls=[action(value="allowed")])),
                          completion("Done")])
    agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
    assert await agent.run("Report the verified result") == "Done"
    assert tool.seen == [{"value": "allowed"}]
    assert not any(event.kind == "retry" for event in events)


async def test_goal_and_constraints_survive_beyond_full_and_summary_windows():
    client = StubClient([completion(record(revision=index + 1)) for index in range(153)])
    agent = Agent(client, ToolRegistry(), "test")
    original = "Build inventory support. Never install global packages."
    for index in range(153):
        await agent.run(original if index == 0 else "continue")
    wire = "\n".join(message.content or "" for message in client.calls[-1]["messages"])
    assert original not in wire
    assert "Use only the project virtual environment" in wire
    assert "Implement inventory support" in wire
    assert '"origin_post": 1' in wire
    assert agent.history.task.source_revision == 153
    assert agent.history.task.sources[1] == [original]
    assert agent.history.posts[0].task_record["goal"] == "Implement inventory support"
    # The 100 older records may use either representation, whichever is smaller.
    assert len(re.findall(r"Conversation Record \((?:Full|Compressed)\):", wire)) == 150
    agent.history.task.record["goal"] = "Later working state"
    assert agent.history.posts[-1].task_record["goal"] == "Implement inventory support"


async def test_new_prompt_replaces_active_prompt_without_forcing_old_source_recall():
    tool = RecordingTool()
    events = []
    client = PlainClient([
        completion(None, [ToolCall("call", "record", {"value": "allowed"})]),
        completion("Done"),
    ])
    agent = Agent(client, ToolRegistry([tool]), "test", context_posts=1, on_event=events.append)
    agent.extend([Message.user("Original task: preserve the database"), Message.assistant("Old answer"),
                  Message.user("Also keep the existing environment"), Message.assistant("Newer answer")])
    assert await agent.run("Current request") == "Done"
    assert tool.seen == [{"value": "allowed"}]
    assert not any(event.kind == "retry" for event in events)
    assert len(client.calls) == 2
    for request in client.calls:
        wire = "\n".join(message.content or "" for message in request["messages"])
        assert "Current request" in wire
        assert '"current_prompt_post": 3' in wire
        assert "Request only recall_history" not in wire


@pytest.mark.parametrize("window", [5, 50])
async def test_current_prompt_and_its_source_survive_compression_without_task_updates(window):
    text = "ORIGINAL USER PROMPT λ\nCONSTRAINT " * 600
    tool = RecordingTool()
    client = PlainClient([completion(None, [ToolCall(str(index), "record", {"value": str(index)})])
                          for index in range(window + 2)] + [completion("Done")])
    events = []
    agent = Agent(client, ToolRegistry([tool]), "test", context_posts=window, on_event=events.append)
    assert await agent.run(text) == "Done"
    await agent.wait_for_compaction()
    assert len(tool.seen) == window + 2
    assert len(client.calls) == window + 3
    for request in client.calls:
        wire = "\n".join(message.content or "" for message in request["messages"])
        assert text in wire
        assert '"current_prompt_post": 1' in wire
    assert "Conversation Record (Compressed):" in "\n".join(m.content or "" for m in client.calls[-1]["messages"])
    assert not any(event.kind == "retry" for event in events)
    assert agent.history.task.record is None


@pytest.mark.parametrize("include_revision", [False, True])
async def test_legacy_task_revision_is_owned_by_harness(include_revision):
    value = json.loads(record(revision=0))
    if not include_revision:
        del value["task"]["source_revision"]
    client = StubClient([completion(json.dumps(value))])
    agent = Agent(client, ToolRegistry(), "test")
    assert await agent.run("Current request") == "Done"
    assert agent.history.task.record["source_revision"] == agent.history.task.source_revision


async def test_context_supplies_retained_prompt_when_legacy_full_post_has_no_copy():
    prompt = "ORIGINAL PROMPT\nUse the local environment."
    messages = [Message.user(prompt), Message.assistant("First answer " * 1000),
                *[Message.assistant("Later answer") for _ in range(5)]]
    history = ConversationHistory()
    history.sync(messages)
    history.posts[0].summary = "Old call summary without original wording."
    # Existing sessions can contain older records without TurnPost.user_prompt.
    for index, post in enumerate(history.posts[1:], 1):
        history.posts[index] = HistoryPost(post.id, post.request, post.messages)
    originals = [message.to_api() for message in messages]
    view = await history.view(messages, [], keep_posts=1, context_length=9000, max_output=1000)
    supplied = [message for message in view if message.role == "user"]
    assert len(supplied) == 1
    assert supplied[0].content == "Current User Request (Full):\n\n" + prompt
    assert message_tokens(view) + 1000 < 9000 * .85
    assert [message.to_api() for message in messages] == originals


async def test_queued_prompt_replaces_retained_prompt_after_current_batch():
    tool = RecordingTool()
    client = PlainClient([completion(None, [ToolCall(str(i), "record", {"value": str(i)})])
                          for i in range(4)] + [completion("Done")])
    agent = Agent(client, ToolRegistry([tool]), "test", context_posts=1)
    queued = False
    def event(value):
        nonlocal queued
        if value.kind == "tool_start" and not queued:
            queued = True
            agent.enqueue("NEW PROMPT: run the targeted tests")
    agent.on_event = event
    assert await agent.run("FIRST PROMPT: inspect the project") == "Done"
    await agent.wait_for_compaction()
    assert '"current_prompt_post": 1' in client.calls[0]["messages"][0].content
    for request in client.calls[1:]:
        assert '"current_prompt_post": 2' in request["messages"][0].content
        assert any((m.content or "").endswith("NEW PROMPT: run the targeted tests") for m in request["messages"] if m.role == "user")
    result = await agent.registry.invoke("recall_history", {"post_id": 2, "section": "user"})
    assert not result.is_error
    assert json.loads(json.loads(result.content)["content"]) == ["NEW PROMPT: run the targeted tests"]


@pytest.mark.parametrize("changes", [
    {"status": "complete", "pending": ["unverified"]},
    {"goal": ""}, {"constraints": ["x" * 2001]},
    {"facts": ["x" * 1900] * 5}, {"source_revision": True},
])
def test_invalid_working_records_are_rejected(changes):
    value = json.loads(record())
    value["task"].update(changes)
    with pytest.raises(ResponseFormatError):
        parse_response(json.dumps(value), False)


async def test_explicit_new_task_preserves_history_and_reset_clears_task(tmp_path):
    client = StubClient([completion(record()), completion(record(goal="New goal"))])
    agent = Agent(client, ToolRegistry(), "test")
    renderer = Renderer(Style(False), io.StringIO(), False)
    session = Session(agent, agent.registry, client, renderer, Workspace(tmp_path), "test", "unused", None, None)
    await agent.run("first task")
    await _handle_command(session, "/task new")
    assert len(agent.history.posts) == 1
    await agent.run("second task")
    assert agent.history.task.sources == {2: ["second task"]}
    await _handle_command(session, "/task")
    assert "New goal" in renderer.stream.getvalue()
    agent.reset()
    assert not agent.history.task.sources and agent.history.task.record is None
