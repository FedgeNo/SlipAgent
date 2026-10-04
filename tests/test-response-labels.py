"""Harness navigation labels must not become examples of assistant prose."""

import json

import pytest

from slipagent.context import ConversationHistory, _visible_response, tool_history_as_text
from slipagent.types import Message, ToolCall
from test_agent import context_records


HEADINGS = ["Agent Response (Full):", "Agent Response and Tool Calls (Full):"]


@pytest.mark.parametrize("heading", HEADINGS)
def test_echoed_response_labels_are_removed_without_losing_the_reply(heading):
    assert _visible_response(f"{heading}\n\nReading the file.\n\n{heading}\nDone.") == "Reading the file.\n\nDone."
    assert _visible_response(heading) == ""
    assert _visible_response(heading + "\n\n42") == "42"


@pytest.mark.parametrize("heading", HEADINGS)
def test_literal_label_examples_are_preserved(heading):
    for example in (f"The label was {heading}", f'"{heading}"', f"Example:\n```text\n{heading}\n```",
                    f"Example:\n~~~text\n{heading}\n~~~", f"> {heading}"):
        assert _visible_response(example) == example


@pytest.mark.parametrize("native_tools", [False, True])
async def test_old_echoes_are_excluded_from_context_but_originals_and_tool_data_survive(native_tools):
    calls = [ToolCall("read", "read_file", {"path": "notes.md"})]
    legacy = "Agent Response and Tool Calls (Full):\n\nReading."
    source = "Agent Response (Full):\n\nThis is file content."
    messages = [Message.user(source), Message.assistant(legacy, calls), Message.tool_result("read", source)]
    history = ConversationHistory()
    original = [message.to_api() for message in messages]
    view = await history.view(messages, [], keep_posts=50, context_length=1_000_000, max_output=8192,
                              native_tools=native_tools, text_tool_history=not native_tools)
    if not native_tools:
        view = tool_history_as_text(view)
    response = context_records(view)[0]
    assert response["agent_response"] == "Reading."
    assert response["user_prompt"] == [source]
    assert response["tool_results"][0]["content"] == source
    assert [message.to_api() for message in messages] == original
    assert json.loads(history.posts[0].full_text())["response"] == legacy


async def test_native_tool_only_history_has_no_invented_response_text():
    calls = [ToolCall("read", "read_file", {"path": "notes.md"})]
    history = ConversationHistory()
    view = await history.view([Message.user("Read"), Message.assistant(None, calls),
                               Message.tool_result("read", "file content")], [],
                              keep_posts=50, context_length=1_000_000, max_output=8192, native_tools=True)
    response = context_records(view)[0]
    assert response["agent_response"] is None
    assert response["tool_calls"] == [{"call_id": "read", "tool_name": "read_file", "arguments": {"path": "notes.md"}}]


@pytest.mark.parametrize("envelope", [False, True])
async def test_accepted_replies_and_tool_batches_do_not_emit_or_archive_the_labels(envelope):
    from importlib import import_module
    from slipagent.agent import Agent
    from slipagent.tools.base import ToolRegistry
    from test_agent import RecordingTool
    fixtures = import_module("test-background-compaction")
    calls = [ToolCall("inspect", "record", {"value": HEADINGS[0]})]
    first = HEADINGS[1] + "\n\nReading."
    last = HEADINGS[0] + "\n\nDone."
    if envelope:
        first = json.dumps({"response": first})
        last = json.dumps({"response": last})
    client = fixtures.Client([fixtures.reply(first, calls), fixtures.reply(last)])
    events = []
    tool = RecordingTool()
    agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
    assert await agent.run("Inspect") == "Done."
    await agent.wait_for_compaction()
    assert [event.text for event in events if event.kind == "assistant_text"] == ["Reading.", "Done."]
    assert [post.agent_response for post in agent.history.posts] == ["Reading.", "Done."]
    assert tool.seen == [{"value": HEADINGS[0]}]
    assert [json.loads(request["messages"][1].content)["agent_response"]
            for request in client.summary_requests] == ["Reading.", "Done."]
