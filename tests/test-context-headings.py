"""Input headings describe context without altering archived originals."""

import json

from slipagent.context import ConversationHistory
from slipagent.protocol import parse_response
from slipagent.types import Message, ToolCall
from test_agent import task_record


def summary(response="Summary", previous=""):
    return parse_response(json.dumps({"task": task_record(), "response": response, "tool_calls": [],
        "previous_tool_responses_compressed": previous, "user_prompt_compressed": "User request summary",
        "agent_response_compressed": "Agent response summary"}), bool(previous))


async def test_full_context_labels_roles_posts_and_named_tool_results_without_changing_originals():
    originals = [Message.system("Project guidance"), Message.user("Read the file"),
        Message.assistant("Reading", [ToolCall("read-1", "read_file", {"path": "app.py"})]),
        Message.tool_result("read-1", "EXACT FILE CONTENT"), Message.user("Now explain it")]
    before = [message.to_api() for message in originals]
    history = ConversationHistory()
    view = await history.view(originals, [], keep_posts=50,
                              context_length=1000000, max_output=8192)
    assert [message.role for message in view] == [message.role for message in originals]
    assert "## System Instructions" in view[0].content
    assert "## Current Turn State" in view[0].content
    assert "## Replies and Tool Calls" in view[0].content
    assert "### Post 1 — User Request (Full)" in view[1].content
    assert "### Post 1 — Agent Response and Tool Calls (Full)" in view[2].content
    assert "### Post 1 — Tool Result: read_file (Full)" in view[3].content
    assert "Call ID: read-1" in view[3].content
    assert view[3].tool_call_id == "read-1" and view[2].tool_calls == originals[2].tool_calls
    assert "## Current User Request — Post 2 (Full)" in view[4].content
    for rendered, original in zip(view[1:], originals[1:]):
        assert rendered.content.endswith(original.content)
    assert [message.to_api() for message in originals] == before
    assert "### Post" not in history.posts[0].full_text()


async def test_compressed_and_full_history_have_distinct_headings():
    history = ConversationHistory()
    messages = [Message.system("System guidance")]
    for index in range(1, 8):
        messages += [Message.user(f"Question {index}"), Message.assistant(f"Answer {index} " * 100)]
        history.sync(messages)
        history.posts[-1].summary = "Whole-turn summary"
    messages.append(Message.user("Current task"))
    view = await history.view(messages, [], keep_posts=1,
                              context_length=1000000, max_output=8192)
    wire = "\n".join(message.content or "" for message in view)
    assert "## Earlier Conversation (Compressed)" in wire
    assert "### Post 1 — Conversation Record (Compressed)" in wire
    assert "### Post 2 — Conversation Record (Compressed)" in wire
    assert "### Post 3 — User Request (Full)" in wire
    assert "### Post 3 — Agent Response (Full)" in wire
    assert "Question 1" not in wire and "Answer 1" not in wire
    assert wire.index("Conversation Record (Compressed)") < wire.index("Post 3 — User Request") < wire.index("Current User Request")
