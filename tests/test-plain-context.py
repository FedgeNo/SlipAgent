"""Context wrappers describe input without modelling numbered Markdown replies."""

import json
import re

import pytest

from slipagent.agent import build_system_prompt
from slipagent.context import ConversationHistory, RecallHistoryTool
from slipagent.instructions import ProjectInstructions
from slipagent.prompts import PromptSections
from slipagent.types import Message, ToolCall


async def test_routine_context_keeps_post_counters_only_in_system_messages():
    history = ConversationHistory()
    messages = [Message.system(build_system_prompt("/project"))]
    for index in range(8):
        messages += [Message.user(f"Inspect file {index}"), Message.assistant("Completed inspection. " * 100)]
    history.sync(messages)
    for post in history.posts:
        post.summary = "Inspected the requested file."
    messages.append(Message.user("Continue"))
    view = await history.view(messages, [], keep_posts=5, context_length=1_000_000, max_output=8192)
    system = "\n".join(message.content or "" for message in view if message.role == "system")
    wire = "\n".join(message.content or "" for message in view if message.role != "system")
    assert not re.search(r"(?m)^#{1,6} ", system + "\n" + wire)
    assert "CURRENT_POST_ID: 9" in system
    assert not re.search(r"\bPost \d+", wire)
    for key in ("CURRENT_POST_ID:", "PREVIOUS_POST_ID:", "TASK_SOURCE_REVISION:",
                '"current_prompt_post":', '"origin_post":', '"source_posts":'):
        assert key not in wire
    assert wire.count("Conversation Record (Compressed):") == 3
    assert wire.count("Conversation Record (Full):") == 5
    assert "Agent Response (Full):" in wire
    assert history.task.current_prompt_post == 9


async def test_source_markdown_and_tool_protocol_are_preserved():
    source = "# Document Title\n\n**Keep this formatting in the file.**\n"
    history = ConversationHistory()
    messages = [Message.user("Read the Markdown file"),
                Message.assistant("Reading", [ToolCall("read", "read_file", {"path": "notes.md"})]),
                Message.tool_result("read", source)]
    before = [message.to_api() for message in messages]
    view = await history.view(messages, [], keep_posts=5, context_length=1_000_000, max_output=8192)
    assert view[-1].content.endswith(source)
    assert view[-1].role == "tool" and view[-1].tool_call_id == "read"
    assert view[-2].tool_calls == messages[1].tool_calls
    assert [message.to_api() for message in messages] == before
    assert ProjectInstructions.render({".": source}).endswith(source)
    assert not ProjectInstructions.render({".": source}).startswith("###")


async def test_record_ids_remain_usable_for_recall_and_stay_in_the_archive():
    history = ConversationHistory()
    history.sync([Message.user("Inspect the frobnicator"), Message.assistant("It works.")])
    tool = RecallHistoryTool(history)
    listing = json.loads((await tool.invoke({"query": "frobnicator"})).content)
    assert "Post 1:" in listing["content"]
    recalled = json.loads((await tool.invoke({"post_id": 1, "sections": ["prompt", "response"]})).content)
    assert json.loads(recalled["content"]) == {"prompt": "Inspect the frobnicator", "response": "It works."}
    assert json.loads(history.posts[0].full_text())["post"] == 1
    assert set(json.loads(history.posts[0].compaction_input())) == {
        "user_prompt", "agent_response", "tool_calls", "tool_results",
    }


def test_prompt_sections_use_plain_labels_without_rewriting_their_content():
    sections = PromptSections()
    sections.add("project", "Project Instructions", "# User's Markdown\nDo the work.", 1)
    assert sections.render() == "Project Instructions:\n# User's Markdown\nDo the work."


@pytest.mark.parametrize("context_length", [9000, 1_000_000])
async def test_counting_backwards_recovers_ids_after_both_history_limits(context_length):
    history = ConversationHistory()
    messages = []
    for index in range(1, 161):
        messages += [Message.user(f"Request_{index:03d}"), Message.assistant(f"Response_{index:03d} " * 100)]
    history.sync(messages)
    for post in history.posts:
        post.summary = f"Summary_{post.id:03d}"
    messages.append(Message.user("Continue"))
    view = await history.view(messages, [], keep_posts=50, context_length=context_length, max_output=1000)
    wire = "\n".join(message.content or "" for message in view if message.role != "system")
    records = re.split(r"Conversation Record \((?:Full|Compressed)\):\n", wire)[1:]
    assert 0 < len(records) <= 150
    current = int(re.search(r"CURRENT_POST_ID: (\d+)", view[0].content)[1])
    recall = RecallHistoryTool(history)
    for distance, record in enumerate(reversed(records), 1):
        expected_id = current - distance
        assert re.search(rf"(?:Summary|Request)_{expected_id:03d}\b", record)
        original = json.loads((await recall.invoke({"post_id": expected_id, "section": "response"})).content)
        assert original["content"] == f"Response_{expected_id:03d} " * 100
