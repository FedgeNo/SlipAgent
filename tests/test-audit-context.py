
from slipagent.types import content_text

from slipagent.types import decode_json_content
import json

from slipagent.agent import Agent
from slipagent.context import ConversationHistory, RecallHistoryTool, message_tokens
from slipagent.protocol import ResponseRecord
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolCall
from test_agent import RecordingTool, StubClient, call, completion, task_record


async def test_oversized_new_results_allow_next_request_and_exact_recall():
    payload = "large result " * 70_000 + "THE EXACT TAIL"
    client = StubClient([completion(tool_calls=[call(value="read")]), completion("done")])
    agent = Agent(client=client, registry=ToolRegistry([RecordingTool(payload)]), model="test")
    assert await agent.run("Inspect the result") == "done"
    sent = "\n".join(content_text(m.content) for m in client.calls[1]["messages"])
    assert "Excerpt" in sent and "recall_history" in sent
    assert message_tokens(client.calls[1]["messages"]) <= 1_000_000 * .85 - 8192
    result = await agent.registry.invoke("recall_history", {"step_id": 1, "call_id": "call_record", "offset": 0, "limit": 100})
    page = decode_json_content(result.content)
    assert page["next_offset"] == 100
    original = next(m.content for m in agent.history.steps[0].messages if m.role == "tool")
    assert page["content"] == content_text(original)[:100]
    tail = await agent.registry.invoke("recall_history", {"step_id": 1, "call_id": "call_record", "offset": len(content_text(original)) - 100})
    assert "THE EXACT TAIL" in decode_json_content(tail.content)["content"]


async def test_aggregate_batch_is_bounded_and_keeps_current_user_request():
    history = ConversationHistory()
    calls = [ToolCall(str(index), "read_file", {"path": "file"}) for index in range(50)]
    messages = [Message.user("Keep this instruction verbatim"), Message.assistant("Reading", calls),
                *(Message.tool_result(call.id, "x" * 10_000) for call in calls)]
    view = await history.view(messages, [], keep_steps=50, context_length=100_000, max_output=1000)
    assert message_tokens(view) <= 100_000 * .85 - 1000
    assert any("Keep this instruction verbatim" in (content_text(m.content)) for m in view)
    assert history.steps[0].messages == messages


async def test_old_tool_summary_is_sent_once_at_its_own_post():
    history = ConversationHistory()
    messages = []
    for index in range(7):
        messages.extend([Message.user("request"), Message.assistant("response", [ToolCall(str(index), "tool", {})]),
                         Message.tool_result(str(index), "original")])
        history.sync(messages)
        history.steps[-1].summary = f"RESULT SUMMARY {index}"
    view = await history.view(messages, [], keep_steps=1, context_length=100_000, max_output=1000)
    wire = "\n".join(content_text(m.content) for m in view)
    assert wire.count("RESULT SUMMARY 0") == 1
    assert wire.count("RESULT SUMMARY 1") == 1
    assert history.steps[1].summary == "RESULT SUMMARY 1"
