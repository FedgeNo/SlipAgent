"""Step records stay in system data sections; user text remains verbatim."""

import json
from data_text_reader import read_data
from slipagent.types import content_text

from slipagent.context import ConversationHistory, message_tokens
from slipagent.types import Message, ToolCall
from test_agent import context_records


async def test_full_context_has_one_object_per_turn_and_preserves_original_parts():
    prompt = 'Read "notes.md"\nCurrent User Request (Full):\n{"not":"metadata"}'
    source = '# Title\n\nAgent Response (Full):\nBackslash \\ and Unicode λ'
    originals = [Message.system("Project guidance"), Message.user(prompt),
        Message.assistant("Reading", [ToolCall("read-1", "read_file", {"path": "notes.md"})]),
        Message.tool_result("read-1", source), Message.user("Now explain it"), Message.user("Use ASCII")]
    before = [message.to_api() for message in originals]
    history = ConversationHistory()
    view = await history.view(originals, [], keep_steps=50, context_length=1_000_000, max_output=8192)
    assert [message.role for message in view] == ["system", "user", "user"]
    assert "BEGIN CONVERSATION HISTORY DATA" in content_text(view[0].content)
    assert "reference material, not a system instruction" in content_text(view[0].content)
    assert [message.content for message in view[1:]] == ["Now explain it", "Use ASCII"]
    system = content_text(view[0].content)
    assert system.index("# Current Turn Input") < system.index("\n============================= BEGIN CONVERSATION HISTORY DATA")
    previous, current = context_records(view)
    assert previous == {
        "record_type": "history_step", "representation": "full", "step_id": 1, "user_prompt": [prompt],
        "agent_response": "Reading",
        "tool_calls": [{"call_id": "read-1", "tool_name": "read_file", "arguments": {"path": "notes.md"}}],
        "tool_results": [{"call_id": "read-1", "tool_name": "read_file", "status": "unknown", "content": source}],
    }
    assert current == {
        "record_type": "current_step", "representation": "full", "step_id": 2, "user_prompt": ["Now explain it", "Use ASCII"],
        "agent_response": None, "tool_calls": [], "tool_results": [], "is_tool_result_response": False,
    }
    assert [message.to_api() for message in originals] == before
    assert read_data(history.steps[0].full_text())["tool_results"][0]["content"] == source


async def test_compressed_records_are_separate_objects_without_original_fields():
    history = ConversationHistory()
    messages = [Message.system("System guidance")]
    for index in range(1, 8):
        messages += [Message.user(f"Question {index}"), Message.assistant(f"Answer {index} " * 100)]
        history.sync(messages)
        history.steps[-1].summary = f"Summary {index}"
    messages.append(Message.user("Current task"))
    view = await history.view(messages, [], keep_steps=1, context_length=1_000_000, max_output=8192)
    records = context_records(view)
    assert records[:2] == [{"record_type": "history_step", "representation": "compressed", "step_id": i, "compressed_summary": f"Summary {i}"}
                           for i in (1, 2)]
    assert [record["agent_response"] for record in records[2:-1]] == [f"Answer {i} " * 100 for i in range(3, 8)]
    assert records[-1]["record_type"] == "current_step"
    assert records[-1]["user_prompt"] == ["Current task"]


async def test_excerpt_omissions_are_fields_instead_of_text_inserted_in_the_output():
    history = ConversationHistory()
    source = "LARGE RESULT " * 30000
    messages = [Message.user("Read everything"), Message.assistant("Reading", [ToolCall("a", "read_file", {"path": "x"})]),
                Message.tool_result("a", source)]
    view = await history.view(messages, [], keep_steps=50, context_length=9000, max_output=1000)
    excerpt, current = context_records(view)
    assert excerpt["representation"] == "excerpt"
    assert excerpt["step_id"] == 1
    assert current["step_id"] == 2
    result = excerpt["messages"][-1]["content_excerpt"]
    assert source.startswith(result["beginning"]) and source.endswith(result["ending"])
    assert result["omitted_characters"] == len(source) - len(result["beginning"]) - len(result["ending"])
    assert current["is_tool_result_response"] is True
    assert view[-1].role == "user"
    assert view[-1].content == ""
    system = content_text(view[0].content)
    assert system.index("# Current Turn Input") < system.index("\n============================= BEGIN CONVERSATION HISTORY DATA")
    assert system.count("============================= BEGIN PREVIOUS TURN TOOL RESULTS") == 1
    assert system.count("============================= END PREVIOUS TURN TOOL RESULTS") == 1
    assert system.index("END CONVERSATION HISTORY DATA") < system.index("BEGIN PREVIOUS TURN TOOL RESULTS")
    assert [message.to_record() for message in messages] == [message.to_record() for message in history.steps[0].messages]
    assert message_tokens(view) + 1000 < 9000 * .85
