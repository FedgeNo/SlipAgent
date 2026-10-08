import json

from slipagent.agent import _tool_message
from slipagent.context import CompletedStep
from slipagent.records import record_message, step_record
from slipagent.tools.base import ToolResult
from slipagent.types import Message, ToolCall


def test_tool_json_is_decoded_at_entry_and_stays_structured_until_transmission():
    call = ToolCall.from_api({"id": "c1", "function": {"name": "inspect", "arguments": '{"path":"notes.json"}'}})
    result = ToolResult.ok('{"files":["notes.json"],"stdout":"hello"}')
    assert result.content == {"files": ["notes.json"], "stdout": "hello"}
    observation = _tool_message(call, result)
    assert observation.content["content"] == result.content
    messages = [Message.user("Inspect the files"), Message.assistant("Checking", [call]), observation]
    record = step_record(messages)
    assert record["tool_calls"][0]["arguments"] == {"path": "notes.json"}
    assert record["tool_results"][0]["content"] == result.content
    packed = record_message(record)
    assert isinstance(packed.content, dict)
    assert json.loads(packed.to_api()["content"]) == record
    source = CompletedStep(1, "Inspect the files", messages).compaction_input()
    assert source["tool_calls"][0]["function"]["arguments"] == {"path": "notes.json"}
    assert source["tool_results"][0]["content"]["content"] == result.content


def test_plain_output_remains_a_string_inside_a_structured_observation():
    result = ToolResult.ok("raw stdout\ninvalid JSON: {")
    observation = _tool_message(ToolCall("c1", "run_command", {}), result)
    assert observation.content["content"] == "raw stdout\ninvalid JSON: {"
    assert json.loads(observation.to_api()["content"]) == observation.content


def test_text_fields_inside_structured_results_remain_literal_text():
    file_contents = '{"setting":true}\n'
    result = ToolResult.ok({"path": "settings.json", "content": file_contents})
    observation = _tool_message(ToolCall("c1", "read_file", {}), result)
    assert observation.content["content"]["content"] == file_contents
    assert json.loads(observation.to_api()["content"])["content"]["content"] == file_contents
    call = ToolCall("c1", "read_file", {"path": "settings.json"})
    assert isinstance(call.to_record()["function"]["arguments"], dict)
    assert isinstance(call.to_api()["function"]["arguments"], str)
