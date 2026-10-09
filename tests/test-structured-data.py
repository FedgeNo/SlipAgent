import json
import httpx
import pytest

from slipagent.agent import _tool_message
from slipagent.context import CompletedStep
from slipagent.records import record_message, step_record
from slipagent.tools.base import ToolResult
from slipagent.types import Message, ToolCall
from data_text_reader import read_data
from slipagent.data_text import MessageSections, render_data
from slipagent.openrouter import OpenRouterClient
from slipagent.tools.files import ReadFileTool, EditFileTool
from slipagent.workspace import Workspace
from test_agent import build_agent, completion, context_records


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
    assert read_data(packed.to_api()["content"]) == record
    source = CompletedStep(1, "Inspect the files", messages).compaction_input()
    assert source["tool_calls"][0]["function"]["arguments"] == {"path": "notes.json"}
    assert source["tool_results"][0]["content"]["content"] == result.content


def test_plain_output_remains_a_string_inside_a_structured_observation():
    result = ToolResult.ok("raw stdout\ninvalid JSON: {")
    observation = _tool_message(ToolCall("c1", "run_command", {}), result)
    assert observation.content["content"] == "raw stdout\ninvalid JSON: {"
    assert read_data(observation.to_api()["content"]) == observation.content


def test_text_fields_inside_structured_results_remain_literal_text():
    file_contents = '{"setting":true}\n'
    result = ToolResult.ok({"path": "settings.json", "content": file_contents})
    observation = _tool_message(ToolCall("c1", "read_file", {}), result)
    assert observation.content["content"]["content"] == file_contents
    assert read_data(observation.to_api()["content"])["content"]["content"] == file_contents
    call = ToolCall("c1", "read_file", {"path": "settings.json"})
    assert isinstance(call.to_record()["function"]["arguments"], dict)
    assert isinstance(call.to_api()["function"]["arguments"], str)


@pytest.mark.parametrize("source", [
    r"quote = 'can\'t'",
    'path = "C:\\new\\test"',
    'json = {"quoted": "value", "slash": "\\\\"}',
    '\tUnicode λ 🐈 and "quotes"',
    'text (12 characters):\nobject (1 fields):\n  content: null\n```\n',
    '', '\n', '\r\n', 'literal \\n versus\nactual newline\n\n',
])
def test_model_text_preserves_literal_values_without_json_escaping(source):
    record = {"content": source, "nested": [source], "odd\nkey": source}
    body = {"messages": [Message.user(record).to_api()]}
    # Decode only the HTTP JSON envelope, as the provider does.
    decoded = json.loads(json.dumps(body, ensure_ascii=False))
    text = decoded["messages"][0]["content"]
    assert read_data(text) == record
    for line in source.split("\n"):
        assert line in text
    assert read_data(render_data(record)) == record


async def test_read_file_text_survives_real_transport_and_exact_edit(tmp_path):
    source = 'value = \'can\\\'t\'\npath = "C:\\new\\test"\n\t# λ\n'
    target = tmp_path / "sample.py"
    target.write_text(source)
    workspace = Workspace(tmp_path)
    read = await ReadFileTool(workspace).invoke({"path": "sample.py"})
    call = ToolCall("read1", "read_file", {"path": "sample.py"})
    observation = _tool_message(call, read)
    record = step_record([Message.user("Edit it"), Message.assistant("Reading", [call]), observation])
    received = []

    def respond(request):
        payload = json.loads(request.content)
        text = payload["messages"][0]["content"]
        received.append(text)
        result = read_data(text)["tool_results"][0]["content"]
        assert result == read.content
        assert "can\\'t" in text
        assert 'path = "C:\\new\\test"' in text
        # Copy the actual read result into an exact edit, removing line numbers only.
        old = "\n".join(line.split("\t", 1)[1] for line in result.split("\n---\n", 1)[1].split("\n"))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "tool_calls": [{
            "id": "edit1", "type": "function", "function": {"name": "edit_file", "arguments": json.dumps({
                "path": "sample.py", "old_string": old, "new_string": "changed = True"
            })}
        }]}, "finish_reason": "tool_calls"}]})

    async with OpenRouterClient(api_key="test-key", transport=httpx.MockTransport(respond)) as client:
        reply = await client.chat(model="test", messages=[Message.user(record)])
    result = await EditFileTool(workspace).invoke(reply.tool_calls[0].arguments)
    assert not result.is_error, result.content
    assert target.read_text() == "changed = True\n"
    assert len(received) == 1


async def test_history_and_compactor_keep_objects_and_literal_file_text(tmp_path):
    source = 'value = "C:\\new\\test"; quote = \'can\\\'t\'\n'
    (tmp_path / "sample.py").write_text(source)
    call = ToolCall("read1", "read_file", {"path": "sample.py"})
    agent, client = build_agent([completion(tool_calls=[call]), completion("Done")],
                               tools=[ReadFileTool(Workspace(tmp_path))])
    await agent.run("Read sample.py")
    await agent.wait_for_compaction()
    context = client.calls[1]["messages"]
    assert isinstance(context[0].content, MessageSections)
    records = context_records(context)
    result = records[0]["tool_results"][0]["content"]
    assert source.rstrip("\n") in result
    assert isinstance(client.summary_calls[0]["messages"][1].content, dict)
    assert source.rstrip("\n") in client.summary_calls[0]["messages"][1].to_api()["content"]
