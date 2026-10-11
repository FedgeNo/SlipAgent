"""Optional reply suggestions remain prompts rather than tool actions."""

import json

import pytest

from slipagent.agent import Agent
from slipagent.protocol import ResponseFormatError, agent_response_format, parse_agent_response
from slipagent.tools.base import ToolRegistry
from slipagent.types import Completion, Message, ToolCall, content_text


@pytest.mark.parametrize("suggestions", [[], ["Continue."], ["  /inspect\nthis  ", "Next", "Then"]])
def test_suggestions_are_optional_verbatim_response_data(suggestions):
    result = parse_agent_response(json.dumps({"response": "Done.", "suggestions": suggestions}), [])
    assert result.text == "Done."
    assert result.suggestions == suggestions
    assert result.calls == []
    assert parse_agent_response("Done.", []).suggestions == []


@pytest.mark.parametrize("suggestions", [None, "Next", [""], ["  "], [123], ["Next"] * 4])
def test_invalid_suggestions_have_a_specific_diagnostic(suggestions):
    with pytest.raises(ResponseFormatError, match="suggestions"):
        parse_agent_response(json.dumps({"response": "Done.", "suggestions": suggestions}), [])


def test_suggestions_are_not_required_by_either_schema():
    for native in (False, True):
        schema = agent_response_format(native_tools=native)["json_schema"]["schema"]
        assert "suggestions" in schema["properties"]
        assert "suggestions" not in schema["required"]


@pytest.mark.parametrize("tool_answer", [False, True])
async def test_finished_reply_emits_suggestions_after_text_without_running_them(tool_answer):
    from slipagent.tools.answer import AnswerTool

    events = []

    class Client:
        async def chat(self, **kwargs):
            if content_text(kwargs["messages"][0].content).lstrip().startswith("# Summarizing the Previous Turn"):
                return Completion(Message.assistant('{"summary":"Completed work.","reasoning_summary":""}'), "stub")
            return Completion(Message.assistant(json.dumps({
                "response": "Done.", "suggestions": ["Extend it.", "Explain it."],
                "tool_calls": [ToolCall("answer", "answer", {"text": "Done."}).to_record()] if tool_answer else [],
            })), "stub")

    agent = Agent(Client(), ToolRegistry([AnswerTool()]), "stub", on_event=events.append)
    assert await agent.run("Build it.") == "Done."
    await agent.wait_for_compaction()
    visible = [(event.kind, event.text) for event in events if event.kind in {"assistant_text", "suggestion"}]
    assert visible == [("assistant_text", "Done."), ("suggestion", "Extend it."), ("suggestion", "Explain it.")]
    assert not agent.pending
    assert any(event.kind == "tool_start" for event in events) == tool_answer


async def test_suggestions_wait_for_a_final_reply_instead_of_tool_batches():
    from test_agent import RecordingTool

    events = []
    responses = [Completion(Message.assistant(json.dumps({
        "response": "Inspecting.", "suggestions": ["Premature."],
        "tool_calls": [ToolCall("read", "record", {"value": "data"}).to_record()],
    })), "stub"), Completion(Message.assistant("Done."), "stub")]

    class Client:
        async def chat(self, **kwargs):
            if content_text(kwargs["messages"][0].content).lstrip().startswith("# Summarizing the Previous Turn"):
                return Completion(Message.assistant('{"summary":"Inspected data.","reasoning_summary":""}'), "stub")
            return responses.pop(0)

    agent = Agent(Client(), ToolRegistry([RecordingTool()]), "stub", on_event=events.append)
    assert await agent.run("Inspect it.") == "Done."
    await agent.wait_for_compaction()
    assert not any(event.kind == "suggestion" for event in events)


async def test_clicked_prompt_bypasses_slash_commands_and_whitespace_trimming(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace
    from slipagent import cli
    from slipagent.terminal import SuggestedPrompt

    prompt = SuggestedPrompt("  /help\nexplain the implementation  ")
    lines = iter([prompt, None])
    runs = []
    renderer = cli.Renderer(cli.Style(False), io.StringIO(), True)
    session = SimpleNamespace(renderer=renderer, mcp=None, reloader=None, extensions={},
                              agent=SimpleNamespace(model="stub"), registry=ToolRegistry(),
                              workspace=SimpleNamespace(root=tmp_path))

    async def nothing(*args):
        pass

    async def read(*args):
        return next(lines)

    async def run(session, text, style):
        runs.append(text)
        return False

    monkeypatch.setattr(cli, "_refresh_quota", nothing)
    monkeypatch.setattr(cli, "_poll_quota", nothing)
    monkeypatch.setattr(cli, "_shutdown", nothing)
    monkeypatch.setattr(cli, "_wait_for_interrupt", nothing)
    monkeypatch.setattr(cli, "_read_line", read)
    monkeypatch.setattr(cli, "_run_request", run)
    assert await cli.run_repl(session) == 0
    assert runs == [prompt]
    assert str(prompt) in renderer.stream.getvalue()
