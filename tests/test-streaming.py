"""Stream real wire fragments through the client, loop, and renderer."""

import asyncio
import io
import json

import httpx
import pytest

from test_agent import summary_response

from slipagent.agent import Agent, AgentEvent
from slipagent.cli import Renderer, Style
from slipagent.context import ContextError, TOOL_SUMMARY_KEY, VisibleStream
from slipagent.openrouter import OpenRouterAPIError, OpenRouterClient, OpenRouterError, RetryPolicy
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolCall
from test_agent import RecordingTool, StubClient, completion, task_record
from test_cli_e2e import StubOpenRouter, run_cli, text_step, tool_step


def reply(post, text, summary="User requested work; model completed it."):
    return json.dumps({"task": task_record(), "response": text, "tool_calls": [],
                       "previous_tool_responses_compressed": "Previous tools returned their results." if post > 1 else "",
                       "user_prompt_compressed": "User requested work.", "agent_response_compressed": summary})


def frame(delta=None, finish=None, **extra):
    return {"model": "test/model", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **extra}


def packet(value):
    return ("data: " + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) + "\n\n").encode()


class Frames(httpx.AsyncByteStream):
    def __init__(self, values):
        self.values = values

    async def __aiter__(self):
        yield b": OPENROUTER PROCESSING\n\n"
        for value in self.values:
            raw = packet(value)
            for index in range(0, len(raw), 7):
                yield raw[index:index + 7]


def streamed(values):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Frames(values))


@pytest.mark.parametrize("model", [
    "test/model",
    "nvidia/nemotron-3-super-120b-a12b:free",
])
async def test_stream_reassembles_unicode_indexed_tools_reasoning_and_final_usage(model):
    captured, chunks = [], []
    values = [
        frame({"reasoning": "Inspect λ", "reasoning_details": [{"type": "reasoning.text", "text": "Inspect λ"}]}),
        frame({"content": "Hello "}), frame({"content": "λ."}),
        frame({"tool_calls": [{"index": 1, "id": "two", "function": {"name": "record", "arguments": '{"value":'}}]}),
        frame({"tool_calls": [{"index": 0, "id": "one", "function": {"name": "record", "arguments": '{"value":"A"}'}}]}),
        frame({"tool_calls": [{"index": 1, "function": {"arguments": '"B"}'}}]}),
        frame(finish="tool_calls"),
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": .01}},
        "[DONE]",
    ]
    def handle(request):
        if request.method == "GET":
            params = ["tools", "response_format", "reasoning"]
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": model, "supported_parameters": params}]})
            return httpx.Response(200, json={"data": {"endpoints": [{"tag": "stub-provider", "context_length": 128000, "supported_parameters": params}]}})
        captured.append(json.loads(request.content))
        if "effort" in captured[-1]["reasoning"]:
            return httpx.Response(404, json={"error": {"message": "No endpoints found that can handle the requested parameters."}})
        return streamed(values)
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        await client.model_capabilities(model)
        result = await client.chat(model=model, messages=[Message.user("hi")], on_delta=lambda *chunk: chunks.append(chunk))
    assert captured[0]["stream"] is True
    assert captured[0]["reasoning"] == {"enabled": True, "exclude": False}
    assert chunks == [("reasoning", "Inspect λ"), ("content", "Hello "), ("content", "λ.")]
    assert result.text == "Hello λ."
    assert [(call.id, call.arguments) for call in result.tool_calls] == [("one", {"value": "A"}), ("two", {"value": "B"})]
    assert result.usage.total_tokens == 15 and result.usage.cost == .01


@pytest.mark.parametrize("details, expected", [
    ({"reasoning_content": "inspect"}, "inspect"),
    ({"reasoning_details": [{"type": "reasoning.summary", "summary": "inspect"}, {"type": "reasoning.encrypted", "data": "opaque"}]}, "inspect"),
    ({"reasoning_details": [{"type": "reasoning.text", "text": "inspect"}, {"type": "reasoning.text", "signature": "opaque"}]}, "inspect"),
])
async def test_reasoning_formats_show_plaintext_without_signatures_or_encrypted_data(details, expected):
    chunks = []
    async with OpenRouterClient("test", transport=httpx.MockTransport(lambda _: streamed([frame(details), frame({"content": "Done"}, "stop"), "[DONE]"]))) as client:
        await client.chat(model="test", messages=[Message.user("work")], on_delta=lambda kind, text: chunks.append((kind, text)))
    assert chunks == [("reasoning", expected), ("content", "Done")]


async def test_current_response_reasoning_is_one_string_without_rewriting_history():
    historical = Message.assistant("Earlier answer")
    historical.reasoning_details = [{"type": "reasoning.text", "text": "OLD ONE"},
                                    {"type": "reasoning.text", "text": "OLD TWO"}]
    captured, chunks = [], []
    def handle(request):
        captured.append(json.loads(request.content))
        return streamed([
            frame({"reasoning": "Inspect λ", "reasoning_details": [{"type": "reasoning.text", "text": "Inspect λ"}]}),
            frame({"reasoning_details": [{"type": "reasoning.text", "text": "\nThen "},
                                          {"type": "reasoning.summary", "summary": "check."}]}),
            frame({"content": "Done"}, "stop"), "[DONE]",
        ])
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        first = await client.chat(model="test", messages=[historical], on_delta=lambda *chunk: chunks.append(chunk))
        second = await client.chat(model="test", messages=[historical, first.message], on_delta=lambda *_: None)
    assert first.message.reasoning == second.message.reasoning == "Inspect λ\nThen check."
    assert "reasoning" not in captured[1]["messages"][1]
    assert "reasoning_details" not in captured[1]["messages"][1]
    assert captured[0]["messages"][0] == captured[1]["messages"][0] == {"role": "assistant", "content": "Earlier answer"}
    assert historical.reasoning_details == [{"type": "reasoning.text", "text": "OLD ONE"},
                                             {"type": "reasoning.text", "text": "OLD TWO"}]
    assert chunks == [("reasoning", "Inspect λ"), ("reasoning", "\nThen "),
                      ("reasoning", "check."), ("content", "Done")]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning", [
    {"reasoning": "Inspect\nλ."},
    {"reasoning_content": "Inspect\nλ."},
    {"reasoning_details": [{"type": "reasoning.text", "text": "Inspect\n"},
                           {"type": "reasoning.summary", "summary": "λ."}]},
])
async def test_json_completion_retains_reasoning_string(reasoning, stream):
    raw = {"choices": [{"message": {"role": "assistant", "content": "Done", **reasoning}, "finish_reason": "stop"}]}
    async with OpenRouterClient("test", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw))) as client:
        result = await client.chat(model="test", messages=[Message.user("work")], on_delta=(lambda *_: None) if stream else None)
    assert result.message.reasoning == "Inspect\nλ."
    assert result.message.to_api()["reasoning"] == "Inspect\nλ."


async def test_signed_stream_assembles_only_current_native_blocks_and_keeps_one_readable_string():
    values = [
        frame({"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Inspect", "signature": None}]}),
        frame({"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": " λ."}]}),
        frame({"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": None, "signature": "signed-first"}]}),
        frame({"reasoning_details": [{"type": "reasoning.encrypted", "index": 0, "data": "opaque-one"},
                                      {"type": "reasoning.encrypted", "index": 0, "data": "opaque-two"}]}),
        frame({"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": " Verify.", "signature": "signed-second"}]}),
        frame({"content": "Done"}, "stop"), "[DONE]",
    ]
    async with OpenRouterClient("test", transport=httpx.MockTransport(lambda _: streamed(values))) as client:
        result = await client.chat(model="test", messages=[Message.user("work")], on_delta=lambda *_: None)
    assert result.message.reasoning == "Inspect λ. Verify."
    assert result.message.to_api()["reasoning_details"] == [
        {"type": "reasoning.text", "index": 0, "text": "Inspect λ.", "signature": "signed-first"},
        {"type": "reasoning.encrypted", "index": 0, "data": "opaque-one"},
        {"type": "reasoning.encrypted", "index": 0, "data": "opaque-two"},
        {"type": "reasoning.text", "index": 0, "text": " Verify.", "signature": "signed-second"},
    ]


@pytest.mark.parametrize("values", [
    [frame({"content": "partial"})],
    [{"error": {"message": "upstream failed"}}],
    ["{broken}"],
    [frame(finish="error")],
])
async def test_incomplete_and_failed_streams_are_errors(values):
    async with OpenRouterClient("test", transport=httpx.MockTransport(lambda _: streamed(values))) as client:
        with pytest.raises(OpenRouterAPIError):
            await client.chat(model="test", messages=[Message.user("work")], on_delta=lambda *_: None)


async def test_network_failure_after_first_chunk_does_not_replay_output():
    requests, chunks = [], []
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield packet(frame({"content": "partial"}))
            raise httpx.ReadError("connection lost")
    def handle(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Broken())
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenRouterError, match="interrupted"):
            await client.chat(model="test", messages=[Message.user("work")], on_delta=lambda *chunk: chunks.append(chunk))
    assert len(requests) == 1
    assert chunks == [("content", "partial")]


async def test_retryable_http_error_before_stream_retries_once():
    requests = []
    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return streamed([frame({"content": "Done"}, "stop"), "[DONE]"])
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle), retry=RetryPolicy(1, 0, 0)) as client:
        result = await client.chat(model="test", messages=[Message.user("work")], on_delta=lambda *_: None)
    assert result.text == "Done" and len(requests) == 2


async def test_agent_streams_only_thoughts_and_keeps_tools_out_of_compressed_fields():
    started, release = asyncio.Event(), asyncio.Event()
    events, requests = [], []
    tool = RecordingTool()
    record = json.loads(reply(1, "Working.", "User asked inspect; model requested record."))
    record["tool_calls"] = [{"id": "c", "name": "record", "arguments": '{"value":"A"}'}]
    content = json.dumps(record)
    class Paused(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield packet(frame({"reasoning": "Inspecting the project."}))
            yield packet(frame({"content": content[:50]}))
            started.set()
            await release.wait()
            yield packet(frame({"content": content[50:]}, "stop", usage={"total_tokens": 15, "cost": .01}))
            yield packet("[DONE]")
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test", "context_length": 1000000}]})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Paused())
        return streamed([frame({"content": reply(2, "Done.")}, "stop"), "[DONE]"])
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
        task = asyncio.create_task(agent.run("inspect"))
        await asyncio.wait_for(started.wait(), 3)
        try:
            assert tool.seen == [] and not task.done()
            assert any(e.kind == "reasoning_delta" and "Inspecting" in e.text for e in events)
            assert not any(e.kind in {"assistant_delta", "assistant_text"} for e in events)
        finally:
            release.set()
        assert await task == "Done."
    assert tool.seen == [{"value": "A"}]
    assert TOOL_SUMMARY_KEY not in agent.history.posts[0].full_text()
    assert agent.history.posts[0].agent_response == "Working."
    assert not any("<slipagent_context>" in e.text for e in events)
    assert "response_format" not in requests[0]
    assert TOOL_SUMMARY_KEY not in tool.parameters["properties"]


async def test_three_invalid_actions_stop_without_spending_the_200_request_budget():
    client = StubClient([completion('{"response":"","tool_calls":[{"name":"record","arguments":"["}]}') for _ in range(200)])
    tool = RecordingTool()
    agent = Agent(client, ToolRegistry([tool]), "test")
    with pytest.raises(ContextError, match="3 consecutive"):
        await agent.run("work")
    assert len(client.calls) == 3 and tool.seen == []
    assert agent.history.posts == []


async def test_invalid_stream_reports_red_error_and_restarts_before_running_tools():
    sink = io.StringIO()
    renderer = Renderer(Style(True), sink, False)
    requests, events = [], []
    tool = RecordingTool()
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "test", "context_length": 1000000}]})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return streamed([
                frame({"reasoning": "First attempt."}),
                frame({"content": "Rejected plan."}),
                frame({"tool_calls": [{"index": 0, "id": "bad", "function": {
                    "name": "record", "arguments": '{"value":',
                }}]}, "tool_calls"), "[DONE]",
            ])
        assert tool.seen == []
        assert "Response rejected" in sink.getvalue()
        return streamed([
            frame({"reasoning": "Starting again."}),
            frame({"content": reply(1, "Accepted response.")}, "stop"), "[DONE]",
        ])
    def emit(event):
        events.append(event)
        renderer.handle(event)
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test", on_event=emit)
        assert await agent.run("work") == "Accepted response."
    retries = [event for event in events if event.kind == "retry"]
    assert len(retries) == 1 and "from the start" in retries[0].text
    assert renderer.style.red("  ✗ " + retries[0].text) in sink.getvalue()
    assert "Thinking: Starting again." in sink.getvalue()
    assert "Rejected plan." not in sink.getvalue()
    assert sink.getvalue().index("Response rejected") < sink.getvalue().index("Accepted response.")
    assert len(agent.history.posts) == 1
    assert "Rejected plan" not in agent.history.posts[0].full_text()
    assert requests[0]["messages"][1:] == requests[1]["messages"][1:]
    assert tool.seen == []


async def test_streamed_literal_metadata_example_stays_in_one_output_block():
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, False)
    text = "Example:\n```\n[Harness context metadata]\n```"
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        return streamed([frame({"content": reply(1, text)}, "stop"), "[DONE]"])
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, ToolRegistry(), "test", on_event=renderer.handle)
        assert await agent.run("show a literal example") == text
    assert sink.getvalue() == "\n" + text + "\n"


def test_reserved_metadata_never_flashes_when_markers_are_split_into_characters():
    value = 'Visible λ.\n<slipagent_context>{"post":1,"summary":"User requested work; model completed it."}</slipagent_context>'
    output = VisibleStream()
    visible = "".join(output.push(character) for character in value)
    assert visible == "Visible λ."
    assert output.finish("Visible λ.") == ""


def test_renderer_appends_chunks_and_shows_thoughts_without_duplicate_replies():
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, False)
    for kind, text in [("reasoning_delta", "Inspect"), ("reasoning_delta", " files."),
                       ("assistant_delta", "Done"), ("assistant_delta", "."), ("stream_end", "")]:
        renderer.handle(AgentEvent(kind=kind, text=text))
    renderer.handle(AgentEvent(kind="tool_start", tool_call=ToolCall("c", "record", {"value": "A"})))
    assert sink.getvalue().splitlines() == ["", "Thinking: Inspect files.", "", "Done.", '  ⚙ record({"value": "A"})']


def test_queued_prompt_interrupts_stream_without_appending_tokens_to_the_prompt():
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, False)
    renderer.handle(AgentEvent(kind="assistant_delta", text="Working."))
    renderer.user_prompt("correction", queued=True)
    renderer.handle(AgentEvent(kind="assistant_delta", text="Continuing."))
    renderer.handle(AgentEvent(kind="stream_end"))
    assert sink.getvalue().splitlines() == [
        "", "Working.", "", "> correction", "  (queued — the agent is still working)", "", "Continuing.",
    ]


def test_one_shot_streams_reply_previews_without_repeating_a_buffered_final_answer():
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, False)
    renderer.show_assistant_text = False
    renderer.handle(AgentEvent(kind="reasoning_delta", text="Inspecting."))
    renderer.handle(AgentEvent(kind="assistant_delta", text="Done."))
    assert sink.getvalue().endswith("Done.")
    renderer.handle(AgentEvent(kind="stream_end"))
    renderer.handle(AgentEvent(kind="assistant_text", text="Done."))
    assert sink.getvalue().splitlines() == ["", "Thinking: Inspecting.", "", "Done."]


def test_real_cli_accepts_tool_only_memory_without_regeneration(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Project notes.\n")
    updated = "Project notes.\nUse agents/ as a scratch pad for audit reports and notes.\n"
    edit = tool_step("edit_file", {"path": "AGENTS.md", "old_string": "Project notes.\n", "new_string": updated,
                                   TOOL_SUMMARY_KEY: "User asked to document agents/ as a scratch pad; model updates AGENTS.md."})
    with StubOpenRouter([edit, text_step(reply(2, "Updated AGENTS.md."))], include_memory=False) as stub:
        result = run_cli("-p", "Document agents/ as a scratch pad in AGENTS.md", "--base-url", stub.base_url, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "AGENTS.md").read_text() == updated
    assert result.stdout.strip() == "Updated AGENTS.md."
    assert len(stub.requests) == 2
    assert TOOL_SUMMARY_KEY not in result.stdout + result.stderr
    assert "Retrying" not in result.stderr
