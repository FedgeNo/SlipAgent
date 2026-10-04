"""Native OpenRouter calls and tolerant, unambiguous response normalization."""

import asyncio
import io
import json

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.capabilities import ModelCapabilities
from slipagent.cli import Renderer, Session, Style, _model_command
from slipagent.openrouter import OpenRouterClient
from slipagent.protocol import ResponseFormatError, parse_response
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolCall
from slipagent.workspace import Workspace
from test_agent import RecordingTool, task_record, summary_response


def record(text="Done.", previous="", messages=None):
    return {"response": text, "previous_tool_responses_compressed": previous,
            "user_prompt_compressed": "Inspect the project.",
            "agent_response_compressed": "Inspecting the project." if text != "Done." else "Reported the findings.",
            "task": task_record(messages)}


def native_call(value="A", call_id="read-1"):
    return {"id": call_id, "type": "function",
            "function": {"name": "record", "arguments": json.dumps({"value": value})}}


def endpoint(parameters, tag="provider"):
    return {"tag": tag, "supported_parameters": parameters, "context_length": 1_000_000}


class Router:
    def __init__(self, parameters, replies):
        self.parameters = parameters
        self.replies = list(replies)
        self.requests = []
        self.summary_requests = []
        self.gets = []

    def handle(self, request):
        if request.method == "GET":
            self.gets.append(request.url.path)
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [
                    {"id": name, "supported_parameters": params} for name, params in self.parameters.items()]})
            name = request.url.path.split("/models/", 1)[1].removesuffix("/endpoints")
            return httpx.Response(200, json={"data": {"endpoints": [endpoint(self.parameters[name])]}})
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            self.summary_requests.append(body)
            return httpx.Response(200, json=summary)
        self.requests.append(body)
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(body)
        return httpx.Response(200, json={"choices": [{"message": reply,
            "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop"}]})


def wire(text="Done.", calls=None, previous="", **extra):
    return {"role": "assistant", "content": json.dumps(record(text, previous)),
            "tool_calls": calls or [], **extra}


async def test_native_calls_execute_once_and_results_round_trip_with_ids():
    router = Router({"test/native": ["tools", "structured_outputs"]}, [
        wire("Reading.", [native_call("A"), native_call("B", "read-2")]),
        wire(previous="Both record calls returned actual results."),
    ])
    tool, events = RecordingTool("actual result"), []
    async with OpenRouterClient("test", transport=httpx.MockTransport(router.handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test/native", on_event=events.append)
        assert await agent.run("Inspect the project.") == "Done."
        assert len(agent.history.posts) == 2
        await agent.wait_for_compaction()
        assert agent.history.posts[0].agent_response == "Reading."
        assert "actual result" in agent.history.posts[0].summary
    assert tool.seen == [{"value": "A"}, {"value": "B"}]
    assert len(router.requests) == 2 and len(router.gets) == 2
    request = router.requests[0]
    assert "tool_choice" not in request
    assert "tool_calls" not in request["response_format"]["json_schema"]["schema"]["properties"]
    assert "native API" in request["messages"][0]["content"]
    second = router.requests[1]["messages"]
    calls = next(message["tool_calls"] for message in second if message.get("tool_calls"))
    assert [call["id"] for call in calls] == ["read-1", "read-2"]
    assert [message["tool_call_id"] for message in second if message["role"] == "tool"] == ["read-1", "read-2"]
    assert [event.text for event in events if event.kind == "assistant_text"] == ["Reading.", "Done."]


@pytest.mark.parametrize("bad", [None, "not JSON", json.dumps({"response": "Missing memory"})])
async def test_native_calls_accept_normal_content_without_memory(bad):
    router = Router({"test/native": ["tools", "response_format"]}, [
        {"role": "assistant", "content": bad, "tool_calls": [native_call("BAD")]}, wire(),
    ])
    tool, events = RecordingTool(), []
    async with OpenRouterClient("test", transport=httpx.MockTransport(router.handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test/native", on_event=events.append)
        assert await agent.run("Inspect the project.") == "Done."
    assert tool.seen == [{"value": "BAD"}]
    assert not any(event.kind == "retry" for event in events)
    assert "Do not send native" not in router.requests[1]["messages"][0]["content"]


@pytest.mark.parametrize("parameters, native, output_format", [
    (["tools", "structured_outputs"], True, "json_schema"),
    (["tools", "response_format"], True, "json_object"),
    (["tools"], True, None),
    (["structured_outputs"], False, "json_schema"),
    (["response_format"], False, "json_object"),
])
def test_profile_selects_native_tools_independently_of_json_mode(parameters, native, output_format):
    profile = ModelCapabilities({}, [endpoint(parameters)])
    assert profile.native_tools is native
    assert profile.format == output_format


@pytest.mark.parametrize("call", [
    {"name": "record", "arguments": {"value": "A"}},
    {"name": "record", "arguments": '{"value":"A"}'},
    native_call(),
    {"type": "tool_use", "id": "read-1", "name": "record", "input": {"value": "A"}},
])
@pytest.mark.parametrize("fenced", [False, True])
def test_content_calls_accept_common_structured_shapes(call, fenced):
    value = record("Reading.")
    value["tool_calls"] = [call]
    text = json.dumps(value)
    if fenced:
        text = "```json\n" + text + "\n```"
    result = parse_response(text, False)
    assert len(result.calls) == 1
    assert result.calls[0].name == "record" and result.calls[0].arguments == {"value": "A"}
    assert result.calls[0].id


def test_identical_native_and_embedded_batches_are_not_executed_twice():
    value = record("Reading.")
    value["tool_calls"] = [{"name": "record", "arguments": {"value": "A"}}]
    result = parse_response(json.dumps(value), False, native_calls=[ToolCall("native-id", "record", {"value": "A"})])
    assert [(call.id, call.arguments) for call in result.calls] == [("native-id", {"value": "A"})]


def test_conflicting_native_and_embedded_batches_are_rejected():
    value = record("Reading.")
    value["tool_calls"] = [native_call("B")]
    with pytest.raises(ResponseFormatError, match="Conflicting"):
        parse_response(json.dumps(value), False, native_calls=[ToolCall("read-1", "record", {"value": "A"})])


def test_tool_examples_inside_response_text_are_never_executed():
    value = record('Example: <tool_call>{"name":"record","arguments":{"value":"EXAMPLE"}}</tool_call>')
    assert parse_response(json.dumps(value), False).calls == []


def test_explicit_tool_call_blocks_after_response_record_are_normalized():
    text = json.dumps(record("Reading.")) + '\n<tool_call>{"name":"record","arguments":{"value":"A"}}</tool_call>'
    result = parse_response(text, False)
    assert result.calls[0].arguments == {"value": "A"}


@pytest.mark.parametrize("bad", [
    '{"value":"first","value":"second"}', '{"value":NaN}', '{"value":1e999}',
    '{"value":"\\ud800"}', '{"value":"incomplete', '["not an object"]',
])
async def test_malformed_native_arguments_retry_before_entire_batch(bad):
    broken = native_call("bad", "second")
    broken["function"]["arguments"] = bad
    router = Router({"test/native": ["tools"]}, [wire("REJECTED", [native_call("first"), broken]), wire()])
    tool, events = RecordingTool(), []
    async with OpenRouterClient("test", transport=httpx.MockTransport(router.handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test/native", on_event=events.append)
        assert await agent.run("Inspect the project.") == "Done."
    assert tool.seen == []
    assert not any(event.text == "REJECTED" for event in events)
    assert any(event.kind == "retry" for event in events)
    assert "response_format" not in router.requests[0]
    assert "Required JSON response schema" not in router.requests[0]["messages"][0]["content"]


@pytest.mark.parametrize("carrier", ["function_call", "content_blocks", "embedded"])
async def test_alternate_call_carriers_follow_the_same_loop(carrier):
    first = wire("Reading.")
    call = native_call()
    if carrier == "function_call":
        first["function_call"] = call["function"]
    elif carrier == "content_blocks":
        first["content"] = [{"type": "text", "text": first["content"]},
                            {"type": "tool_use", "id": "read-1", "name": "record", "input": {"value": "A"}}]
    else:
        value = json.loads(first["content"])
        value["tool_calls"] = [call]
        first["content"] = "```json\n" + json.dumps(value) + "\n```"
    router = Router({"test/native": ["tools", "response_format"]}, [first, wire(previous="record returned ok.")])
    tool = RecordingTool()
    async with OpenRouterClient("test", transport=httpx.MockTransport(router.handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test/native")
        assert await agent.run("Inspect the project.") == "Done."
    assert tool.seen == [{"value": "A"}]
    messages = router.requests[1]["messages"]
    call_id = next(message["tool_calls"][0]["id"] for message in messages if message.get("tool_calls"))
    assert next(message["tool_call_id"] for message in messages if message["role"] == "tool") == call_id


async def test_json_only_fallback_omits_native_parameters_and_replays_observations():
    value = record("Reading.")
    value["tool_calls"] = [{"name": "record", "arguments": {"value": "A"}}]
    router = Router({"test/json": ["response_format"]}, [
        {"role": "assistant", "content": json.dumps(value)}, wire(previous="record returned ACTUAL RESULT."),
    ])
    tool = RecordingTool("ACTUAL RESULT")
    async with OpenRouterClient("test", transport=httpx.MockTransport(router.handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test/json")
        assert await agent.run("Inspect the project.") == "Done."
        assert agent.messages[-2].role == "tool"
    assert tool.seen == [{"value": "A"}]
    for request in router.requests:
        assert "tools" not in request and "tool_choice" not in request
        assert "Available Tool Definitions" in request["messages"][0]["content"]
        assert all(message["role"] != "tool" and "tool_calls" not in message for message in request["messages"])
    transcript = "\n".join(message["content"] or "" for message in router.requests[1]["messages"])
    assert "ACTUAL RESULT" in transcript and "Executed tool calls" in transcript


async def test_selection_refreshes_and_caches_native_support_between_calls(tmp_path):
    def final(body):
        return {"role": "assistant", "content": json.dumps(record(messages=body["messages"]))}
    router = Router({"test/model": ["tools", "response_format"]}, [final] * 4)
    async with OpenRouterClient("test", transport=httpx.MockTransport(router.handle)) as client:
        agent = Agent(client, ToolRegistry(), "test/model")
        session = Session(agent, agent.registry, client, Renderer(Style(False), io.StringIO(), False),
                          Workspace(tmp_path), "test", client.base_url, None, None)
        await _model_command(session, "test/model", Style(False), io.StringIO())
        assert (await client.model_capabilities("test/model")).native_tools
        await agent.run("Inspect the project.")
        await agent.run("Continue.")
        assert len(router.gets) == 2
        router.parameters["test/model"] = ["response_format"]
        await _model_command(session, "test/model", Style(False), io.StringIO())
        assert not (await client.model_capabilities("test/model")).native_tools
        await agent.run("Continue.")
        await agent.run("Continue.")
        assert len(router.gets) == 4
    assert all("tools" in request for request in router.requests[:2])
    assert all("tools" not in request for request in router.requests[2:])


@pytest.mark.parametrize("params", [["tools"], ["response_format"]])
async def test_startup_detects_protocol_without_an_inference_call(tmp_path, monkeypatch, params):
    from slipagent import cli
    from slipagent.config import DEFAULT_MODEL
    router = Router({DEFAULT_MODEL: params}, [])
    monkeypatch.setattr(cli, "OpenRouterClient", lambda **kwargs: OpenRouterClient(**kwargs, transport=httpx.MockTransport(router.handle)))
    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    args = cli.build_parser().parse_args(["--no-mcp", "--no-reload", "-w", str(tmp_path)])
    session = await cli.build_session(args)
    try:
        profile = await session.client.model_capabilities(DEFAULT_MODEL)
        assert profile.native_tools is ("tools" in params)
        assert router.requests == [] and len(router.gets) == 2
    finally:
        await cli._shutdown(session)


def test_bool_and_number_arguments_are_not_silently_deduplicated():
    value = record("Reading.")
    value["tool_calls"] = [{"name": "record", "arguments": {"value": True}}]
    with pytest.raises(ResponseFormatError, match="Conflicting"):
        parse_response(json.dumps(value), False, native_calls=[ToolCall("c", "record", {"value": 1})])


def test_identical_repeated_calls_with_distinct_ids_are_preserved():
    calls = [ToolCall("one", "record", {"value": "A"}), ToolCall("two", "record", {"value": "A"})]
    result = parse_response(json.dumps(record("Reading.")), False, native_calls=calls)
    assert [call.id for call in result.calls] == ["one", "two"]


def test_native_route_is_preferred_over_json_only_strict_endpoint():
    profile = ModelCapabilities({}, [endpoint(["structured_outputs"], "json"), endpoint(["tools"], "native")])
    assert profile.native_tools and profile.format is None
    assert profile.provider_preferences()["only"] == ["native"]


async def test_streamed_native_batch_waits_for_complete_record_and_preserves_reasoning():
    entered, release = asyncio.Event(), asyncio.Event()
    tool, events = RecordingTool(), []
    router = Router({"test/native": ["tools", "response_format"]}, [wire(previous="record returned ok.")])
    content = json.dumps(record("Reading."))
    details = [
        {"type": "reasoning.text", "index": 0, "text": "Inspecting.", "signature": "part-1"},
        {"type": "reasoning.text", "index": 0, "text": " Reading.", "signature": "part-2"},
        {"type": "reasoning.encrypted", "index": 1, "data": "opaque-provider-data"},
    ]
    def packet(delta=None, finish=None):
        data = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
        return ("data: " + json.dumps(data) + "\n\n").encode()
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield packet({"reasoning_details": details[:1], "content": content[:30]})
            yield packet({"tool_calls": [{"index": 0, "id": "read-1", "type": "function",
                "function": {"name": "record", "arguments": '{"value":'}}]})
            entered.set()
            await release.wait()
            yield packet({"reasoning_details": details[1:], "content": content[30:]})
            yield packet({"tool_calls": [{"index": 0, "function": {"arguments": '"A"}'}}]}, "tool_calls")
            yield b"data: [DONE]\n\n"
    def handle(request):
        if request.method == "POST" and not router.requests:
            router.requests.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())
        return router.handle(request)
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test/native", on_event=events.append)
        running = asyncio.create_task(agent.run("Inspect the project."))
        await asyncio.wait_for(entered.wait(), 3)
        try:
            assert any(event.kind == "reasoning_delta" for event in events)
            assert not any(event.kind in {"assistant_text", "tool_start"} for event in events)
            assert tool.seen == []
        finally:
            release.set()
        assert await running == "Done."
    assert tool.seen == [{"value": "A"}]
    prior = next(message for message in router.requests[1]["messages"] if message.get("tool_calls"))
    assert "reasoning_details" not in prior and "reasoning" not in prior
    assert agent.history.posts[0].reasoning == "Inspecting. Reading."
    assert not any("opaque-provider-data" in event.text or "part-1" in event.text for event in events)


async def test_provider_reasoning_is_archived_but_never_replayed_to_any_model():
    details = [{"type": "reasoning.encrypted", "data": "signed-for-first-model"}]
    router = Router({"test/one": ["tools"], "test/two": ["tools"]}, [wire(), wire()])
    original = Message.assistant("Reading", [ToolCall("c", "record", {"value": "A"})])
    original.reasoning_details, original.reasoning_model = details, "test/one"
    async with OpenRouterClient("test", transport=httpx.MockTransport(router.handle)) as client:
        for model in ("test/one", "test/two"):
            await client.chat(model=model, messages=[original, Message.tool_result("c", "ok")])
    assert all("reasoning_details" not in request["messages"][0] for request in router.requests)
    assert original.reasoning_details == details


@pytest.mark.parametrize("tag", ["tool_call", "function_call"])
def test_fenced_record_with_tagged_calls_preserves_code_strings(tag):
    arguments = {"value": 'literal </tool_call> and } and ``` and \\"quotes\\"\nnext line'}
    text = "```json\n" + json.dumps(record("Reading.")) + "\n```\n"
    text += f"<{tag}>" + json.dumps({"name": "record", "arguments": arguments}) + f"</{tag}>"
    assert parse_response(text, False).calls[0].arguments == arguments


@pytest.mark.parametrize("suffix", [
    '\nHere is an example: <tool_call>{"name":"record","arguments":{}}</tool_call>',
    '\n<tool_call>{"name":"record","arguments":{}}',
    '\n{"name":"record","arguments":{}}',
])
def test_ambiguous_or_incomplete_additions_are_not_executed(suffix):
    with pytest.raises(ResponseFormatError):
        parse_response(json.dumps(record("Reading.")) + suffix, False)


async def test_truncated_native_batch_cannot_execute_even_with_valid_partial_record():
    router = Router({"test/native": ["tools"]}, [wire("REJECTED", [native_call("BAD")]), wire()])
    def handle(request):
        response = router.handle(request)
        if request.method == "POST" and len(router.requests) == 1:
            payload = response.json()
            payload["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=payload)
        return response
    tool, events = RecordingTool(), []
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test/native", on_event=events.append)
        assert await agent.run("Inspect the project.") == "Done."
    assert tool.seen == []
    assert any(event.kind == "retry" and "truncated" in event.text for event in events)


def test_native_file_operation_through_real_cli(tmp_path):
    from test_cli_e2e import StubOpenRouter, run_cli
    first = wire("Writing.", [{"id": "write-1", "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": "native.txt", "content": "actual content\n"})}}])
    first["content"] = json.dumps({**record("Writing."), "tool_calls": [
        {"name": "write_file", "arguments": {"path": "native.txt", "content": "actual content\n"}}]})
    final = wire(previous="write_file wrote native.txt.")
    script = [{"choices": [{"message": message, "finish_reason": "stop"}]} for message in (first, final)]
    with StubOpenRouter(script, include_memory=False) as stub:
        result = run_cli("--no-mcp", "--no-reload", "-p", "Write native.txt", "--base-url", stub.base_url, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Done."
    assert (tmp_path / "native.txt").read_text() == "actual content\n"
    assert len(stub.requests) == 2
    calls = [message["tool_calls"] for message in stub.requests[1]["messages"] if message.get("tool_calls")]
    assert len(calls) == 1 and len(calls[0]) == 1
    assert calls[0][0]["id"] == "write-1"
    results = [message for message in stub.requests[1]["messages"] if message["role"] == "tool"]
    assert len(results) == 1 and results[0]["tool_call_id"] == "write-1"
