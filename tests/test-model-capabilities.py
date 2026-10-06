"""Select requests from endpoint metadata without paid inference calls."""

import io
import json

import httpx
import pytest

from test_agent import summary_response

from slipagent.agent import Agent
from slipagent.cli import Renderer, Session, Style, _model_command
from slipagent import cli
from slipagent.config import DEFAULT_MODEL
from slipagent.openrouter import OpenRouterClient, OpenRouterError
from slipagent.tools.base import ToolRegistry
from slipagent.workspace import Workspace
from slipagent.capabilities import ModelCapabilities
from test_agent import RecordingTool, context_body, task_record


def endpoint(parameters, context=262144, tag="provider", **extra):
    return {"tag": tag, "supported_parameters": parameters, "context_length": context,
            "max_completion_tokens": 32768, "status": 0, **extra}


def test_context_uses_catalog_when_endpoint_omits_it():
    profile = ModelCapabilities({"context_length": 4096}, [endpoint(["tools", "response_format"], context=None)])
    assert profile.context_length == 4096


def test_unknown_context_cannot_borrow_another_endpoints_limit():
    with pytest.raises(ValueError, match="context_length"):
        ModelCapabilities({}, [endpoint(["tools", "response_format"], None, "unknown"),
                               endpoint(["tools", "response_format"], 1_000_000, "known")])


@pytest.mark.parametrize("efforts", [[], ["none"]])
def test_explicitly_disabled_reasoning_is_never_enabled(efforts):
    params = ["tools", "response_format", "reasoning"]
    profile = ModelCapabilities({}, [endpoint(params, reasoning={"supported_efforts": efforts})])
    assert profile.reasoning is None and not profile.include_reasoning


def test_high_effort_does_not_route_to_unknown_or_disabled_effort_endpoints():
    params = ["tools", "response_format", "reasoning"]
    profile = ModelCapabilities({}, [
        endpoint(params, tag="high", reasoning={"supported_efforts": ["high"]}),
        endpoint(params, tag="disabled", reasoning={"supported_efforts": []}),
        endpoint(params, tag="unknown"),
    ])
    assert profile.reasoning["effort"] == "high"
    assert profile.provider_preferences()["only"] == ["high"]


def test_base_provider_explicitly_excludes_incompatible_variant():
    params = ["tools", "response_format", "reasoning"]
    profile = ModelCapabilities({}, [
        endpoint(params, 1_000_000, "provider", reasoning={"supported_efforts": ["high"]}),
        endpoint(params, 8192, "provider/turbo", reasoning={"supported_efforts": ["low"]}),
    ])
    assert profile.provider_preferences() == {"require_parameters": True, "only": ["provider"], "ignore": ["provider/turbo"]}
    assert profile.context_length == 1_000_000


@pytest.mark.parametrize("limit", [0, -1, "1000", True])
def test_malformed_endpoint_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="context_length"):
        ModelCapabilities({}, [endpoint(["tools", "response_format"], limit)])


@pytest.mark.parametrize("context, max_output", [(1, None), (262144, 40000)])
async def test_infeasible_selection_preserves_active_model(tmp_path, context, max_output):
    from slipagent.config import Config, save_model_choice
    save_model_choice("test/current")
    params = ["tools", "response_format", "max_tokens"]
    models = [{"id": "test/new", "supported_parameters": params}]
    requests, gets = [], []
    async with OpenRouterClient("test", transport=transport_for(models, {"test/new": [endpoint(params, context)]}, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/current", max_tokens=max_output)
        session = Session(agent, agent.registry, client, Renderer(Style(False), io.StringIO(), False), Workspace(tmp_path), "test", client.base_url, None, None)
        output = io.StringIO()
        await _model_command(session, "test/new", Style(False), output)
        assert agent.model == "test/current"
        assert Config.from_env(environ={"OPENROUTER_API_KEY": "test"}).model == "test/current"
        assert "could not" in output.getvalue()
        assert "test/new" not in client._capabilities
    assert requests == []


async def test_startup_preflight_failure_closes_all_created_clients(tmp_path, monkeypatch):
    params = ["tools", "response_format"]
    requests, gets, clients, registries = [], [], [], []
    models = [{"id": "test/tiny", "supported_parameters": params}]
    transport = transport_for(models, {"test/tiny": [endpoint(params, 1)]}, requests, gets)
    original_registry = cli.build_default_registry
    def new_client(**kwargs):
        client = OpenRouterClient(**kwargs, transport=transport)
        clients.append(client)
        return client
    def new_registry(*args):
        registry = original_registry(*args)
        registries.append(registry)
        return registry
    monkeypatch.setattr(cli, "OpenRouterClient", new_client)
    monkeypatch.setattr(cli, "build_default_registry", new_registry)
    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    args = cli.build_parser().parse_args(["--no-reload", "--no-mcp", "--model", "test/tiny", "-w", str(tmp_path)])
    with pytest.raises(OpenRouterError, match="context budget"):
        await cli.build_session(args)
    assert clients[0]._client.is_closed
    assert registries[0].get("fetch_page")._client.is_closed
    assert requests == []


def record(text="Done.", **extra):
    return {"task": task_record(), "response": text, "tool_calls": [], "previous_tool_responses_compressed": "",
            "user_prompt_compressed": "Do the task.", "agent_response_compressed": "Done.", **extra}


def transport_for(models, endpoints, requests, gets, replies=None):
    def handle(request):
        if request.method == "GET":
            gets.append(request.url.path)
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": models})
            model = request.url.path.split("/models/", 1)[1].removesuffix("/endpoints")
            return httpx.Response(200, json={"data": {"endpoints": endpoints[model]}})
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        content = replies[len(requests)-1] if replies else record()
        system = body["messages"][0]["content"]
        native = "Replies and Native Tool Calls:" in system
        text = json.dumps({"response": content["response"], "tool_calls": content["tool_calls"]})
        message = {"role": "assistant", "content": text}
        if native:
            message["content"] = json.dumps({"response": content["response"]}) if body.get("response_format") else content["response"]
            message["tool_calls"] = [{"id": call["id"], "type": "function",
                                      "function": {"name": call["name"], "arguments": call["arguments"]}}
                                     for call in content["tool_calls"]]
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
    return httpx.MockTransport(handle)


@pytest.mark.parametrize("parameters, efforts, expected", [
    (["reasoning", "reasoning_effort"], ["low", "max", "high"], {"effort": "max", "exclude": False}),
    (["reasoning", "reasoning_effort"], ["low", "medium"], {"effort": "medium", "exclude": False}),
    (["reasoning"], None, {"enabled": True, "exclude": False}),
    (["reasoning_effort"], None, {"enabled": True, "exclude": False}),
    ([], ["high"], None),
    (["include_reasoning"], ["high"], None),
])
async def test_reasoning_uses_only_advertised_choices(parameters, efforts, expected):
    params = ["tools", "response_format", *parameters]
    model = {"id": "test/model", "context_length": 1000000, "supported_parameters": params}
    if efforts is not None:
        model["reasoning"] = {"supported_efforts": efforts}
    requests, gets = [], []
    async with OpenRouterClient("test", transport=transport_for([model], {"test/model": [endpoint(params)]}, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/model")
        assert await agent.run("Do the task.") == "Done."
        assert await agent.run("Again.") == "Done."
        assert await agent._context_length() == 262144
    assert requests[0].get("reasoning") == expected
    assert requests[0].get("include_reasoning") is (True if parameters == ["include_reasoning"] else None)
    assert "response_format" not in requests[0]
    assert "user_prompt_compressed" not in requests[0]["messages"][0]["content"]
    assert "plain assistant reply text" in requests[0]["messages"][0]["content"]
    assert gets == ["/api/v1/models", "/api/v1/models/test/model/endpoints"]


async def test_format_and_context_use_only_compatible_endpoints():
    params = ["tools", "response_format", "reasoning"]
    model = {"id": "test/model", "context_length": 1000000, "supported_parameters": params}
    endpoints = [endpoint(params, 1000000, "loose"),
                 endpoint([*params, "structured_outputs"], 128000, "strict"),
                 endpoint([*params, "structured_outputs"], 8000, "offline", status=1)]
    requests, gets = [], []
    async with OpenRouterClient("test", transport=transport_for([model], {"test/model": endpoints}, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/model")
        assert await agent.run("Do the task.") == "Done."
        assert await agent._context_length() == 128000
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[0]["provider"] == {"require_parameters": True, "only": ["strict"]}


async def test_json_object_retries_before_reply_or_tool_execution():
    params = ["tools", "response_format", "reasoning"]
    model = {"id": "test/model", "supported_parameters": params}
    call = {"id": "one", "name": "record", "arguments": '{"value":"A"}'}
    invalid = record("REJECTED REPLY", tool_calls=[{**call, "arguments": "["}])
    valid = record("Reading.", tool_calls=[call], agent_response_compressed="Read pending.")
    final = record(previous_tool_responses_compressed="Tool returned its result.")
    requests, gets, events = [], [], []
    tool = RecordingTool("actual result")
    async with OpenRouterClient("test", transport=transport_for([model], {"test/model": [endpoint(params)]}, requests, gets, [invalid, valid, final])) as client:
        agent = Agent(client, ToolRegistry([tool]), "test/model", on_event=events.append)
        assert await agent.run("Do the task.") == "Done."
    assert tool.seen == [{"value": "A"}]
    assert not any(event.text == "REJECTED REPLY" for event in events)
    assert [event.kind for event in events].index("retry") < [event.kind for event in events].index("tool_start")
    assert "last response was rejected" in requests[1]["messages"][0]["content"]
    assert json.loads(requests[2]["messages"][-2]["content"])["tool_results"][0]["content"] == "actual result"
    assert all(request.get("response_format") == requests[0].get("response_format") for request in requests)


async def test_selection_loads_capabilities_and_switches_without_stale_settings(tmp_path):
    from slipagent.config import Config
    strict = ["tools", "response_format", "structured_outputs", "reasoning", "reasoning_effort"]
    plain = ["tools", "response_format"]
    models = [{"id": "test/strict", "supported_parameters": strict, "reasoning": {"supported_efforts": ["low", "high"]}},
              {"id": "test/plain", "supported_parameters": plain}]
    requests, gets = [], []
    async with OpenRouterClient("test", transport=transport_for(models, {"test/strict": [endpoint(strict)], "test/plain": [endpoint(plain)]}, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/strict")
        renderer = Renderer(Style(False), io.StringIO(), False)
        session = Session(agent, agent.registry, client, renderer, Workspace(tmp_path), "test", client.base_url, None, None)
        await _model_command(session, "test/plain", Style(False), io.StringIO())
        assert Config.from_env(environ={"OPENROUTER_API_KEY": "test"}).model == "test/plain"
        assert gets[-1].endswith("test/plain/endpoints")
        await agent.run("Do the task.")
        await _model_command(session, "test/strict", Style(False), io.StringIO())
        assert Config.from_env(environ={"OPENROUTER_API_KEY": "test"}).model == "test/strict"
        await agent.run("Do the task.")
        await _model_command(session, "test/plain", Style(False), io.StringIO())
        await agent.run("Do the task.")
    assert "reasoning" not in requests[0] and "reasoning" not in requests[2]
    assert requests[1]["reasoning"]["effort"] == "high"
    assert requests[1]["response_format"]["type"] == "json_schema"
    assert gets.count("/api/v1/models/test/plain/endpoints") == 2


async def test_unsupported_model_fails_before_inference():
    params = ["temperature"]
    requests, gets = [], []
    model = {"id": "test/model", "supported_parameters": params}
    async with OpenRouterClient("test", transport=transport_for([model], {"test/model": [endpoint(params)]}, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/model")
        with pytest.raises(OpenRouterError, match="JSON"):
            await agent.run("Do the task.")
    assert requests == []


@pytest.mark.parametrize("bad_endpoints", [[], [endpoint(["temperature"])], [endpoint(["max_tokens"])],
                                           [endpoint(["tools", "response_format"], status=1)], "invalid"])
async def test_rejected_selection_preserves_current_model_and_conversation(tmp_path, bad_endpoints):
    params = ["tools", "response_format", "reasoning"]
    models = [{"id": slug, "supported_parameters": params} for slug in ["test/current", "test/new"]]
    requests, gets = [], []
    async with OpenRouterClient("test", transport=transport_for(models, {"test/current": [endpoint(params)], "test/new": bad_endpoints}, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/current")
        await agent.run("Do the task.")
        original = list(agent.messages)
        renderer = Renderer(Style(True), io.StringIO(), False)
        session = Session(agent, agent.registry, client, renderer, Workspace(tmp_path), "test", client.base_url, None, None)
        out = io.StringIO()
        await _model_command(session, "test/new", renderer.style, out)
        assert "could not load model properties" in out.getvalue()
        assert "\x1b[" in out.getvalue()
        assert agent.model == "test/current" and agent.messages == original
        assert await agent.run("Continue.") == "Done."


async def test_failed_reselection_retains_cached_working_properties(tmp_path):
    params = ["tools", "response_format", "reasoning"]
    models = [{"id": "test/current", "supported_parameters": params}]
    endpoints = {"test/current": [endpoint(params)]}
    requests, gets = [], []
    async with OpenRouterClient("test", transport=transport_for(models, endpoints, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/current")
        await agent.run("Do the task.")
        endpoints["test/current"] = [endpoint(["temperature"])]
        session = Session(agent, agent.registry, client, Renderer(Style(False), io.StringIO(), False), Workspace(tmp_path), "test", client.base_url, None, None)
        out = io.StringIO()
        await _model_command(session, "test/current", Style(False), out)
        assert "could not load model properties" in out.getvalue()
        assert await agent.run("Continue.") == "Done."
    assert "response_format" not in requests[-1]
    assert requests[-1]["reasoning"] == {"enabled": True, "exclude": False}


@pytest.mark.parametrize("selected", [DEFAULT_MODEL, "test/chosen"])
async def test_startup_fetches_model_properties_and_remembers_explicit_choice(tmp_path, monkeypatch, selected):
    from slipagent.config import Config
    params = ["tools", "response_format", "reasoning", "reasoning_effort"]
    models = [{"id": selected, "supported_parameters": params, "reasoning": {"supported_efforts": ["low", "high"]}}]
    requests, gets = [], []
    transport = transport_for(models, {selected: [endpoint(params, context=128000)]}, requests, gets)
    original = OpenRouterClient
    monkeypatch.setattr(cli, "OpenRouterClient", lambda **kwargs: original(**kwargs, transport=transport))
    monkeypatch.setenv("SLIPAGENT_NO_DOTENV", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    arguments = ["--no-mcp", "--no-reload", "-w", str(tmp_path)]
    if selected != DEFAULT_MODEL:
        arguments.extend(["--model", selected])
    session = await cli.build_session(cli.build_parser().parse_args(arguments))
    try:
        assert session.agent.model == selected
        assert Config.from_env(environ={"OPENROUTER_API_KEY": "test"}).model == selected
        assert gets == ["/api/v1/models", f"/api/v1/models/{selected}/endpoints"]
        assert requests == []
        assert await session.agent._context_length() == 128000
        assert len(gets) == 2
    finally:
        await cli._shutdown(session)


async def test_reselection_refreshes_reasoning_and_context_properties(tmp_path):
    params = ["tools", "response_format", "reasoning", "reasoning_effort"]
    models = [{"id": "test/model", "supported_parameters": params, "reasoning": {"supported_efforts": ["low", "high"]}}]
    endpoints = {"test/model": [endpoint(params, context=1000000)]}
    requests, gets = [], []
    async with OpenRouterClient("test", transport=transport_for(models, endpoints, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/model")
        await agent.run("Do the task.")
        models[0]["reasoning"] = {"supported_efforts": ["low", "medium"]}
        endpoints["test/model"] = [endpoint(params, context=128000)]
        session = Session(agent, agent.registry, client, Renderer(Style(False), io.StringIO(), False), Workspace(tmp_path), "test", client.base_url, None, None)
        await _model_command(session, "test/model", Style(False), io.StringIO())
        await agent.run("Continue.")
        assert await agent._context_length() == 128000
    assert requests[0]["reasoning"]["effort"] == "high"
    assert requests[1]["reasoning"]["effort"] == "medium"
    assert gets.count("/api/v1/models") == 2


async def test_non_text_model_is_rejected_on_selection(tmp_path):
    params = ["tools", "response_format"]
    models = [{"id": "test/image", "supported_parameters": params, "architecture": {"output_modalities": ["image"]}}]
    requests, gets = [], []
    async with OpenRouterClient("test", transport=transport_for(models, {"test/image": [endpoint(params)]}, requests, gets)) as client:
        agent = Agent(client, ToolRegistry(), "test/current")
        session = Session(agent, agent.registry, client, Renderer(Style(False), io.StringIO(), False), Workspace(tmp_path), "test", client.base_url, None, None)
        out = io.StringIO()
        await _model_command(session, "test/image", Style(False), out)
        assert "must support text" in out.getvalue()
        assert agent.model == "test/current"
    assert requests == []
