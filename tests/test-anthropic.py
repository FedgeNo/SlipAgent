"""Claude catalog discovery, paid routing, and compatible inference."""

import io
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent import cli, providers, sessionstate
from slipagent.agent import Agent
from slipagent.api import APIAuthError, APIConfigError, APIResponseError, RetryPolicy
from slipagent.config import Config
from slipagent.compressionproviders import CompressionProviders
from slipagent.modelrouting import eligible_free
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ModelInfo, ToolSpec
from slipagent.workspace import Workspace


MODEL = "claude-sonnet-4-6"


def model_entry(model=MODEL, **overrides):
    return {"id": model, "display_name": "Claude Sonnet", "type": "model",
            "max_input_tokens": 200000, "max_tokens": 64000, "lifecycle": "active", **overrides}


def page(*entries, **pagination):
    return {"data": list(entries), "has_more": False, **pagination}


async def test_catalog_pagination_limits_and_paid_eligibility():
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.params.get("after_id"):
            return httpx.Response(200, json=page(model_entry("claude-other"),
                                                model_entry("retired", lifecycle="retired")))
        return httpx.Response(200, json=page(model_entry(pricing={"prompt": "0", "completion": "0"}),
                                            has_more=True, last_id=MODEL))

    async with providers.create_client("anthropic", api_key="sk-ant-test", transport=httpx.MockTransport(respond)) as client:
        models = await client.list_models()
        assert [model.id for model in models] == [MODEL, "claude-other"]
        assert all(model.provider == "anthropic" and model.context_length == 200000 for model in models)
        assert not any(eligible_free(model) for model in models)
        assert all(cli._price_cell(model.pricing) == "" for model in models)
        assert client.catalog_context_length(MODEL) == 200000
        profile = await client.model_capabilities(MODEL)
        assert profile.native_tools and profile.format is None
        assert profile.context_length == profile.max_prompt_tokens == 200000
        assert profile.max_completion_tokens == 64000
        with pytest.raises(APIConfigError, match="available model"):
            await client.model_capabilities("retired")
        assert await client.key_info() is None
    assert len(requests) == 2
    assert requests[1].url.params["after_id"] == MODEL
    assert all(request.url.path == "/v1/models" for request in requests)
    assert all(request.headers["authorization"] == "Bearer sk-ant-test" for request in requests)
    assert all(request.headers["anthropic-version"] == "2026-10-21" for request in requests)


@pytest.mark.parametrize("raw", [
    {"data": None}, {"data": [None]}, page({"display_name": "missing ID"}),
    page(model_entry(), has_more=True), page(model_entry(), has_more=True, last_id=""),
    page(model_entry(), has_more=True, last_id=123),
    page(model_entry(), has_more=True, last_id=MODEL),
])
async def test_malformed_catalog_and_repeated_cursors_are_rejected(raw):
    async with providers.create_client("anthropic", api_key="dummy", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=raw))) as client:
        with pytest.raises(APIResponseError, match="Anthropic"):
            await client.list_models()


@pytest.mark.parametrize("field,value", [(field, value) for field in ("max_input_tokens", "max_tokens")
                                        for value in (None, True, 0, -1, "200000")])
async def test_missing_or_invalid_limits_prevent_selection(field, value):
    async with providers.create_client("anthropic", api_key="dummy", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=page(model_entry(**{field: value}))))) as client:
        with pytest.raises(APIConfigError, match="valid token limits"):
            await client.model_capabilities(MODEL)


async def test_catalog_auth_error_is_not_an_empty_catalog():
    async with providers.create_client("anthropic", api_key="bad", retry=RetryPolicy(max_retries=0),
            transport=httpx.MockTransport(lambda request: httpx.Response(
                401, json={"error": {"type": "authentication_error", "message": "Invalid API key"}}))) as client:
        with pytest.raises(APIAuthError):
            await client.list_models()


async def test_streamed_tools_and_nonstreaming_tool_results_use_shared_transport():
    requests = []
    events = [
        {"choices": [{"delta": {"content": "Checking."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "type": "function",
                                                "function": {"name": "inspect", "arguments": '{"path":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}]},
                      "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}},
    ]

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=page(model_entry()))
        payload = json.loads(request.content)
        if payload.get("stream"):
            data = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"},
                                                    "finish_reason": "stop"}]})

    deltas, snapshots = [], []
    async with providers.create_client("anthropic", api_key="sk-ant-test", transport=httpx.MockTransport(respond)) as client:
        await client.model_capabilities(MODEL)
        messages = [Message.system("Be helpful"), Message.user("Inspect a.py")]
        response = await client.chat(model=MODEL, messages=messages, tools=[ToolSpec("inspect", "Inspect", {})],
                                     temperature=0.5, session_id="session", on_delta=lambda *delta: deltas.append(delta),
                                     on_request=snapshots.append)
        assert response.tool_calls[0].arguments == {"path": "a.py"}
        assert response.usage.prompt_tokens == 15 and response.usage.completion_tokens == 8
        assert deltas == [("content", "Checking.")]
        messages.extend([response.message, Message.tool_result("call-1", "File contents")])
        final = await client.chat(model=MODEL, messages=messages, max_tokens=1024)
        assert final.message.content == "Done"
        with pytest.raises(APIConfigError, match="output limit"):
            await client.chat(model=MODEL, messages=messages, max_tokens=64001)
        with pytest.raises(APIConfigError, match="tool_choice"):
            await client.chat(model=MODEL, messages=messages, tool_choice="required")
    body = json.loads(requests[1].content)
    assert body == snapshots[0]
    assert body["max_tokens"] == 64000 and body["tools"][0]["function"]["name"] == "inspect"
    assert not {"session_id", "temperature", "response_format", "top_p"}.intersection(body)
    assert requests[-1].url.path == "/v1/chat/completions"
    final_body = json.loads(requests[-1].content)
    assert final_body["max_tokens"] == 1024
    assert final_body["messages"][-1] == {"role": "tool", "tool_call_id": "call-1", "content": "File contents"}


def test_provider_keys_and_configuration():
    assert providers.key_provider("sk-ant-api03-example") == "anthropic"
    assert providers.key_provider("sk-ant-") is None
    assert providers.active_providers({"ANTHROPIC_API_KEY": "test"}) == ["anthropic"]
    config = Config.from_env(provider="anthropic", environ={"ANTHROPIC_API_KEY": "test"})
    assert config.model == MODEL and config.base_url == "https://api.anthropic.com/v1"
    assert config.api_key == "test"
    assert cli.build_parser().parse_args(["--provider", "anthropic"]).provider == "anthropic"


async def test_cli_lists_authenticated_claude_catalog(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    create = providers.create_client
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=page(model_entry()))

    def factory(provider, **options):
        return create(provider, **options, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(providers, "create_client", factory)
    monkeypatch.setattr(cli, "fetch_catalog", providers.fetch_catalog)
    launcher = cli.Launcher(cli.build_parser().parse_args(["--list-models", "--no-color"]))
    status = await launcher.list_models()
    captured = capsys.readouterr()
    assert status == 0, captured.err
    output = captured.out
    assert "anthropic" in output and MODEL in output
    assert len(requests) == 1 and requests[0].headers["authorization"] == "Bearer sk-ant-test"


async def test_existing_model_picker_selects_claude_without_using_it_for_compaction(tmp_path, monkeypatch):
    async with providers.create_client("openrouter", api_key="dummy") as client:
        registry = ToolRegistry([])
        session = cli.Session(Agent(client, registry, "test/model"), registry, client,
                              cli.Renderer(cli.Style(False), io.StringIO(), False),
                              Workspace(tmp_path), client.api_key, client.base_url, None, "SlipAgent")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia")
        create = providers.create_client

        def factory(provider, **options):
            return create(provider, **options, transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=page(model_entry()))))

        monkeypatch.setattr(sessionstate, "create_client", factory)
        model = ModelInfo(MODEL, provider="anthropic", score=100,
                          pricing={"prompt": "unknown", "completion": "unknown"})
        free = ModelInfo("test/free", provider="nvidia", score=90, pricing={"prompt": "0", "completion": "0"})
        monkeypatch.setattr(cli.Session, "catalog", AsyncMock(return_value=[model, free]))
        compression = CompressionProviders(session.model_routing())
        out = io.StringIO()
        try:
            assert await compression.background() == [free]
            await cli.ModelPicker(session, cli.Style(False), out).select(f"anthropic::{MODEL}", catalog=[model])
            assert session.client.provider == "anthropic" and session.agent.model == MODEL
            assert not eligible_free(model)
            assert await compression.background() == [free]
        finally:
            await cli.Repl(session).shutdown()
