"""Kilo catalog selection, capabilities, and wire requests stay provider-specific."""

import io
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent import cli
from slipagent.agent import Agent
from slipagent.api import APIConfigError
from slipagent.config import Config
from slipagent.kilo import KiloClient
from slipagent.model_catalog import coding_models
from slipagent.providers import active_providers, create_client
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ModelInfo, ToolSpec
from slipagent.workspace import Workspace


def catalog_entry(model="kilo-auto/free"):
    return {"id": model, "context_length": 262144,
            "pricing": {"prompt": "0", "completion": "0"},
            "top_provider": {"max_completion_tokens": 32768},
            "supported_parameters": ["tools", "tool_choice", "reasoning", "max_tokens", "temperature"],
            "opencode": {"variants": {
                effort: {"reasoning": {"effort": effort}} for effort in ("none", "low", "high")}}}


def test_kilo_config_activates_only_with_its_key():
    assert "kilo" not in active_providers({})
    config = Config.from_env(environ={"KILO_API_KEY": "dummy-kilo"})
    assert config.provider == "kilo" and config.model == "kilo-auto/free"
    assert config.base_url == KiloClient.default_base_url
    config = Config.from_env(provider="kilo", environ={"KILO_API_KEY": "dummy-kilo",
        "KILO_MODEL": "test/coder", "KILO_BASE_URL": "https://example.invalid/gateway"})
    assert config.model == "test/coder" and config.base_url == "https://example.invalid/gateway"


async def test_kilo_catalog_and_chat_use_gateway_contract():
    requests = []
    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [catalog_entry()]})
        assert request.url.path == "/api/gateway/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "reasoning": "Inspecting.", "content": "Ready.",
            "tool_calls": [{"id": "1", "type": "function", "function": {
                "name": "probe", "arguments": '{"text":"a\\nb"}'}}]}, "finish_reason": "tool_calls"}]})
    async with create_client("kilo", api_key="dummy", transport=httpx.MockTransport(handle)) as client:
        models = await client.list_models()
        assert models[0].selector == "kilo::kilo-auto/free"
        profile = await client.model_capabilities(models[0].id)
        assert profile.native_tools and profile.format is None
        assert profile.context_length == 262144 and profile.max_completion_tokens == 32768
        result = await client.chat(model=models[0].id, messages=[Message.user("hello")],
            tools=[ToolSpec("probe", "Inspect", {})], tool_choice="auto", max_tokens=100, session_id="private")
        assert result.tool_calls[0].arguments == {"text": "a\nb"}
        assert await client.key_info() is None
        with pytest.raises(APIConfigError, match="output limit"):
            await client.chat(model=models[0].id, messages=[Message.user("hello")], max_tokens=32769)
        with pytest.raises(APIConfigError, match="does not list"):
            await client.model_capabilities("tencent/hy3:free")
    body = json.loads(requests[-1].content)
    assert body["reasoning"] == {"effort": "high", "exclude": False}
    assert body["max_tokens"] == 100
    assert not {"provider", "session_id", "chat_template_kwargs"}.intersection(body)
    assert len(requests) == 2


async def test_kilo_catalog_refresh_and_preview_do_not_replace_active_profile():
    entry = catalog_entry()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"data": [entry]}))
    async with KiloClient("dummy", transport=transport) as client:
        initial = await client.model_capabilities(entry["id"])
        entry["context_length"] = 100000
        preview = await client.model_capabilities(entry["id"], refresh=True, store=False)
        assert preview.context_length == 100000
        assert await client.model_capabilities(entry["id"]) is initial
        entry["supported_parameters"] = ["max_tokens"]
        with pytest.raises(APIConfigError, match="No endpoint"):
            await client.model_capabilities(entry["id"], refresh=True)


async def test_kilo_catalog_is_public_and_filters_specialists(tmp_path, monkeypatch):
    entries = [catalog_entry(name) for name in (
        "nvidia/nemotron-3.5-content-safety:free", "meta-llama/llama-guard-4-12b",
        "google/lyria-3-pro-preview", "tencent/hy-mt2-7b", "inference-net/schematron-v2-small",
        "google/gemini-3-pro-image", "tencent/hy3", "new/coding-model")]
    entries[-2]["pricing"] = {"prompt": "0.000001", "completion": "0.000002"}
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"data": entries})
    async with KiloClient("", catalog_only=True, transport=httpx.MockTransport(handle)) as client:
        models = await client.list_models()
        assert "authorization" not in requests[0].headers
        assert [m.id for m in coding_models(models)] == ["tencent/hy3", "new/coding-model"]
        assert models[-2].pricing["prompt"] == "0.000001"
    monkeypatch.setenv("KILO_API_KEY", "dummy")
    fetch = AsyncMock(return_value=[ModelInfo("new/coding-model", provider="kilo")])
    monkeypatch.setattr(cli, "fetch_catalog", fetch)
    from slipagent.openrouter import OpenRouterClient
    async with OpenRouterClient("dummy") as client:
        monkeypatch.setattr(client, "list_models", AsyncMock(return_value=[]))
        registry = ToolRegistry([])
        session = cli.Session(Agent(client, registry, "test/model"), registry, client,
            cli.Renderer(cli.Style(False), io.StringIO(), False), Workspace(tmp_path),
            client.api_key, client.base_url, None, "SlipAgent")
        assert [m.selector for m in await session.catalog()] == ["kilo::new/coding-model"]
        assert fetch.call_args.args == ("kilo",)
