"""Mistral discovery, capabilities, and completion wire compatibility."""

import json
from importlib import import_module

import httpx
import pytest

from slipagent.api import APIConfigError
from slipagent.config import Config
from slipagent.mistral import MistralClient
from slipagent.providers import active_providers, fetch_catalog
from slipagent.types import Message, ToolSpec


def catalog():
    return {"data": [
        {"id": "mistral-small-version", "aliases": ["mistral-small-latest"],
         "max_context_length": 128000,
         "capabilities": {"completion_chat": True, "function_calling": True}},
        {"id": "text-only", "max_context_length": 32000,
         "capabilities": {"completion_chat": True, "function_calling": False}},
        {"id": "embedding", "capabilities": {"completion_chat": False}},
    ]}


async def test_catalog_and_native_tool_request():
    requests = []
    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=catalog())
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}]})
    async with MistralClient("dummy", transport=httpx.MockTransport(handle)) as client:
        models = await client.list_models()
        assert {m.id for m in models} == {"mistral-small-version", "mistral-small-latest", "text-only"}
        assert all(m.provider == "mistral" and m.pricing["prompt"] == "unknown" for m in models)
        profile = await client.model_capabilities("mistral-small-latest")
        assert profile.native_tools and profile.context_length == 128000
        assert client.catalog_context_length("mistral-small-latest") == 128000
        assert not (await client.model_capabilities("text-only")).native_tools
        await client.chat(model="mistral-small-latest", messages=[Message.user("Inspect")],
                          tools=[ToolSpec("inspect", "Inspect", {})], tool_choice="required", session_id="test")
        with pytest.raises(APIConfigError, match="chat completion"):
            await client.model_capabilities("embedding")
    assert requests[0].headers["authorization"] == "Bearer dummy"
    body = json.loads(requests[-1].content)
    assert body["tool_choice"] == "any" and body["tools"][0]["function"]["name"] == "inspect"
    assert "session_id" not in body and "provider" not in body


async def test_catalog_auth_is_provider_scoped(monkeypatch):
    captured = []
    class Catalog:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def list_models(self):
            return []
    def create(provider, **options):
        captured.append((provider, options["api_key"]))
        return Catalog()
    monkeypatch.setattr("slipagent.providers.create_client", create)
    await fetch_catalog("mistral", api_key="dummy-mistral")
    await fetch_catalog("openrouter", api_key="dummy-router")
    assert captured == [("mistral", "dummy-mistral"), ("openrouter", "")]
    with pytest.raises(APIConfigError, match="MISTRAL_API_KEY"):
        await fetch_catalog("mistral", api_key="")


async def test_mistral_stream_omits_unsupported_options():
    def handle(request):
        body = json.loads(request.content)
        assert body["stream"] is True and "stream_options" not in body
        chunk = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]}
        return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})
    deltas = []
    async with MistralClient("dummy", transport=httpx.MockTransport(handle)) as client:
        result = await client.chat(model="mistral-small-latest", messages=[Message.user("Hello")],
                                   on_delta=lambda kind, text: deltas.append((kind, text)))
    assert result.message.content == "Hello" and deltas


@pytest.mark.parametrize("model_id", ["labs-leanstral-1-5", "labs-leanstral-1-5-1"])
async def test_mistral_activation_routes_its_ranked_free_model(tmp_path, model_id):
    env = {"MISTRAL_API_KEY": "dummy"}
    assert active_providers(env) == ["mistral"]
    config = Config.from_env(provider="mistral", environ=env)
    assert config.model == "mistral-small-latest" and config.api_key == "dummy"
    data = {"data": [{"id": model_id, "aliases": ["leanstral-alias"],
                      "max_context_length": 262144, "capabilities": {"completion_chat": True}}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    async with MistralClient("dummy", transport=transport) as client:
        session = import_module("test-providers").session_for(client, tmp_path)
        session.transport = transport
        session._catalog = await client.list_models()
        assert session._catalog[0].score is not None
        assert session._catalog[1].score == session._catalog[0].score
        session.configure_compaction()
        router = session.agent._compactor().routing
        try:
            await session.model_routing().enable()
            assert session.agent.model == model_id
            destination, model = await router.select(set())
            assert destination.provider == "mistral" and model.id == model_id
        finally:
            await router.aclose()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cached", [False, True])
async def test_listed_alias_inherits_free_price_and_rank(reverse, cached, tmp_path):
    from slipagent import startupcache
    from slipagent.modelrouting import eligible_free
    data = {"data": [
        {"id": "labs-leanstral-1-5-1", "aliases": ["labs-leanstral-1-5"],
         "max_context_length": 262144, "capabilities": {"completion_chat": True}},
        {"id": "labs-leanstral-1-5", "aliases": ["labs-leanstral-1-5-1"],
         "max_context_length": 262144, "capabilities": {"completion_chat": True}},
    ]}
    if reverse:
        data["data"].reverse()
    async with MistralClient("dummy", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data))) as client:
        models = await client.list_models()
        if cached:
            startupcache.save_catalogs(models, {"mistral": "dummy-identity"})
            models = startupcache.cached_catalog({"mistral": "dummy-identity"})
            assert models is not None
        assert len(models) == 2
        assert all(eligible_free(model) and model.score is not None for model in models)
        assert models[0].score == models[1].score
        session = import_module("test-providers").session_for(client, tmp_path)
        session.model_routing().catalog_notice(models)
        assert "New free models" not in session.renderer.stream.getvalue()


@pytest.mark.parametrize("stream", [True, False])
async def test_thinking_chunks_and_text_transitions_preserve_reasoning(stream):
    thinking = {"type": "thinking", "thinking": [{"type": "text", "text": "Consider "}]}
    closing = {"type": "thinking", "thinking": [{"type": "text", "text": "this."}]}
    answer = {"type": "text", "text": "Hello"}
    def handle(request):
        if not stream:
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": [thinking, closing, answer, {"type": "text", "text": " world"}]},
                "finish_reason": "stop"}]})
        chunks = ["", [thinking], [closing, answer], " world"]
        events = [{"choices": [{"index": 0, "delta": {"content": content},
                               "finish_reason": "stop" if i == 3 else None}]}
                  for i, content in enumerate(chunks)]
        return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events)
                              + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
    captured, deltas = [], []
    async with MistralClient("dummy", transport=httpx.MockTransport(handle)) as client:
        result = await client.chat(model="labs-leanstral-1-5-1", messages=[Message.user("Hello")],
                                   on_response=captured.append,
                                   on_delta=(lambda kind, text: deltas.append((kind, text))) if stream else None)
    assert result.message.content == "Hello world"
    assert result.message.reasoning == "Consider this."
    assert result.response_error is None
    assert "Consider " in json.dumps(captured) and " world" in json.dumps(captured)
    if stream:
        assert deltas == [("reasoning", "Consider "), ("reasoning", "this."),
                          ("content", "Hello"), ("content", " world")]
