"""Google catalog authentication, limits, and compatible completions."""

import json

import httpx
import pytest

from slipagent.api import APIConfigError, APIResponseError
from slipagent.google import GoogleClient
from slipagent.modelpricing import free_label
from slipagent.types import Message


def entry(name):
    return {"name": "models/" + name, "inputTokenLimit": 1000000, "outputTokenLimit": 65536,
            "supportedGenerationMethods": ["generateContent"]}


async def test_catalog_pagination_auth_limits_and_completion():
    requests = []
    def handle(request):
        requests.append(request)
        if request.method == "GET":
            assert request.url.path == "/v1beta/models"
            assert request.headers["x-goog-api-key"] == "dummy"
            assert "authorization" not in request.headers
            if request.url.params.get("pageToken"):
                return httpx.Response(200, json={"models": [entry("gemini-paid")]})
            return httpx.Response(200, json={"models": [entry("gemini-3.8-flash"), entry("gemini-image")],
                                            "nextPageToken": "next"})
        assert request.headers["authorization"] == "Bearer dummy"
        assert "x-goog-api-key" not in request.headers
        body = json.loads(request.content)
        assert "session_id" not in body and "tools" not in body
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"},
                                                     "finish_reason": "stop"}]})
    async with GoogleClient("dummy", transport=httpx.MockTransport(handle)) as client:
        models = await client.list_models()
        assert [m.id for m in models] == ["gemini-3.8-flash", "gemini-paid"]
        assert free_label(models[0].pricing) == "free tier"
        assert free_label(models[1].pricing) is None
        profile = await client.model_capabilities("gemini-3.8-flash")
        assert profile.context_length == 1000000 and profile.max_completion_tokens == 65536
        assert not profile.native_tools and profile.format == "json_object"
        result = await client.chat(model="gemini-3.8-flash", messages=[Message.user("Hello")], session_id="local")
        assert result.message.content == "Done"
        with pytest.raises(APIConfigError):
            await client.model_capabilities("absent")


async def test_repeated_page_token_rejected():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": [], "nextPageToken": "same"}))
    async with GoogleClient("dummy", transport=transport) as client:
        with pytest.raises(APIResponseError, match="page token"):
            await client.list_models()


@pytest.mark.parametrize("model", ["gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemini-3-flash-preview"])
async def test_additional_free_models_are_visible_ranked_and_compatible(model, monkeypatch):
    from slipagent.modelrouting import eligible_free
    gemma = model.startswith("gemma-")

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"models": [entry(model)]})
        body = json.loads(request.content)
        assert ("response_format" in body) == (not gemma)
        assert "tools" not in body
        if gemma:
            assert "extra_body" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "Done"}, "finish_reason": "stop"}]})

    async with GoogleClient("dummy", transport=httpx.MockTransport(handle)) as client:
        models = await client.list_models()
        assert len(models) == 1 and models[0].rank is not None
        assert eligible_free(models[0])
        monkeypatch.setenv("GEMINI_BILLING_TIER", "paid")
        assert eligible_free(models[0]) == gemma
        from slipagent.protocol import model_response_format
        profile = await client.model_capabilities(model)
        response_format = model_response_format(model, profile, native_tools=False)
        await client.chat(model=model, messages=[Message.system("Instructions"), Message.user("Work")],
                          extra_body={"response_format": response_format} if response_format is not None else {})


async def test_compatible_stream():
    def handle(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        chunk = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]}
        return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})
    deltas = []
    async with GoogleClient("dummy", transport=httpx.MockTransport(handle)) as client:
        result = await client.chat(model="gemini-3.8-flash", messages=[Message.user("Hello")],
                                   on_delta=lambda kind, text: deltas.append((kind, text)))
    assert result.message.content == "Hello" and deltas


@pytest.mark.parametrize("stream", [False, True])
async def test_thought_summaries_requested_and_received(stream):
    def handle(request):
        body = json.loads(request.content)
        assert body["extra_body"]["google"]["thinking_config"] == {"include_thoughts": True}
        assert "reasoning_effort" not in body
        message = {"role": "assistant", "content": "Answer", "reasoning_content": "Considering the problem."}
        if stream:
            chunks = [{"choices": [{"delta": {"reasoning_content": "Considering the problem."}}]},
                      {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]}]
            return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
    deltas = []
    async with GoogleClient("dummy", transport=httpx.MockTransport(handle)) as client:
        result = await client.chat(model="gemini-3.8-flash", messages=[Message.user("Explain")],
                                   on_delta=(lambda kind, text: deltas.append((kind, text))) if stream else None)
    assert result.message.content == "Answer"
    assert result.message.reasoning == "Considering the problem."
    if stream:
        assert deltas == [("reasoning", "Considering the problem."), ("content", "Answer")]
