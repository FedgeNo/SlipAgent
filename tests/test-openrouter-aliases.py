"""Latest aliases borrow their advertised target's endpoint capabilities."""

import json

import httpx
import pytest

from slipagent.openrouter import OpenRouterClient, OpenRouterConfigError
from slipagent.types import Message, ToolSpec


ALIAS = "~anthropic/claude-haiku-latest"


def catalog(target):
    return [{"id": ALIAS, "alias_target": {"slug": target}, "supported_parameters": ["tools"]},
            {"id": "anthropic/haiku-one", "supported_parameters": ["tools"]},
            {"id": "anthropic/haiku-two", "supported_parameters": ["tools"]}]


async def test_alias_uses_target_endpoints_and_preserves_requested_model():
    requests = []
    target = "anthropic/haiku-one"

    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": catalog(target)})
        if request.method == "GET":
            endpoints = [{"tag": "anthropic", "supported_parameters": ["tools", "max_tokens"],
                          "context_length": 200000}] if f"/{target}/" in request.url.path else []
            return httpx.Response(200, json={"data": {"endpoints": endpoints}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Done"}, "finish_reason": "stop"}]})

    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        profile = await client.model_capabilities(ALIAS)
        assert profile.native_tools and profile.context_length == 200000
        assert profile.format is None
        assert await client.model_capabilities(ALIAS) is profile
        await client.chat(model=ALIAS, messages=[Message.user("Inspect")], tools=[ToolSpec("inspect", "Inspect", {})])
    assert [request.url.path for request in requests if request.method == "GET"] == [
        "/api/v1/models", f"/api/v1/models/{target}/endpoints"]
    payload = json.loads(requests[-1].content)
    assert payload["model"] == ALIAS and payload["tools"][0]["function"]["name"] == "inspect"
    assert payload["provider"]["only"] == ["anthropic"]


async def test_refresh_follows_changed_target_without_publishing_unaccepted_profile():
    target = ["anthropic/haiku-one"]

    def handle(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": catalog(target[0])})
        context = 200000 if "/haiku-one/" in request.url.path else 1000000
        return httpx.Response(200, json={"data": {"endpoints": [
            {"tag": "anthropic", "supported_parameters": ["tools"], "context_length": context}]}})

    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        original = await client.model_capabilities(ALIAS)
        assert original.context_length == 200000
        target[0] = "anthropic/haiku-two"
        refreshed = await client.model_capabilities(ALIAS, refresh=True, store=False)
        assert refreshed.context_length == 1000000
        assert await client.model_capabilities(ALIAS) is original


@pytest.mark.parametrize("alias_target", [{}, {"slug": ""}, {"slug": 7}, {"slug": ALIAS},
                                        {"slug": "missing/model"}, "malformed"])
async def test_invalid_alias_targets_fail_without_guessing_capabilities(alias_target):
    models = [{"id": ALIAS, "alias_target": alias_target, "supported_parameters": ["tools"]}]
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": models}))) as client:
        with pytest.raises(OpenRouterConfigError, match="alias"):
            await client.model_capabilities(ALIAS)
