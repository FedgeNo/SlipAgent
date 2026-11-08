"""Z.ai reviewed model table, free Flash models, capabilities and request wire."""

import json

import httpx
import pytest

from slipagent.api import APIConfigError
from slipagent.modelrouting import eligible_free
from slipagent.providers import active_providers, create_client
from slipagent.types import Message, ToolSpec
from slipagent.zai import ZaiClient


async def test_catalog_needs_no_request_and_marks_only_the_free_flash_models():
    requests = []
    transport = httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(500))
    async with ZaiClient("dummy", transport=transport) as client:
        models = {model.id: model for model in await client.list_models()}
    assert not requests
    assert {model_id for model_id, model in models.items() if eligible_free(model)} == {
        "glm-4.7-flash", "glm-4.5-flash", "glm-4.6v-flash"}
    assert models["glm-5.3"].pricing == {"prompt": "1.4e-06", "completion": "4.4e-06"}
    assert models["glm-4.7-flash"].context_length == 200000
    # Z.ai's bare IDs share the z-ai namespace used by benchmark scores.
    from slipagent.modelscores import SCORES
    assert models["glm-5.3"].score == SCORES["z-ai/glm-5.3"]
    assert models["glm-4.7-flash"].score == SCORES["z-ai/glm-4.7-flash"] is not None


async def test_capabilities_and_request_enable_thinking_with_native_tools():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"},
                                                      "finish_reason": "stop"}]})

    async with ZaiClient("dummy", transport=httpx.MockTransport(handle)) as client:
        profile = await client.model_capabilities("glm-4.7-flash")
        assert profile.native_tools and profile.context_length == 200000 and profile.max_completion_tokens == 131072
        with pytest.raises(APIConfigError, match="No reviewed Z.ai profile"):
            await client.model_capabilities("glm-unknown")
        await client.chat(model="glm-4.7-flash", messages=[Message.user("Inspect")],
                          tools=[ToolSpec("inspect", "Inspect", {})], session_id="test", request_profile=profile)
    body = json.loads(requests[-1].read())
    assert requests[-1].url.path == "/api/paas/v4/chat/completions"
    assert body["thinking"] == {"type": "enabled"} and "session_id" not in body and body["tools"]


def test_provider_is_registered_and_activated_by_its_key():
    assert isinstance(create_client("zai", api_key="dummy"), ZaiClient)
    assert "zai" in active_providers({"ZAI_API_KEY": "key"})
