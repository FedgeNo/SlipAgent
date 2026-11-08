"""AIHubMix catalog parsing, free-route detection, capabilities and request wire."""

import json

import httpx
import pytest

from slipagent.aihubmix import AIHubMixClient
from slipagent.api import APIConfigError
from slipagent.modelrouting import eligible_free
from slipagent.providers import active_providers, create_client
from slipagent.types import Message, ToolSpec


def entry(model_id, **values):
    base = {"model_id": model_id, "model_name": model_id, "types": "llm", "output_modalities": "text",
            "context_length": 200000, "max_output": 131072, "retire_stage": "active",
            "pricing": {"input": 0.6, "output": 2.2}, "features": "", "reasoning_options": []}
    return {**base, **values}


def catalog():
    return {"data": [
        entry("coding-glm-5.3-free", vendor="zhipu", context_length=1048576, pricing={"input": 0, "output": 0},
              features="thinking,tools,function_calling,structured_outputs",
              reasoning_options=[{"type": "toggle"}, {"type": "effort", "values": ["low", "high", "max"]}]),
        entry("coding-glm-5.3", vendor="zhipu", pricing={"input": 1.4, "output": 4.4},
              features="tools,function_calling"),
        # A zero price without the -free suffix is not a free route.
        entry("promo-model", pricing={"input": 0, "output": 0}, features="tool_calling"),
        entry("crouton-alpha-free", pricing={"input": 0, "output": 0}, max_output=0,
              reasoning_options=[{"type": "effort", "values": ["minimal", "max", "xhigh"]}]),
        entry("gemma-4-31b-it-free", vendor="google", pricing={"input": 0, "output": 0},
              features="reasoning,tool_calling,long_context",
              reasoning_options=[{"type": "effort", "values": ["minimal", "high"]}]),
        entry("retired-free", pricing={"input": 0, "output": 0}, retire_stage="retired"),
        entry("no-context-free", pricing={"input": 0, "output": 0}, context_length=0),
        entry("image-model", types="image_generation", output_modalities="image"),
    ]}


def handler(requests):
    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=catalog())
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"},
                                                      "finish_reason": "stop"}]})
    return handle


async def test_catalog_reads_the_public_endpoint_and_trusts_only_suffixed_free_routes():
    requests = []
    async with AIHubMixClient("dummy", transport=httpx.MockTransport(handler(requests))) as client:
        models = {model.id: model for model in await client.list_models()}
    assert requests[0].url.path == "/api/v1/models" and requests[0].url.params["type"] == "llm"
    assert set(models) == {"coding-glm-5.3-free", "coding-glm-5.3", "promo-model",
                           "crouton-alpha-free", "gemma-4-31b-it-free"}
    assert {model_id for model_id, model in models.items() if eligible_free(model)} == {
        "coding-glm-5.3-free", "crouton-alpha-free", "gemma-4-31b-it-free"}
    assert models["promo-model"].pricing == {}
    assert models["coding-glm-5.3"].pricing == {"prompt": "1.4e-06", "completion": "4.4e-06"}
    assert all(model.provider == "aihubmix" for model in models.values())
    # Vendor namespaces reach existing benchmark scores; unknown models stay unscored.
    from slipagent.modelscores import SCORES
    assert models["coding-glm-5.3-free"].score == SCORES["z-ai/glm-5.3"]
    assert models["gemma-4-31b-it-free"].score == SCORES["google/gemma-4-31b-it"]
    assert models["crouton-alpha-free"].score is None


async def test_capabilities_and_request_use_published_tools_formats_and_highest_effort():
    requests = []
    async with AIHubMixClient("dummy", transport=httpx.MockTransport(handler(requests))) as client:
        glm = await client.model_capabilities("coding-glm-5.3-free")
        assert glm.native_tools and glm.format == "json_schema" and glm.context_length == 1048576
        crouton = await client.model_capabilities("crouton-alpha-free")
        assert not crouton.native_tools and crouton.format is None and crouton.max_completion_tokens is None
        with pytest.raises(APIConfigError, match="active chat model"):
            await client.model_capabilities("retired-free")
        assert client.reasoning_effort("crouton-alpha-free") == "max"
        assert client.reasoning_effort("gemma-4-31b-it-free") == "high"
        assert client.reasoning_effort("coding-glm-5.3") is None
        await client.chat(model="coding-glm-5.3-free", messages=[Message.user("Inspect")],
                          tools=[ToolSpec("inspect", "Inspect", {})], session_id="test", request_profile=glm)
    body = json.loads(requests[-1].read())
    assert requests[-1].url.path == "/v1/chat/completions"
    assert body["reasoning_effort"] == "max" and "session_id" not in body


def test_scores_come_from_the_id_alone_so_cached_catalogs_keep_them():
    from slipagent import startupcache
    from slipagent.modelscores import SCORES, reviewed_model
    from slipagent.types import ModelInfo
    expected = {"coding-glm-5.3-free": "z-ai/glm-5.3", "coding-kimi-k3-free": "moonshotai/kimi-k3",
                "kimi-for-coding-free": "moonshotai/kimi-k2.7-code", "xiaomi-mimo-v2.6-pro-free": "xiaomi/mimo-v2.6-pro",
                "mimo-v2-flash-free": "xiaomi/mimo-v2-flash", "nemotron-3.5-lightning-free": "nvidia/nemotron-3.5-lightning",
                "dots-3-note-preview-free": "dots-studio/dots-3-note-preview", "xing4.0-29b-a4b-free": "xingchen-agi/xing4.0-29b-a4b",
                "k2.6-code-preview-free": "moonshotai/kimi-k2.6", "glm-4.7-flash-free": "z-ai/glm-4.7-flash"}
    for model_id, canonical in expected.items():
        assert ModelInfo(model_id, provider="aihubmix").score == SCORES[canonical] is not None, model_id
    # Reviewed gaps are recognised, so they are not announced as newly released.
    assert reviewed_model("aihubmix", "agents-a1-free") and reviewed_model("aihubmix", "crouton-alpha-free")
    assert ModelInfo("coding-glm-5.3-free", provider="openrouter").score is None
    startupcache.save_catalogs([ModelInfo("coding-glm-5.3-free", provider="aihubmix",
                                          pricing={"prompt": "0", "completion": "0"})], {"aihubmix": "id"})
    cached = startupcache.cached_catalog({"aihubmix": "id"})
    assert cached and cached[0].score == SCORES["z-ai/glm-5.3"]


def test_provider_is_registered_and_activated_by_its_key():
    assert isinstance(create_client("aihubmix", api_key="dummy"), AIHubMixClient)
    assert "aihubmix" in active_providers({"AIHUBMIX_API_KEY": "key"})
    assert "aihubmix" not in active_providers({"AIHUBMIX_API_KEY": " "})
