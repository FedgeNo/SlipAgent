"""Catalog coverage, shared model identity, and endpoint-specific free eligibility."""

import json
from pathlib import Path

import httpx
import pytest

from slipagent.model_catalog import coding_models
from slipagent.modelranks import MODEL_ROUTERS, model_rank
from slipagent.modelrouting import eligible_free
from slipagent.nvidia import NvidiaClient
from slipagent.protocol import model_response_format
from slipagent.types import Message, ModelInfo, ToolSpec


@pytest.mark.parametrize("provider, model", [
    ("nvidia", "stepfun-ai/step-5-preview"),
    ("openrouter", "stepfun/step-5-preview:free"),
    ("kilo", "stepfun/step-5-preview-free"),
])
def test_step_five_rank_is_shared_and_precedes_ultra(provider, model):
    rank = model_rank(provider, model)
    assert rank is not None
    assert rank == model_rank("openrouter", "stepfun/step-5-preview")
    assert rank < model_rank("nvidia", "nvidia/nemotron-3-ultra-550b-a55b")


@pytest.mark.parametrize("direct, hosted", [
    ("deepseek-ai/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash"),
    ("nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3.5-super-vl"),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3.5-lightning"),
    ("nvidia/nemotron-nano-3-30b-a3b", "nvidia/nemotron-3-nano-30b-a3b"),
])
def test_provider_aliases_have_one_rank(direct, hosted):
    rank = model_rank("nvidia", direct)
    assert rank is not None
    assert rank == model_rank("openrouter", hosted + ":free")


def test_rank_never_makes_a_paid_endpoint_free():
    model = ModelInfo("stepfun/step-5-preview", provider="kilo",
                      pricing={"prompt": "0.000001", "completion": "0.0000027"})
    assert model.rank is not None
    assert not eligible_free(model)
    assert model_rank("kilo", "stepfun/step-5-preview-free-experimental") is None


def test_reviewed_free_catalog_exposes_only_documented_benchmark_gaps():
    catalog = json.loads((Path(__file__).parent / "fixtures/free-model-catalog.json").read_text())
    missing = [model.selector for provider, ids in catalog.items()
               for model in coding_models([ModelInfo(mid, provider=provider) for mid in ids])
               if model.rank is None and model.id not in MODEL_ROUTERS]
    # The 2026-10-15 review found general benchmarks, but no verified coding
    # results for these two. Preserve the ordinary unranked-model behavior.
    assert missing == ["nvidia::aisingapore/sea-lion-7b-instruct", "nvidia::zyphra/zamba2-7b-instruct"]


@pytest.mark.parametrize("model, native, limit, output", [
    ("stepfun-ai/step-5-preview", True, 1_000_000, 1_048_576),
    ("deepseek-ai/deepseek-v4.1-flash", False, 1_000_000, 1_048_576),
    ("z-ai/glm-5.3", False, 1_000_000, None),
    ("z-ai/glm-5.3-flash", False, 1_000_000, None),
    ("moonshotai/kimi-k2.6", True, 262_144, 65_536),
    ("nvidia/nemotron-3.5-super-vl-120b-a12b", True, 1_000_000, 131_072),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", False, 1_000_000, 32_768),
    ("nvidia/nemotron-3-super-120b-a12b", False, 1_000_000, 32_768),
    ("openai/gpt-oss-20b", True, 128_000, 4_096),
])
async def test_ranked_nvidia_models_can_be_selected_and_requested(model, native, limit, output):
    requests = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": model}]})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Done"}, "finish_reason": "stop"}]})

    async with NvidiaClient("dummy", transport=httpx.MockTransport(handle)) as client:
        entry, = await client.list_models()
        assert eligible_free(entry) and entry.rank is not None
        profile = await client.model_capabilities(model)
        assert (profile.native_tools, profile.context_length, profile.max_completion_tokens) == (native, limit, output)
        response_format = model_response_format(model, profile, native_tools=native)
        await client.chat(model=model, messages=[Message.system("Instructions"), Message.user("Work")],
                          max_tokens=1024, tools=[ToolSpec("probe", "Probe", {})] if native else None,
                          extra_body={"response_format": response_format} if response_format is not None else {})
    assert requests[0]["model"] == model
    assert ("tools" in requests[0]) == native
    assert "chat_template_kwargs" not in requests[0]
    assert "response_format" not in requests[0]


@pytest.mark.parametrize("model", ["google/gemma-4-31b-it", "google/diffusiongemma-26b-a4b-it"])
async def test_nvidia_gemma_preserves_instructions_in_supported_roles(model):
    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": model}]})
        body = json.loads(request.content)
        assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
        assert "Instructions" in body["messages"][0]["content"]
        assert "First request" in body["messages"][0]["content"]
        assert body["messages"][-1]["content"] == "Next request"
        assert "response_format" not in body and "tools" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "Done"}, "finish_reason": "stop"}]})

    async with NvidiaClient("dummy", transport=httpx.MockTransport(handle)) as client:
        profile = await client.model_capabilities(model)
        assert not profile.native_tools and profile.format is None
        result = await client.chat(model=model, messages=[Message.system("Instructions"),
            Message.user("First request"), Message.assistant("Reply"), Message.user("Next request")])
        assert result.message.content == "Done"
