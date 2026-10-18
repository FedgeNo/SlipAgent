"""Catalog coverage, shared model identity, and endpoint-specific free eligibility."""

import json
from pathlib import Path

import httpx
import pytest

from slipagent.model_catalog import coding_models
from slipagent.modelscores import MODEL_ROUTERS, SCORES, model_score, reviewed_model
from slipagent.modelrouting import eligible_free
from slipagent.nvidia import NvidiaClient
from slipagent.protocol import model_response_format
from slipagent.types import Message, ModelInfo, ToolSpec


@pytest.mark.parametrize("provider, model", [
    ("nvidia", "stepfun-ai/step-5-preview"),
    ("openrouter", "stepfun/step-5-preview:free"),
    ("kilo", "stepfun/step-5-preview-free"),
])
def test_step_five_score_is_shared_and_precedes_ultra(provider, model):
    score = model_score(provider, model)
    assert score is not None
    assert score == model_score("openrouter", "stepfun/step-5-preview")
    assert score > model_score("nvidia", "nvidia/nemotron-3-ultra-550b-a55b")


@pytest.mark.parametrize("direct, hosted", [
    ("deepseek-ai/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash"),
    ("nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3.5-super-vl"),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3.5-lightning"),
    ("nvidia/nemotron-nano-3-30b-a3b", "nvidia/nemotron-3-nano-30b-a3b"),
])
def test_provider_aliases_have_one_score(direct, hosted):
    score = model_score("nvidia", direct)
    assert score is not None
    assert score == model_score("openrouter", hosted + ":free")


def test_score_never_makes_a_paid_endpoint_free():
    model = ModelInfo("stepfun/step-5-preview", provider="kilo",
                      pricing={"prompt": "0.000001", "completion": "0.0000027"})
    assert model.score is not None
    assert not eligible_free(model)
    assert model_score("kilo", "stepfun/step-5-preview-free-experimental") is None


def test_reviewed_free_catalog_exposes_only_documented_benchmark_gaps():
    catalog = json.loads((Path(__file__).parent / "fixtures/free-model-catalog.json").read_text())
    missing = [model.selector for provider, ids in catalog.items()
               for model in coding_models([ModelInfo(mid, provider=provider) for mid in ids])
               if model.score is None and model.id not in MODEL_ROUTERS]
    # Visual and general-knowledge results cannot supply text-coding scores.
    assert missing == ["nvidia::aisingapore/sea-lion-7b-instruct"]
    assert all(reviewed_model(*selector.split("::", 1)) for selector in missing)


def test_scores_are_bounded_estimates_and_have_stable_aliases():
    assert all(score is None or 0 <= score <= 100 for score in SCORES.values())
    assert model_score("kilo", "reflection/beam-free") == model_score("kilo", "reflection/beam")
    assert model_score("openrouter", "xingchen-agi/xing4.0-29b-a4b:free") is not None
    assert model_score("kilo", "reflection/beam-free-experimental") is None


def test_model_display_uses_estimated_score_and_sorts_highest_first():
    import io
    from slipagent.cli import ModelPicker, Style
    from slipagent.modelpricing import model_order
    free = {"prompt": "0", "completion": "0"}
    high = ModelInfo("high", score=90.5, pricing=free)
    low = ModelInfo("low", score=0.0, pricing=free)
    unknown = ModelInfo("unknown", pricing=free)
    paid = ModelInfo("paid", score=100.0, pricing={"prompt": "1", "completion": "1"})
    assert sorted([paid, low, unknown, high], key=model_order) == [high, low, unknown, paid]
    assert "estimated coding score: 90.5/100" in ModelPicker(None, Style(False), io.StringIO()).label(high)
    assert "estimated coding score: 0.0/100" in ModelPicker(None, Style(False), io.StringIO()).label(low)
    assert "score:" not in ModelPicker(None, Style(False), io.StringIO()).label(unknown)


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
        assert eligible_free(entry) and entry.score is not None
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
