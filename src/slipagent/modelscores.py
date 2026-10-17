"""Estimated text-coding scores; higher is better (2026-10-16).

Only the final scores belong in runtime metadata. See docs/model-scores.md
for benchmark calibration, validation, and uncertainty.
"""

from __future__ import annotations

# Adding a model never renumbers existing models. Keep unavailable IDs so their
# scores apply if a provider lists them again. None records a reviewed gap.
SCORES: dict[str, float | None] = {
    "deepseek/deepseek-v4.1-flash": 95.3,
    "google/gemini-3.8-flash": 95.2,
    "z-ai/glm-5.3": 93.7,
    "stepfun/step-5-preview": 93.6,
    "z-ai/glm-5.3-flash": 92.1,
    "google/gemini-3.6-flash": 87.6,
    "reflection/beam": 83.6,
    "poolside/laguna-s-2.1": 82.8,
    "dots-studio/dots-3-note-preview": 82.7,
    "moonshotai/kimi-k2.6": 81.9,
    "mistral/labs-leanstral-1-5-1": 79.7,
    "thinkingmachines/inkling-small": 79.7,
    "thinkingmachines/inkling": 78.1,
    "google/gemini-3-flash-preview": 78.0,
    "nvidia/nemotron-3.5-super-vl": 74.9,
    "xingchen-agi/xing4.0-29b-a4b": 74.4,
    "nvidia/nemotron-3-ultra-550b-a55b": 74.1,
    "meta/muse-glimmer-30b": 72.6,
    "google/gemini-3.5-flash-lite": 70.9,
    "poolside/laguna-xs-2.1": 67.4,
    "cohere/north-mini-code": 65.0,
    "google/gemma-4-31b-it": 64.1,
    "mistralai/mistral-large-2-instruct": 63.8,
    "nvidia/nemotron-3-super-120b-a12b": 61.2,
    "openai/gpt-oss-20b": 60.7,
    "google/gemini-2.5-pro": 58.0,
    "google/gemma-4-26b-a4b-it": 53.4,
    "google/diffusiongemma-26b-a4b-it": 50.1,
    "google/gemini-3.1-flash-lite": 48.2,
    "google/gemma-3-12b-it": 47.3,
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": 46.9,
    "nvidia/nemotron-3.5-lightning": 45.9,
    "google/gemini-2.5-flash": 44.2,
    "nvidia/nemotron-3-nano-30b-a3b": 43.9,
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": 43.5,
    "liquid/lfm-2.5-2.6b": 39.6,
    "meta/llama-3.2-90b-vision-instruct": 38.8,
    "mistralai/codestral-22b-instruct-v0.1": 32.8,
    "nvidia/llama-3.1-nemotron-70b-instruct": 31.4,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 30.8,
    "01-ai/yi-large": 30.6,
    "meta/codellama-70b": 30.4,
    "nvidia/llama-3.1-nemotron-51b-instruct": 30.2,
    "nvidia/nemotron-4-340b-instruct": 29.5,
    "meta/llama-3.2-11b-vision-instruct": 28.9,
    "ai21labs/jamba-1.5-large-instruct": 27.6,
    "google/gemma-3-4b-it": 27.6,
    "nvidia/mistral-nemo-minitron-8b-8k-instruct": 27.6,
    "microsoft/phi-3.5-moe-instruct": 27.0,
    "databricks/dbrx-instruct": 26.5,
    "mistralai/mistral-large": 24.6,
    "nv-mistralai/mistral-nemo-12b-instruct": 23.8,
    "google/gemini-2.5-flash-lite": 23.0,
    "ibm/granite-3.0-8b-instruct": 21.9,
    "ibm/granite-34b-code-instruct": 20.2,
    "google/codegemma-1.1-7b": 19.0,
    "ibm/granite-8b-code-instruct": 17.4,
    "bigcode/starcoder2-15b": 16.8,
    "stealth/glyph-cluster": 16.3,
    "zyphra/zamba2-7b-instruct": 16.2,
    "mistralai/mistral-7b-instruct-v0.3": 15.6,
    "mistralai/mixtral-8x22b-v0.1": 11.2,
    "google/codegemma-7b": 11.0,
    "ibm/granite-3.0-3b-a800m-instruct": 9.2,
    "meta/llama2-70b": 6.8,
    "microsoft/phi-3-vision-128k-instruct": 5.6,
    "google/gemma-2b": 4.1,
    "google/recurrentgemma-2b": 4.0,
    "aisingapore/sea-lion-7b-instruct": None,
}

# Exact aliases preserve identity without guessing about new variants or routers.
MODEL_ALIASES = {
    "deepseek-ai/deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "meta-llama/codellama-70b-instruct": "meta/codellama-70b",
    "meta-llama/llama-2-70b-chat": "meta/llama2-70b",
    "meta-llama/llama-3.2-11b-vision-instruct": "meta/llama-3.2-11b-vision-instruct",
    "meta-llama/llama-3.2-90b-vision-instruct": "meta/llama-3.2-90b-vision-instruct",
    "nvidia/nemotron-3.5-lightning-30b-a3b": "nvidia/nemotron-3.5-lightning",
    "nvidia/nemotron-3.5-super-vl-120b-a12b": "nvidia/nemotron-3.5-super-vl",
    "nvidia/nemotron-nano-3-30b-a3b": "nvidia/nemotron-3-nano-30b-a3b",
    "reflection/beam-free": "reflection/beam",
    "stepfun-ai/step-5-preview": "stepfun/step-5-preview",
    "stepfun/step-5-preview-free": "stepfun/step-5-preview",
}

# Routers choose changing models and cannot have a model benchmark score.
MODEL_ROUTERS = {"openrouter/free", "kilo-auto/free"}


def _identity(provider: str, model: str) -> str:
    model = model.removesuffix(":free")
    if provider in {"google", "mistral"} and "/" not in model:
        model = f"{provider}/{model}"
    return MODEL_ALIASES.get(model, model)


def model_score(provider: str, model: str) -> float | None:
    return SCORES.get(_identity(provider, model))


def reviewed_model(provider: str, model: str) -> bool:
    """Distinguish newly seen models from reviewed models lacking evidence."""
    return _identity(provider, model) in SCORES
