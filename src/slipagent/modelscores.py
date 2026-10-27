"""Estimated text-coding scores; higher is better (2026-10-26).

Only the final scores belong in runtime metadata. See docs/model-scores.md
for benchmark calibration, validation, and uncertainty.
"""

from __future__ import annotations

# Adding a model never renumbers existing models. Keep unavailable IDs so their
# scores apply if a provider lists them again. None records a reviewed gap.
SCORES: dict[str, float | None] = {
    "minimax/minimax-m3.1-flash": 96.2,
    "deepseek/deepseek-v4.1-flash": 95.9,
    "google/gemini-3.8-flash": 95.5,
    "z-ai/glm-5.3": 94.5,
    "z-ai/glm-5.3-flash": 93.1,
    "stepfun/step-5-preview": 92.4,
    "google/gemini-3.6-flash": 89.1,
    "qwen/qwen3.8-27b": 85.0,
    "reflection/beam": 85.0,
    "poolside/laguna-s-2.1": 84.8,
    "dots-studio/dots-3-note-preview": 82.9,
    "moonshotai/kimi-k2.6": 82.8,
    "mistral/labs-leanstral-1-5-1": 79.7,
    "thinkingmachines/inkling-small": 78.6,
    "thinkingmachines/inkling": 78.4,
    "google/gemini-3-flash-preview": 78.0,
    "nvidia/nemotron-3-ultra-550b-a55b": 75.2,
    "nvidia/nemotron-3.5-super-vl": 74.9,
    "xingchen-agi/xing4.0-29b-a4b": 74.7,
    "recursive/flint-v1.0": 73.3,
    "meta/muse-glimmer-30b": 73.0,
    "google/gemini-3.5-flash-lite": 71.6,
    "poolside/laguna-xs-2.1": 68.1,
    "google/gemma-4-31b-it": 67.2,
    "cohere/north-mini-code": 65.6,
    "mistralai/mistral-large-2-instruct": 63.8,
    "google/gemini-2.5-pro": 59.4,
    "nvidia/nemotron-3-super-120b-a12b": 59.4,
    "google/gemma-4-26b-a4b-it": 54.8,
    "openai/gpt-oss-20b": 52.7,
    "openai/gpt-oss-120b": 52.1,
    "google/diffusiongemma-26b-a4b-it": 50.1,
    "google/gemini-3.1-flash-lite": 49.1,
    "google/gemma-3-12b-it": 47.3,
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": 46.9,
    "nvidia/nemotron-3.5-lightning": 45.9,
    "nvidia/nemotron-3-nano-30b-a3b": 45.5,
    "google/gemini-2.5-flash": 44.2,
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": 43.5,
    "liquid/lfm-2.5-2.6b": 39.6,
    "meta/llama-3.2-90b-vision-instruct": 35.0,
    "nvidia/nemotron-3.5-bolt-e4b": 34.7,
    "mistralai/codestral-22b-instruct-v0.1": 33.9,
    "nvidia/llama-3.1-nemotron-51b-instruct": 30.2,
    "nvidia/llama-3.1-nemotron-70b-instruct": 29.9,
    "nvidia/nemotron-4-340b-instruct": 29.5,
    "01-ai/yi-large": 29.1,
    "databricks/dbrx-instruct": 28.1,
    "meta/codellama-70b": 28.0,
    "ai21labs/jamba-1.5-large-instruct": 27.6,
    "google/gemma-3-4b-it": 27.6,
    "mistralai/mixtral-8x22b-v0.1": 27.6,
    "nvidia/mistral-nemo-minitron-8b-8k-instruct": 27.6,
    "microsoft/phi-3.5-moe-instruct": 27.0,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 26.9,
    "ibm/granite-34b-code-instruct": 26.8,
    "meta/llama-3.2-11b-vision-instruct": 25.7,
    "nv-mistralai/mistral-nemo-12b-instruct": 23.8,
    "ibm/granite-8b-code-instruct": 23.2,
    "google/gemini-2.5-flash-lite": 23.0,
    "mistralai/mistral-large": 22.2,
    "ibm/granite-3.0-8b-instruct": 20.1,
    "google/codegemma-1.1-7b": 19.0,
    "stealth/glyph-cluster": 16.3,
    "mistralai/mistral-7b-instruct-v0.3": 15.7,
    "zyphra/zamba2-7b-instruct": 15.6,
    "bigcode/starcoder2-15b": 15.2,
    "google/codegemma-7b": 14.7,
    "ibm/granite-3.0-3b-a800m-instruct": 9.2,
    "meta/llama2-70b": 6.5,
    "microsoft/phi-3-vision-128k-instruct": 5.6,
    "google/gemma-2b": 4.2,
    "google/recurrentgemma-2b": 4.0,
    "groq/allam-2-7b": 3.6,
    "aisingapore/sea-lion-7b-instruct": None,
    "openai/gpt-oss-safeguard-20b": None,
}

# Exact aliases preserve identity without guessing about new variants or routers.
MODEL_ALIASES = {
    "deepseek-ai/deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "meta-llama/codellama-70b-instruct": "meta/codellama-70b",
    "meta-llama/llama-2-70b-chat": "meta/llama2-70b",
    "meta-llama/llama-3.2-11b-vision-instruct": "meta/llama-3.2-11b-vision-instruct",
    "meta-llama/llama-3.2-90b-vision-instruct": "meta/llama-3.2-90b-vision-instruct",
    "minimaxai/minimax-m3.1-flash": "minimax/minimax-m3.1-flash",
    "mistral/labs-leanstral-1-5": "mistral/labs-leanstral-1-5-1",
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
    if provider in {"google", "groq", "mistral"} and "/" not in model:
        model = f"{provider}/{model}"
    return MODEL_ALIASES.get(model, model)


def model_score(provider: str, model: str) -> float | None:
    return SCORES.get(_identity(provider, model))


def reviewed_model(provider: str, model: str) -> bool:
    """Distinguish newly seen models from reviewed models lacking evidence."""
    return _identity(provider, model) in SCORES
