"""Estimated text-coding scores; higher is better (2026-10-27).

Only the final scores belong in runtime metadata. See docs/model-scores.md
for benchmark calibration, validation, and uncertainty.
"""

from __future__ import annotations

# Adding a model never renumbers existing models. Keep unavailable IDs so their
# scores apply if a provider lists them again. None records a reviewed gap.
SCORES: dict[str, float | None] = {
    "minimax/minimax-m3.1-flash": 96.4,
    "deepseek/deepseek-v4.1-flash": 96.1,
    "google/gemini-3.8-flash": 95.7,
    "z-ai/glm-5.3": 94.7,
    "z-ai/glm-5.3-flash": 93.4,
    "stepfun/step-5-preview": 92.7,
    "google/gemini-3.6-flash": 89.5,
    "qwen/qwen3.8-27b": 85.7,
    "reflection/beam": 85.5,
    "poolside/laguna-s-2.1": 85.4,
    "dots-studio/dots-3-note-preview": 83.2,
    "moonshotai/kimi-k2.6": 83.2,
    "mistral/labs-leanstral-1-5-1": 79.5,
    "thinkingmachines/inkling-small": 79.2,
    "thinkingmachines/inkling": 78.7,
    "nvidia/nemotron-3.5-super-vl": 76.5,
    "google/gemini-3-flash-preview": 76.1,
    "nvidia/nemotron-3-ultra-550b-a55b": 75.8,
    "xingchen-agi/xing4.0-29b-a4b": 75.1,
    "recursive/flint-v1.0": 73.7,
    "meta/muse-glimmer-30b": 73.4,
    "google/gemini-3.5-flash-lite": 72.5,
    "poolside/laguna-xs-2.1": 68.5,
    "google/gemma-4-31b-it": 68.4,
    "cohere/north-mini-code": 66.1,
    "google/gemini-2.5-pro": 60.2,
    "nvidia/nemotron-3-super-120b-a12b": 59.8,
    "google/gemma-4-26b-a4b-it": 55.7,
    "google/diffusiongemma-26b-a4b-it": 54.8,
    "openai/gpt-oss-20b": 53.4,
    "openai/gpt-oss-120b": 52.6,
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": 51.6,
    "google/gemini-3.1-flash-lite": 50.3,
    "nvidia/nemotron-3.5-lightning": 46.9,
    "google/gemini-2.5-flash": 46.5,
    "nvidia/nemotron-3-nano-30b-a3b": 46.2,
    "mistralai/mistral-large-2-instruct": 46.1,
    "liquid/lfm-2.5-2.6b": 44.2,
    "google/gemini-2.5-flash-lite": 24.7,
    "mistralai/codestral-22b-instruct-v0.1": 23.0,
    "google/gemma-3-12b-it": 21.7,
    "nvidia/llama-3.1-nemotron-70b-instruct": 20.3,
    "01-ai/yi-large": 19.6,
    "meta/codellama-70b": 18.5,
    "mistralai/mixtral-8x22b-v0.1": 18.4,
    "databricks/dbrx-instruct": 18.2,
    "ibm/granite-34b-code-instruct": 17.9,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 17.7,
    "nvidia/nemotron-4-340b-instruct": 16.9,
    "ai21labs/jamba-1.5-large-instruct": 15.6,
    "nvidia/mistral-nemo-minitron-8b-8k-instruct": 15.6,
    "ibm/granite-8b-code-instruct": 15.2,
    "microsoft/phi-3.5-moe-instruct": 15.2,
    "mistralai/mistral-large": 14.5,
    "nv-mistralai/mistral-nemo-12b-instruct": 13.8,
    "google/gemma-3-4b-it": 10.7,
    "google/codegemma-1.1-7b": 10.2,
    "mistralai/mistral-7b-instruct-v0.3": 9.9,
    "ibm/granite-3.0-8b-instruct": 9.8,
    "zyphra/zamba2-7b-instruct": 9.8,
    "bigcode/starcoder2-15b": 9.5,
    "google/codegemma-7b": 9.1,
    "ibm/granite-3.0-3b-a800m-instruct": 4.3,
    "meta/llama2-70b": 3.9,
    "google/gemma-2b": 2.4,
    "groq/allam-2-7b": 2.1,
    "google/recurrentgemma-2b": 2.0,
    "aisingapore/sea-lion-7b-instruct": None,
    "nvidia/llama-3.1-nemotron-51b-instruct": None,
    "nvidia/nemotron-3.5-bolt-e4b": None,
    "openai/gpt-oss-safeguard-20b": None,
    "stealth/glyph-cluster": None,
}

# Exact aliases preserve identity without guessing about new variants or routers.
MODEL_ALIASES = {
    "deepseek-ai/deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "meta-llama/codellama-70b-instruct": "meta/codellama-70b",
    "meta-llama/llama-2-70b-chat": "meta/llama2-70b",
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
