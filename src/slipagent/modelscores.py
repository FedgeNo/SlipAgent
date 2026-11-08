"""Estimated text-coding scores; higher is better (2026-10-30).

Only the final scores belong in runtime metadata. See docs/model-scores.md
for benchmark calibration, validation, and uncertainty.
"""

from __future__ import annotations

# Routing preference for missing benchmarks, just above Gemini 3.8 Flash (95.6).
# This is not a measured or calibrated coding score.
UNKNOWN_MODEL_SCORE = 95.7

# Adding a model never renumbers existing models. Keep unavailable IDs so their
# scores apply if a provider lists them again. None records a reviewed gap.
SCORES: dict[str, float | None] = {
    "minimax/minimax-m3.1-flash": 96.3,
    "deepseek/deepseek-v4.1-flash": 96.2,
    "google/gemini-3.8-flash": 95.6,
    "xiaomi/mimo-v2.6-pro": 95.5,
    "z-ai/glm-5.3": 95.0,
    "inclusionai/ling-3.1-flash-1019": 94.7,
    "mistralai/mistral-large-4-1t-a52b": 94.0,
    "moonshotai/kimi-k3": 93.8,
    "z-ai/glm-5.3-flash": 93.7,
    "xiaomi/mimo-v2.6-flash": 93.5,
    "stepfun/step-5-preview": 93.2,
    "google/gemini-3.6-flash": 90.1,
    "z-ai/glm-5.2": 89.3,
    "qwen/qwen3.8-27b": 86.7,
    "poolside/laguna-s-2.1": 86.0,
    "reflection/beam": 85.7,
    "moonshotai/kimi-k2.7-code": 84.4,
    "dots-studio/dots-3-note-preview": 83.7,
    "minimax/minimax-m3": 83.6,
    "z-ai/glm-5.1": 83.0,
    "moonshotai/kimi-k2.6": 81.7,
    "minimax/minimax-m2.7": 81.5,
    "internlm/intern-s2": 81.1,
    "qwen/qwen3.6-plus-preview": 80.5,
    "minimax/minimax-m2.5": 80.2,
    "mistral/labs-leanstral-1-5-1": 79.6,
    "z-ai/glm-5": 79.4,
    "thinkingmachines/inkling-small": 78.4,
    "thinkingmachines/inkling": 77.3,
    "google/gemini-3-flash-preview": 76.7,
    "nvidia/nemotron-3-ultra-550b-a55b": 76.1,
    "nvidia/nemotron-3.5-super-vl": 76.0,
    "xingchen-agi/xing4.0-29b-a4b": 75.8,
    "recursive/flint-v1.0": 74.4,
    "meta/muse-glimmer-30b": 74.1,
    "google/gemini-3.5-flash-lite": 73.0,
    "minimax/minimax-m2.1": 72.9,
    "z-ai/glm-4.7": 71.3,
    "minimax/minimax-m2": 70.7,
    "xiaomi/mimo-v2-flash": 69.9,
    "poolside/laguna-xs-2.1": 68.8,
    "google/gemma-4-31b-it": 67.9,
    "cohere/north-mini-code": 66.2,
    "z-ai/glm-4.6": 65.8,
    "google/gemini-2.5-pro": 60.7,
    "nvidia/nemotron-3-super-120b-a12b": 60.1,
    "openai/gpt-oss-20b": 56.1,
    "google/gemma-4-26b-a4b-it": 56.0,
    "openai/gpt-oss-120b": 56.0,
    "google/diffusiongemma-26b-a4b-it": 54.2,
    "z-ai/glm-4.7-flash": 53.9,
    "nvidia/nemotron-3.5-lightning": 52.6,
    "google/gemini-3.1-flash-lite": 52.0,
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": 51.0,
    "google/gemini-2.5-flash": 46.2,
    "nvidia/nemotron-3-nano-30b-a3b": 45.9,
    "mistralai/mistral-large-2-instruct": 45.8,
    "liquid/lfm-2.5-2.6b": 43.6,
    "liquid/lfm-3-1.7b": 42.4,
    "google/gemini-2.5-flash-lite": 24.5,
    "mistralai/codestral-22b-instruct-v0.1": 22.8,
    "google/gemma-3-12b-it": 21.4,
    "nvidia/llama-3.1-nemotron-70b-instruct": 20.1,
    "01-ai/yi-large": 19.5,
    "meta/codellama-70b": 18.3,
    "mistralai/mixtral-8x22b-v0.1": 18.3,
    "databricks/dbrx-instruct": 18.1,
    "ibm/granite-34b-code-instruct": 17.8,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 17.5,
    "nvidia/nemotron-4-340b-instruct": 16.7,
    "ai21labs/jamba-1.5-large-instruct": 15.5,
    "nvidia/mistral-nemo-minitron-8b-8k-instruct": 15.5,
    "ibm/granite-8b-code-instruct": 15.1,
    "microsoft/phi-3.5-moe-instruct": 15.1,
    "mistralai/mistral-large": 14.4,
    "nv-mistralai/mistral-nemo-12b-instruct": 13.7,
    "google/gemma-3-4b-it": 10.6,
    "google/codegemma-1.1-7b": 10.1,
    "mistralai/mistral-7b-instruct-v0.3": 9.8,
    "ibm/granite-3.0-8b-instruct": 9.7,
    "zyphra/zamba2-7b-instruct": 9.7,
    "bigcode/starcoder2-15b": 9.4,
    "google/codegemma-7b": 9.1,
    "ibm/granite-3.0-3b-a800m-instruct": 4.3,
    "meta/llama2-70b": 3.8,
    "google/gemma-2b": 2.4,
    "groq/allam-2-7b": 2.1,
    "google/recurrentgemma-2b": 2.0,
    "aisingapore/sea-lion-7b-instruct": None,
    "internscience/agents-a1": None,
    "nvidia/llama-3.1-nemotron-51b-instruct": None,
    "nvidia/nemotron-3.5-bolt-e4b": None,
    "openai/gpt-oss-safeguard-20b": None,
    "stealth/crouton-alpha": None,
    "stealth/glyph-cluster": None,
    "z-ai/glm-4.5-flash": None,
    "z-ai/glm-5-turbo": None,
}

# Exact aliases preserve identity without guessing about new variants or routers.
MODEL_ALIASES = {
    "deepseek-ai/deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "inclusionai/ling-3.1-flash-1019-free": "inclusionai/ling-3.1-flash-1019",
    "meta-llama/codellama-70b-instruct": "meta/codellama-70b",
    "meta-llama/llama-2-70b-chat": "meta/llama2-70b",
    "minimaxai/minimax-m3.1-flash": "minimax/minimax-m3.1-flash",
    "mistral/labs-leanstral-1-5": "mistral/labs-leanstral-1-5-1",
    "moonshotai/k2.6-code-preview": "moonshotai/kimi-k2.6",
    "moonshotai/kimi-for-coding": "moonshotai/kimi-k2.7-code",
    "nvidia/nemotron-3.5-lightning-30b-a3b": "nvidia/nemotron-3.5-lightning",
    "nvidia/nemotron-3.5-super-vl-120b-a12b": "nvidia/nemotron-3.5-super-vl",
    "nvidia/nemotron-nano-3-30b-a3b": "nvidia/nemotron-3-nano-30b-a3b",
    "reflection/beam-free": "reflection/beam",
    "stepfun-ai/step-5-preview": "stepfun/step-5-preview",
    "stepfun/step-5-preview-free": "stepfun/step-5-preview",
}

# Routers choose changing models and cannot have a model benchmark score.
MODEL_ROUTERS = {"openrouter/free", "kilo-auto/free"}


# AIHubMix serves free routes as bare IDs ("coding-glm-5.3-free"). Map them by
# ID alone so cached catalogs and the current model score like a live catalog.
AIHUBMIX_NAMESPACES = (
    ("xiaomi-mimo-", "xiaomi/mimo-"), ("mimo-", "xiaomi/mimo-"), ("glm-", "z-ai/glm-"),
    ("kimi-", "moonshotai/kimi-"), ("k2.", "moonshotai/k2."), ("minimax-", "minimax/minimax-"),
    ("nemotron-", "nvidia/nemotron-"), ("gemma-", "google/gemma-"), ("north-", "cohere/north-"),
    ("dots-", "dots-studio/dots-"), ("lfm-", "liquid/lfm-"), ("xing", "xingchen-agi/xing"),
    ("qwen", "qwen/qwen"), ("intern-", "internlm/intern-"), ("agents-", "internscience/agents-"),
    ("crouton-", "stealth/crouton-"), ("flint-", "recursive/flint-"),
)


def _identity(provider: str, model: str) -> str:
    model = model.removesuffix(":free")
    if provider == "aihubmix" and "/" not in model:
        model = model.removesuffix("-free").removeprefix("coding-")
        model = next((namespace + model[len(prefix):] for prefix, namespace in AIHUBMIX_NAMESPACES
                      if model.startswith(prefix)), model)
    if provider in {"google", "groq", "mistral"} and "/" not in model:
        model = f"{provider}/{model}"
    elif provider == "zai" and "/" not in model:
        model = f"z-ai/{model}"
    return MODEL_ALIASES.get(model, model)


def model_score(provider: str, model: str) -> float | None:
    return SCORES.get(_identity(provider, model))


def reviewed_model(provider: str, model: str) -> bool:
    """Distinguish newly seen models from reviewed models lacking evidence."""
    return _identity(provider, model) in SCORES
