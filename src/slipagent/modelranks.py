"""Manually refreshed overall coding ranks; lower is better (2026-10-15)."""

from __future__ import annotations

# Store the ranking, rather than the benchmark inputs used to decide it.
RANKS = {
    "deepseek/deepseek-v4.1-flash": 1,
    "google/gemini-3.8-flash": 2,
    "stepfun/step-5-preview": 3,
    "z-ai/glm-5.3": 4,
    "google/gemini-3.6-flash": 5,
    "z-ai/glm-5.3-flash": 6,
    "dots-studio/dots-3-note-preview": 7,
    "poolside/laguna-s-2.1": 8,
    "moonshotai/kimi-k2.6": 9,
    "thinkingmachines/inkling-small": 10,
    "thinkingmachines/inkling": 11,
    "nvidia/nemotron-3.5-super-vl": 12,
    "google/gemini-3.5-flash-lite": 13,
    "nvidia/nemotron-3-ultra-550b-a55b": 14,
    "meta/muse-glimmer-30b": 15,
    "google/gemini-3-flash-preview": 16,
    "poolside/laguna-xs-2.1": 17,
    "google/gemini-2.5-pro": 18,
    "cohere/north-mini-code": 19,
    "nvidia/nemotron-3-super-120b-a12b": 20,
    "google/gemma-4-31b-it": 21,
    "google/gemma-4-26b-a4b-it": 22,
    "openai/gpt-oss-20b": 23,
    "google/diffusiongemma-26b-a4b-it": 24,
    "google/gemini-3.1-flash-lite": 25,
    "nvidia/nemotron-3.5-lightning": 26,
    "google/gemini-2.5-flash": 27,
    "mistral/labs-leanstral-1-5-1": 28,
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": 29,
    "nvidia/nemotron-3-nano-30b-a3b": 30,
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": 31,
    "stealth/glyph-cluster": 32,
    "liquid/lfm-2.5-2.6b": 33,
    "google/gemini-2.5-flash-lite": 34,
    "mistralai/mistral-large-2-instruct": 35,
    "google/gemma-3-12b-it": 36,
    "meta/llama-3.2-90b-vision-instruct": 37,
    "mistralai/codestral-22b-instruct-v0.1": 38,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 39,
    "nvidia/llama-3.1-nemotron-51b-instruct": 40,
    "nvidia/llama-3.1-nemotron-70b-instruct": 41,
    "mistralai/mixtral-8x22b-v0.1": 42,
    "nvidia/nemotron-4-340b-instruct": 43,
    "microsoft/phi-3.5-moe-instruct": 44,
    "ai21labs/jamba-1.5-large-instruct": 45,
    "meta/llama-3.2-11b-vision-instruct": 46,
    "nvidia/mistral-nemo-minitron-8b-8k-instruct": 47,
    "nv-mistralai/mistral-nemo-12b-instruct": 48,
    "databricks/dbrx-instruct": 49,
    "meta/codellama-70b": 50,
    "01-ai/yi-large": 51,
    "google/gemma-3-4b-it": 52,
    "ibm/granite-3.0-8b-instruct": 53,
    "ibm/granite-34b-code-instruct": 54,
    "google/codegemma-1.1-7b": 55,
    "ibm/granite-8b-code-instruct": 56,
    "google/codegemma-7b": 57,
    "microsoft/phi-3-vision-128k-instruct": 58,
    "mistralai/mistral-7b-instruct-v0.3": 59,
    "bigcode/starcoder2-15b": 60,
    "mistralai/mistral-large": 61,
    "ibm/granite-3.0-3b-a800m-instruct": 62,
    "meta/llama2-70b": 63,
    "google/recurrentgemma-2b": 64,
    "google/gemma-2b": 65,
}

# Exact aliases preserve identity without guessing about new variants or routers.
MODEL_ALIASES = {
    "stepfun-ai/step-5-preview": "stepfun/step-5-preview",
    "stepfun/step-5-preview-free": "stepfun/step-5-preview",
    "deepseek-ai/deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "nvidia/nemotron-3.5-super-vl-120b-a12b": "nvidia/nemotron-3.5-super-vl",
    "nvidia/nemotron-3.5-lightning-30b-a3b": "nvidia/nemotron-3.5-lightning",
    "nvidia/nemotron-nano-3-30b-a3b": "nvidia/nemotron-3-nano-30b-a3b",
    "meta-llama/llama-3.2-90b-vision-instruct": "meta/llama-3.2-90b-vision-instruct",
    "meta-llama/llama-3.2-11b-vision-instruct": "meta/llama-3.2-11b-vision-instruct",
    "meta-llama/codellama-70b-instruct": "meta/codellama-70b",
    "meta-llama/llama-2-70b-chat": "meta/llama2-70b",
}

# Routers choose changing models and cannot have a model benchmark rank.
MODEL_ROUTERS = {"openrouter/free", "kilo-auto/free"}


def model_rank(provider: str, model: str) -> int | None:
    model = model.removesuffix(":free")
    if provider in {"google", "mistral"} and "/" not in model:
        model = f"{provider}/{model}"
    return RANKS.get(MODEL_ALIASES.get(model, model))
