"""Manually refreshed overall coding ranks; lower is better (2026-10-15)."""

from __future__ import annotations

# Store the ranking, rather than the benchmark inputs used to decide it.
RANKS = {
    "google/gemini-3.8-flash": 1,
    "google/gemini-3.6-flash": 2,
    "dots-studio/dots-3-note-preview": 3,
    "poolside/laguna-s-2.1": 4,
    "thinkingmachines/inkling-small": 5,
    "thinkingmachines/inkling": 6,
    "nvidia/nemotron-3.5-super-vl": 7,
    "google/gemini-3.5-flash-lite": 8,
    "nvidia/nemotron-3-ultra-550b-a55b": 9,
    "poolside/laguna-xs-2.1": 10,
    "google/gemini-2.5-pro": 11,
    "cohere/north-mini-code": 12,
    "nvidia/nemotron-3-super-120b-a12b": 13,
    "google/gemma-4-31b-it": 14,
    "google/gemma-4-26b-a4b-it": 15,
    "google/gemini-3.1-flash-lite": 16,
    "nvidia/nemotron-3.5-lightning": 17,
    "google/gemini-2.5-flash": 18,
    "mistral/labs-leanstral-1-5-1": 19,
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": 20,
    "stealth/glyph-cluster": 21,
    "liquid/lfm-2.5-2.6b": 22,
    "google/gemini-2.5-flash-lite": 23,
}

# Routers choose changing models and cannot have a model benchmark rank.
MODEL_ROUTERS = {"openrouter/free", "kilo-auto/free"}


def model_rank(provider: str, model: str) -> int | None:
    model = model.removesuffix(":free")
    if provider in {"google", "mistral"} and "/" not in model:
        model = f"{provider}/{model}"
    return RANKS.get(model)
