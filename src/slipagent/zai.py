"""Z.ai's official GLM Chat Completions endpoint and reviewed model limits."""

from __future__ import annotations

import copy
from typing import Any

from .api import APIClient, APIConfigError
from .capabilities import RequestProfile
from .types import ModelInfo

# Z.ai's catalog route requires a key and does not publish prices or limits.
# Limits and USD-per-million prices follow docs.z.ai/guides/overview/pricing
# and the GLM model guides, as collected by models.dev on 2026-11-08. A price
# of zero marks the Flash models Z.ai lists as free.
MODELS: dict[str, tuple[int, int, float, float]] = {
    "glm-4.7-flash": (200_000, 131_072, 0, 0),
    "glm-4.5-flash": (131_072, 98_304, 0, 0),
    "glm-4.6v-flash": (128_000, 32_768, 0, 0),
    "glm-5.3": (1_000_000, 131_072, 1.4, 4.4),
    "glm-5.3-flash": (1_000_000, 131_072, 0.15, 0.5),
    "glm-5.3-flashx": (1_000_000, 131_072, 0.37, 1.25),
    "glm-5.2": (1_000_000, 131_072, 1.4, 4.4),
    "glm-5.1": (200_000, 131_072, 1.4, 4.4),
    "glm-5": (204_800, 131_072, 1, 3.2),
    "glm-5-turbo": (200_000, 131_072, 1.2, 4),
    "glm-5v-turbo": (200_000, 131_072, 1.2, 4),
    "glm-4.7": (204_800, 131_072, 0.6, 2.2),
    "glm-4.7-flashx": (200_000, 131_072, 0.07, 0.4),
    "glm-4.6": (204_800, 131_072, 0.6, 2.2),
    "glm-4.6v": (128_000, 32_768, 0.3, 0.9),
    "glm-4.5": (131_072, 98_304, 0.6, 2.2),
    "glm-4.5-air": (131_072, 98_304, 0.2, 1.1),
    "glm-4.5v": (64_000, 16_384, 0.6, 1.8),
}


def per_token(price: float) -> str:
    """OpenRouter-style per-token price text from a per-million price."""
    return "0" if price == 0 else f"{price / 1_000_000:.12g}"


class ZaiClient(APIClient):
    """Z.ai direct access with a reviewed model table; every listed model takes tools."""

    provider = "zai"
    default_base_url = "https://api.z.ai/api/paas/v4"
    default_model = "glm-4.7-flash"
    key_env = "ZAI_API_KEY"
    model_env = "ZAI_MODEL"
    base_url_env = "ZAI_BASE_URL"

    async def list_models(self) -> list[ModelInfo]:
        models = []
        self._model_properties = {}
        for model_id, (context, output, prompt, completion) in MODELS.items():
            pricing = {"prompt": per_token(prompt), "completion": per_token(completion)}
            self._model_properties[model_id] = {"id": model_id, "context_length": context,
                                                "max_completion_tokens": output, "pricing": pricing}
            models.append(ModelInfo(model_id, context_length=context, pricing=pricing, provider=self.provider))
        return models

    async def model_capabilities(self, model: str, *, refresh: bool = False,
                                 store: bool = True) -> RequestProfile:
        if not hasattr(self, "_capabilities"):
            self._capabilities = {}
        cached = self._capabilities.get(model)
        if cached is not None and not refresh:
            return copy.deepcopy(cached)
        limits = MODELS.get(model)
        if limits is None:
            raise APIConfigError(f"No reviewed Z.ai profile for {model!r}; choose one of {', '.join(MODELS)}.")
        profile = RequestProfile(parameters={"max_tokens", "temperature", "tools", "tool_choice"},
                                 context_length=limits[0], max_completion_tokens=limits[1])
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
        from .reasoning import reasoning_options
        for key, value in reasoning_options(self.provider, payload["model"]).items():
            payload.setdefault(key, value)
