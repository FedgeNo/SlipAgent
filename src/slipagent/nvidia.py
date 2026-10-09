"""NVIDIA's hosted Chat Completions endpoint and explicit model profiles."""

from __future__ import annotations

import copy
from typing import Any

from .api import APIClient, APIConfigError
from .capabilities import RequestProfile
from .types import ModelInfo

ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class NvidiaClient(APIClient):
    """NVIDIA direct access; its minimal catalog does not advertise capabilities."""

    provider = "nvidia"
    default_base_url = "https://integrate.api.nvidia.com/v1"
    default_model = ULTRA_MODEL
    key_env = "NVIDIA_API_KEY"
    model_env = "NVIDIA_MODEL"
    base_url_env = "NVIDIA_BASE_URL"

    def __init__(self, api_key: str, *, profiles: dict[str, RequestProfile] | None = None,
                 reasoning_effort: str = "high", reasoning_budget: int | None = None,
                 **options: Any) -> None:
        if reasoning_effort not in {"none", "medium", "high"}:
            raise APIConfigError("NVIDIA reasoning_effort must be none, medium, or high.")
        if reasoning_budget is not None and (isinstance(reasoning_budget, bool)
                or not isinstance(reasoning_budget, int) or not -1 <= reasoning_budget <= 32768):
            raise APIConfigError("NVIDIA reasoning_budget must be -1 or 0–32768 tokens.")
        # Catalog/model card: build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b.
        # Output limit: docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b-infer.
        self.profiles = {ULTRA_MODEL: RequestProfile(
            parameters={"tools", "tool_choice", "temperature", "max_tokens"},
            context_length=1_000_000, max_completion_tokens=32768,
        )}
        if profiles:
            self.profiles.update(copy.deepcopy(profiles))
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget = reasoning_budget
        super().__init__(api_key, **options)

    async def list_models(self) -> list[ModelInfo]:
        models = await super().list_models()
        for model in models:
            model.pricing = {"prompt": "unknown", "completion": "unknown"}
            profile = self.profiles.get(model.id)
            if profile is not None:
                model.context_length = profile.context_length
                self._model_properties[model.id]["context_length"] = profile.context_length
            if model.id == ULTRA_MODEL:
                model.pricing = {"prompt": "0", "completion": "0"}
        return models

    def replacement_options(self) -> dict[str, Any]:
        return {**super().replacement_options(), "profiles": self.profiles,
                "reasoning_effort": self.reasoning_effort, "reasoning_budget": self.reasoning_budget}

    async def model_capabilities(self, model: str, *, refresh: bool = False,
                                 store: bool = True) -> RequestProfile:
        if not hasattr(self, "_capabilities"):
            self._capabilities = {}
        cached = self._capabilities.get(model)
        if cached is not None and not refresh:
            return cached
        if not hasattr(self, "_model_properties") or refresh:
            await self.list_models()
        if model not in self._model_properties:
            raise APIConfigError(f"NVIDIA does not list model {model!r}.")
        if model not in self.profiles:
            raise APIConfigError(f"No verified NVIDIA capability profile for {model!r}. "
                                 "Supply a RequestProfile through NvidiaClient(profiles=...).")
        profile = copy.deepcopy(self.profiles[model])
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
        if payload.get("model") == ULTRA_MODEL:
            template = {"enable_thinking": self.reasoning_effort != "none"}
            if self.reasoning_effort == "medium":
                template["medium_effort"] = True
            payload.setdefault("chat_template_kwargs", template)
            if self.reasoning_budget is not None:
                payload.setdefault("reasoning_budget", self.reasoning_budget)
        temperature = payload.get("temperature")
        if temperature is not None and temperature > 1:
            raise APIConfigError("NVIDIA temperature must be at most 1.")
