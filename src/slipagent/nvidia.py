"""NVIDIA's hosted Chat Completions endpoint and explicit model profiles."""

from __future__ import annotations

import copy
from typing import Any

from .api import APIClient, APIConfigError
from .capabilities import RequestProfile
from .data_text import MessageSections, RenderedContent
from .nvidiaprofiles import USER_INSTRUCTIONS, builtin_profile
from .types import ModelInfo

ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
MISTRAL_LARGE_4_MODEL = "mistralai/mistral-large-4-1t-a52b"


class NvidiaClient(APIClient):
    """NVIDIA direct access; its minimal catalog does not advertise capabilities."""

    provider = "nvidia"
    default_base_url = "https://integrate.api.nvidia.com/v1"
    max_temperature = 1.0
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
        self.profiles = copy.deepcopy(profiles or {})
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget = reasoning_budget
        super().__init__(api_key, **options)

    async def list_models(self) -> list[ModelInfo]:
        models = await super().list_models()
        for model in models:
            # NVIDIA's hosted API catalog is available through its free trial.
            # Pricing and endpoint capabilities are separate facts.
            model.pricing = {"prompt": "0", "completion": "0"}
            profile = self.profiles.get(model.id)
            if profile is None:
                profile = builtin_profile(model.id)
            if profile is not None:
                model.context_length = profile.context_length
                self._model_properties[model.id]["context_length"] = profile.context_length
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
        profile = self.profiles.get(model)
        if profile is None:
            profile = builtin_profile(model)
        if profile is None:
            raise APIConfigError(f"No verified NVIDIA capability profile for {model!r}. "
                                 "Supply a RequestProfile through NvidiaClient(profiles=…).")
        profile = copy.deepcopy(profile)
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
        if payload.get("model") == MISTRAL_LARGE_4_MODEL:
            if self.reasoning_effort not in {"none", "high"}:
                raise APIConfigError("NVIDIA Mistral Large 4 reasoning_effort must be none or high.")
            # Omitting this field serves non-reasoning mode on this endpoint.
            payload.setdefault("reasoning_effort", self.reasoning_effort)
        if payload.get("model") in USER_INSTRUCTIONS:
            messages: list[dict[str, Any]] = []
            for message in payload["messages"]:
                message = dict(message)
                if message["role"] == "system":
                    message["role"] = "user"
                if messages and message["role"] == messages[-1]["role"] == "user":
                    messages[-1]["content"] = RenderedContent(MessageSections([
                        messages[-1]["content"], "\n\n", message["content"],
                    ]))
                else:
                    messages.append(message)
            payload["messages"] = messages
        if payload.get("model") == ULTRA_MODEL:
            template = {"enable_thinking": self.reasoning_effort != "none"}
            if self.reasoning_effort == "medium":
                template["medium_effort"] = True
            payload.setdefault("chat_template_kwargs", template)
            if self.reasoning_budget is not None:
                payload.setdefault("reasoning_budget", self.reasoning_budget)
        temperature = payload.get("temperature")
        if temperature is not None and temperature > self.max_temperature:
            raise APIConfigError(f"NVIDIA temperature must be at most {self.max_temperature:g}.")
