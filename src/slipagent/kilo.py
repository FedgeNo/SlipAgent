"""Kilo Gateway chat access and live catalog capabilities."""

from __future__ import annotations

import copy
from typing import Any

from .api import APIClient, APIConfigError
from .capabilities import ModelCapabilities, RequestProfile


class KiloClient(APIClient):
    """Use Kilo's catalog contract without OpenRouter-only metadata requests."""

    provider = "kilo"
    default_base_url = "https://api.kilo.ai/api/gateway"
    default_model = "kilo-auto/free"
    key_env = "KILO_API_KEY"
    model_env = "KILO_MODEL"
    base_url_env = "KILO_BASE_URL"

    async def model_capabilities(self, model: str, *, refresh: bool = False,
                                 store: bool = True) -> RequestProfile:
        cached = self._capabilities.get(model)
        if cached is not None and not refresh:
            return cached
        if not hasattr(self, "_model_properties") or refresh:
            await self.list_models()
        if model not in self._model_properties:
            raise APIConfigError(f"Kilo does not list model {model!r}.")
        properties = copy.deepcopy(self._model_properties[model])
        parameters = properties.get("supported_parameters")
        if not isinstance(parameters, list) or any(not isinstance(value, str) for value in parameters):
            raise APIConfigError(f"Kilo does not advertise valid capabilities for {model!r}.")
        # Kilo publishes limits and reasoning variants in its catalog; it does
        # not expose OpenRouter's per-model endpoint or account-quota routes.
        top = properties.get("top_provider")
        if isinstance(top, dict):
            for name in ("context_length", "max_prompt_tokens", "max_completion_tokens"):
                if properties.get(name) is None and top.get(name) is not None:
                    properties[name] = top[name]
        options = properties.get("opencode")
        variants = options.get("variants") if isinstance(options, dict) else None
        if isinstance(variants, dict) and "reasoning" not in properties:
            efforts = []
            for variant in variants.values():
                reasoning = variant.get("reasoning") if isinstance(variant, dict) else None
                if isinstance(reasoning, dict) and "effort" in reasoning:
                    efforts.append(reasoning["effort"])
            if efforts:
                properties["reasoning"] = {"supported_efforts": efforts}
        try:
            profile = ModelCapabilities(properties, [{"tag": "kilo", "supported_parameters": parameters}])
        except ValueError as exc:
            raise APIConfigError(f"Cannot use {model} on Kilo: {exc}") from exc
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
        if isinstance(profile, ModelCapabilities):
            if profile.reasoning is not None:
                payload["reasoning"] = profile.reasoning
            elif profile.include_reasoning:
                payload["include_reasoning"] = True
