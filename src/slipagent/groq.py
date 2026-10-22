"""Groq chat completions and live catalog capabilities."""

from __future__ import annotations

from typing import Any

from .api import APIClient, APIConfigError
from .capabilities import RequestProfile
from .types import ModelInfo


def _limit(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


class GroqClient(APIClient):
    provider = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    default_model = "openai/gpt-oss-120b"
    key_env = "GROQ_API_KEY"
    model_env = "GROQ_MODEL"
    base_url_env = "GROQ_BASE_URL"
    catalog_requires_auth = True

    @staticmethod
    def chat_capable(properties: dict[str, Any]) -> bool:
        """Text-in, text-out models that advertise features; speech and guard models do not."""
        return (properties.get("active") is not False and properties.get("output_modalities") == ["text"]
                and isinstance(properties.get("supported_features"), list))

    async def list_models(self) -> list[ModelInfo]:
        models = await super().list_models()
        result = []
        for model in models:
            properties = self._model_properties[model.id]
            if not self.chat_capable(properties):
                continue
            model.context_length = _limit(properties.get("context_window"))
            # Groq lists per-token prices for billed models and omits the price of free ones.
            pricing = properties.get("pricing")
            model.pricing = pricing if isinstance(pricing, dict) else {"prompt": "0", "completion": "0"}
            result.append(model)
        return result

    async def model_capabilities(self, model: str, *, refresh: bool = False,
                                 store: bool = True) -> RequestProfile:
        cached = self._capabilities.get(model)
        if cached is not None and not refresh:
            return cached
        if not hasattr(self, "_model_properties") or refresh:
            await self.list_models()
        properties = self._model_properties.get(model)
        if properties is None or not self.chat_capable(properties):
            raise APIConfigError(f"Groq does not list an active chat model {model!r}.")
        context = _limit(properties.get("context_window"))
        if context is None:
            raise APIConfigError(f"Groq does not advertise a valid context limit for {model!r}.")
        features = properties["supported_features"]
        sampling = properties.get("supported_sampling_parameters")
        parameters = {name for name in ("max_tokens", "temperature", "top_p")
                      if isinstance(sampling, list) and name in sampling}
        if "tools" in features:
            parameters.update({"tools", "tool_choice"})
        if "json_mode" in features:
            parameters.add("response_format")
        profile = RequestProfile(parameters=parameters, context_length=context,
                                 max_completion_tokens=_limit(properties.get("max_completion_tokens")),
                                 format="json_object" if "json_mode" in features else None)
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
