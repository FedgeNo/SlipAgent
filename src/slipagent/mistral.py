"""Mistral chat completions and authenticated model capabilities."""

from __future__ import annotations

from typing import Any

from .api import APIClient, APIConfigError
from .capabilities import RequestProfile
from .types import ModelInfo


class MistralClient(APIClient):
    provider = "mistral"
    default_base_url = "https://api.mistral.ai/v1"
    default_model = "mistral-small-latest"
    key_env = "MISTRAL_API_KEY"
    model_env = "MISTRAL_MODEL"
    base_url_env = "MISTRAL_BASE_URL"
    catalog_requires_auth = True

    async def list_models(self) -> list[ModelInfo]:
        models = await super().list_models()
        result = []
        for model in models:
            properties = self._model_properties[model.id]
            capabilities = properties.get("capabilities")
            if not isinstance(capabilities, dict) or capabilities.get("completion_chat") is not True:
                continue
            properties["context_length"] = properties.get("max_context_length")
            model.context_length = properties["context_length"]
            # The catalog describes models, not the account's free-plan allowance.
            model.pricing = {"prompt": "unknown", "completion": "unknown"}
            result.append(model)
            for alias in properties.get("aliases") or []:
                if isinstance(alias, str) and alias not in self._model_properties:
                    self._model_properties[alias] = {**properties, "id": alias}
                    result.append(ModelInfo(id=alias, provider=self.provider,
                                            context_length=model.context_length, pricing=model.pricing))
        return result

    async def model_capabilities(self, model: str, *, refresh: bool = False,
                                 store: bool = True) -> RequestProfile:
        cached = self._capabilities.get(model)
        if cached is not None and not refresh:
            return cached
        if not hasattr(self, "_model_properties") or refresh:
            await self.list_models()
        properties = self._model_properties.get(model, {})
        capabilities = properties.get("capabilities", {})
        limit = properties.get("max_context_length")
        if not isinstance(capabilities, dict) or capabilities.get("completion_chat") is not True:
            raise APIConfigError(f"Mistral does not advertise chat completion support for {model!r}.")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise APIConfigError(f"Mistral does not advertise a valid context limit for {model!r}.")
        parameters = {"max_tokens", "temperature", "top_p", "response_format"}
        if capabilities.get("function_calling") is True:
            parameters.update({"tools", "tool_choice"})
        profile = RequestProfile(parameters=parameters, context_length=limit, format="json_object")
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
        # Mistral includes stream usage without OpenAI's stream_options field.
        payload.pop("stream_options", None)
        if payload.get("tool_choice") == "required":
            payload["tool_choice"] = "any"
