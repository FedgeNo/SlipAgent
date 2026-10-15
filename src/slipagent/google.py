"""Google's compatible completion API with native catalog token limits."""

from __future__ import annotations

from typing import Any
import httpx

from .api import APIClient, APIConfigError, APIResponseError
from .capabilities import RequestProfile
from .types import ModelInfo

# Free-tier availability, not a claim about the account's billing tier.
# https://ai.google.dev/gemini-api/docs/pricing (2026-10-15)
FREE_TIER_MODELS = {
    "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
}


class GoogleClient(APIClient):
    provider = "google"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    default_model = "gemini-3.8-flash"
    key_env = "GEMINI_API_KEY"
    model_env = "GEMINI_MODEL"
    base_url_env = "GEMINI_BASE_URL"
    catalog_requires_auth = True

    def _catalog_auth(self, request: httpx.Request) -> httpx.Request:
        # Native metadata uses an API-key header; completions use bearer auth.
        request.headers.pop("Authorization", None)
        request.headers["x-goog-api-key"] = self.api_key
        return request

    async def list_models(self) -> list[ModelInfo]:
        properties: dict[str, dict[str, Any]] = {}
        token = ""
        seen: set[str] = set()
        while True:
            params = {"pageSize": "1000"}
            if token:
                params["pageToken"] = token
            raw = await self._request("GET", "/../models", params=params, auth=self._catalog_auth)
            entries = raw.get("models")
            if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
                raise APIResponseError("Google returned a malformed model catalog.")
            for entry in entries:
                name = entry.get("name")
                if not isinstance(name, str):
                    raise APIResponseError("Google returned a model without a name.")
                model = name.removeprefix("models/")
                if (not model.startswith("gemini-") or "generateContent" not in entry.get("supportedGenerationMethods", [])
                        or any(part in model for part in ("image", "audio", "tts", "live", "robotics", "computer-use"))):
                    continue
                limit = entry.get("inputTokenLimit")
                output = entry.get("outputTokenLimit")
                if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (limit, output)):
                    raise APIResponseError(f"Google returned invalid token limits for {model}.")
                properties[model] = {**entry, "id": model, "context_length": limit}
            token = raw.get("nextPageToken", "")
            if not token:
                break
            if not isinstance(token, str) or token in seen:
                raise APIResponseError("Google returned an invalid catalog page token.")
            seen.add(token)
        self._model_properties = properties
        return [ModelInfo(id=model, name=entry.get("displayName"), context_length=entry["context_length"],
                          provider=self.provider, pricing={"prompt": "unknown", "completion": "unknown",
                                                          "free_tier": model in FREE_TIER_MODELS})
                for model, entry in properties.items()]

    async def model_capabilities(self, model: str, *, refresh: bool = False,
                                 store: bool = True) -> RequestProfile:
        cached = self._capabilities.get(model)
        if cached is not None and not refresh:
            return cached
        if not hasattr(self, "_model_properties") or refresh:
            await self.list_models()
        entry = self._model_properties.get(model)
        if entry is None:
            raise APIConfigError(f"Google does not list a compatible chat model {model!r}.")
        # Embedded tool calls fit SlipAgent's compressed history without losing
        # the opaque signatures required to replay Gemini native function calls.
        profile = RequestProfile(parameters={"max_tokens", "temperature", "top_p", "response_format"},
                                 context_length=entry["inputTokenLimit"],
                                 max_prompt_tokens=entry["inputTokenLimit"],
                                 max_completion_tokens=entry["outputTokenLimit"], format="json_object")
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
