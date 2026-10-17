"""Google's compatible completion API with native catalog token limits."""

from __future__ import annotations

from typing import Any
import os
import httpx

from .api import APIClient, APIConfigError, APIResponseError
from .capabilities import RequestProfile
from .types import ModelInfo
from .requestbody import RequestBody
from .upload import PacedRequestBody, UploadPacer

# Free-tier availability, not a claim about the account's billing tier.
# https://ai.google.dev/gemini-api/docs/pricing (2026-10-15)
FREE_TIER_MODELS = {
    "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
    "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
}
FREE_MODELS = {"gemma-4-31b-it", "gemma-4-26b-a4b-it"}
# Leave headroom below the reported 250,000 input TPM free-tier limit.
FREE_TIER_TOKENS_PER_MINUTE = 200000


class GoogleClient(APIClient):
    provider = "google"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    default_model = "gemini-3.8-flash"
    key_env = "GEMINI_API_KEY"
    model_env = "GEMINI_MODEL"
    base_url_env = "GEMINI_BASE_URL"
    catalog_requires_auth = True

    @staticmethod
    def free_tier_limited(model: str) -> bool:
        """Whether requests for this model count against the free tier's input allowance."""
        tier = os.environ.get("GEMINI_BILLING_TIER", "free").strip().lower()
        if tier not in {"free", "paid"}:
            raise APIConfigError("GEMINI_BILLING_TIER must be free or paid.")
        return model in FREE_MODELS or (tier == "free" and model in FREE_TIER_MODELS)

    def request_body(self, payload: dict[str, Any]) -> RequestBody:
        if not self.free_tier_limited(payload["model"]):
            return super().request_body(payload)
        pacer = getattr(self, "_upload_pacer", None)
        if pacer is None:
            pacer = self._upload_pacer = UploadPacer(FREE_TIER_TOKENS_PER_MINUTE)
        return PacedRequestBody(payload, pacer)

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
                if ((not model.startswith("gemini-") and model not in FREE_MODELS)
                        or "generateContent" not in entry.get("supportedGenerationMethods", [])
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
                          provider=self.provider, pricing=({"prompt": "0", "completion": "0"} if model in FREE_MODELS
                              else {"prompt": "unknown", "completion": "unknown", "free_tier": model in FREE_TIER_MODELS}))
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
        parameters = {"max_tokens", "temperature", "top_p"}
        response_format = None
        if not model.startswith("gemma-"):
            parameters.add("response_format")
            response_format = "json_object"
        # One request cannot exceed the free tier's per-minute input allowance.
        prompt_limit = entry["inputTokenLimit"]
        if self.free_tier_limited(model):
            prompt_limit = min(prompt_limit, FREE_TIER_TOKENS_PER_MINUTE)
        profile = RequestProfile(parameters=parameters,
                                 context_length=entry["inputTokenLimit"],
                                 max_prompt_tokens=prompt_limit,
                                 max_completion_tokens=entry["outputTokenLimit"], format=response_format)
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        payload.pop("session_id", None)
        model = payload["model"]
        properties = getattr(self, "_model_properties", {}).get(model, {})
        if properties.get("thinking") is True or model.startswith(("gemini-2.5-", "gemini-3")):
            # Request Google's visible summaries while retaining its default thinking level.
            extra = payload.setdefault("extra_body", {}).setdefault("google", {})
            extra.setdefault("thinking_config", {})["include_thoughts"] = True
