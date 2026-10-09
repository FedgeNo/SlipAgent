"""OpenRouter routing, capabilities, and account metadata."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from .api import APIClient
from .api import (
    DEFAULT_TEMPERATURE as DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT as DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES as DEFAULT_MAX_RETRIES,
    RETRYABLE_STATUS as RETRYABLE_STATUS,
    RetryPolicy as RetryPolicy,
    _build_error as _build_error,
    _StreamCompletion as _StreamCompletion,
    _parse_completion as _parse_completion,
    _parse_retry_after as _parse_retry_after,
    APIConfigError as OpenRouterConfigError,
    APITransportError as OpenRouterTransportError,
    APIAuthError as OpenRouterAuthError,
    APIContextError as OpenRouterContextError,
    APILimitError as OpenRouterLimitError,
    APIResponseError as OpenRouterAPIError,
    APIError as OpenRouterError,
)
from .capabilities import ModelCapabilities, RequestProfile
from .types import KeyInfo

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(APIClient):
    """OpenRouter provider with live routing metadata and key quotas."""

    provider = "openrouter"
    default_base_url = DEFAULT_BASE_URL
    default_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    key_env = "OPENROUTER_API_KEY"
    model_env = "OPENROUTER_MODEL"
    base_url_env = "OPENROUTER_BASE_URL"

    def request_headers(self, http_referer: str | None, app_title: str | None) -> dict[str, str]:
        from . import __version__
        return {
            "HTTP-Referer": http_referer or "https://github.com/FedgeNo/SlipAgent",
            "X-OpenRouter-Title": app_title or "SlipAgent",
            "X-OpenRouter-Categories": "cli-agent",
            "User-Agent": os.environ.get("OPENROUTER_USER_AGENT") or f"SlipAgent/{__version__}",
        }

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        payload.setdefault("provider", profile.provider_preferences() if isinstance(profile, ModelCapabilities)
                           else {"require_parameters": True})
        if isinstance(profile, ModelCapabilities):
            if profile.reasoning is not None:
                payload["reasoning"] = profile.reasoning
            elif profile.include_reasoning:
                payload["include_reasoning"] = True

    async def model_capabilities(self, model: str, *, refresh: bool = False, store: bool = True) -> RequestProfile | None:
        """Fetch endpoint properties; cache them between explicit selections."""
        cache: dict[str, RequestProfile | None] = getattr(self, "_capabilities", {})
        self._capabilities = cache
        if model in cache and not refresh:
            return cache[model]
        if not hasattr(self, "_model_properties"):
            await self.list_models()
        properties = self._model_properties.get(model)
        # Some compatible gateways expose only the older minimal catalog.
        if properties is not None and not isinstance(properties.get("supported_parameters"), list):
            if store and model not in cache:
                cache[model] = None
            return None
        raw = await self._request("GET", f"/models/{quote(model, safe='/')}/endpoints")
        data = raw.get("data")
        if properties is None and isinstance(data, dict):
            properties = data
        endpoints = data.get("endpoints") if isinstance(data, dict) else None
        if not isinstance(endpoints, list) or any(not isinstance(entry, dict) or not isinstance(entry.get("supported_parameters"), list)
                                                or any(not isinstance(value, str) for value in entry["supported_parameters"]) for entry in endpoints):
            raise OpenRouterAPIError("OpenRouter returned malformed model endpoint properties.")
        try:
            if properties is None:
                raise ValueError("No model properties were supplied by the catalog or endpoint metadata.")
            capabilities = ModelCapabilities(properties, endpoints)
        except ValueError as exc:
            raise OpenRouterConfigError(f"Cannot use {model}: {exc}") from exc
        if store:
            self.cache_capabilities(model, capabilities)
        return capabilities

    async def key_info(self) -> KeyInfo:
        """Return this key's identity, spend limit, and quota.

        Unlike `/models`, this endpoint requires a valid key, which makes it
        the correct way to verify credentials.
        """
        raw = await self._request("GET", "/key")
        data = raw.get("data")
        try:
            if not isinstance(data, dict):
                raise ValueError("data must be a key object")
            return KeyInfo.from_api(data)
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise OpenRouterAPIError(f"OpenRouter returned malformed key information: {exc}") from exc


__all__ = ['OpenRouterConfigError', 'OpenRouterTransportError', 'OpenRouterAuthError', 'OpenRouterContextError', 'OpenRouterLimitError', 'OpenRouterAPIError', 'OpenRouterError', 'OpenRouterClient', 'RetryPolicy', 'DEFAULT_TEMPERATURE', 'RETRYABLE_STATUS', '_build_error', '_StreamCompletion']
