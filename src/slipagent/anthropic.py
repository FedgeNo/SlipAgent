"""Claude's compatible completions and authenticated native model catalog."""

from __future__ import annotations

from typing import Any

from .api import APIClient, APIConfigError, APIResponseError
from .capabilities import RequestProfile
from .types import ModelInfo


class AnthropicClient(APIClient):
    provider = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"
    default_model = "claude-sonnet-4-6"
    key_env = "ANTHROPIC_API_KEY"
    model_env = "ANTHROPIC_MODEL"
    base_url_env = "ANTHROPIC_BASE_URL"
    catalog_requires_auth = True
    max_temperature = 1.0

    def request_headers(self, http_referer: str | None, app_title: str | None) -> dict[str, str]:
        return {"anthropic-version": "2026-10-21"}

    async def list_models(self) -> list[ModelInfo]:
        properties: dict[str, dict[str, Any]] = {}
        cursor = ""
        seen: set[str] = set()
        while True:
            params = {"limit": "1000"}
            if cursor:
                params["after_id"] = cursor
            raw = await self._request("GET", "/models", params=params)
            entries = raw.get("data")
            if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
                raise APIResponseError("Anthropic returned a malformed model catalog.")
            for entry in entries:
                model = entry.get("id")
                if not isinstance(model, str) or not model:
                    raise APIResponseError("Anthropic returned a model without an ID.")
                if entry.get("lifecycle") == "retired":
                    continue
                properties[model] = {**entry, "context_length": entry.get("max_input_tokens")}
            if not raw.get("has_more", False):
                break
            next_cursor = raw.get("last_id")
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen:
                raise APIResponseError("Anthropic returned an invalid catalog page cursor.")
            cursor = next_cursor
            seen.add(cursor)
        self._model_properties = properties
        # The catalog supplies limits, not prices; Claude has no free inference tier.
        return [ModelInfo(id=model, name=entry.get("display_name"), provider=self.provider,
                          context_length=entry["context_length"],
                          pricing={"prompt": "unknown", "completion": "unknown"})
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
            raise APIConfigError(f"Anthropic does not list an available model {model!r}.")
        context, output = entry.get("max_input_tokens"), entry.get("max_tokens")
        if any(type(value) is not int or value <= 0 for value in (context, output)):
            raise APIConfigError(f"Anthropic does not advertise valid token limits for {model!r}.")
        # Compatible completions support native tools, but newer Claude models
        # reject forced tool use and sampling overrides. Keep provider defaults.
        profile = RequestProfile(parameters={"max_tokens", "tools", "tool_choice"},
                                 context_length=context, max_prompt_tokens=context,
                                 max_completion_tokens=output)
        profile.tool_choices = [{"auto": True, "none": True, "required": False, "function": False}]
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        for name in ("session_id", "temperature", "top_p", "response_format", "reasoning_effort"):
            payload.pop(name, None)
        # Anthropic requires an output budget; match the harness's usual reserve.
        payload.setdefault("max_tokens", min(8192, profile.max_completion_tokens or 8192) if profile else 8192)
