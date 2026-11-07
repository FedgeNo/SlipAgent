"""Claude's native Messages API and authenticated model capabilities."""

from __future__ import annotations

from typing import Any, Callable

from .api import APIClient, APIConfigError, APIResponseError, _StreamCompletion
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
    completion_path = '/messages'

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
        profile = RequestProfile(parameters={"max_tokens", "tools", "tool_choice"},
                                 context_length=context, max_prompt_tokens=context,
                                 max_completion_tokens=output)
        profile.tool_choices = [{"auto": True, "none": True, "required": False, "function": False}]
        if store:
            self.cache_capabilities(model, profile)
        return profile

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        for name in ("session_id", "temperature", "top_p", "response_format", "reasoning_effort", 'thinking'):
            payload.pop(name, None)
        if 'output_config' in payload:
            payload['output_config'] = dict(payload['output_config'])
            payload['output_config'].pop('effort', None)
            if not payload['output_config']:
                payload.pop('output_config')
        # Anthropic requires an output budget; use the model's advertised limit.
        payload.setdefault("max_tokens", (profile.max_completion_tokens or 8192) if profile else 8192)
        from .anthropicwire import message_payload
        message_payload(payload)

    def completion_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        from .anthropicwire import completion_data
        try:
            return completion_data(raw)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise APIResponseError(f'Anthropic returned a malformed message: {exc}') from exc

    def stream_completion(self, on_delta: Callable[[str, str], None], identity_mode: str) -> _StreamCompletion:
        from .anthropicwire import AnthropicStream
        return AnthropicStream(on_delta)

    def response_events(self, events: list[str]) -> dict[str, Any]:
        # Native block-start/delta events cannot be merged as OpenAI choice deltas.
        import json
        records = []
        for event in events:
            try:
                records.append(json.loads(event))
            except ValueError:
                records.append({'unparsed_event': event})
        return {'events': records}
