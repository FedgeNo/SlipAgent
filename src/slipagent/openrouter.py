"""Async client for the OpenRouter Chat Completions API.

Speaks the REST API directly over httpx. Retries the failure modes that are
worth retrying (rate limits, transient 5xx, network errors) with exponential
backoff, and surfaces everything else as a typed error with the provider's own
message attached.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from .types import Completion, KeyInfo, Message, ModelInfo, ToolSpec, Usage
from .capabilities import ModelCapabilities
from .protocol import ResponseFormatError, normalize_calls

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 1.0

# Statuses worth another attempt: transient upstream and rate-limit conditions.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenRouterError(Exception):
    """Base class for every failure raised by this client."""


class OpenRouterConfigError(OpenRouterError):
    """The client is missing required configuration, such as an API key."""


class OpenRouterAPIError(OpenRouterError):
    """The API returned an error response."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.retry_after: str | None = None
        self.partial_response = ""
        self.usage = Usage()


class OpenRouterTransportError(OpenRouterAPIError):
    """An interrupted request can be retried before accepting any actions."""


class OpenRouterAuthError(OpenRouterAPIError):
    """The API key was missing, malformed, or rejected."""


class OpenRouterContextError(OpenRouterAPIError):
    """The provider explicitly rejected the input for exceeding its context."""


class OpenRouterLimitError(OpenRouterAPIError):
    """The key's spend/credit limit was reached.

    OpenRouter reports this as 403 ("Key limit exceeded (total limit)") rather
    than 402, so it is deliberately distinct from a bad key.
    """


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = DEFAULT_MAX_RETRIES
    initial_backoff: float = 1.0
    max_backoff: float = 30.0


class OpenRouterClient:
    """Thin async wrapper over the OpenRouter REST endpoints."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        http_referer: str | None = None,
        app_title: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise OpenRouterConfigError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key."
            )
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryPolicy()

        from . import __version__

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": http_referer or "https://github.com/FedgeNo/SlipAgent",
            "X-OpenRouter-Title": app_title or "SlipAgent",
            "X-OpenRouter-Categories": "cli-agent",
            "User-Agent": os.environ.get("OPENROUTER_USER_AGENT") or f"SlipAgent/{__version__}",
        }

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        on_request: Callable[[str], None] | None = None,
        request_profile: ModelCapabilities | None = None,
        single_attempt: bool = False,
        disable_timeout: bool = False,
    ) -> Completion:
        """Request a completion; an isolated job may supply its frozen profile.

        Ordinary calls use the selected model's cached capabilities. Background
        compaction passes request_profile so later reselection cannot change
        routing, temperature support, or reasoning for an already archived step.
        """
        if not messages:
            raise OpenRouterError("chat() requires at least one message")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.to_api() for message in messages],
        }
        # Also cover direct callers and older archived messages. These are new
        # payload dictionaries; the recorded originals remain untouched.
        for wire in payload["messages"]:
            wire.pop("reasoning", None)
            wire.pop("reasoning_details", None)
        if tools:
            payload["tools"] = [spec.to_api() for spec in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
        if session_id:
            # Sticky routing: keeps the whole agent run on one provider so
            # prompt caching actually hits.
            payload["session_id"] = session_id
        if extra_body:
            payload.update(extra_body)

        capabilities = request_profile or getattr(self, "_capabilities", {}).get(model)
        if capabilities is not None:
            if "temperature" in capabilities.parameters:
                payload.setdefault("temperature", DEFAULT_TEMPERATURE)
            else:
                payload.pop("temperature", None)
            try:
                capabilities.validate_output_limit(max_tokens)
            except ValueError as exc:
                raise OpenRouterConfigError(str(exc)) from exc
            if tool_choice is not None:
                choice = tool_choice if isinstance(tool_choice, str) else "function"
                if "tool_choice" not in capabilities.parameters or any(isinstance(choices, dict) and not choices.get(choice, False) for choices in capabilities.tool_choices):
                    raise OpenRouterConfigError(f"The selected endpoints do not support tool_choice={choice}.")
            if capabilities.reasoning is not None:
                payload["reasoning"] = capabilities.reasoning
            elif capabilities.include_reasoning:
                payload["include_reasoning"] = True

        if on_delta is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if on_request is not None:
            # Capture the final request body before transport, without headers.
            on_request(json.dumps(payload, ensure_ascii=False))
        if on_delta is not None:
            return await self._stream_request(payload, on_delta, single_attempt=single_attempt, disable_timeout=disable_timeout)
        transport_options: dict[str, Any] = {"timeout": None} if disable_timeout else {}
        raw = await self._request("POST", "/chat/completions", single_attempt=single_attempt, json=payload, **transport_options)
        return _parse_completion(raw)

    async def _stream_request(self, payload: dict[str, Any], on_delta: Callable[[str, str], None], *, single_attempt: bool = False, disable_timeout: bool = False) -> Completion:
        transport_options: dict[str, Any] = {"timeout": None} if disable_timeout else {}
        observed = False
        retries = 0 if single_attempt else self.retry.max_retries
        for attempt in range(retries + 1):
            state: _StreamCompletion | None = None
            try:
                async with self._client.stream("POST", f"{self.base_url}/chat/completions", json=payload, **transport_options) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        error = _build_error(response)
                        error.retry_after = response.headers.get("retry-after")
                        if response.status_code in RETRYABLE_STATUS and attempt < retries:
                            await asyncio.sleep(self._backoff(attempt, response.headers.get("retry-after")))
                            continue
                        raise error
                    # Gateways and test servers may return a regular completion.
                    if "application/json" in response.headers.get("content-type", ""):
                        await response.aread()
                        raw = _decode_json(response)
                        completion = _parse_completion(raw)
                        wire = raw["choices"][0]["message"]
                        try:
                            for text in _reasoning_text(wire):
                                on_delta("reasoning", text)
                            if completion.text:
                                on_delta("content", completion.text)
                        except OpenRouterAPIError as exc:
                            exc.usage = completion.usage
                            exc.partial_response = json.dumps(wire, ensure_ascii=False)
                            raise
                        return completion
                    state = _StreamCompletion(on_delta)
                    data: list[str] = []
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data.append(line[5:].lstrip(" "))
                        elif not line and data:
                            event = "\n".join(data)
                            data.clear()
                            if event == "[DONE]":
                                return state.completion()
                            observed = True
                            state.add(event)
                    if data:
                        event = "\n".join(data)
                        if event != "[DONE]":
                            observed = True
                            state.add(event)
                    return state.completion()
            except (httpx.TimeoutException, httpx.TransportError, httpx.DecodingError) as exc:
                # Visible partial output needs an agent-level retry notification.
                if observed or attempt == retries:
                    error = OpenRouterTransportError(f"OpenRouter stream interrupted: {exc}")
                    if state is not None:
                        state.attach_diagnostics(error)
                    raise error from exc
                await asyncio.sleep(self._backoff(attempt))
            except OpenRouterAPIError as exc:
                if state is not None:
                    state.attach_diagnostics(exc)
                raise
        raise OpenRouterError("OpenRouter stream failed after retries")

    async def list_models(self) -> list[ModelInfo]:
        """Return the model catalog.

        Note: `GET /models` is public and does not authenticate, so a successful
        call says nothing about whether the key is valid. Use `key_info()`.
        """
        raw = await self._request("GET", "/models")
        data = raw.get("data")
        try:
            if not isinstance(data, list) or any(not isinstance(entry, dict) for entry in data):
                raise ValueError("data must be a list of model objects")
            models = [ModelInfo.from_api(entry) for entry in data]
            self._model_properties = {entry["id"]: entry for entry in data}
            return models
        except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise OpenRouterAPIError(f"OpenRouter returned a malformed catalog: {exc}") from exc

    async def model_capabilities(self, model: str, *, refresh: bool = False, store: bool = True) -> ModelCapabilities | None:
        """Fetch endpoint properties; cache them between explicit selections."""
        cache: dict[str, ModelCapabilities | None] = getattr(self, "_capabilities", {})
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

    def cache_capabilities(self, model: str, capabilities: ModelCapabilities) -> None:
        """Publish a profile only after the caller's selection preflight succeeds."""
        if not hasattr(self, "_capabilities"):
            self._capabilities = {}
        self._capabilities[model] = capabilities

    def catalog_context_length(self, model: str) -> int | None:
        """Read the already fetched catalog without another metadata request."""
        properties = getattr(self, "_model_properties", {}).get(model)
        return ModelInfo.from_api(properties).context_length if properties is not None else None

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

    async def _request(
        self, method: str, path: str, *, single_attempt: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        retries = 0 if single_attempt else self.retry.max_retries
        for attempt in range(retries + 1):
            is_final = attempt == retries
            try:
                response = await self._client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError, httpx.DecodingError) as exc:
                last_error = exc
                if is_final:
                    raise OpenRouterTransportError(
                        f"Network error talking to OpenRouter after "
                        f"{attempt + 1} attempt(s): {exc}"
                    ) from exc
                await asyncio.sleep(self._backoff(attempt))
                continue

            if response.status_code < 400:
                raw = _decode_json(response)
                error_body = raw.get("error")
                code = error_body.get("code") if isinstance(error_body, dict) else None
                if type(code) is not int or code not in RETRYABLE_STATUS:
                    return raw
                # Providers can report transient failures inside an HTTP 200 body.
                response = httpx.Response(code, headers=response.headers, json=raw)

            error = _build_error(response)
            error.retry_after = response.headers.get("retry-after")
            if response.status_code in RETRYABLE_STATUS and not is_final:
                last_error = error
                await asyncio.sleep(self._backoff(attempt, response.headers.get("retry-after")))
                continue
            raise error

        raise OpenRouterError(f"Request failed after retries: {last_error}")

    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        header_delay = _parse_retry_after(retry_after)
        if header_delay is not None:
            return min(header_delay, self.retry.max_backoff)
        # Full jitter keeps a fleet of agents from retrying in lockstep.
        ceiling = min(self.retry.initial_backoff * (2**attempt), self.retry.max_backoff)
        return random.uniform(ceiling / 2, ceiling)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        # HTTP-date form; not worth parsing precisely, fall back to backoff.
        return None


def _reasoning_text(delta: dict[str, Any]) -> list[str]:
    for key in ("reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return [value]
    texts = []
    for detail in delta.get("reasoning_details") or []:
        if isinstance(detail, dict) and detail.get("type") in {"reasoning.text", "reasoning.summary"}:
            value = detail.get("text") or detail.get("summary")
            if isinstance(value, str) and value:
                texts.append(value)
    return texts


class _StreamCompletion:
    """Assemble text, indexed tool fragments, and final usage from SSE events."""

    def __init__(self, on_delta: Callable[[str, str], None]) -> None:
        self.on_delta = on_delta
        self.content = ""
        self.calls: dict[int, dict[str, Any]] = {}
        self.model = ""
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.reasoning_details: list[dict[str, Any]] = []
        self.reasoning: list[str] = []
        self.function_call: dict[str, Any] = {}

    def attach_diagnostics(self, error: OpenRouterAPIError) -> None:
        error.partial_response = json.dumps({"content": self.content, "tool_calls": self.calls,
                                             "reasoning": "".join(self.reasoning)}, ensure_ascii=False)[:16000]
        try:
            error.usage = Usage.from_api(self.usage)
        except (ValueError, TypeError, AttributeError, OverflowError):
            pass

    def _add_reasoning_detail(self, detail: dict[str, Any]) -> None:
        """Reconstruct native blocks only for the response being received.

        Preserve provider metadata in the local original. Adjacent text
        fragments can join when their metadata agrees; encrypted
        blocks and distinct identities/signatures must remain separate.
        """
        kind = detail.get("type")
        field = {"reasoning.text": "text", "reasoning.summary": "summary"}.get(kind) if isinstance(kind, str) else None
        previous = self.reasoning_details[-1] if self.reasoning_details else None
        if field and previous is not None and previous.get("type") == detail.get("type"):
            compatible = all(value is None or previous.get(key) is None or previous[key] == value
                             for key, value in detail.items() if key != field)
            previous_text, text = previous.get(field), detail.get(field)
            if compatible and (previous_text is None or isinstance(previous_text, str)) and (text is None or isinstance(text, str)):
                previous[field] = (previous_text or "") + (text or "")
                for key, value in detail.items():
                    if key != field and previous.get(key) is None:
                        previous[key] = copy.deepcopy(value)
                return
        self.reasoning_details.append(copy.deepcopy(detail))

    def add(self, data: str) -> None:
        try:
            raw = json.loads(data)
            if not isinstance(raw, dict):
                raise ValueError("stream event must be an object")
            if raw.get("error"):
                if _is_context_overflow(str(raw["error"])):
                    raise OpenRouterContextError(f"OpenRouter stream context limit exceeded: {raw['error']}")
                code = raw["error"].get("code") if isinstance(raw["error"], dict) else None
                raise OpenRouterAPIError(f"OpenRouter stream failed: {raw['error']}",
                                         status_code=code if type(code) is int else None)
            self.model = raw.get("model") or self.model
            if raw.get("usage") is not None:
                self.usage = raw["usage"]
            for choice in raw.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                if choice.get("finish_reason"):
                    self.finish_reason = choice["finish_reason"]
                    if self.finish_reason == "error":
                        raise OpenRouterTransportError("OpenRouter stream ended with a provider error")
                delta = choice.get("delta") or {}
                details = delta.get("reasoning_details")
                if details is not None:
                    if not isinstance(details, list) or any(not isinstance(detail, dict) for detail in details):
                        raise ValueError("reasoning_details must be an array of objects")
                    for detail in details:
                        self._add_reasoning_detail(detail)
                for text in _reasoning_text(delta):
                    self.reasoning.append(text)
                    self.on_delta("reasoning", text)
                content = delta.get("content")
                if content is not None:
                    content, blocks = _content_parts(content)
                    if blocks:
                        raise ValueError("stream tool calls must use indexed delta.tool_calls")
                    self.content += content
                    if content:
                        self.on_delta("content", content)
                for fragment in delta.get("tool_calls") or []:
                    index = fragment.get("index")
                    if type(index) is not int or index < 0:
                        raise ValueError("tool fragments need a nonnegative index")
                    call = self.calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if set(fragment) - {"index", "id", "type", "function"}:
                        call["invalid_fields"] = True
                    if "type" in fragment:
                        call["type"] = fragment["type"]
                    call["id"] += fragment.get("id") or ""
                    function = fragment.get("function") or {}
                    if set(function) - {"name", "arguments"}:
                        call["invalid_fields"] = True
                    call["function"]["name"] += function.get("name") or ""
                    call["function"]["arguments"] += function.get("arguments") or ""
                legacy = delta.get("function_call")
                if legacy is not None:
                    if not isinstance(legacy, dict):
                        raise ValueError("function_call must be an object")
                    for field in ("name", "arguments"):
                        self.function_call[field] = self.function_call.get(field, "") + (legacy.get(field) or "")
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise OpenRouterAPIError(f"Malformed OpenRouter stream: {exc}") from exc

    def completion(self) -> Completion:
        if self.finish_reason is None:
            raise OpenRouterTransportError("OpenRouter stream ended before a complete response")
        return _parse_completion({
            "model": self.model, "usage": self.usage,
            "choices": [{"finish_reason": self.finish_reason, "message": {
                "role": "assistant", "content": self.content or None,
                "tool_calls": [self.calls[index] for index in sorted(self.calls)],
                "function_call": self.function_call or None,
                "reasoning_details": self.reasoning_details or None,
                "reasoning": "".join(self.reasoning),
            }}],
        })


def _decode_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise OpenRouterAPIError(
            "OpenRouter returned a non-JSON response.",
            status_code=response.status_code,
            body=response.text[:2000],
        ) from exc
    if not isinstance(payload, dict):
        raise OpenRouterAPIError(
            "OpenRouter returned an unexpected JSON shape.",
            status_code=response.status_code,
            body=response.text[:2000],
        )
    return payload


def _is_context_overflow(detail: str) -> bool:
    """Recognize explicit input-size errors in either HTTP or streamed payloads."""
    lower = detail.casefold()
    overflow = any(code in lower for code in ("context_length_exceeded", "context_window_exceeded", "prompt_too_long"))
    overflow = overflow or (any(term in lower for term in ("context length", "context window", "input tokens", "prompt tokens"))
                            and any(term in lower for term in ("exceed", "too long", "too many", "requested", "resulted in")))
    return overflow


def _build_error(response: httpx.Response) -> OpenRouterAPIError:
    status = response.status_code
    detail = _extract_error_message(response)
    if status in {400, 413} and _is_context_overflow(detail):
        return OpenRouterContextError(
            f"OpenRouter context limit exceeded ({status}): {detail}",
            status_code=status, body=response.text[:2000],
        )

    if status == 401:
        return OpenRouterAuthError(
            f"OpenRouter rejected the API key (401): {detail}",
            status_code=status,
            body=response.text[:2000],
        )
    if status == 402 or status == 403 and "key limit exceeded" in detail.casefold():
        return OpenRouterLimitError(
            f"OpenRouter key limit reached ({status}): {detail}. "
            f"Raise or reset the key's limit in the OpenRouter dashboard.",
            status_code=status,
            body=response.text[:2000],
        )
    if status == 429:
        return OpenRouterLimitError(
            f"OpenRouter rate limit hit ({status}): {detail}",
            status_code=status,
            body=response.text[:2000],
        )
    return OpenRouterAPIError(
        f"OpenRouter request failed ({status}): {detail}",
        status_code=status,
        body=response.text[:2000],
    )


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        text = response.text.strip()
        return text[:500] if text else "no response body"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if message:
                return f"{message} (code={code})" if code else str(message)
        if isinstance(error, str):
            return error
        if payload.get("message"):
            return str(payload["message"])
    return "no error detail"


def _content_parts(content: Any) -> tuple[str, list[dict[str, Any]]]:
    """Read text/tool blocks without mistaking reasoning or unknown data for text."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise ResponseFormatError("Assistant content must be text or an array of content blocks.")
    texts, calls = [], []
    for block in content:
        if not isinstance(block, dict):
            raise ResponseFormatError("Assistant content blocks must be objects.")
        if block.get("type") in {"text", "output_text"} and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif block.get("type") == "tool_use":
            calls.append(block)
        else:
            raise ResponseFormatError("Unsupported assistant content block; expected text or tool_use.")
    return "".join(texts), calls


def _parse_completion(raw: dict[str, Any]) -> Completion:
    if raw.get("error") and _is_context_overflow(str(raw["error"])):
        raise OpenRouterContextError(f"OpenRouter context limit exceeded: {raw['error']}")
    choices = raw.get("choices") or []
    if not choices:
        raise OpenRouterAPIError(
            f"OpenRouter response contained no choices: {str(raw)[:500]}"
        )

    try:
        if not isinstance(choices, list) or not isinstance(choices[0], dict):
            raise ValueError("choices must contain objects")
        first = choices[0]
        wire = first.get("message")
        if not isinstance(wire, dict) or wire.get("role", "assistant") != "assistant":
            raise ValueError("message must be an assistant object")
        content = ""
        calls = []
        response_error = None
        response_excerpt = ""
        try:
            content, blocks = _content_parts(wire.get("content"))
            if wire.get("function_call") is not None or blocks:
                raise ResponseFormatError("Unexpected tool-call carrier in the API response.")
            raw_calls = wire.get("tool_calls", [])
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                raise ResponseFormatError("API tool_calls must be an array.")
            for call in raw_calls:
                if (not isinstance(call, dict) or set(call) != {"id", "type", "function"}
                        or call["type"] != "function" or not isinstance(call["id"], str)
                        or not call["id"].strip() or not isinstance(call["function"], dict)
                        or set(call["function"]) != {"name", "arguments"}
                        or not isinstance(call["function"]["arguments"], str)):
                    raise ResponseFormatError("API calls require id, type=function, and function with name and JSON-encoded arguments.")
                if not call["function"]["arguments"].strip():
                    raise ResponseFormatError("API arguments must be a JSON-encoded object string; use '{}' for no arguments.")
            calls = normalize_calls(raw_calls)
        except ResponseFormatError as exc:
            response_error = str(exc)
            response_excerpt = json.dumps({"invalid_message_prefix": json.dumps(wire, ensure_ascii=True)[:1500]})
        message = Message.assistant(content or None, calls)
        details = wire.get("reasoning_details")
        if details is not None:
            if not isinstance(details, list) or any(not isinstance(detail, dict) for detail in details):
                raise ValueError("reasoning_details must be an array of objects")
            # Ordinary reasoning is one string. Keep opaque provider metadata
            # in the local original; outgoing requests exclude both forms.
            if any(detail.get("signature") or detail.get("type") == "reasoning.encrypted" for detail in details):
                message.reasoning_details = copy.deepcopy(details)
        message.reasoning = "".join(_reasoning_text(wire)) or None
        return Completion(
            message=message,
            model=raw.get("model", ""),
            finish_reason=first.get("finish_reason"),
            usage=Usage.from_api(raw.get("usage")),
            response_error=response_error, response_excerpt=response_excerpt,
        )
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise OpenRouterAPIError(f"OpenRouter returned a malformed completion: {exc}") from exc
