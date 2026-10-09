"""Shared async transport for OpenAI-compatible Chat Completions providers.

Speaks the REST API directly over httpx. Retries the failure modes that are
worth retrying (rate limits, transient 5xx, network errors) with exponential
backoff, and surfaces everything else as a typed error with the provider's own
message attached.
"""

from __future__ import annotations

import asyncio
import json
import random
import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .types import Completion, KeyInfo, Message, ModelInfo, ToolSpec, Usage
from .capabilities import RequestProfile
from .protocol import ResponseFormatError, normalize_calls, combine_calls

DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 1.0

# Statuses worth another attempt: transient upstream and rate-limit conditions.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class APIError(Exception):
    """Base class for every failure raised by this client."""


class APIConfigError(APIError):
    """The client is missing required configuration, such as an API key."""


class APIResponseError(APIError):
    """The API returned an error response."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.retry_after: str | None = None
        self.partial_response = ""
        self.usage = Usage()


class APITransportError(APIResponseError):
    """An interrupted request can be retried before accepting any actions."""


class APIAuthError(APIResponseError):
    """The API key was missing, malformed, or rejected."""


class APIContextError(APIResponseError):
    """The provider explicitly rejected the input for exceeding its context."""


class APILimitError(APIResponseError):
    """The provider reported a credit, spending, or rate limit."""


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = DEFAULT_MAX_RETRIES
    initial_backoff: float = 1.0
    max_backoff: float = 60.0
    # Explicit policies retain a finite retry count; application defaults extend
    # recovery after the silent retries, up to the maximum interval.
    extended: bool = False


class APIClient:
    """Shared OpenAI-compatible transport; providers supply metadata and options."""

    provider = ""
    default_base_url = ""
    default_model = ""
    key_env = ""
    model_env = ""
    base_url_env = ""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        http_referer: str | None = None,
        app_title: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        catalog_only: bool = False,
        on_retry: Callable[[str, int], None] | None = None,
    ) -> None:
        if not api_key.strip() and not catalog_only:
            raise APIConfigError(
                f"Missing {self.provider} API key. Set {self.key_env} or pass --api-key."
            )
        self.api_key = api_key.strip()
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.retry = retry or RetryPolicy(extended=True)
        self.on_retry = on_retry
        self._capabilities: dict[str, RequestProfile | None] = {}

        from . import __version__

        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"SlipAgent/{__version__}",
            **self.request_headers(http_referer, app_title),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> APIClient:
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
        on_request: Callable[[dict[str, Any]], None] | None = None,
        request_profile: RequestProfile | None = None,
        single_attempt: bool = False,
        disable_timeout: bool = False,
    ) -> Completion:
        """Request a completion; an isolated job may supply its frozen profile.

        Ordinary calls use the selected model's cached capabilities. Background
        compaction passes request_profile so later reselection cannot change
        routing, temperature support, or reasoning for an already archived step.
        """
        if not messages:
            raise APIError("chat() requires at least one message")
        if not self.api_key:
            raise APIConfigError(f"Set {self.key_env} before using {self.provider} inference.")

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
            payload["max_tokens"] = max_tokens
        if session_id:
            # Providers supporting sticky routing can use this session hint
            # to improve cache reuse; it does not guarantee a cache hit.
            payload["session_id"] = session_id
        if extra_body:
            payload.update(extra_body)

        capabilities = request_profile or getattr(self, "_capabilities", {}).get(model)
        if capabilities is not None:
            from .inference import sampling_defaults
            for name, value in sampling_defaults(model, capabilities).items():
                payload.setdefault(name, int(value) if name == "top_k" else value)
            if "temperature" in capabilities.parameters:
                payload.setdefault("temperature", DEFAULT_TEMPERATURE)
            else:
                payload.pop("temperature", None)
            try:
                capabilities.validate_output_limit(max_tokens)
            except ValueError as exc:
                raise APIConfigError(str(exc)) from exc
            if tool_choice is not None:
                choice = tool_choice if isinstance(tool_choice, str) else "function"
                if "tool_choice" not in capabilities.parameters or any(isinstance(choices, dict) and not choices.get(choice, False) for choices in capabilities.tool_choices):
                    raise APIConfigError(f"The selected endpoints do not support tool_choice={choice}.")

        if on_delta is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        self.prepare_payload(payload, capabilities)
        if on_request is not None:
            # Capture the final request body before transport, without headers.
            on_request(copy.deepcopy(payload))
        if on_delta is not None:
            return await self._stream_request(payload, on_delta, single_attempt=single_attempt, disable_timeout=disable_timeout)
        transport_options: dict[str, Any] = {"timeout": None} if disable_timeout else {}
        raw = await self._request("POST", "/chat/completions", single_attempt=single_attempt, json=payload, **transport_options)
        return _parse_completion(raw)

    async def _stream_request(self, payload: dict[str, Any], on_delta: Callable[[str, str], None], *, single_attempt: bool = False, disable_timeout: bool = False) -> Completion:
        transport_options: dict[str, Any] = {"timeout": None} if disable_timeout else {}
        observed = False
        attempt = 0
        while True:
            state: _StreamCompletion | None = None
            try:
                async with self._client.stream("POST", f"{self.base_url}/chat/completions", json=payload, **transport_options) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        error = _build_error(response)
                        error.retry_after = response.headers.get("retry-after")
                        if not single_attempt and response.status_code in RETRYABLE_STATUS and await self.wait_retry(attempt, error):
                            attempt += 1
                            continue
                        raise error
                    # Gateways and test servers may return a regular completion.
                    if "application/json" in response.headers.get("content-type", ""):
                        await response.aread()
                        raw = _decode_json(response)
                        embedded = raw.get("error")
                        code = embedded.get("code") if isinstance(embedded, dict) else None
                        if type(code) is int:
                            raise _build_error(httpx.Response(code, headers=response.headers, json=raw))
                        completion = _parse_completion(raw)
                        wire = raw["choices"][0]["message"]
                        try:
                            for text in _reasoning_text(wire):
                                on_delta("reasoning", text)
                            if completion.text:
                                on_delta("content", completion.text)
                        except APIResponseError as exc:
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
                error = APITransportError(f"API provider stream interrupted: {exc}")
                if state is not None:
                    state.attach_diagnostics(error)
                if observed or single_attempt or not await self.wait_retry(attempt, error):
                    raise error from exc
                attempt += 1
            except APIResponseError as exc:
                if state is not None:
                    state.attach_diagnostics(exc)
                exc.retry_after = exc.retry_after or response.headers.get("retry-after")
                # HTTP 200 can contain an overload instead of a completion.
                # Partial output returns to the agent for visible recovery.
                no_partial = not observed if state is None else not (
                    state.content or state.reasoning or state.reasoning_details or state.calls or state.function_call or state.tool_blocks or state.usage)
                if (not single_attempt and exc.status_code in RETRYABLE_STATUS and no_partial
                        and await self.wait_retry(attempt, exc)):
                    attempt += 1
                    observed = False
                    continue
                raise

    async def list_models(self) -> list[ModelInfo]:
        """Return the model catalog.

        Catalog access does not establish inference access. Account metadata
        is available through key_info() only on providers that support it.
        """
        raw = await self._request("GET", "/models")
        data = raw.get("data")
        try:
            if not isinstance(data, list) or any(not isinstance(entry, dict) for entry in data):
                raise ValueError("data must be a list of model objects")
            models = [ModelInfo.from_api(entry) for entry in data]
            for model in models:
                model.provider = self.provider
            self._model_properties = {entry["id"]: entry for entry in data}
            return models
        except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise APIResponseError(f"API provider returned a malformed catalog: {exc}") from exc

    async def model_capabilities(self, model: str, *, refresh: bool = False, store: bool = True) -> RequestProfile | None:
        raise NotImplementedError

    def cache_capabilities(self, model: str, capabilities: RequestProfile) -> None:
        """Publish a profile only after the caller's selection preflight succeeds."""
        if not hasattr(self, "_capabilities"):
            self._capabilities = {}
        self._capabilities[model] = capabilities

    def catalog_context_length(self, model: str) -> int | None:
        """Read the already fetched catalog without another metadata request."""
        properties = getattr(self, "_model_properties", {}).get(model)
        return ModelInfo.from_api(properties).context_length if properties is not None else None

    async def key_info(self) -> KeyInfo | None:
        """Return authoritative account metadata, or None when unsupported."""
        return None

    def prepare_payload(self, payload: dict[str, Any], profile: RequestProfile | None) -> None:
        """Apply provider-specific fields before diagnostics and transport."""

    def request_headers(self, http_referer: str | None, app_title: str | None) -> dict[str, str]:
        return {}

    def replacement_options(self) -> dict[str, Any]:
        """Settings retained when replacing a credential on this provider."""
        return {"retry": self.retry, "on_retry": self.on_retry}

    async def _request(
        self, method: str, path: str, *, single_attempt: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError, httpx.DecodingError) as exc:
                error: APIResponseError = APITransportError(f"Network error talking to API provider after {attempt + 1} attempt(s): {exc}")
                if single_attempt or not await self.wait_retry(attempt, error):
                    raise error from exc
                attempt += 1
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
            if not single_attempt and response.status_code in RETRYABLE_STATUS and await self.wait_retry(attempt, error):
                attempt += 1
                continue
            raise error

    def retry_delay(self, attempt: int, retry_after: str | None = None) -> float | None:
        """Stop instead of scheduling an interval beyond the one-minute ceiling."""
        maximum = min(60.0, self.retry.max_backoff)
        header_delay = _parse_retry_after(retry_after)
        if header_delay is not None and header_delay > maximum:
            return None
        if attempt >= self.retry.max_retries:
            if not self.retry.extended or self.retry.initial_backoff <= 0:
                return None
            if self.retry.initial_backoff * 2 ** attempt > maximum:
                return None
        return min(maximum, self._backoff(attempt, retry_after))

    async def wait_retry(self, attempt: int, error: APIResponseError, *,
                         stop: asyncio.Event | None = None,
                         on_retry: Callable[[str, int], None] | None = None) -> bool:
        """Wait silently first, then report a cancellable countdown on each retry."""
        delay = self.retry_delay(attempt, error.retry_after)
        if delay is None or stop is not None and stop.is_set():
            return False
        callback = on_retry or self.on_retry
        visible = attempt >= self.retry.max_retries and callback is not None
        try:
            remaining = delay
            while remaining > 0:
                if visible and callback is not None:
                    callback(f"{self.provider}: {error}", math.ceil(remaining))
                interval = min(1.0, remaining) if visible else remaining
                if stop is None:
                    await asyncio.sleep(interval)
                else:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=interval)
                        return False
                    except TimeoutError:
                        pass
                remaining = max(0.0, remaining - interval)
            return True
        finally:
            if visible and callback is not None:
                callback("", 0)

    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        header_delay = _parse_retry_after(retry_after)
        if header_delay is not None:
            return min(header_delay, self.retry.max_backoff)
        # Randomize within the upper half of the interval to stagger retries.
        ceiling = min(self.retry.initial_backoff * (2**attempt), self.retry.max_backoff)
        return random.uniform(ceiling / 2, ceiling)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
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
        self.tool_blocks: list[dict[str, Any]] = []

    def attach_diagnostics(self, error: APIResponseError) -> None:
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
                    raise APIContextError(f"API provider stream context limit exceeded: {raw['error']}")
                code = raw["error"].get("code") if isinstance(raw["error"], dict) else None
                raise APIResponseError(f"API provider stream failed: {raw['error']}",
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
                        raise APITransportError("API provider stream ended with a provider error")
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
                    self.tool_blocks.extend(blocks)
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
            raise APIResponseError(f"Malformed API provider stream: {exc}") from exc

    def completion(self) -> Completion:
        if self.finish_reason is None:
            raise APITransportError("API provider stream ended before a complete response")
        return _parse_completion({
            "model": self.model, "usage": self.usage,
            "choices": [{"finish_reason": self.finish_reason, "message": {
                "role": "assistant", "content": ([{"type": "text", "text": self.content}, *self.tool_blocks]
                                                  if self.tool_blocks else self.content or None),
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
        raise APIResponseError(
            "API provider returned a non-JSON response.",
            status_code=response.status_code,
            body=response.text[:2000],
        ) from exc
    if not isinstance(payload, dict):
        raise APIResponseError(
            "API provider returned an unexpected JSON shape.",
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


def _build_error(response: httpx.Response) -> APIResponseError:
    status = response.status_code
    detail = _extract_error_message(response)
    if status in {400, 413} and _is_context_overflow(detail):
        return APIContextError(
            f"API provider context limit exceeded ({status}): {detail}",
            status_code=status, body=response.text[:2000],
        )

    if status == 401:
        return APIAuthError(
            f"API provider rejected the API key (401): {detail}",
            status_code=status,
            body=response.text[:2000],
        )
    if status == 402 or status == 403 and "key limit exceeded" in detail.casefold():
        return APILimitError(
            f"API provider key limit reached ({status}): {detail}. "
            f"Check the account limits in your API provider's dashboard.",
            status_code=status,
            body=response.text[:2000],
        )
    if status == 429:
        return APILimitError(
            f"API provider rate limit hit ({status}): {detail}",
            status_code=status,
            body=response.text[:2000],
        )
    return APIResponseError(
        f"API provider request failed ({status}): {detail}",
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
        raise APIContextError(f"API provider context limit exceeded: {raw['error']}")
    choices = raw.get("choices") or []
    if not choices:
        raise APIResponseError(
            f"API provider response contained no choices: {str(raw)[:500]}"
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
            calls = combine_calls(normalize_calls(raw_calls), normalize_calls(wire.get("function_call")), normalize_calls(blocks))
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
        raise APIResponseError(f"API provider returned a malformed completion: {exc}") from exc
