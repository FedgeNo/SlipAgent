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
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TYPE_CHECKING

import httpx

from .types import Completion, KeyInfo, Message, ModelInfo, ToolSpec, Usage
from .capabilities import RequestProfile
from .protocol import ResponseFormatError, normalize_calls, combine_calls
from .prompts import model_prompt_profile
from .thoughts import ThoughtStream

if TYPE_CHECKING:
    from .requestbody import RequestBody

DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE: float | None = None

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


class APIConnectionError(APITransportError):
    """A broken connection does not establish that the selected model failed."""


class APIOutputSkipped(APIResponseError):
    """The user moved on to another model before this one produced output."""


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
    max_temperature = 2.0
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
        temperature: float | None = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        session_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        on_request: Callable[[dict[str, Any]], None] | None = None,
        on_request_body: Callable[[Any], Awaitable[None] | None] | None = None,
        on_response: Callable[[dict[str, Any]], None] | None = None,
        on_raw_response: Callable[[httpx.Response], None] | None = None,
        request_profile: RequestProfile | None = None,
        single_attempt: bool = False,
        disable_timeout: bool = False,
        first_output_timeout: float | None = None,
        on_first_output: Callable[[FirstOutputDeadline | None], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> Completion:
        """Request a completion; an isolated job may supply its frozen profile.

        Ordinary calls use the selected model's cached capabilities. Background
        compaction passes request_profile so later reselection cannot change
        routing, temperature support, or reasoning for an already archived step.
        first_output_timeout bounds streaming startup; background calls omit it.
        on_first_output receives each attempt's deadline while it waits for
        output, then None. on_phase receives a short description of what the
        streaming request is doing, and an empty string when it ends.
        """
        if not messages:
            raise APIError("chat() requires at least one message")
        if not self.api_key:
            raise APIConfigError(f"Set {self.key_env} before using {self.provider} inference.")

        from .requestbody import wire_message
        payload: dict[str, Any] = {
            "model": model,
            "messages": [wire_message(message) for message in messages],
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
            if "temperature" not in capabilities.parameters:
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
        body = self.request_body(payload)
        if on_phase is not None and hasattr(body, "on_wait"):
            body.on_wait = on_phase
        if on_request is not None:
            # Capture the final request body before transport, without headers.
            on_request(body.read())
        if on_request_body is not None:
            # An async observer may persist the body; transport waits for it.
            observed = on_request_body(body)
            if observed is not None:
                await observed
        if on_delta is not None:
            try:
                return await self._stream_request(payload, on_delta, single_attempt=single_attempt, disable_timeout=disable_timeout, on_response=on_response, request_body=body,
                                                  first_output_timeout=first_output_timeout, on_raw_response=on_raw_response,
                                                  on_first_output=on_first_output, on_phase=on_phase)
            finally:
                if on_phase is not None:
                    on_phase("")
        transport_options: dict[str, Any] = {"timeout": None} if disable_timeout else {}
        raw = await self._request("POST", "/chat/completions", single_attempt=single_attempt, content=body, headers={"Content-Length": str(body.size)}, on_response=on_response, on_raw_response=on_raw_response, **transport_options)
        return _parse_completion(raw)

    async def _stream_request(self, payload: dict[str, Any], on_delta: Callable[[str, str], None], *, single_attempt: bool = False, disable_timeout: bool = False,
                              on_response: Callable[[dict[str, Any]], None] | None = None, request_body: Any = None,
                              first_output_timeout: float | None = None,
                              on_raw_response: Callable[[httpx.Response], None] | None = None,
                              on_first_output: Callable[[FirstOutputDeadline | None], None] | None = None,
                              on_phase: Callable[[str], None] | None = None) -> Completion:
        body = request_body if request_body is not None else self.request_body(payload)
        transport_options: dict[str, Any] = {"timeout": None} if disable_timeout else {}
        observed = False
        attempt = 0
        carried: float | None = None
        phase = on_phase or (lambda text: None)
        announced: set[str] = set()
        deliver = on_delta

        def on_delta(kind: str, chunk: str) -> None:
            # The first reasoning or reply text of an attempt names the phase once,
            # which also shows where a stuck request stopped.
            if kind not in announced:
                announced.add(kind)
                if kind == "reasoning":
                    phase("Thinking")
                elif kind == "content":
                    phase("Receiving reply")
            deliver(kind, chunk)

        while True:
            announced.clear()
            state: _StreamCompletion | None = None
            # An HTTP error whose retry was already declined is not reconsidered.
            declined: APIResponseError | None = None
            captured: dict[str, Any] | None = None
            # Raw events are merged once when the attempt ends.
            events: list[str] = []
            data: list[str] = []
            def capture_event(event: str) -> None:
                if on_response is not None and event != "[DONE]":
                    events.append(event)
            phase(f"Sending request to {self.provider}")
            # A retry after an error keeps the running deadline, so repeated
            # 429s cannot extend the wait; only a 200 response restarts it.
            first_output = asyncio.timeout(first_output_timeout) if carried is None else asyncio.timeout_at(carried)
            deadline = None if first_output_timeout is None else FirstOutputDeadline(first_output)
            content = body if deadline is None else _PacingExcluded(body, deadline)
            if deadline is not None and on_first_output is not None:
                on_first_output(deadline)
            try:
                async with first_output, self._client.stream("POST", f"{self.base_url}/chat/completions", content=content, headers={"Content-Length": str(body.size)}, **transport_options) as response:
                    if on_raw_response is not None:
                        on_raw_response(response)
                    if response.status_code >= 400:
                        await response.aread()
                        captured = ResponseReader()._response_body(response)
                        error = _build_error(response)
                        error.retry_after = response.headers.get("retry-after")
                        if not single_attempt and response.status_code in RETRYABLE_STATUS and await self.wait_retry(attempt, error):
                            attempt += 1
                            continue
                        declined = error
                        raise error
                    phase("Waiting for first output")
                    if deadline is not None and first_output.when() is not None and not (deadline.held or deadline.skipped):
                        first_output.reschedule(asyncio.get_running_loop().time() + first_output_timeout)
                    # Gateways and test servers may return a regular completion.
                    if "application/json" in response.headers.get("content-type", ""):
                        await response.aread()
                        captured = ResponseReader()._response_body(response)
                        raw = ResponseReader()._decode_json(response)
                        embedded = raw.get("error")
                        code = embedded.get("code") if isinstance(embedded, dict) else None
                        if type(code) is int:
                            raise _build_error(httpx.Response(code, headers=response.headers, json=raw))
                        completion = _parse_completion(raw)
                        wire = raw["choices"][0]["message"]
                        try:
                            if completion.message.reasoning:
                                on_delta("reasoning", completion.message.reasoning)
                        except APIResponseError as exc:
                            exc.usage = completion.usage
                            exc.partial_response = json.dumps(wire, ensure_ascii=False)
                            raise
                        return completion
                    profile = model_prompt_profile(payload["model"])
                    identity_mode = profile.get("stream_tool_identity", "delta")
                    if "stream_tool_identity" not in profile and self.provider == "nvidia" and "minimax" in payload["model"].lower():
                        identity_mode = "snapshot"
                    state = _StreamCompletion(on_delta, identity_mode=identity_mode,
                                              index_mode=profile.get("stream_tool_indices", "index"))
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data.append(line[5:].lstrip(" "))
                        elif not line and data:
                            event = "\n".join(data)
                            data.clear()
                            capture_event(event)
                            if event == "[DONE]":
                                return state.completion()
                            observed = True
                            state.add(event)
                            if first_output.when() is not None and (
                                    any(state.content_parts) or state.reasoning
                                    or any(detail.get("text") or detail.get("summary") or detail.get("data")
                                           for detail in state.reasoning_details)
                                    or any(call["function"]["name"] or state.has_arguments(index)
                                           for index, call in state.calls.items())
                                    or state.function_call or state.tool_blocks):
                                first_output.reschedule(None)
                                if not announced:
                                    # Output that is neither reasoning nor reply text, such as a tool call.
                                    phase("")
                                if on_first_output is not None:
                                    on_first_output(None)
                    if data:
                        event = "\n".join(data)
                        data.clear()
                        capture_event(event)
                        if event != "[DONE]":
                            observed = True
                            state.add(event)
                    return state.completion()
            except TimeoutError as exc:
                if deadline is None or not first_output.expired():
                    raise
                error = (APIOutputSkipped("Skipped by the user before any model output.") if deadline.skipped else
                         APIResponseError(f"No model output within {first_output_timeout:g} seconds from request start."))
                if state is not None:
                    state.attach_diagnostics(error)
                raise error from exc
            except (httpx.TimeoutException, httpx.TransportError, httpx.DecodingError) as exc:
                # Visible partial output needs an agent-level retry notification.
                error_type = APIConnectionError if isinstance(exc, httpx.TransportError) else APITransportError
                error = error_type(f"API provider stream interrupted: {exc}")
                if state is not None:
                    state.attach_diagnostics(error)
                if observed or single_attempt or not await self.wait_retry(attempt, error):
                    raise error from exc
                attempt += 1
            except APIResponseError as exc:
                if exc is declined:
                    raise
                if state is not None:
                    state.attach_diagnostics(exc)
                exc.retry_after = exc.retry_after or response.headers.get("retry-after")
                # HTTP 200 can contain an overload instead of a completion.
                # Partial output returns to the agent for visible recovery.
                no_partial = not observed if state is None else not (
                    any(state.content_parts) or state.reasoning or state.reasoning_details or state.calls or state.function_call or state.tool_blocks or state.usage)
                if (not single_attempt and exc.status_code in RETRYABLE_STATUS and no_partial
                        and await self.wait_retry(attempt, exc)):
                    attempt += 1
                    observed = False
                    continue
                raise
            finally:
                carried = first_output.when() if deadline is not None else None
                if deadline is not None and on_first_output is not None:
                    on_first_output(None)
                if on_response is not None:
                    if data:
                        capture_event("\n".join(data))
                    on_response(captured if captured is not None else _merge_events(events))

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
            from .modelscores import model_score
            for model in models:
                model.provider = self.provider
                model.score = model_score(model.provider, model.id)
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

    def request_body(self, payload: dict[str, Any]) -> RequestBody:
        from .requestbody import RequestBody
        return RequestBody(payload)

    def request_headers(self, http_referer: str | None, app_title: str | None) -> dict[str, str]:
        return {}

    def replacement_options(self) -> dict[str, Any]:
        """Settings retained when replacing a credential on this provider."""
        return {"retry": self.retry, "on_retry": self.on_retry}

    async def _request(
        self, method: str, path: str, *, single_attempt: bool = False,
        on_response: Callable[[dict[str, Any]], None] | None = None,
        on_raw_response: Callable[[httpx.Response], None] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                async with self._client.stream(method, url, **kwargs) as response:
                    if on_raw_response is not None:
                        on_raw_response(response)
                    await response.aread()
            except (httpx.TimeoutException, httpx.TransportError, httpx.DecodingError) as exc:
                error_type = APIConnectionError if isinstance(exc, httpx.TransportError) else APITransportError
                error: APIResponseError = error_type(f"Network error talking to API provider after {attempt + 1} attempt(s): {exc}")
                if single_attempt or not await self.wait_retry(attempt, error):
                    raise error from exc
                attempt += 1
                continue

            if on_response is not None:
                on_response(ResponseReader()._response_body(response))
            if response.status_code < 400:
                raw = ResponseReader()._decode_json(response)
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
            if maximum <= 0 or attempt > math.log2(maximum / self.retry.initial_backoff):
                return None
        return min(maximum, self._backoff(attempt, retry_after))

    async def wait_retry(self, attempt: int, error: APIResponseError, *,
                         stop: asyncio.Event | None = None,
                         on_retry: Callable[[str, int], None] | None = None,
                         persistent: bool = False) -> bool:
        """Wait silently first, then report a cancellable countdown on each retry."""
        delay = self.retry_delay(attempt, error.retry_after)
        if delay is None and persistent:
            # Background work remains queued; honor longer provider cooldowns.
            delay = max(1.0, _parse_retry_after(error.retry_after) or min(60.0, self.retry.max_backoff))
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


class ResponseReader:
    """Read provider responses: completions, streamed reasoning and content, errors, and retry hints."""

    def _parse_retry_after(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            seconds = float(value.strip())
            return max(0.0, seconds) if math.isfinite(seconds) else None
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                return None

    def _reasoning_text(self, delta: dict[str, Any]) -> list[str]:
        for key in ("reasoning", "reasoning_content"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return [value]
        texts: list[str] = []
        content = delta.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    parts = block.get("thinking")
                    if isinstance(parts, list):
                        texts.extend(part["text"] for part in parts if isinstance(part, dict)
                                     and part.get("type") == "text" and isinstance(part.get("text"), str))
        for detail in delta.get("reasoning_details") or []:
            if isinstance(detail, dict) and detail.get("type") in {"reasoning.text", "reasoning.summary"}:
                value = detail.get("text") or detail.get("summary")
                if isinstance(value, str) and value:
                    texts.append(value)
        return texts

    def _decode_json(self, response: httpx.Response) -> dict[str, Any]:
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

    def _is_context_overflow(self, detail: str) -> bool:
        """Recognize explicit input-size errors in either HTTP or streamed payloads."""
        lower = detail.casefold()
        overflow = any(code in lower for code in ("context_length_exceeded", "context_window_exceeded", "prompt_too_long"))
        overflow = overflow or (any(term in lower for term in ("context length", "context window", "input tokens", "prompt tokens"))
                                and any(term in lower for term in ("exceed", "too long", "too many", "requested", "resulted in")))
        return overflow

    def _build_error(self, response: httpx.Response) -> APIResponseError:
        status = response.status_code
        detail = self._extract_error_message(response)
        if status in {400, 413} and self._is_context_overflow(detail):
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

    def _extract_error_message(self, response: httpx.Response) -> str:
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

    def _response_body(self, response: httpx.Response) -> dict[str, Any]:
        """Capture the response body before validation, without request headers."""
        try:
            value = response.json()
        except ValueError:
            return {"body": response.text}
        return value if isinstance(value, dict) else {"body": value}

    def _joined(self, value: Any) -> Any:
        if isinstance(value, _TextParts):
            return value.text()
        if isinstance(value, dict):
            return {key: self._joined(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._joined(item) for item in value]
        return value

    def _merge_events(self, events: list[str]) -> dict[str, Any]:
        """Assemble captured stream events for diagnostics in one linear pass."""
        captured: dict[str, Any] = {}
        for event in events:
            try:
                value = json.loads(event)
            except ValueError:
                captured.setdefault("unparsed_events", []).append(event)
                continue
            if isinstance(value, dict):
                self._merge_response(captured, value)
            else:
                captured.setdefault("unparsed_events", []).append(value)
        result: dict[str, Any] = self._joined(captured)
        return result

    def _merge_response(self, target: dict[str, Any], incoming: dict[str, Any]) -> None:
        """Assemble provider deltas for diagnostics independently of tool parsing.

        Indexed arrays retain choice/call/reasoning-block identities. Text fragments
        join in place; unknown fields remain available for inspecting provider formats.
        """
        for key, value in incoming.items():
            if value is None and key in target:
                continue
            if key == "content" and (isinstance(value, list) or isinstance(target.get(key), list)):
                # Content can switch from strings to thinking blocks and back.
                # Preserve ordered fragments rather than merging unrelated block positions.
                previous = target.get(key)
                if isinstance(previous, _TextParts):
                    previous = previous.text()
                if not isinstance(previous, list):
                    target[key] = [{"type": "text", "text": previous}] if isinstance(previous, str) and previous else []
                if isinstance(value, list):
                    target[key].extend(copy.deepcopy(value))
                elif isinstance(value, str):
                    target[key].append({"type": "text", "text": value})
                else:
                    target[key].append(copy.deepcopy(value))
                continue
            if isinstance(value, dict):
                if not isinstance(target.get(key), dict):
                    target[key] = {}
                self._merge_response(target[key], value)
            elif isinstance(value, list):
                if key in {"choices", "tool_calls", "reasoning_details", "content"}:
                    items = target.setdefault(key, [])
                    for position, part in enumerate(value):
                        if not isinstance(part, dict):
                            items.append(copy.deepcopy(part))
                            continue
                        index = part.get("index", position)
                        match = next((item for i, item in enumerate(items)
                                      if isinstance(item, dict) and item.get("index", i) == index), None)
                        if match is None:
                            match = {}
                            items.append(match)
                        self._merge_response(match, part)
                else:
                    target.setdefault(key, []).extend(copy.deepcopy(value))
            elif isinstance(value, str) and key in {"content", "reasoning", "reasoning_content", "text", "summary", "arguments", "name", "refusal"}:
                previous = target.get(key)
                if isinstance(previous, _TextParts):
                    previous.parts.append(value)
                else:
                    target[key] = _TextParts([previous, value]) if isinstance(previous, str) else value
            else:
                target[key] = value

    def _content_parts(self, content: Any) -> tuple[str, list[dict[str, Any]]]:
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
            elif block.get("type") == "thinking":
                parts = block.get("thinking")
                if not isinstance(parts, list) or any(not isinstance(part, dict)
                        or part.get("type") != "text" or not isinstance(part.get("text"), str) for part in parts):
                    raise ResponseFormatError("Thinking content must contain text blocks.")
            else:
                raise ResponseFormatError("Unsupported assistant content block; expected text or tool_use.")
        return "".join(texts), calls

    def _parse_completion(self, raw: dict[str, Any]) -> Completion:
        if raw.get("error") and self._is_context_overflow(str(raw["error"])):
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
                content, blocks = self._content_parts(wire.get("content"))
                calls = combine_calls(normalize_calls(wire.get("tool_calls")),
                                      normalize_calls(wire.get("function_call")), normalize_calls(blocks))
            except ResponseFormatError as exc:
                response_error = str(exc)
                response_excerpt = json.dumps(
                    {"invalid_message_prefix": json.dumps(wire, ensure_ascii=True)[:1500]}, indent=2,
                )
            separated: dict[str, list[str]] = {"content": [], "reasoning": []}
            thoughts = ThoughtStream(lambda kind, text: separated[kind].append(text))
            thoughts.add(content)
            thoughts.finish()
            message = Message.assistant("".join(separated["content"]) or None, calls)
            details = wire.get("reasoning_details")
            if details is not None:
                if not isinstance(details, list) or any(not isinstance(detail, dict) for detail in details):
                    raise ValueError("reasoning_details must be an array of objects")
                # Ordinary reasoning is one string. Keep opaque provider metadata
                # in the local original; outgoing requests exclude both forms.
                if any(detail.get("signature") or detail.get("type") == "reasoning.encrypted" for detail in details):
                    message.reasoning_details = copy.deepcopy(details)
            message.reasoning = "".join([*self._reasoning_text(wire), *separated["reasoning"]]) or None
            return Completion(
                message=message,
                model=raw.get("model", ""),
                finish_reason=first.get("finish_reason"),
                usage=Usage.from_api(raw.get("usage")),
                response_error=response_error, response_excerpt=response_excerpt,
            )
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise APIResponseError(f"API provider returned a malformed completion: {exc}") from exc


# Entry points for other modules; plugins override the methods above.
def _parse_retry_after(value: str | None) -> float | None:
    return ResponseReader()._parse_retry_after(value)


def _build_error(response: httpx.Response) -> APIResponseError:
    return ResponseReader()._build_error(response)


def _merge_events(events: list[str]) -> dict[str, Any]:
    return ResponseReader()._merge_events(events)


def _parse_completion(raw: dict[str, Any]) -> Completion:
    return ResponseReader()._parse_completion(raw)


class FirstOutputDeadline:
    """A working request's wait for output, which the user can end early."""

    def __init__(self, timeout: asyncio.Timeout) -> None:
        self.timeout = timeout
        self.skipped = False
        self.held = False

    def remaining(self) -> float | None:
        """Seconds left, or None while the deadline is suspended or disarmed."""
        when = self.timeout.when()
        return None if when is None else max(0.0, when - asyncio.get_running_loop().time())

    def hold(self) -> None:
        """Wait for this model without a deadline; other request failures still apply."""
        if self.skipped:
            return
        self.held = True
        try:
            self.timeout.reschedule(None)
        except RuntimeError:
            # The request enters its timeout right after announcing the deadline.
            asyncio.get_running_loop().call_soon(self._apply_hold)

    def _apply_hold(self) -> None:
        if self.held:
            try:
                self.timeout.reschedule(None)
            except RuntimeError:
                pass

    def skip(self) -> None:
        """Expire now; a suspended deadline expires when it resumes."""
        self.skipped = True
        was_held, self.held = self.held, False
        if self.timeout.when() is not None or was_held:
            try:
                self.timeout.reschedule(asyncio.get_running_loop().time())
            except RuntimeError:
                pass


class _PacingExcluded(httpx.AsyncByteStream):
    """Extend a deadline by time spent producing the body, such as upload pacing waits.

    Network writes happen between chunks, so a stalled upload still counts.
    """

    def __init__(self, body: httpx.AsyncByteStream, deadline: FirstOutputDeadline) -> None:
        self.body, self.deadline = body, deadline

    async def __aiter__(self) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        chunks = aiter(self.body)
        timeout = self.deadline.timeout
        while True:
            when = timeout.when()
            remaining = None if when is None else when - loop.time()
            timeout.reschedule(None)
            try:
                chunk = await anext(chunks)
            except StopAsyncIteration:
                return
            finally:
                if self.deadline.skipped:
                    timeout.reschedule(loop.time())
                elif remaining is not None and not self.deadline.held:
                    timeout.reschedule(loop.time() + remaining)
            yield chunk


class _StreamCompletion:
    """Assemble text, indexed tool fragments, and final usage from SSE events."""

    def __init__(self, on_delta: Callable[[str, str], None], *, identity_mode: str = "delta", index_mode: str = "index") -> None:
        self.on_delta = on_delta
        self.identity_mode = identity_mode
        self.index_mode = index_mode
        self.active_slots: dict[int, int] = {}
        # Streamed text is forwarded as it arrives and joined once at the end.
        self.content_parts: list[str] = []
        self.thoughts = ThoughtStream(self._text_delta)
        self.calls: dict[int, dict[str, Any]] = {}
        # String argument fragments per call; None while a non-string value is held.
        self.argument_parts: dict[int, list[str] | None] = {}
        self.model = ""
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.reasoning_details: list[dict[str, Any]] = []
        self.reasoning: list[str] = []
        self.function_call: dict[str, Any] = {}
        self.tool_blocks: list[dict[str, Any]] = []

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    def has_arguments(self, index: int) -> bool:
        parts = self.argument_parts.get(index, [])
        return any(parts) if parts is not None else bool(self.calls[index]["function"]["arguments"])

    def joined_calls(self) -> dict[int, dict[str, Any]]:
        for index, parts in self.argument_parts.items():
            if parts is not None:
                joined = "".join(parts)
                self.argument_parts[index] = [joined]
                self.calls[index]["function"]["arguments"] = joined
        return self.calls

    def attach_diagnostics(self, error: APIResponseError) -> None:
        error.partial_response = json.dumps({"content": self.content, "tool_calls": self.joined_calls(),
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
                if ResponseReader()._is_context_overflow(str(raw["error"])):
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
                if self.finish_reason is not None:
                    # Usage may follow the completed choice, but late deltas
                    # cannot reopen its reply or introduce additional tools.
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
                for text in ResponseReader()._reasoning_text(delta):
                    self.reasoning.append(text)
                    self.on_delta("reasoning", text)
                content = delta.get("content")
                if content is not None:
                    content, blocks = ResponseReader()._content_parts(content)
                    self.tool_blocks.extend(blocks)
                    self.thoughts.add(content)
                fragments = delta.get("tool_calls") or []
                if isinstance(fragments, dict):
                    fragments = [fragments]
                for position, fragment in enumerate(fragments):
                    if type(fragment.get("id")) is int:
                        fragment = {**fragment, "id": str(fragment["id"])}
                    index = fragment.get("index")
                    if self.index_mode == "id":
                        raw_index = index if type(index) is int and index >= 0 else position
                        identifier = fragment.get("id")
                        if identifier:
                            matching = next((i for i, c in self.calls.items() if c["id"] == identifier), None)
                            slot = matching if matching is not None else max(self.calls, default=-1) + 1
                            self.active_slots[raw_index] = slot
                        index = self.active_slots.get(raw_index, raw_index)
                    if type(index) is not int or index < 0:
                        # IDs identify interleaved calls; otherwise use their
                        # position in the delta, including single-call streams.
                        identifier = fragment.get("id")
                        matching = next((i for i, c in self.calls.items()
                                         if identifier and c["id"] == identifier), None)
                        index = matching if matching is not None else position
                        if (matching is None and identifier and index in self.calls
                                and self.calls[index]["id"] not in ("", identifier)):
                            index = max(self.calls) + 1
                    call = self.calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    identifier = fragment.get("id")
                    if isinstance(identifier, str) and identifier and identifier != call["id"]:
                        if call["id"] and self.identity_mode == "snapshot":
                            raise ValueError("Conflicting streamed tool-call IDs at one index")
                        call["id"] = identifier if self.identity_mode == "snapshot" or self.index_mode == "id" else call["id"] + identifier
                    function = fragment.get("function")
                    if not isinstance(function, dict):
                        function = fragment
                    name = function.get("name") or fragment.get("name")
                    if isinstance(name, str) and name:
                        if call["function"]["name"] and self.identity_mode == "snapshot" and name != call["function"]["name"]:
                            raise ValueError("Conflicting streamed tool names at one index")
                        call["function"]["name"] = name if self.identity_mode == "snapshot" else call["function"]["name"] + name
                    arguments = next((source[field] for source in (function, fragment)
                                      for field in ("arguments", "input", "parameters")
                                      if source.get(field) is not None), None)
                    if arguments is not None:
                        parts = self.argument_parts.setdefault(index, [])
                        if isinstance(arguments, str) and parts is not None:
                            parts.append(arguments)
                        elif isinstance(arguments, str):
                            self.argument_parts[index] = [arguments]
                        else:
                            call["function"]["arguments"] = arguments
                            self.argument_parts[index] = None
                legacy = delta.get("function_call")
                if legacy is not None:
                    if not isinstance(legacy, dict):
                        raise ValueError("function_call must be an object")
                    for field in ("name", "arguments"):
                        value = legacy.get(field) or ""
                        if value:
                            previous = self.function_call.get(field, "")
                            if field == "name" and self.identity_mode == "snapshot" and previous and value != previous:
                                raise ValueError("Conflicting streamed legacy tool names")
                            self.function_call[field] = value if field == "name" and self.identity_mode == "snapshot" else self.function_call.get(field, "") + value
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise APIResponseError(f"Malformed API provider stream: {exc}") from exc

    def _text_delta(self, kind: str, text: str) -> None:
        if kind == "reasoning":
            self.reasoning.append(text)
        else:
            self.content_parts.append(text)
        self.on_delta(kind, text)

    def completion(self) -> Completion:
        if self.finish_reason is None:
            raise APITransportError("API provider stream ended before a complete response")
        try:
            self.thoughts.finish()
        except ValueError as exc:
            raise APIResponseError(str(exc)) from exc
        calls = self.joined_calls()
        content = self.content
        return _parse_completion({
            "model": self.model, "usage": self.usage,
            "choices": [{"finish_reason": self.finish_reason, "message": {
                "role": "assistant", "content": ([{"type": "text", "text": content}, *self.tool_blocks]
                                                  if self.tool_blocks else content or None),
                "tool_calls": [calls[index] for index in sorted(calls)],
                "function_call": self.function_call or None,
                "reasoning_details": self.reasoning_details or None,
                "reasoning": "".join(self.reasoning),
            }}],
        })


class _TextParts:
    """Text fragments joined once after all diagnostic events are merged."""

    def __init__(self, parts: list[str]) -> None:
        self.parts = parts

    def text(self) -> str:
        return "".join(self.parts)


