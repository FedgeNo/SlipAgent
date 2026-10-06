"""The OpenRouter client, exercised through a mocked httpx transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from slipagent.openrouter import (
    OpenRouterAPIError,
    OpenRouterAuthError,
    OpenRouterClient,
    OpenRouterConfigError,
    OpenRouterError,
    OpenRouterLimitError,
    RetryPolicy,
)
from slipagent.types import Message, ToolCall, ToolSpec

FAST_RETRY = RetryPolicy(max_retries=2, initial_backoff=0.001, max_backoff=0.002)


def make_client(handler: Any, retry: RetryPolicy = FAST_RETRY) -> OpenRouterClient:
    return OpenRouterClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        retry=retry,
    )


def chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "gen-1",
        "model": "anthropic/claude-sonnet-4.5",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello there."},
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "total_tokens": 15,
            "cost": 0.0012,
        },
    }
    body.update(overrides)
    return body


def tool_call_body() -> dict[str, Any]:
    return chat_body(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "README.md"}',
                            },
                        }
                    ],
                },
            }
        ]
    )


# --------------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------------- #


async def test_requires_api_key() -> None:
    with pytest.raises(OpenRouterConfigError, match="API key"):
        OpenRouterClient(api_key="  ")


async def test_sends_expected_request_shape() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=chat_body())

    client = make_client(handler)
    await client.chat(
        model="anthropic/claude-sonnet-4.5",
        messages=[Message.system("be brief"), Message.user("hi")],
        tools=[ToolSpec(name="read_file", description="read", parameters={})],
        temperature=0.2,
        max_tokens=512,
        session_id="sess-1",
    )
    await client.aclose()

    assert captured["url"].endswith("/api/v1/chat/completions")
    assert captured["auth"] == "Bearer test-key"

    body = captured["body"]
    assert body["model"] == "anthropic/claude-sonnet-4.5"
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert body["temperature"] == 0.2
    assert body["max_completion_tokens"] == 512
    assert body["session_id"] == "sess-1"


async def test_omits_optional_parameters_when_unset() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=chat_body())

    client = make_client(handler)
    await client.chat(model="m", messages=[Message.user("hi")])
    await client.aclose()

    assert "temperature" not in captured["body"]
    assert "tools" not in captured["body"]
    assert "max_completion_tokens" not in captured["body"]


async def test_rejects_empty_message_list() -> None:
    client = make_client(lambda r: httpx.Response(200, json=chat_body()))

    with pytest.raises(OpenRouterError, match="at least one message"):
        await client.chat(model="m", messages=[])


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


async def test_parses_text_completion() -> None:
    client = make_client(lambda r: httpx.Response(200, json=chat_body()))
    completion = await client.chat(model="m", messages=[Message.user("hi")])
    await client.aclose()

    assert completion.text == "Hello there."
    assert completion.finish_reason == "stop"
    assert completion.model == "anthropic/claude-sonnet-4.5"
    assert completion.usage.total_tokens == 15
    assert completion.usage.cost == 0.0012
    assert not completion.wants_tools


async def test_parses_tool_calls() -> None:
    client = make_client(lambda r: httpx.Response(200, json=tool_call_body()))
    completion = await client.chat(model="m", messages=[Message.user("hi")])
    await client.aclose()

    assert completion.wants_tools
    assert len(completion.tool_calls) == 1
    assert completion.tool_calls[0].name == "read_file"
    assert completion.tool_calls[0].arguments == {"path": "README.md"}


async def test_rejects_empty_tool_arguments() -> None:
    body = tool_call_body()
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = ""
    client = make_client(lambda r: httpx.Response(200, json=body))

    completion = await client.chat(model="m", messages=[Message.user("hi")])
    await client.aclose()

    assert completion.response_error
    assert completion.tool_calls == []


async def test_response_without_choices_is_an_error() -> None:
    client = make_client(lambda r: httpx.Response(200, json={"id": "x"}))

    with pytest.raises(OpenRouterAPIError, match="no choices"):
        await client.chat(model="m", messages=[Message.user("hi")])


async def test_non_json_response_is_an_error() -> None:
    client = make_client(lambda r: httpx.Response(200, text="<html>nope</html>"))

    with pytest.raises(OpenRouterAPIError, match="non-JSON"):
        await client.chat(model="m", messages=[Message.user("hi")])


# --------------------------------------------------------------------------- #
# Errors and retries
# --------------------------------------------------------------------------- #


async def test_auth_error_is_typed() -> None:
    client = make_client(
        lambda r: httpx.Response(401, json={"error": {"message": "No auth", "code": 401}})
    )

    with pytest.raises(OpenRouterAuthError, match="rejected the API key"):
        await client.chat(model="m", messages=[Message.user("hi")])


async def test_credit_error_is_limit_typed() -> None:
    client = make_client(
        lambda r: httpx.Response(402, json={"error": {"message": "insufficient credits"}})
    )

    with pytest.raises(OpenRouterLimitError, match="insufficient credits"):
        await client.chat(model="m", messages=[Message.user("hi")])


async def test_403_is_a_limit_error_not_an_auth_error() -> None:
    """OpenRouter signals a $0-limit key with 403, not 401."""
    client = make_client(
        lambda r: httpx.Response(
            403,
            json={"error": {"message": "Key limit exceeded (total limit).",
                            "code": 403}},
        )
    )

    with pytest.raises(OpenRouterLimitError, match="Key limit exceeded") as caught:
        await client.chat(model="m", messages=[Message.user("hi")])

    assert not isinstance(caught.value, OpenRouterAuthError)
    assert "dashboard" in str(caught.value)


def test_guardrail_403_does_not_claim_a_key_spending_limit():
    from slipagent.openrouter import _build_error
    error = _build_error(httpx.Response(403, json={"error": {"message": "Request blocked by account guardrail"}}))
    assert not isinstance(error, OpenRouterLimitError)
    assert "guardrail" in str(error)
    assert "Raise or reset" not in str(error)


async def test_401_is_an_auth_error() -> None:
    client = make_client(
        lambda r: httpx.Response(401, json={"error": {"message": "User not found.",
                                                        "code": 401}})
    )

    with pytest.raises(OpenRouterAuthError, match="rejected the API key"):
        await client.chat(model="m", messages=[Message.user("hi")])


async def test_rate_limit_error_is_retried_and_surfaces_as_limit() -> None:
    client = make_client(
        lambda r: httpx.Response(429, json={"error": {"message": "slow down"}})
    )

    with pytest.raises(OpenRouterLimitError, match="rate limit"):
        await client.chat(model="m", messages=[Message.user("hi")])


async def test_retries_rate_limit_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=chat_body())

    client = make_client(handler)
    completion = await client.chat(model="m", messages=[Message.user("hi")])
    await client.aclose()

    assert attempts["n"] == 3
    assert completion.text == "Hello there."


async def test_gives_up_after_max_retries() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    client = make_client(handler)

    with pytest.raises(OpenRouterAPIError, match="unavailable"):
        await client.chat(model="m", messages=[Message.user("hi")])

    # One initial attempt plus max_retries.
    assert attempts["n"] == 3


async def test_does_not_retry_client_errors() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    client = make_client(handler)

    with pytest.raises(OpenRouterAPIError, match="bad request"):
        await client.chat(model="m", messages=[Message.user("hi")])

    assert attempts["n"] == 1


async def test_network_errors_are_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = make_client(handler, RetryPolicy(max_retries=1, initial_backoff=0.001))

    with pytest.raises(OpenRouterError, match="Network error"):
        await client.chat(model="m", messages=[Message.user("hi")])


# --------------------------------------------------------------------------- #
# Model catalog
# --------------------------------------------------------------------------- #


async def test_list_models_parses_catalog() -> None:
    client = make_client(
        lambda r: httpx.Response(
            200,
            json={
                "data": [
                    {"id": "a/b", "name": "B", "context_length": 200000},
                    {"id": "c/d", "name": "D", "context_length": 8000},
                ]
            },
        )
    )

    models = await client.list_models()
    await client.aclose()

    assert [m.id for m in models] == ["a/b", "c/d"]
    assert models[0].context_length == 200000


async def test_tool_call_round_trips_through_serialization() -> None:
    """The wire format we send back must be what OpenRouter expects."""
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.py", "limit": 10})
    payload = call.to_api()

    assert payload["type"] == "function"
    assert json.loads(payload["function"]["arguments"]) == {"path": "a.py", "limit": 10}
    assert ToolCall.from_api(payload) == call


@pytest.mark.parametrize("payload", [
    {"choices": ["invalid"]}, {"choices": [{"message": "invalid"}]},
    chat_body(usage={"prompt_tokens": "invalid"}),
])
async def test_malformed_completions_raise_typed_errors(payload) -> None:
    client = make_client(lambda r: httpx.Response(200, json=payload))
    try:
        with pytest.raises(OpenRouterAPIError):
            await client.chat(model="m", messages=[Message.user("go")])
    finally:
        await client.aclose()


@pytest.mark.parametrize("message", [
    {"content": ["invalid"]}, {"tool_calls": ["invalid"]},
    {"tool_calls": [{"function": "invalid"}]},
])
async def test_malformed_model_output_preserves_usage_for_protocol_retry(message) -> None:
    async with make_client(lambda r: httpx.Response(200, json=chat_body(choices=[{"message": message}]))) as client:
        result = await client.chat(model="m", messages=[Message.user("go")])
    assert result.response_error and result.response_excerpt
    assert result.tool_calls == []
    assert result.usage.total_tokens == 15


def test_non_object_tool_arguments_are_preserved_as_invalid() -> None:
    call = ToolCall.from_api({"id": "c", "function": {"name": "tool", "arguments": [1]}})
    assert "__invalid_arguments__" in call.arguments


@pytest.mark.parametrize("endpoint, data", [("key_info", []), ("key_info", {"limit": "invalid"}),
                                         ("list_models", {}), ("list_models", ["invalid"])])
async def test_malformed_catalog_and_key_raise_typed_errors(endpoint, data) -> None:
    client = make_client(lambda r: httpx.Response(200, json={"data": data}))
    try:
        with pytest.raises(OpenRouterAPIError):
            await getattr(client, endpoint)()
    finally:
        await client.aclose()
