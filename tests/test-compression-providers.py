"""Background routing uses separate clients and excludes the working provider."""

import json
from importlib import import_module
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent.capabilities import RequestProfile
from slipagent.compaction import StepCompactor
from slipagent.compressionproviders import CompressionProviders, CompressionUnavailableError, ULTRA_MODEL
from slipagent.context import CompletedStep
from slipagent.historymerge import schedule_history_merge
from slipagent.providers import create_client
from slipagent.types import Message
from test_agent import StubClient

history_with_posts = import_module("test-history-merge").history_with_posts


@pytest.fixture
def routing(monkeypatch):
    selected = ["openrouter"]
    options = {name: {"api_key": "dummy-" + name} for name in ("openrouter", "nvidia", "kilo")}
    requests = []
    clients = []

    def factory(provider, **kwargs):
        def handle(request):
            body = json.loads(request.content)
            requests.append((provider, body))
            content = {"summary": "The user asked for inspection; the tools returned observations."}
            if "Summarizing the Previous Turn" in body["messages"][0]["content"]:
                content["reasoning_summary"] = ""
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": json.dumps(content)}, "finish_reason": "stop"}]})
        client = create_client(provider, transport=httpx.MockTransport(handle), **kwargs)
        profile = RequestProfile(parameters={"max_tokens", "temperature"}, context_length=1_000_000,
                                 max_completion_tokens=4096 if provider == "nvidia" else 8192)
        monkeypatch.setattr(client, "model_capabilities", AsyncMock(return_value=profile))
        clients.append(client)
        return client

    monkeypatch.setattr("slipagent.compressionproviders.create_client", factory)
    router = CompressionProviders(lambda: selected[0], lambda: options)
    return router, selected, options, requests, clients


async def test_turn_and_group_compression_share_rotation_and_own_limits(routing):
    router, _, _, requests, clients = routing
    errors = []
    owner = StepCompactor(lambda usage: None, errors.append)
    owner.routing = router
    primary = StubClient([])
    try:
        for number in (1, 2):
            step = CompletedStep(number, "Inspect", [Message.user("Inspect"), Message.assistant("Done")])
            owner.submit(step, primary, "unrelated-working-model", None, 50)
            await owner.wait()
            assert step.summary and step.compaction_status == "complete"
        history, _ = history_with_posts(32)
        schedule_history_merge(owner, history.steps, primary, "working-model", None, 50)
        await owner.wait()
        assert not errors
        assert [provider for provider, _ in requests] == ["nvidia", "kilo", "nvidia"]
        assert [body["max_tokens"] for _, body in requests] == [4096, 8192, 4096]
        assert [body["model"] for _, body in requests] == [ULTRA_MODEL, ULTRA_MODEL + ":free", ULTRA_MODEL]
        assert all(not body.get("tools") for _, body in requests)
        assert getattr(history.steps[20], "historical_summary", None)
    finally:
        await owner.aclose()
    assert all(client._client.is_closed for client in clients)


async def test_rotation_follows_provider_switch_and_single_alternative(routing):
    router, selected, options, _, _ = routing
    try:
        assert (await router.select())[0].provider == "nvidia"
        selected[0] = "nvidia"
        assert (await router.select())[0].provider == "openrouter"
        assert (await router.select())[0].provider == "kilo"
        del options["openrouter"]
        assert (await router.select())[0].provider == "kilo"
        del options["kilo"]
        with pytest.raises(CompressionUnavailableError, match="other than"):
            await router.select()
    finally:
        await router.aclose()


async def test_retry_uses_next_alternative_provider(routing, monkeypatch):
    from slipagent.api import APIClient, APIResponseError
    router, _, _, _, _ = routing
    attempts = []
    original = APIClient.chat

    async def chat(client, **kwargs):
        attempts.append(client.provider)
        if len(attempts) == 1:
            raise APIResponseError("rate limited", status_code=429)
        return await original(client, **kwargs)

    monkeypatch.setattr(APIClient, "chat", chat)
    monkeypatch.setattr(APIClient, "wait_retry", AsyncMock(return_value=True))
    owner = StepCompactor(lambda usage: None, lambda error: pytest.fail(error))
    owner.routing = router
    step = CompletedStep(1, "Inspect", [Message.user("Inspect"), Message.assistant("Done")])
    try:
        owner.submit(step, StubClient([]), "working", None, 50)
        await owner.wait()
        assert attempts == ["nvidia", "kilo"]
        assert step.summary
    finally:
        await owner.aclose()


async def test_missing_alternative_preserves_original_without_calling_working_provider(routing):
    router, _, options, requests, _ = routing
    options.pop("nvidia")
    options.pop("kilo")
    errors = []
    owner = StepCompactor(lambda usage: None, errors.append)
    owner.routing = router
    step = CompletedStep(1, "Inspect", [Message.user("Inspect"), Message.assistant("Original answer")])
    try:
        owner.submit(step, StubClient([]), "working", None, 50)
        await owner.wait()
        assert step.compaction_status == "failed"
        assert step.summary is None
        assert "Original answer" in step.full_text()
        owner.submit(step, StubClient([]), "working", None, 50)
        await owner.wait()
        assert len(errors) == 1 and "other than" in errors[0]
        assert not requests
    finally:
        await owner.aclose()
