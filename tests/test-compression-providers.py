"""All background attempts share ranked free provider rotation."""

import asyncio
import json
from importlib import import_module
from unittest.mock import AsyncMock
from types import SimpleNamespace

import httpx
import pytest

from slipagent.capabilities import RequestProfile
from slipagent.compaction import StepCompactor
from slipagent.compressionproviders import CompressionProviders
from slipagent.modelrouting import ModelRouting
from slipagent.nvidia import ULTRA_MODEL
from slipagent.context import CompletedStep
from slipagent.historymerge import schedule_history_merge
from slipagent.providers import create_client
from slipagent.types import Message, ModelInfo
from test_agent import StubClient, completion

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
    class Catalog:
        @property
        def client(self):
            return SimpleNamespace(provider=selected[0])

        agent = SimpleNamespace(model=ULTRA_MODEL + ":free")
        notices = []
        renderer = SimpleNamespace(emit=notices.append)
        extra = []

        @property
        def _catalog(self):
            return [ModelInfo(ULTRA_MODEL if name == "nvidia" else ULTRA_MODEL + ":free",
                              provider=name, pricing={"prompt": "0", "completion": "0"}) for name in options] + self.extra

        async def catalog(self):
            return self._catalog

        def compression_options(self):
            return options

    router = CompressionProviders(ModelRouting(Catalog()))
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
        assert all("temperature" not in body for _, body in requests)
        assert getattr(history.steps[20], "historical_summary", None)
    finally:
        await owner.aclose()
    assert all(client._client.is_closed for client in clients)


async def test_rotation_follows_provider_switch_and_single_alternative(routing):
    router, selected, options, _, _ = routing
    try:
        assert (await router.select(set()))[0].provider == "nvidia"
        selected[0] = "nvidia"
        router.models.session.agent.model = ULTRA_MODEL
        assert (await router.select(set()))[0].provider == "kilo"
        assert (await router.select(set()))[0].provider == "openrouter"
        del options["openrouter"]
        assert (await router.select(set()))[0].provider == "kilo"
        del options["kilo"]
        assert (await router.select(set()))[0].provider == "nvidia"
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


async def test_single_provider_compresses_on_working_provider(routing):
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
        assert step.compaction_status == "complete"
        assert step.summary is not None
        assert "Original answer" in step.full_text()
        owner.submit(step, StubClient([]), "working", None, 50)
        await owner.wait()
        assert not errors
        assert [provider for provider, _ in requests] == ["openrouter", "openrouter"]
        assert len(router.models.session.notices) == 1
        assert "Add more provider API keys" in router.models.session.notices[0]
    finally:
        await owner.aclose()


async def test_paid_working_model_allows_its_provider_free_model(routing):
    router, _, _, _, _ = routing
    session = router.models.session
    session.agent.model = "paid"
    session.extra.append(ModelInfo("paid", provider="openrouter", rank=1,
                                   pricing={"prompt": "1", "completion": "1"}))
    try:
        choices = [await router.select(set()) for _ in range(4)]
        assert [client.provider for client, model in choices] == ["openrouter", "nvidia", "kilo", "openrouter"]
        assert all(model.id != "paid" for client, model in choices)
    finally:
        await router.aclose()


async def test_unknown_working_price_keeps_its_provider_reserved(routing):
    router, _, _, _, _ = routing
    router.models.session.agent.model = "unknown-price"
    router.models.session.extra.append(ModelInfo("unknown-price", provider="openrouter"))
    try:
        choices = [await router.select(set()) for _ in range(3)]
        assert [client.provider for client, model in choices] == ["nvidia", "kilo", "nvidia"]
    finally:
        await router.aclose()


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("failure", [400, 401, 429, 503, "malformed", "timeout", "capabilities", "budget"])
async def test_every_failure_moves_to_another_provider(routing, monkeypatch, grouped, failure):
    from slipagent.api import APIClient, APIResponseError
    router, _, _, _, clients = routing
    attempts = []
    original = APIClient.chat
    create = import_module("slipagent.compressionproviders").create_client

    def factory(provider, **kwargs):
        client = create(provider, **kwargs)
        if provider == "nvidia" and failure == "capabilities":
            client.model_capabilities.side_effect = APIResponseError("incompatible", 400)
        if provider == "nvidia" and failure == "budget":
            client.model_capabilities.return_value = RequestProfile(parameters={"max_tokens"}, context_length=100)
        return client

    async def chat(client, **kwargs):
        attempts.append(client.provider)
        if client.provider == "nvidia":
            if isinstance(failure, int):
                raise APIResponseError("unavailable", failure)
            if failure == "timeout":
                raise TimeoutError("timeout")
            return completion("bad JSON")
        return await original(client, **kwargs)

    monkeypatch.setattr("slipagent.compressionproviders.create_client", factory)
    monkeypatch.setattr(APIClient, "chat", chat)
    monkeypatch.setattr(APIClient, "wait_retry", AsyncMock(return_value=True))
    monkeypatch.setattr("slipagent.compaction.asyncio.sleep", AsyncMock())
    owner = StepCompactor(lambda usage: None, lambda error: pytest.fail(error))
    owner.routing = router
    step = CompletedStep(1, "Inspect", [Message.user("Inspect"), Message.assistant("Original")])
    try:
        if grouped:
            history, _ = history_with_posts(32)
            schedule_history_merge(owner, history.steps, StubClient([]), "working", None, 50)
        else:
            owner.submit(step, StubClient([]), "working", None, 50)
        await owner.wait()
        expected = ["kilo"] if failure in {"capabilities", "budget"} else ["nvidia", "kilo"]
        assert attempts == expected
        assert [client.provider for client in clients] == ["nvidia", "kilo"]
        if grouped:
            assert getattr(history.steps[20], "historical_summary", None)
        else:
            assert step.summary and "Original" in step.full_text()
        if failure in {429, 503, "timeout", "budget"}:
            assert not router.models.failed["compression"]
        elif failure == 401:
            assert router.models.failed_providers == {"nvidia"}
            assert all(model.provider != "nvidia" for model in await router.models.candidates("working"))
        else:
            assert router.models.failed["compression"] == {f"nvidia::{ULTRA_MODEL}"}
            assert f"nvidia::{ULTRA_MODEL}" not in router.models.failed["working"]
    finally:
        await owner.aclose()


async def test_permanent_failure_uses_next_rank_on_later_provider_rotation(routing, monkeypatch):
    from slipagent.api import APIClient, APIResponseError
    router, _, _, _, _ = routing
    router.models.session.extra.append(ModelInfo("best:free", provider="kilo", rank=1,
                                                pricing={"prompt": "0", "completion": "0"}))
    attempts = []
    original = APIClient.chat

    async def chat(client, **kwargs):
        attempts.append((client.provider, kwargs["model"]))
        if len(attempts) <= 2:
            raise APIResponseError("incompatible", 400)
        return await original(client, **kwargs)

    monkeypatch.setattr(APIClient, "chat", chat)
    monkeypatch.setattr("slipagent.compaction.asyncio.sleep", AsyncMock())
    owner = StepCompactor(lambda usage: None, lambda error: pytest.fail(error))
    owner.routing = router
    step = CompletedStep(1, "Inspect", [Message.user("Inspect"), Message.assistant("Original")])
    try:
        owner.submit(step, StubClient([]), "working", None, 50)
        await owner.wait()
        assert attempts == [("nvidia", ULTRA_MODEL), ("kilo", "best:free"), ("kilo", ULTRA_MODEL + ":free")]
        assert step.summary
    finally:
        await owner.aclose()


async def test_exhausted_invalid_replies_preserve_originals_and_finish(routing, monkeypatch):
    from slipagent.api import APIClient
    router, _, _, _, _ = routing
    attempts = []

    async def chat(client, **kwargs):
        attempts.append(client.provider)
        return completion("invalid summary")

    monkeypatch.setattr(APIClient, "chat", chat)
    monkeypatch.setattr("slipagent.compaction.asyncio.sleep", AsyncMock())
    errors = []
    owner = StepCompactor(lambda usage: None, errors.append)
    owner.routing = router
    step = CompletedStep(1, "Inspect", [Message.user("Inspect"), Message.assistant("Original")])
    try:
        owner.submit(step, StubClient([]), "working", None, 50)
        await owner.wait()
        assert attempts == ["nvidia", "kilo"] * 3
        assert step.summary is None and step.compaction_status == "failed"
        assert "Original" in step.full_text() and len(errors) == 1
    finally:
        await owner.aclose()


async def test_concurrent_jobs_rotate_and_retry_on_another_provider(routing, monkeypatch):
    from slipagent.api import APIClient, APIResponseError
    router, _, _, _, _ = routing
    owner = StepCompactor(lambda usage: None, lambda error: pytest.fail(error))
    owner.routing = router
    monkeypatch.setattr(APIClient, "wait_retry", AsyncMock(return_value=True))
    second_started = asyncio.Event()
    attempts = []

    async def first(client, model, profile, reserve):
        attempts.append(("first", client.provider))
        if len(attempts) == 1:
            await second_started.wait()
            raise APIResponseError("overloaded", 503)
        return "ready"

    async def second(client, model, profile, reserve):
        attempts.append(("second", client.provider))
        second_started.set()
        return "ready"

    try:
        results = await asyncio.gather(owner.request(first, StubClient([]), "working", None, 50, 0),
                                       owner.request(second, StubClient([]), "working", None, 50, 0))
        assert results == ["ready", "ready"]
        assert attempts == [("first", "nvidia"), ("second", "kilo"), ("first", "kilo")]
    finally:
        await owner.aclose()


async def test_reset_during_lookup_clears_inflight_job_exclusions(routing, monkeypatch):
    from slipagent.api import APIResponseError
    router, _, _, _, _ = routing
    owner = StepCompactor(lambda usage: None, lambda error: pytest.fail(error))
    owner.routing = router
    reset_lookup = False
    attempts = []

    async def catalog():
        nonlocal reset_lookup
        if reset_lookup:
            reset_lookup = False
            router.models.reset()
        return router.models.session._catalog

    async def retry_wait(delay):
        nonlocal reset_lookup
        # A concurrent job consumes the next slot before the user's reset.
        await router.select(set())
        reset_lookup = True

    async def operation(client, model, profile, reserve):
        attempts.append(client.provider)
        if len(attempts) == 1:
            raise APIResponseError("unusable", 400)
        return "ready"

    monkeypatch.setattr(router.models.session, "catalog", catalog)
    monkeypatch.setattr("slipagent.compaction.asyncio.sleep", retry_wait)
    try:
        assert await owner.request(operation, StubClient([]), "working", None, 50, 0) == "ready"
        assert attempts == ["nvidia", "nvidia"]
        assert not router.models.failed["compression"]
    finally:
        await owner.aclose()
