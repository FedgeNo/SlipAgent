"""Provider changes and catalog refreshes preserve routing exclusions."""

from importlib import import_module
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent.api import APIResponseError
from slipagent.capabilities import RequestProfile
from slipagent.compressionproviders import CompressionProviders
from slipagent.modelscores import model_score
from slipagent.providers import create_client
from slipagent.types import ModelInfo

session_for = import_module("test-providers").session_for
entry = import_module("test-optimal").entry


@pytest.mark.parametrize("endpoint", [None, "https://generativelanguage.googleapis.com/v1beta/openai/",
                                    "https://example.invalid/google/"])
@pytest.mark.parametrize("status", [429, 401])
async def test_provider_switch_does_not_restore_the_failed_model(tmp_path, monkeypatch, endpoint, status):
    from slipagent import sessionstate
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-google")
    if endpoint is None:
        monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("GEMINI_BASE_URL", endpoint)
    clients = []

    def factory(provider, **options):
        client = create_client(provider, **options, transport=httpx.MockTransport(
            lambda request: pytest.fail("Selection should use the stubbed profile")))
        monkeypatch.setattr(client, "model_capabilities", AsyncMock(return_value=RequestProfile(
            parameters={"tools", "max_tokens"}, context_length=1_000_000)))
        clients.append(client)
        return client

    monkeypatch.setattr(sessionstate, "create_client", factory)
    session = session_for(factory("openrouter", api_key="dummy"), tmp_path)
    session._catalog = [entry("openrouter", "first", 1), entry("google", "second", 2),
                        entry("openrouter", "third", 3)]
    routing = session.model_routing()
    try:
        await routing.enable()
        await routing.fallback(APIResponseError("No output within the deadline"))
        assert session.client.provider == "google" and session.agent.model == "second"
        await routing.fallback(APIResponseError("Provider failed", status_code=status))
        assert session.client.provider == "openrouter" and session.agent.model == "third"
        assert routing.failed["working"] == {"openrouter::first", "google::second"}
        assert routing.failed_providers == ({"google"} if status == 401 else set())
    finally:
        await session.agent._compactor().aclose()
        for client in clients:
            await client.aclose()


@pytest.mark.parametrize("purpose", ["working", "compression"])
async def test_catalog_expiry_skips_deleted_models_and_restores_returning_models(tmp_path, monkeypatch, purpose):
    from slipagent import sessionstate
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(sessionstate, "time", SimpleNamespace(monotonic=lambda: clock.now), raising=False)
    best = entry("openrouter", "stepfun/step-5-preview:free", 1)
    other = entry("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", 2)
    retained_rank = model_score(best.provider, best.id)
    async with create_client("openrouter", api_key="dummy") as client:
        catalog = AsyncMock(side_effect=[[best, other], [other], [best, other]])
        monkeypatch.setattr(client, "list_models", catalog)
        monkeypatch.setattr(client, "model_capabilities", AsyncMock(return_value=RequestProfile(
            parameters={"tools", "max_tokens"}, context_length=1_000_000)))
        session = session_for(client, tmp_path)
        routing = session.model_routing()
        compression = CompressionProviders(routing)
        try:
            await routing.enable()
            assert session.agent.model == best.id
            clock.now += 299
            assert (await routing.candidates(purpose))[0] is best
            assert catalog.await_count == 1
            clock.now += 2
            results = await asyncio.gather(*(routing.candidates(purpose) for _ in range(3)))
            assert results == [[other]] * 3
            assert catalog.await_count == 2
            if purpose == "working":
                await routing.prepare_working()
                assert session.agent.model == other.id
            else:
                assert (await compression.select(set()))[1] is other
            assert model_score(best.provider, best.id) == retained_rank
            assert best.selector not in routing.failed[purpose]
            clock.now += 301
            if purpose == "working":
                await routing.enable()
                assert session.agent.model == best.id
            else:
                assert (await compression.select(set()))[1] is best
            assert catalog.await_count == 3
        finally:
            await compression.aclose()
            await session.agent._compactor().aclose()


async def test_explicit_catalog_refresh_accepts_an_empty_provider_catalog(tmp_path, monkeypatch):
    best = entry("openrouter", "stepfun/step-5-preview:free", 1)
    retained_rank = model_score(best.provider, best.id)
    async with create_client("openrouter", api_key="dummy") as client:
        catalog = AsyncMock(side_effect=[[best], [], [best]])
        monkeypatch.setattr(client, "list_models", catalog)
        session = session_for(client, tmp_path)
        routing = session.model_routing()
        assert await routing.candidates("working") == [best]
        await session.catalog(refresh=True)
        assert await routing.candidates("working") == []
        assert await routing.candidates("compression", include_failed=True) == []
        assert model_score(best.provider, best.id) == retained_rank
        await session.catalog(refresh=True)
        assert await routing.candidates("working") == [best]


async def test_catalog_connection_failure_preserves_known_available_models(tmp_path, monkeypatch):
    from slipagent import sessionstate
    from slipagent.api import APIConnectionError
    monkeypatch.setenv("KILO_API_KEY", "dummy-kilo")
    best = entry("openrouter", "first", 1)
    other = entry("kilo", "second", 2)
    monkeypatch.setattr(sessionstate, "fetch_catalog", AsyncMock(return_value=[other]))
    async with create_client("openrouter", api_key="dummy") as client:
        monkeypatch.setattr(client, "list_models", AsyncMock(side_effect=[
            [best], APIConnectionError("Local connection failed"), []]))
        session = session_for(client, tmp_path)
        routing = session.model_routing()
        assert await routing.candidates("working") == [best, other]
        await session.catalog(refresh=True)
        assert await routing.candidates("working") == [best, other]
        assert session.extensions["catalog_errors"] == ["openrouter: Local connection failed"]
        await session.catalog(refresh=True)
        assert await routing.candidates("working") == [other]


async def test_cached_catalog_scores_follow_table_updates_without_network(tmp_path, monkeypatch):
    from slipagent.modelscores import SCORES
    best = ModelInfo("reflection/beam-free", provider="kilo", pricing={"prompt": "0", "completion": "0"})
    async with create_client("kilo", api_key="dummy") as client:
        session = session_for(client, tmp_path)
        session._catalog = [best]
        monkeypatch.setattr(client, "list_models", AsyncMock(side_effect=AssertionError("Unexpected refresh")))
        monkeypatch.setitem(SCORES, "reflection/beam", 88.0)
        assert (await session.catalog())[0].score == 88.0
        monkeypatch.setitem(SCORES, "reflection/beam", None)
        assert (await session.catalog())[0].score is None


@pytest.mark.parametrize("purpose", ["working", "compression"])
async def test_new_models_and_zero_scores_sort_consistently_with_catalog(tmp_path, monkeypatch, purpose):
    from slipagent.modelpricing import model_order
    monkeypatch.setenv("KILO_API_KEY", "dummy-kilo")
    free = {"prompt": "0", "completion": "0"}
    models = [ModelInfo("xingchen-agi/xing4.0-29b-a4b:free", pricing=free),
              ModelInfo("reflection/beam-free", provider="kilo", pricing=free),
              ModelInfo("tie", provider="kilo", score=model_score("openrouter", "xingchen-agi/xing4.0-29b-a4b"), pricing=free),
              ModelInfo("zero", score=0.0, pricing=free)]
    async with create_client("openrouter", api_key="dummy") as client:
        session = session_for(client, tmp_path)
        session._catalog = models + [ModelInfo("reflection/beam", provider="kilo",
                                               pricing={"prompt": "1", "completion": "1"})]
        routing = session.model_routing()
        candidates = await routing.candidates(purpose)
        assert candidates == sorted(models, key=model_order)
        assert [model.id for model in candidates] == ["reflection/beam-free", "tie",
                                                     "xingchen-agi/xing4.0-29b-a4b:free", "zero"]
