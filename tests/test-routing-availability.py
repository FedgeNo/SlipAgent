"""Provider changes and catalog refreshes preserve routing exclusions."""

from importlib import import_module
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent.api import APIResponseError
from slipagent.capabilities import RequestProfile
from slipagent.providers import create_client

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
