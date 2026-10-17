"""Provider detection installs credentials without changing the working model."""

import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent import cli, compressionproviders, sessionstate
from slipagent.agent import Agent
from slipagent.api import RetryPolicy
from slipagent.capabilities import RequestProfile
from slipagent.config import Config, load_dotenv
from slipagent.providers import create_client, provider_class
from slipagent.tools import ToolRegistry
from slipagent.types import ModelInfo
from slipagent.workspace import Workspace


def token(payload):
    parts = [{"alg": "HS256", "typ": "JWT"}, payload]
    return ".".join(base64.urlsafe_b64encode(json.dumps(part).encode()).decode().rstrip("=")
                    for part in parts) + ".fake-signature"


@pytest.fixture
async def key_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    requests = []
    clients = []

    def factory(provider, **options):
        def respond(request):
            requests.append((provider, request))
            if provider == "openrouter":
                return httpx.Response(200, json={"data": {"label": "test key", "usage": 0}})
            if provider == "google":
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, json={"data": []})
        options["transport"] = httpx.MockTransport(respond)
        options["retry"] = RetryPolicy(max_retries=0)
        client = create_client(provider, **options)
        clients.append(client)
        return client

    monkeypatch.setattr(sessionstate, "create_client", factory)
    client = factory("openrouter", api_key="sk-or-v1-original")
    registry = ToolRegistry()
    out = io.StringIO()
    renderer = cli.Renderer(cli.Style(False), out, False)
    session = cli.Session(Agent(client, registry, "test/model"), registry, client,
                          renderer, Workspace(tmp_path), client.api_key, client.base_url,
                          None, "test")
    try:
        yield session, out, requests, clients
    finally:
        await cli._shutdown(session)
        for client in clients:
            await client.aclose()


@pytest.mark.parametrize("key, provider", [
    ("sk-or-v1-new", "openrouter"),
    ("nvapi-new", "nvidia"),
    ("mstrl_new", "mistral"),
    ("AIza-new", "google"),
    ("AQ.new", "google"),
    (token({"kiloUserId": "test-user", "version": 3}), "kilo"),
])
async def test_detected_key_is_saved_to_its_provider(key_session, tmp_path, monkeypatch, key, provider):
    session, out, requests, clients = key_session
    original = session.client
    session.free_calls = 37
    routing = session.model_routing()
    routing.enabled = True
    routing.failed_providers.update({provider, "untouched"})
    session._catalog = []
    path = tmp_path / ".env"
    path.write_text("# Keep this\nOPENROUTER_API_KEY=old\nUNRELATED=value\n")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ask = AsyncMock(return_value="yes")
    monkeypatch.setattr(cli, "_ask_input", ask)

    await cli._key_command(session, key, cli.Style(False), out)

    assert f"{provider_class(provider).key_env}={key}\n" in path.read_text()
    assert "# Keep this\n" in path.read_text() and "UNRELATED=value\n" in path.read_text()
    assert session.compression_options()[provider]["api_key"] == key
    assert session._catalog is None
    assert provider not in routing.failed_providers and "untouched" in routing.failed_providers
    assert routing.enabled and session.agent.model == "test/model"
    assert key not in out.getvalue()
    assert all(name == provider for name, request in requests)
    assert ask.await_count == 1
    if provider != "openrouter":
        assert "OPENROUTER_API_KEY=old\n" in path.read_text()
        assert session.client is original and not original._client.is_closed
        assert session.free_calls == 37
        assert all(client._client.is_closed for client in clients[1:])
    else:
        assert session.client is not original and original._client.is_closed
    loaded = {}
    load_dotenv(tmp_path, environ=loaded)
    restored = Config.from_env(provider=provider, environ=loaded)
    assert restored.api_key == key and restored.provider == provider


async def test_unrecognized_key_needs_provider_without_sending_it(key_session, tmp_path):
    session, out, requests, clients = key_session
    original = session.client
    await cli._key_command(session, "opaque-secret", cli.Style(False), out)
    assert "provider" in out.getvalue() and "/key <provider> <key>" in out.getvalue()
    assert "opaque-secret" not in out.getvalue()
    assert not requests and len(clients) == 1
    assert session.client is original
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("candidate", [
    "eyJ.abc.fake", token({"sub": "unrelated-user"}), token(["kiloUserId"]),
    token({"kiloUserId": "", "version": 3}), token({"kiloUserId": "test", "version": True}),
])
async def test_unrelated_or_malformed_token_needs_provider(key_session, candidate):
    session, out, requests, clients = key_session
    await cli._key_command(session, candidate, cli.Style(False), out)
    assert "Could not identify" in out.getvalue()
    assert not requests and len(clients) == 1


async def test_unrecognized_key_prompts_for_provider(key_session, monkeypatch):
    session, out, requests, clients = key_session
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ask = AsyncMock(side_effect=["mistral", "n"])
    monkeypatch.setattr(cli, "_ask_input", ask)
    await cli._key_command(session, "opaque-secret", cli.Style(False), out)
    assert session.compression_options()["mistral"]["api_key"] == "opaque-secret"
    assert len(requests) == 1 and requests[0][0] == "mistral"
    assert "provider" in ask.call_args_list[0].args[1]


async def test_explicit_provider_accepts_opaque_key(key_session):
    session, out, requests, clients = key_session
    await cli._key_command(session, "kilo opaque-secret", cli.Style(False), out)
    assert session.compression_options()["kilo"]["api_key"] == "opaque-secret"
    assert session.client.provider == "openrouter"
    assert "authentication will be checked on inference" in out.getvalue()
    assert not requests


async def test_provider_mismatch_leaves_credentials_untouched(key_session):
    session, out, requests, clients = key_session
    await cli._key_command(session, "google mstrl_new", cli.Style(False), out)
    assert "mistral" in out.getvalue() and "google" in out.getvalue()
    assert not requests and len(clients) == 1
    assert "google" not in session.compression_options()


async def test_hidden_key_input_uses_detection(key_session, monkeypatch):
    session, out, requests, clients = key_session
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ask = AsyncMock(side_effect=["nvapi-hidden", "n"])
    monkeypatch.setattr(cli, "_ask_input", ask)
    await cli._key_command(session, "", cli.Style(False), out)
    assert ask.call_args_list[0].kwargs["password"] is True
    assert session.compression_options()["nvidia"]["api_key"] == "nvapi-hidden"
    assert session.client.provider == "openrouter"


@pytest.mark.parametrize("provider, key", [("openrouter", "sk-or-v1-bad"), ("mistral", "mstrl_bad")])
async def test_rejected_key_keeps_live_state_and_saved_key(key_session, tmp_path, monkeypatch, provider, key):
    session, out, requests, clients = key_session
    original = session.client
    original_catalog = session._catalog = []
    session.free_calls = 37
    session.extensions["provider_options"] = {"mistral": {"api_key": "mstrl_old"}}
    path = tmp_path / ".env"
    path.write_text("OPENROUTER_API_KEY=old\nMISTRAL_API_KEY=mstrl_old\n")
    original_text = path.read_text()
    factory = sessionstate.create_client

    def rejected(provider, **options):
        client = factory(provider, **options)
        from slipagent.api import APIAuthError
        method = "key_info" if provider == "openrouter" else "list_models"
        monkeypatch.setattr(client, method, AsyncMock(side_effect=APIAuthError("invalid key", 401)))
        return client

    monkeypatch.setattr(sessionstate, "create_client", rejected)
    await cli._key_command(session, key, cli.Style(False), out)
    assert "rejected" in out.getvalue()
    assert session.client is original and not original._client.is_closed
    assert session.free_calls == 37 and session._catalog is original_catalog
    assert session.compression_options()["mistral"]["api_key"] == "mstrl_old"
    assert path.read_text() == original_text
    assert clients[-1]._client.is_closed


async def test_cancelled_verification_closes_candidate(key_session, monkeypatch):
    session, out, requests, clients = key_session
    original = session.client
    factory = sessionstate.create_client

    def cancelled(provider, **options):
        client = factory(provider, **options)
        monkeypatch.setattr(client, "key_info", AsyncMock(side_effect=asyncio.CancelledError))
        return client

    monkeypatch.setattr(sessionstate, "create_client", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await cli._key_command(session, "sk-or-v1-new", cli.Style(False), out)
    assert session.client is original and not original._client.is_closed
    assert clients[-1]._client.is_closed


async def test_installed_key_reaches_catalog_model_selection_and_compression(key_session, monkeypatch):
    session, out, requests, clients = key_session
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nvidia.example.test/v1")
    await cli._key_command(session, "nvapi-new", cli.Style(False), out)
    model = ModelInfo("test/new", provider="nvidia", score=99.0, pricing={"prompt": "0", "completion": "0"})
    catalog = AsyncMock(return_value=[model])
    monkeypatch.setattr(sessionstate, "fetch_catalog", catalog)
    monkeypatch.setattr(session.client, "list_models", AsyncMock(return_value=[]))
    assert await session.catalog() == [model]
    assert catalog.call_args.kwargs["api_key"] == "nvapi-new"
    assert catalog.call_args.kwargs["base_url"] == "https://nvidia.example.test/v1"

    monkeypatch.setattr(compressionproviders, "create_client", sessionstate.create_client)
    session.configure_compaction()
    compressor = session.agent._compactor().routing
    client, selected = await compressor.select(set())
    assert selected is model and client.api_key == "nvapi-new"
    assert client.base_url == "https://nvidia.example.test/v1"

    monkeypatch.setattr(provider_class("nvidia"), "model_capabilities",
                        AsyncMock(return_value=RequestProfile(parameters=set(), context_length=10000)))
    monkeypatch.setattr(Agent, "_context_view", AsyncMock())
    await session.select_model(model)
    assert session.api_key == "nvapi-new" and session.client.provider == "nvidia"
    assert session.base_url == "https://nvidia.example.test/v1"


@pytest.mark.parametrize("key", ["\n", "nvapi-key\nother", "AQ.\u2603", "nvapi-\x00bad"])
async def test_invalid_key_characters_do_not_reach_transport(key_session, key):
    session, out, requests, clients = key_session
    await cli._replace_key(session, cli.Style(False), out, candidate=key)
    assert not requests and len(clients) == 1
    assert "ASCII" in out.getvalue()
