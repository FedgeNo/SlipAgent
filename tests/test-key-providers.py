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
from slipagent.config import Config, data_directory, load_dotenv
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
        await cli.Repl(session).shutdown()
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
    path = data_directory() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Keep this\nOPENROUTER_API_KEY=old\nUNRELATED=value\n")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ask = AsyncMock(return_value="yes")
    monkeypatch.setattr(cli.Renderer, "ask_input", ask)

    await cli.KeyEntry(session, cli.Style(False), out).command(key)

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
    load_dotenv(environ=loaded)
    restored = Config.from_env(provider=provider, environ=loaded)
    assert restored.api_key == key and restored.provider == provider


async def test_unrecognized_key_needs_provider_without_sending_it(key_session, tmp_path):
    session, out, requests, clients = key_session
    original = session.client
    await cli.KeyEntry(session, cli.Style(False), out).command("opaque-secret")
    assert "provider" in out.getvalue() and "/key <provider> <key>" in out.getvalue()
    assert "opaque-secret" not in out.getvalue()
    assert not requests and len(clients) == 1
    assert session.client is original
    assert not (data_directory() / ".env").exists()


@pytest.mark.parametrize("candidate", [
    "eyJ.abc.fake", token({"sub": "unrelated-user"}), token(["kiloUserId"]),
    token({"kiloUserId": "", "version": 3}), token({"kiloUserId": "test", "version": True}),
])
async def test_unrelated_or_malformed_token_needs_provider(key_session, candidate):
    session, out, requests, clients = key_session
    await cli.KeyEntry(session, cli.Style(False), out).command(candidate)
    assert "Could not identify" in out.getvalue()
    assert not requests and len(clients) == 1


async def test_unrecognized_key_prompts_for_provider(key_session, monkeypatch):
    session, out, requests, clients = key_session
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ask = AsyncMock(side_effect=["mistral", "n"])
    monkeypatch.setattr(cli.Renderer, "ask_input", ask)
    await cli.KeyEntry(session, cli.Style(False), out).command("opaque-secret")
    assert session.compression_options()["mistral"]["api_key"] == "opaque-secret"
    assert len(requests) == 1 and requests[0][0] == "mistral"
    assert "provider" in ask.call_args_list[0].args[0]


async def test_explicit_provider_accepts_opaque_key(key_session):
    session, out, requests, clients = key_session
    await cli.KeyEntry(session, cli.Style(False), out).command("kilo opaque-secret")
    assert session.compression_options()["kilo"]["api_key"] == "opaque-secret"
    assert session.client.provider == "openrouter"
    assert "authentication will be checked on inference" in out.getvalue()
    assert not requests


async def test_provider_mismatch_leaves_credentials_untouched(key_session):
    session, out, requests, clients = key_session
    await cli.KeyEntry(session, cli.Style(False), out).command("google mstrl_new")
    assert "mistral" in out.getvalue() and "google" in out.getvalue()
    assert not requests and len(clients) == 1
    assert "google" not in session.compression_options()


async def test_hidden_key_input_uses_detection(key_session, monkeypatch):
    session, out, requests, clients = key_session
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ask = AsyncMock(side_effect=["nvapi-hidden", "n"])
    monkeypatch.setattr(cli.Renderer, "ask_input", ask)
    await cli.KeyEntry(session, cli.Style(False), out).command("")
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
    path = data_directory() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
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
    await cli.KeyEntry(session, cli.Style(False), out).command(key)
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
        await cli.KeyEntry(session, cli.Style(False), out).command("sk-or-v1-new")
    assert session.client is original and not original._client.is_closed
    assert clients[-1]._client.is_closed


async def test_installed_key_reaches_catalog_model_selection_and_compression(key_session, monkeypatch):
    session, out, requests, clients = key_session
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nvidia.example.test/v1")
    await cli.KeyEntry(session, cli.Style(False), out).command("nvapi-new")
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
    await cli.KeyEntry(session, cli.Style(False), out).replace(candidate=key)
    assert not requests and len(clients) == 1
    assert "ASCII" in out.getvalue()


async def test_key_listing_masks_keys_and_names_their_sources(key_session, monkeypatch):
    import os
    from slipagent import config
    session, out, requests, clients = key_session
    path = data_directory() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("NVIDIA_API_KEY=nvapi-saved-secret-value-1234\n")
    monkeypatch.setenv("MISTRAL_API_KEY", "mstrl_shellsecretvalue5678")
    monkeypatch.setattr(config, "_DOTENV_NAMES", set())
    monkeypatch.setattr(config, "_SESSION_NAMES", set())
    session.extensions["provider_options"] = {"kilo": {"api_key": "kilo-session-secret-abcd"}}
    await cli.KeyEntry(session, cli.Style(False), out).command("list")
    text = out.getvalue()
    assert "openrouter  sk-or-v1-…inal  active" in text
    assert "nvapi-…1234" in text and str(path) in text
    assert "mstrl_…5678" in text and "exported by the shell" in text
    assert "kilo…abcd" in text and "this session" in text
    for secret in ("nvapi-saved-secret-value-1234", "mstrl_shellsecretvalue5678",
                   "kilo-session-secret-abcd", "sk-or-v1-original"):
        assert secret not in text
    assert not requests


async def test_key_delete_removes_saved_process_and_session_copies(key_session, monkeypatch):
    import os
    from slipagent import config
    session, out, requests, clients = key_session
    path = data_directory() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Keep this\nexport NVIDIA_API_KEY=nvapi-saved\nUNRELATED=value\n")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-saved")
    monkeypatch.setattr(config, "_DOTENV_NAMES", {"NVIDIA_API_KEY"})
    monkeypatch.setattr(config, "_SESSION_NAMES", set())
    session.extensions["provider_options"] = {"nvidia": {"api_key": "nvapi-saved"}}
    routing = session.model_routing()
    routing.failed_providers.update({"nvidia", "untouched"})
    session._catalog = []
    entry = cli.KeyEntry(session, cli.Style(False), out)

    await entry.command("delete nvidia")
    assert path.read_text() == "# Keep this\nUNRELATED=value\n"
    assert "NVIDIA_API_KEY" not in os.environ
    assert "nvidia" not in session.extensions["provider_options"]
    assert "nvidia" not in routing.failed_providers and "untouched" in routing.failed_providers
    assert session._catalog is None
    assert "nvidia key deleted" in out.getvalue() and str(path) in out.getvalue()

    # The provider serving the current model keeps its key.
    await entry.command("delete openrouter")
    assert "in use by the current model" in out.getvalue()
    assert session.client.api_key == "sk-or-v1-original"

    # A shell export is cleared for this process only, with a warning.
    monkeypatch.setenv("MISTRAL_API_KEY", "mstrl_shell")
    await entry.command("delete mistral")
    assert "MISTRAL_API_KEY" not in os.environ
    assert "your shell exports it" in out.getvalue()

    await entry.command("delete nvidia")
    assert "no nvidia key is configured" in out.getvalue()
    await entry.command("delete bogus")
    assert "usage: /key delete <provider>" in out.getvalue()
    assert not requests


async def test_key_menu_confirms_before_deleting(key_session, monkeypatch):
    import os
    from types import SimpleNamespace
    from slipagent import config
    session, out, requests, clients = key_session
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-saved-value-9999")
    monkeypatch.setattr(config, "_DOTENV_NAMES", set())
    monkeypatch.setattr(config, "_SESSION_NAMES", set())
    choose = AsyncMock(side_effect=["nvidia", "cancel", "nvidia", "delete"])
    session.renderer.terminal = SimpleNamespace(choose=choose)
    entry = cli.KeyEntry(session, cli.Style(False), out)

    await entry.command("")
    assert "NVIDIA_API_KEY" in os.environ and "kept" in out.getvalue()
    title, options = choose.await_args_list[0].args
    assert title == "API Keys" and options[0] == ("new", "Enter a new key")
    assert any(key == "nvidia" and "nvapi-…9999" in label and "nvapi-saved-value-9999" not in label for key, label in options)
    question = choose.await_args_list[1].args[0]
    assert question.startswith("Are you sure you want to delete this key?") and "nvapi-…9999" in question

    await entry.command("")
    assert "NVIDIA_API_KEY" not in os.environ and "nvidia key deleted" in out.getvalue()
    assert choose.await_count == 4


async def test_expiring_and_expired_keys_warn_once_and_show_in_listing(key_session):
    import math
    from datetime import datetime, timedelta, timezone
    from slipagent.types import KeyInfo
    session, out, requests, clients = key_session
    quota = cli.Quota(session)
    soon = (datetime.now(timezone.utc) + timedelta(days=2, hours=12)).isoformat()
    quota.cache(KeyInfo(label="k", expires_at=soon))
    quota.cache(KeyInfo(label="k", expires_at=soon))
    assert out.getvalue().count("openrouter API key expires in 3 days") == 1
    assert "rotate it with /key <key>" in out.getvalue()
    far = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    quota.cache(KeyInfo(label="k", expires_at=far))
    assert "expires in 30" not in out.getvalue()
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    quota.cache(KeyInfo(label="k", expires_at=past))
    assert "openrouter API key expired on" in out.getvalue()
    assert "/key delete openrouter" in out.getvalue()
    await cli.KeyEntry(session, cli.Style(False), out).command("list")
    assert f"expired {(datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()}" in out.getvalue()
    cli.KeyEntry(session, cli.Style(False), out).describe(KeyInfo(label="k", expires_at=past))
    assert "(expired)" in out.getvalue()


async def test_rejected_key_in_quota_poll_warns_once_with_remedy(key_session, monkeypatch):
    from slipagent.api import APIAuthError
    session, out, requests, clients = key_session
    monkeypatch.setattr(session.client, "key_info",
                        AsyncMock(side_effect=APIAuthError("API provider rejected the API key (401): expired")))
    for _ in range(2):
        session.quota_checked_at = None
        await cli.Quota(session).refresh()
    text = out.getvalue()
    assert text.count("rejected its API key during the quota check") == 1
    assert "/key delete openrouter" in text and "/key <key>" in text
    # A failed run names the same remedy.
    assert cli._failure_text(session, APIAuthError("API provider rejected the API key (401)")).endswith(
        "Replace it with /key <key> or remove it with /key delete openrouter.")
    assert cli._failure_text(session, ValueError("other")) == "other"
