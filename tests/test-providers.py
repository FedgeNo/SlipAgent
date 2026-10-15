"""Provider selection, public catalogs, and NVIDIA wire compatibility."""

import slipagent.sessionstate as sessionstate

import io
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent import cli
from slipagent.agent import Agent
from slipagent.api import APIConfigError, APIResponseError, RetryPolicy
from slipagent.config import Config, ConfigError, save_model_choice
from slipagent.nvidia import NvidiaClient, ULTRA_MODEL
from slipagent.openrouter import OpenRouterClient
from slipagent.providers import create_client, active_providers
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ModelInfo, ToolSpec
from slipagent.workspace import Workspace


def completion(text="Done."):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def transport_for(provider, requests, fail_profile=False):
    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": ULTRA_MODEL, "supported_parameters": ["tools", "max_tokens"]}]})
        if request.url.path.endswith("/endpoints"):
            if fail_profile:
                return httpx.Response(400, json={"error": "unsupported"})
            return httpx.Response(200, json={"data": {"endpoints": [{"tag": "test", "supported_parameters": ["tools", "max_tokens"], "context_length": 1_000_000}]}})
        return httpx.Response(200, json=completion())
    return httpx.MockTransport(handle)


def session_for(client, tmp_path):
    registry = ToolRegistry([])
    agent = Agent(client, registry, ULTRA_MODEL)
    return cli.Session(agent, registry, client, cli.Renderer(cli.Style(False), io.StringIO(), False),
                       Workspace(tmp_path), client.api_key, client.base_url, None, "SlipAgent")


def test_provider_config_keeps_credentials_and_preferences_separate():
    env = {"OPENROUTER_API_KEY": "router-key", "NVIDIA_API_KEY": "nvidia-key"}
    save_model_choice("router/model", "openrouter")
    save_model_choice(ULTRA_MODEL, "nvidia")
    assert Config.from_env(environ=env).provider == "nvidia"
    assert Config.from_env(provider="nvidia", environ=env).api_key == "nvidia-key"
    assert Config.from_env(provider="openrouter", environ=env).model == "router/model"
    assert Config.from_env(provider="nvidia", environ=env).model == ULTRA_MODEL
    with pytest.raises(ConfigError, match="NVIDIA_API_KEY"):
        Config.from_env(provider="nvidia", environ={"OPENROUTER_API_KEY": "router-key"})
    with pytest.raises(ConfigError, match="Unknown API provider"):
        Config.from_env(provider="invalid", environ=env)


async def test_nvidia_payload_uses_its_own_parameters_and_no_router_headers():
    requests = []
    async with create_client("nvidia", api_key="dummy", transport=transport_for("nvidia", requests),
                             reasoning_effort="medium", reasoning_budget=128) as client:
        profile = await client.model_capabilities(ULTRA_MODEL)
        assert profile.native_tools and profile.format is None
        snapshots = []
        await client.chat(model=ULTRA_MODEL, messages=[Message.user("hello")], max_tokens=256,
                          session_id="session", tools=[ToolSpec("probe", "probe", {})], on_request=snapshots.append)
        assert await client.key_info() is None
        with pytest.raises(APIConfigError, match="output limit"):
            await client.chat(model=ULTRA_MODEL, messages=[Message.user("hello")], max_tokens=32769)
    request = next(r for r in requests if r.method == "POST")
    body = json.loads(request.content)
    assert body["max_tokens"] == 256
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "medium_effort": True}
    assert body["reasoning_budget"] == 128
    assert not {"provider", "session_id", "max_completion_tokens", "reasoning", "response_format"}.intersection(body)
    assert not any("openrouter" in name.lower() for name in request.headers)
    assert snapshots[0] == body
    assert all(not r.url.path.endswith("/key") for r in requests)


async def test_catalog_only_client_sends_no_credentials_and_cannot_infer():
    requests = []
    async with create_client("nvidia", api_key="", catalog_only=True,
                             transport=transport_for("nvidia", requests)) as client:
        models = await client.list_models()
        assert models[0].selector == f"nvidia::{ULTRA_MODEL}"
        assert "authorization" not in requests[0].headers
        with pytest.raises(APIConfigError, match="NVIDIA_API_KEY"):
            await client.chat(model=ULTRA_MODEL, messages=[Message.user("hello")])


async def test_unknown_nvidia_capabilities_are_not_invented():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "unknown/model"}]}))
    async with NvidiaClient("dummy", transport=transport) as client:
        models = await client.list_models()
        assert models[0].context_length is None
        assert cli._price_cell(models[0].pricing) != "free"
        with pytest.raises(APIConfigError, match="No verified NVIDIA capability profile"):
            await client.model_capabilities("unknown/model")


async def test_streamed_overload_is_retried_before_tool_call_is_delivered():
    attempts = []
    def handle(request):
        attempts.append(request)
        if len(attempts) <= 3:
            events = [{"error": {"message": "Service temporarily overloaded", "code": 503}}]
        else:
            events = [{"choices": [{"delta": {"reasoning_content": "Checking."}}]},
                      {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "type": "function", "function": {"name": "probe", "arguments": '{"value":7}'}}]}, "finish_reason": "tool_calls"}]}]
        content = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=content)
    deltas = []
    async with NvidiaClient("dummy", transport=httpx.MockTransport(handle),
                            retry=RetryPolicy(max_retries=3, initial_backoff=0, max_backoff=0)) as client:
        result = await client.chat(model=ULTRA_MODEL, messages=[Message.user("probe")], on_delta=lambda *delta: deltas.append(delta))
    assert len(attempts) == 4
    assert deltas == [("reasoning", "Checking.")]
    assert result.tool_calls[0].arguments == {"value": 7}


async def test_search_combines_providers_and_duplicate_ids_require_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia")
    async with OpenRouterClient("dummy", transport=transport_for("openrouter", [])) as client:
        session = session_for(client, tmp_path)
        alternate = ModelInfo(ULTRA_MODEL, provider="nvidia")
        fetch = AsyncMock(return_value=[alternate])
        monkeypatch.setattr(sessionstate, "fetch_catalog", fetch)
        models = await session.catalog()
        assert {model.provider for model in models} == {"openrouter", "nvidia"}
        fetch.assert_awaited_once()
        out = io.StringIO()
        await cli._model_command(session, ULTRA_MODEL, cli.Style(False), out)
        assert "Select a provider explicitly" in out.getvalue()
        assert session.client is client
        choose = AsyncMock(return_value=None)
        session.renderer.terminal = type("Terminal", (), {"choose": choose})()
        await cli._models_command(session, "nemotron", cli.Style(False), out)
        options = choose.call_args.args[1]
        assert {value for value, label in options} == {f"openrouter::{ULTRA_MODEL}", f"nvidia::{ULTRA_MODEL}"}
        assert all("key needed" not in label for value, label in options)


async def test_partial_catalog_failure_preserves_other_results(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia")
    async with OpenRouterClient("dummy", transport=transport_for("openrouter", [])) as client:
        session = session_for(client, tmp_path)
        monkeypatch.setattr(sessionstate, "fetch_catalog", AsyncMock(side_effect=APIResponseError("unavailable", 503)))
        assert len(await session.catalog()) == 1
        out = io.StringIO()
        await cli._models_command(session, "nemotron", cli.Style(False), out)
        assert "catalog unavailable: nvidia" in out.getvalue()


@pytest.mark.parametrize('provider', ['openrouter', 'nvidia'])
async def test_model_switch_does_not_wait_for_background_compression(tmp_path, monkeypatch, provider):
    import asyncio
    monkeypatch.setenv('NVIDIA_API_KEY', 'dummy-nvidia')
    async with OpenRouterClient('dummy', transport=transport_for('openrouter', [])) as client:
        session = session_for(client, tmp_path)
        def factory(provider, **kwargs):
            kwargs['transport'] = transport_for(provider, [])
            return create_client(provider, **kwargs)
        monkeypatch.setattr(sessionstate, 'create_client', factory)
        compactor = session.agent._compactor()
        job = asyncio.create_task(asyncio.Event().wait())
        compactor.jobs.add(job)
        job.add_done_callback(compactor.jobs.discard)
        try:
            await asyncio.wait_for(session.select_model(ModelInfo(ULTRA_MODEL, provider=provider)), 1)
            assert session.client.provider == provider
            assert job.cancelled() if provider == 'nvidia' else not job.done()
        finally:
            compactor.reset()
            await compactor.wait()
            await session.client.aclose()


async def test_provider_switch_is_atomic_and_key_replacement_keeps_provider(tmp_path, monkeypatch):
    async with OpenRouterClient("dummy", transport=transport_for("openrouter", [])) as client:
        session = session_for(client, tmp_path)
        original_messages = list(session.agent.messages)
        entry = ModelInfo(ULTRA_MODEL, provider="nvidia")
        with pytest.raises(ConfigError, match="NVIDIA_API_KEY"):
            await session.select_model(entry)
        assert session.client is client and session.agent.messages == original_messages
        monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia")
        replacements = []
        def factory(provider, **kwargs):
            kwargs["transport"] = transport_for(provider, [])
            replacement = create_client(provider, **kwargs)
            replacements.append(replacement)
            return replacement
        monkeypatch.setattr(sessionstate, "create_client", factory)
        await session.select_model(entry)
        assert session.client.provider == "nvidia" and client._client.is_closed
        assert session.agent.messages == original_messages
        await session.use_api_key("another-dummy")
        assert session.client.provider == "nvidia" and session.api_key == "another-dummy"
        await session.select_model(ModelInfo(ULTRA_MODEL))
        assert session.client.provider == "openrouter" and session.api_key == "dummy"
        assert "nvidia" in active_providers(keys={"nvidia": session.extensions["provider_options"]["nvidia"]["api_key"]})
        await session.select_model(entry)
        assert session.client.provider == "nvidia" and session.api_key == "another-dummy"
        await session.client.aclose()
        assert all(replacement._client.is_closed for replacement in replacements)


async def test_failed_candidate_closes_itself_and_preserves_active_model(tmp_path, monkeypatch):
    async with NvidiaClient("dummy", transport=transport_for("nvidia", [])) as client:
        session = session_for(client, tmp_path)
        replacement = OpenRouterClient("dummy", transport=transport_for("openrouter", [], fail_profile=True))
        monkeypatch.setattr(sessionstate, "create_client", lambda *args, **kwargs: replacement)
        with pytest.raises(APIResponseError):
            await session.select_model(ModelInfo(ULTRA_MODEL))
        assert replacement._client.is_closed and not client._client.is_closed
        assert session.agent.client is client and session.agent.model == ULTRA_MODEL


async def test_model_preflight_uses_candidate_profile_without_switching_early(tmp_path, monkeypatch):
    from slipagent import agent as agent_module, contextbuilder
    old = "old/native"
    candidate = ULTRA_MODEL
    lookups = []

    def profile(model):
        lookups.append(model)
        return {"tool_channel": "embedded" if model == candidate else "native", "sections": []}

    monkeypatch.setattr(agent_module, "model_prompt_profile", profile)
    monkeypatch.setattr(contextbuilder, "model_prompt_profile", profile)
    original = Agent._context_view
    async def preview(self, *args, **kwargs):
        assert self.model == old
        result = await original(self, *args, **kwargs)
        assert self.model == old
        return result

    monkeypatch.setattr(Agent, "_context_view", preview)
    async with OpenRouterClient("dummy", transport=transport_for("openrouter", [])) as client:
        session = session_for(client, tmp_path)
        session.agent.model = old
        await session.select_model(ModelInfo(candidate))
        assert lookups and set(lookups) == {candidate}
        assert session.agent.model == candidate


async def test_provider_without_key_is_not_queried_and_activation_refreshes_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    async with OpenRouterClient("session-key", transport=transport_for("openrouter", [])) as client:
        session = session_for(client, tmp_path)
        fetch = AsyncMock(return_value=[ModelInfo(ULTRA_MODEL, provider="nvidia")])
        monkeypatch.setattr(sessionstate, "fetch_catalog", fetch)
        monkeypatch.setenv("NVIDIA_API_KEY", "   ")
        assert {model.provider for model in await session.catalog()} == {"openrouter"}
        fetch.assert_not_awaited()
        monkeypatch.setenv("NVIDIA_API_KEY", "dummy-nvidia")
        assert {model.provider for model in await session.catalog()} == {"openrouter", "nvidia"}
        fetch.assert_awaited_once()
        monkeypatch.delenv("NVIDIA_API_KEY")
        assert {model.provider for model in await session.catalog()} == {"openrouter"}
        assert fetch.await_count == 1


def test_only_nvidia_key_selects_nvidia_automatically():
    config = Config.from_env(environ={"NVIDIA_API_KEY": "dummy"})
    assert config.provider == "nvidia" and config.model == ULTRA_MODEL
    assert active_providers({"OPENROUTER_API_KEY": "  ", "NVIDIA_API_KEY": "dummy"}) == ["nvidia"]


@pytest.mark.parametrize("provider,model", [
    ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ("nvidia", ULTRA_MODEL), ("kilo", "kilo-auto/free"),
])
async def test_provider_command_selects_and_remembers_free_default(tmp_path, monkeypatch, provider, model):
    async with OpenRouterClient("dummy") as client:
        session = session_for(client, tmp_path)
        entry = ModelInfo(model, provider=provider, pricing={"prompt": "0", "completion": "0"})
        catalog = AsyncMock(return_value=[entry])
        select = AsyncMock()
        monkeypatch.setattr(cli.Session, "catalog", catalog)
        monkeypatch.setattr(cli.Session, "select_model", select)
        await cli._execute_command(session, f"/provider {provider}")
        select.assert_awaited_once_with(entry)
        catalog.assert_awaited_once_with(refresh=True)
        config = Config.from_env(environ={"OPENROUTER_API_KEY": "dummy", "NVIDIA_API_KEY": "dummy", "KILO_API_KEY": "dummy"})
        assert (config.provider, config.model) == (provider, model)


@pytest.mark.parametrize("pricing", [{}, {"prompt": "unknown", "completion": "0"},
    {"prompt": "0.000001", "completion": "0"},
    {"prompt": "0", "completion": "0", "request": "0.01"}])
async def test_provider_command_rejects_paid_or_unknown_pricing(tmp_path, monkeypatch, pricing):
    async with OpenRouterClient("dummy") as client:
        session = session_for(client, tmp_path)
        entry = ModelInfo("kilo-auto/free", provider="kilo", pricing=pricing)
        monkeypatch.setattr(cli.Session, "catalog", AsyncMock(return_value=[entry]))
        select = AsyncMock()
        monkeypatch.setattr(cli.Session, "select_model", select)
        await cli._execute_command(session, "/provider kilo")
        select.assert_not_awaited()
        assert "not advertised as free" in session.renderer.stream.getvalue()


async def test_provider_command_does_not_substitute_missing_default(tmp_path, monkeypatch):
    async with OpenRouterClient("dummy") as client:
        session = session_for(client, tmp_path)
        monkeypatch.setattr(cli.Session, "catalog", AsyncMock(return_value=[
            ModelInfo("other/free", provider="kilo", pricing={"prompt": "0", "completion": "0"})]))
        select = AsyncMock()
        monkeypatch.setattr(cli.Session, "select_model", select)
        await cli._execute_command(session, "/provider kilo")
        select.assert_not_awaited()
        assert "unknown model: kilo::kilo-auto/free" in session.renderer.stream.getvalue()


async def test_provider_command_queues_during_work_and_help_accepts_slash(tmp_path):
    async with OpenRouterClient("dummy") as client:
        session = session_for(client, tmp_path)
        session.agent.running = True
        await cli._execute_command(session, "/provider kilo")
        assert session.extensions["deferred_commands"] == ["/provider kilo"]
        assert cli._command_help("provider") == cli._command_help("/provider")
        assert "free default" in cli._command_help("provider")


async def test_list_models_activates_provider_from_explicit_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    fetch = AsyncMock(return_value=[ModelInfo(ULTRA_MODEL, provider="nvidia")])
    monkeypatch.setattr(cli, "fetch_catalog", fetch)
    args = cli.build_parser().parse_args(["--list-models", "--provider", "nvidia", "--api-key", "dummy"])
    assert await cli._list_models(args) == 0
    fetch.assert_awaited_once()
    assert fetch.call_args.args == ("nvidia",)
    output = capsys.readouterr()
    assert f"nvidia::{ULTRA_MODEL}" in output.out
    assert "openrouter::" not in output.out


@pytest.mark.parametrize("provider", ["openrouter", "nvidia"])
async def test_coding_listings_hide_curated_ids_and_keep_unknown_models(provider, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("NVIDIA_API_KEY", "dummy")
    entries = [ModelInfo(name, provider=provider) for name in (
        "nvidia/embed-qa-4", "nvidia/nemotron-4-340b-reward", ULTRA_MODEL,
        "new/unknown-model", "new/embed-named-coder",
    )]
    fetch = AsyncMock(return_value=entries)
    monkeypatch.setattr(cli, "fetch_catalog", fetch)
    monkeypatch.setattr(sessionstate, "fetch_catalog", fetch)
    async with create_client(provider, api_key="dummy", transport=transport_for(provider, [])) as client:
        monkeypatch.setattr(client, "list_models", AsyncMock(return_value=entries))
        session = session_for(client, tmp_path)
        out = io.StringIO()
        await cli._models_command(session, "", cli.Style(False), out)
        listing = out.getvalue()
        assert "nvidia/embed-qa-4" not in listing
        assert "nvidia/nemotron-4-340b-reward" not in listing
        assert ULTRA_MODEL in listing
        assert "new/unknown-model" in listing
        assert "new/embed-named-coder" in listing
        assert len(await client.list_models()) == 5

    args = cli.build_parser().parse_args(["--list-models"])
    assert await cli._list_models(args) == 0
    listing = capsys.readouterr().out
    assert "nvidia/embed-qa-4" not in listing
    assert "nvidia/nemotron-4-340b-reward" not in listing
    assert ULTRA_MODEL in listing
    assert "new/unknown-model" in listing
    assert "new/embed-named-coder" in listing
