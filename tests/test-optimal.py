"""Ranked free routing, user control, and failure isolation."""

from importlib import import_module
from unittest.mock import AsyncMock
from types import SimpleNamespace
import json

import httpx
import pytest

from slipagent.api import APIConfigError, APIResponseError
from slipagent.capabilities import RequestProfile
from slipagent.modelrouting import ModelRouting
from slipagent.providers import create_client
from slipagent.types import ModelInfo

session_for = import_module("test-providers").session_for
RecordingTool = import_module("test-background-compaction").RecordingTool


def entry(provider, model, rank, *, free=True):
    return ModelInfo(model, provider=provider, rank=rank,
                     pricing={"prompt": "0" if free else "1", "completion": "0" if free else "1"})


def test_rank_metadata_is_shared_across_hosts_and_unknowns_stay_unranked():
    direct = ModelInfo("nvidia/nemotron-3-ultra-550b-a55b", provider="nvidia")
    hosted = ModelInfo(direct.id + ":free")
    assert direct.rank == hosted.rank and direct.rank is not None
    assert ModelInfo("new/model:free").rank is None
    assert ModelInfo("gemini-3.8-flash", provider="google").rank == ModelInfo("google/gemini-3.8-flash").rank


@pytest.mark.parametrize("model", ["stepfun-ai/step-5-preview", "deepseek-ai/deepseek-v4.1-flash"])
async def test_optimal_selects_ranked_nvidia_catalog_models_through_preflight(tmp_path, monkeypatch, model):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    calls = []

    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, json={"data": [
            {"id": "nvidia/nemotron-3-ultra-550b-a55b"}, {"id": model},
        ]})

    async with create_client("nvidia", api_key="dummy", transport=httpx.MockTransport(handle)) as client:
        session = session_for(client, tmp_path)
        try:
            await session.model_routing().enable()
            assert session.agent.model == model
            assert session.client is client
            assert session.model_routing().enabled
            assert not session.model_routing().failed["working"]
            assert set(calls) == {"GET"}
            assert f"Optimal selected nvidia::{model}" in session.renderer.stream.getvalue()
        finally:
            await session.agent._compactor().aclose()


async def test_optimal_reset_retries_best_and_manual_selection_disables(tmp_path, monkeypatch):
    client = create_client("openrouter", api_key="dummy")
    session = session_for(client, tmp_path)
    session._catalog = [entry("openrouter", "first", 1), entry("openrouter", "second", 2)]
    router = ModelRouting(session)
    selected = []

    async def select(model, *, automatic=False):
        selected.append(model.id)
        session.agent.model = model.id

    monkeypatch.setattr(type(session), "select_model", lambda self, model, **kw: select(model, **kw))
    try:
        await router.enable()
        await router.fallback(APIResponseError("unusable", status_code=400))
        assert selected == ["first", "second"]
        await router.enable()
        assert selected == ["first", "second", "first"]
        assert not router.failed["working"]
        router.enabled = False
        assert not await router.fallback(APIResponseError("unusable", status_code=400))
    finally:
        await client.aclose()


@pytest.fixture
def optimal_session(tmp_path, monkeypatch):
    responses = []
    requests = []

    def handle(request):
        body = json.loads(request.content)
        if "Summarizing the Previous Turn" in body["messages"][0]["content"]:
            return httpx.Response(200, json={"choices": [{"message": {"content":
                json.dumps({"summary": "The work and its observations were preserved.", "reasoning_summary": ""})},
                "finish_reason": "stop"}]})
        requests.append(body)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, int):
            return httpx.Response(result, json={"error": "model unavailable"})
        return httpx.Response(200, json={"choices": [{"message": result, "finish_reason": "stop"}]})

    client = create_client("openrouter", api_key="dummy", transport=httpx.MockTransport(handle))
    monkeypatch.setattr(client, "model_capabilities", AsyncMock(return_value=RequestProfile(
        parameters={"tools", "tool_choice", "temperature", "max_tokens"}, context_length=1_000_000)))
    monkeypatch.setattr(client, "wait_retry", AsyncMock(return_value=False))
    session = session_for(client, tmp_path)
    session._catalog = [entry("openrouter", "first", 1), entry("openrouter", "second", 2),
                        entry("openrouter", "paid", 0, free=False), entry("openrouter", "unknown", None)]
    return session, responses, requests


async def test_fallback_keeps_history_and_never_replays_tools(optimal_session):
    session, responses, requests = optimal_session
    tool = RecordingTool("The existing tool result")
    session.registry.register(tool)
    responses.extend([
        {"role": "assistant", "content": "Inspecting.", "tool_calls": [
            {"id": "one", "type": "function", "function": {"name": "record", "arguments": '{"value":"A"}'}}]},
        400, {"role": "assistant", "content": "Done."},
    ])
    try:
        await session.model_routing().enable()
        assert await session.agent.run("Complete this original task") == "Done."
        assert [request["model"] for request in requests] == ["first", "first", "second"]
        assert tool.seen == [{"value": "A"}]
        last = json.dumps(requests[-1]["messages"])
        assert "Complete this original task" in last
        assert "The existing tool result" in last and "model unavailable" in last
        assert session.agent._requests == 3
    finally:
        await session.agent._compactor().aclose()
        await session.client.aclose()


@pytest.mark.parametrize("failure", [400, 429, 503, {"role": "assistant", "content": "{\"tool_calls\": [{}]}"}])
async def test_unusable_working_models_fall_back_after_their_recovery(optimal_session, failure):
    session, responses, requests = optimal_session
    malformed = isinstance(failure, dict)
    responses.extend([failure] * (3 if malformed else 1) + [{"content": "Recovered."}])
    try:
        await session.model_routing().enable()
        assert await session.agent.run("Work") == "Recovered."
        assert [request["model"] for request in requests] == ["first"] * (3 if malformed else 1) + ["second"]
        assert session.model_routing().failed["working"] == {"openrouter::first"}
    finally:
        await session.agent._compactor().aclose()
        await session.client.aclose()


async def test_manual_selection_disables_optimal_and_paid_is_never_auto_selected(optimal_session):
    session, _, _ = optimal_session
    try:
        await session.model_routing().enable()
        await session.select_model(entry("openrouter", "paid", 0, free=False))
        assert session.agent.model == "paid"
        assert not session.model_routing().enabled
        await session.model_routing().enable()
        assert session.agent.model == "first"
    finally:
        await session.client.aclose()


async def test_unknown_notice_is_once_per_new_model_and_excludes_routers(optimal_session):
    session, _, _ = optimal_session
    router = session.model_routing()
    session._catalog += [entry("openrouter", "openrouter/free", None)]
    try:
        await router.candidates("working")
        await router.candidates("compression")
        output = session.renderer.stream.getvalue()
        assert output.count("New free models") == 1
        assert "openrouter::unknown" in output and "Update SlipAgent" in output
        assert "openrouter/free" not in output
    finally:
        await session.client.aclose()


async def test_reset_clears_both_purposes_and_provider_failures(optimal_session):
    from slipagent.cli import _handle_command
    session, _, _ = optimal_session
    routing = session.model_routing()
    try:
        await routing.enable()
        for command in ("/optimal on", "/optimal reset"):
            routing.failed["working"].add("openrouter::first")
            routing.failed["compression"].add("openrouter::first")
            routing.failed_providers.add("openrouter")
            session.agent.model = "second"
            await _handle_command(session, command)
            assert session.agent.model == "first"
            assert not any(routing.failed.values()) and not routing.failed_providers
        await routing.fallback(APIResponseError("unavailable", status_code=400))
        await _handle_command(session, "/optimal off")
        assert not routing.enabled and session.agent.model == "second"
        assert routing.failed["working"] == {"openrouter::first"}
    finally:
        await session.client.aclose()


async def test_normal_mode_keeps_error_and_selected_model(optimal_session):
    session, responses, requests = optimal_session
    responses.append(400)
    session.agent.model = "first"
    try:
        with pytest.raises(APIResponseError):
            await session.agent.run("Work")
        assert [r["model"] for r in requests] == ["first"]
        assert session.agent.model == "first"
    finally:
        await session.client.aclose()


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError,
                                      httpx.ReadTimeout, httpx.WriteError, httpx.ProxyError])
async def test_connection_failure_preserves_model_and_failure_state(optimal_session, failure):
    session, responses, requests = optimal_session
    responses.append(failure("Connection unavailable"))
    routing = session.model_routing()
    try:
        await routing.enable()
        with pytest.raises(APIResponseError, match="Connection unavailable"):
            await session.agent.run("Work")
        assert [request["model"] for request in requests] == ["first"]
        assert session.agent.model == "first"
        assert not any(routing.failed.values()) and not routing.failed_providers
        assert "keeping" in session.renderer.stream.getvalue().lower()
        responses.append({"content": "Connection restored."})
        assert await session.agent.run("Continue") == "Connection restored."
        assert [request["model"] for request in requests] == ["first", "first"]
    finally:
        await session.agent._compactor().aclose()
        await session.client.aclose()


async def test_preflight_connection_failure_keeps_candidates_eligible(optimal_session, monkeypatch):
    from slipagent.api import APIConnectionError
    session, _, _ = optimal_session
    routing = session.model_routing()
    monkeypatch.setattr(session.client, "model_capabilities", AsyncMock(
        side_effect=APIConnectionError("Metadata connection failed")))
    try:
        with pytest.raises(APIConnectionError):
            await routing.enable()
        assert not any(routing.failed.values()) and not routing.failed_providers
        assert session.client.model_capabilities.await_count == 1
    finally:
        await session.client.aclose()


@pytest.mark.parametrize("command", ["/optimal on", "/optimal reset", "/optimal off"])
async def test_optimal_command_at_tool_boundary_keeps_task_running(optimal_session, monkeypatch, command):
    import asyncio
    from slipagent import cli
    from slipagent.tools.base import Tool, ToolResult
    session, responses, requests = optimal_session
    entered, release = asyncio.Event(), asyncio.Event()

    class GatedTool(Tool):
        name = "gated"
        description = "Wait for the test."
        parameters = {"type": "object", "properties": {}}

        async def run(self):
            entered.set()
            await release.wait()
            return ToolResult.ok("The tool ran once.")

    session.registry.register(GatedTool())
    responses.extend([
        {"tool_calls": [{"id": "one", "type": "function", "function": {
            "name": "gated", "arguments": "{}"}}]},
        {"content": "Finished."},
    ])
    async def read(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(cli, "_read_line", read)
    routing = session.model_routing()
    await routing.enable()
    await routing.fallback(APIResponseError("Unavailable", status_code=400))
    task = asyncio.create_task(cli._run_request(session, "Complete this task", session.renderer.style))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await cli._handle_command(session, command)
        assert not session.agent.stop_requested
        assert session.agent.model == "second"
        release.set()
        assert not await asyncio.wait_for(task, 5)
        assert [request["model"] for request in requests] == ["second", "second" if command.endswith("off") else "first"]
        assert not session.agent.stopped
        assert not session.work.deferred_commands
        assert "Complete this task" in json.dumps(requests[-1]["messages"])
        assert "The tool ran once." in json.dumps(requests[-1]["messages"])
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await session.agent._compactor().aclose()
        await session.client.aclose()


async def test_all_failed_routes_stay_stopped_until_reset(optimal_session):
    session, responses, requests = optimal_session
    responses.extend([400, 400, {"content": "Reset worked."}])
    try:
        await session.model_routing().enable()
        with pytest.raises(APIConfigError, match="exhausted"):
            await session.agent.run("Work")
        with pytest.raises(APIConfigError, match="exhausted"):
            await session.agent.run("Try the available models")
        assert [r["model"] for r in requests] == ["first", "second"]
        await session.model_routing().enable()
        assert await session.agent.run("Retry from the top") == "Reset worked."
        assert requests[-1]["model"] == "first"
    finally:
        await session.agent._compactor().aclose()
        await session.client.aclose()


async def test_failed_enable_cannot_run_the_previous_paid_model(optimal_session):
    session, responses, requests = optimal_session
    session.agent.model = "paid"
    session._catalog = [model for model in session._catalog if model.id == "paid"]
    responses.append({"content": "This must not be sent."})
    try:
        with pytest.raises(APIConfigError, match="exhausted"):
            await session.model_routing().enable()
        with pytest.raises(APIConfigError, match="exhausted"):
            await session.agent.run("Work")
        assert not requests
    finally:
        await session.agent._compactor().aclose()
        await session.client.aclose()


async def test_tool_errors_are_feedback_and_do_not_change_models(optimal_session):
    session, responses, requests = optimal_session
    tool = RecordingTool("Test command failed", is_error=True)
    session.registry.register(tool)
    responses.extend([
        {"tool_calls": [{"id": "one", "type": "function", "function": {
            "name": "record", "arguments": '{"value":"test"}'}}]},
        {"content": "The test failed; this is task feedback."},
    ])
    try:
        await session.model_routing().enable()
        await session.agent.run("Test this code")
        assert [r["model"] for r in requests] == ["first", "first"]
        assert "Test command failed" in json.dumps(requests[-1]["messages"])
        assert not session.model_routing().failed["working"]
    finally:
        await session.agent._compactor().aclose()
        await session.client.aclose()


async def test_provider_auth_failure_skips_its_other_models(optimal_session, monkeypatch):
    import slipagent.sessionstate as sessionstate
    session, responses, requests = optimal_session
    monkeypatch.setenv("KILO_API_KEY", "dummy-kilo")
    session._catalog.append(entry("kilo", "alternate", 3))
    replacements = []
    captured = []

    def factory(provider, **options):
        def handle(request):
            captured.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "Recovered"}, "finish_reason": "stop"}]})
        client = create_client(provider, **options, transport=httpx.MockTransport(handle))
        monkeypatch.setattr(client, "model_capabilities", AsyncMock(return_value=RequestProfile(
            parameters={"tools", "max_tokens", "temperature"}, context_length=1_000_000)))
        replacements.append(client)
        return client

    monkeypatch.setattr(sessionstate, "create_client", factory)
    responses.append(401)
    try:
        await session.model_routing().enable()
        original = session.client
        assert await session.agent.run("Inspect") == "Recovered"
        assert [r["model"] for r in requests] == ["first"]
        assert session.client.provider == "kilo" and original._client.is_closed
        assert captured[0].headers["authorization"] == "Bearer dummy-kilo"
        assert session.model_routing().failed_providers == {"openrouter"}
        session.agent.reset()
        assert not session.model_routing().failed_providers
    finally:
        await session.agent._compactor().aclose()
        await session.client.aclose()


async def test_credentials_change_clears_only_that_provider_failures(optimal_session, monkeypatch):
    session, _, _ = optimal_session
    monkeypatch.setenv("KILO_API_KEY", "old-dummy")
    routing = session.model_routing()
    try:
        await routing.candidates("working")
        routing.failed["working"].update({"openrouter::first", "kilo::alternate"})
        routing.failed["compression"].add("kilo::alternate")
        routing.failed_providers.add("kilo")
        monkeypatch.setenv("KILO_API_KEY", "new-dummy")
        await routing.candidates("working")
        assert routing.failed["working"] == {"openrouter::first"}
        assert not routing.failed["compression"] and not routing.failed_providers
    finally:
        await session.client.aclose()


async def test_billed_google_account_is_excluded_from_free_automation(optimal_session, monkeypatch):
    session, _, _ = optimal_session
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-google")
    session._catalog.append(ModelInfo("gemini-3.8-flash", provider="google", pricing={"free_tier": True}))
    try:
        assert any(m.provider == "google" for m in await session.model_routing().candidates("working"))
        monkeypatch.setenv("GEMINI_BILLING_TIER", "paid")
        for purpose in ("working", "compression"):
            assert all(m.provider != "google" for m in await session.model_routing().candidates(purpose))
    finally:
        await session.client.aclose()


@pytest.mark.parametrize("selection", [None, "openrouter::paid"])
@pytest.mark.parametrize("command", ["model", "models"])
async def test_model_pickers_disable_optimal_only_after_selection(optimal_session, monkeypatch, command, selection):
    from slipagent import cli
    session, _, _ = optimal_session
    monkeypatch.setattr(type(session), "catalog", AsyncMock(return_value=session._catalog))
    try:
        await session.model_routing().enable()
        session.renderer.terminal = SimpleNamespace(choose=AsyncMock(return_value=selection))
        handler = cli._model_command if command == "model" else cli._models_command
        await handler(session, "", session.renderer.style, session.renderer.stream)
        assert session.model_routing().enabled is (selection is None)
        assert session.agent.model == ("first" if selection is None else "paid")
    finally:
        session.renderer.terminal = None
        await session.client.aclose()


@pytest.mark.parametrize("explicit", [[], ["--provider", "openrouter"],
                                       ["--model", "nvidia/nemotron-3-ultra-550b-a55b:free"]])
async def test_startup_defaults_to_best_rank_unless_explicitly_selected(tmp_path, monkeypatch, explicit):
    from slipagent import cli
    from slipagent.config import DEFAULT_MODEL
    requests = []
    best = "dots-studio/dots-3-note-preview:free"
    properties = {"context_length": 1_000_000, "supported_parameters": ["tools", "max_tokens"],
                  "pricing": {"prompt": "0", "completion": "0"}}

    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{**properties, "id": model} for model in (DEFAULT_MODEL, best)]})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json={"data": {"endpoints": [{**properties, "tag": "stub"}]}})
        return httpx.Response(200, json={"data": {"is_free_tier": True}})

    monkeypatch.setattr(cli, "create_client", lambda provider, **options:
                        create_client(provider, **options, transport=httpx.MockTransport(handle)))
    monkeypatch.setattr("slipagent.environment.ProjectEnvironment.snapshot", AsyncMock(return_value={}))
    monkeypatch.setattr(cli.RepositoryMap, "snapshot", AsyncMock(return_value=""))
    args = cli.build_parser().parse_args(["--api-key", "dummy", "--workspace", str(tmp_path),
                                        "--no-mcp", "--no-reload", "--no-session", *explicit])
    session = await cli.build_session(args)
    try:
        assert session.model_routing().enabled is (not explicit)
        assert session.agent.model == (DEFAULT_MODEL if explicit else best)
        assert all(request.method == "GET" for request in requests)
        assert sum(request.url.path.endswith("/models") for request in requests) == 1
        if not explicit:
            await session.model_routing().fallback(APIResponseError("unusable", status_code=400))
            assert session.agent.model == DEFAULT_MODEL
            assert session.model_routing().failed["working"]
    finally:
        await cli._shutdown(session)
    if not explicit:
        fresh = await cli.build_session(args)
        try:
            assert fresh.agent.model == best
            assert not fresh.model_routing().failed["working"]
        finally:
            await cli._shutdown(fresh)


async def test_failed_startup_closes_automatically_selected_provider(tmp_path, monkeypatch):
    from slipagent import cli, providers, sessionstate
    from slipagent.config import DEFAULT_MODEL
    clients = []

    def client(provider, **options):
        def handle(request):
            assert request.method == "GET"
            if provider == "google":
                return httpx.Response(200, json={"models": [{
                    "name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"],
                    "inputTokenLimit": 1_000_000, "outputTokenLimit": 32_768,
                }]})
            return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL,
                "pricing": {"prompt": "0", "completion": "0"}}]})

        result = create_client(provider, **options, transport=httpx.MockTransport(handle))
        clients.append(result)
        return result

    def fail(session, module):
        assert session.client.provider == "google"
        raise RuntimeError("reload startup failed")

    for module in (cli, providers, sessionstate):
        monkeypatch.setattr(module, "create_client", client)
    monkeypatch.setattr(sessionstate, "fetch_catalog", providers.fetch_catalog)
    monkeypatch.setattr(cli, "RuntimeFrame", fail)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-google")
    monkeypatch.setattr("slipagent.environment.ProjectEnvironment.snapshot", AsyncMock(return_value={}))
    monkeypatch.setattr(cli.RepositoryMap, "snapshot", AsyncMock(return_value=""))
    args = cli.build_parser().parse_args(["--api-key", "dummy", "--workspace", str(tmp_path),
                                        "--no-mcp", "--no-session"])
    with pytest.raises(RuntimeError, match="reload startup failed"):
        await cli.build_session(args)
    assert {client.provider for client in clients} == {"openrouter", "google"}
    assert all(client._client.is_closed for client in clients)
