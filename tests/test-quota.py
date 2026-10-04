"""Quota updates follow the wall clock while the REPL remains usable."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from slipagent import cli
from slipagent.agent import Agent
from slipagent.openrouter import OpenRouterAuthError, OpenRouterClient
from slipagent.tools import ToolRegistry
from slipagent.types import KeyInfo


def quota_payload(remaining):
    return {"data": {"free_model_daily_requests": {
        "used": 100 - remaining, "limit": 100, "remaining": remaining,
    }}}


@pytest.fixture
def quota_session(workspace):
    client = OpenRouterClient(api_key="test-key", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=quota_payload(80)),
    ))
    registry = ToolRegistry()
    return cli.Session(
        agent=Agent(client=client, registry=registry, model="test/model"),
        registry=registry, client=client,
        renderer=cli.Renderer(cli.Style(False), io.StringIO(), verbose=False),
        workspace=workspace, api_key=client.api_key, base_url=client.base_url,
        http_referer=None, app_title="slipagent", transport=client._client._transport,
    )


async def test_quota_updates_after_fifteen_minutes_and_can_increase(quota_session, monkeypatch):
    session = quota_session
    clock = [1000.0]
    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    info = AsyncMock(side_effect=[
        KeyInfo.from_api(quota_payload(80)["data"]),
        KeyInfo.from_api(quota_payload(100)["data"]),
    ])
    monkeypatch.setattr(session.client, "key_info", info)
    try:
        await cli._refresh_quota(session)
        assert session.free_calls == 80
        clock[0] += 899
        await cli._refresh_quota(session)
        assert info.await_count == 1
        clock[0] += 1
        await cli._refresh_quota(session)
        assert session.free_calls == 100
        assert info.await_count == 2
    finally:
        await session.client.aclose()


@pytest.mark.parametrize("working", [False, True])
async def test_repl_refreshes_quota_without_input_or_waiting_for_model_and_stops_on_exit(
    quota_session, monkeypatch, working,
):
    session = quota_session
    refreshed = asyncio.Event()
    calls = []

    async def key_info():
        calls.append(len(calls))
        if len(calls) >= 2:
            refreshed.set()
        return KeyInfo.from_api(quota_payload(80 if len(calls) == 1 else 100)["data"])

    first = True

    async def read_line(session, style):
        nonlocal first
        if working and first:
            first = False
            return "go"
        await refreshed.wait()
        return None

    async def run(self, prompt):
        await refreshed.wait()
        return "done"

    monkeypatch.setattr(cli, "QUOTA_REFRESH_INTERVAL", .02, raising=False)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO())
    monkeypatch.setattr(cli, "_read_line", read_line)
    monkeypatch.setattr(session.client, "key_info", key_info)
    monkeypatch.setattr(Agent, "run", run)
    async with asyncio.timeout(2):
        assert await cli.run_repl(session) == 0
    assert session.free_calls == 100
    count = len(calls)
    await asyncio.sleep(.05)
    assert len(calls) == count
    assert session.client._client.is_closed


async def test_quota_refresh_ignores_response_for_replaced_key(quota_session, monkeypatch):
    session = quota_session
    started = asyncio.Event()
    finish = asyncio.Event()

    async def old_key_info():
        started.set()
        await finish.wait()
        return KeyInfo.from_api(quota_payload(10)["data"])

    monkeypatch.setattr(session.client, "key_info", old_key_info)
    refresh = asyncio.create_task(cli._refresh_quota(session))
    try:
        await started.wait()
        await session.use_api_key("replacement-key")
        cli._cache_quota(session, KeyInfo.from_api(quota_payload(100)["data"]))
        finish.set()
        await refresh
        assert session.free_calls == 100
    finally:
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)
        await session.client.aclose()


async def test_successful_refresh_without_quota_clears_old_readout(quota_session, monkeypatch):
    session = quota_session
    session.free_calls = 80
    monkeypatch.setattr(session.client, "key_info", AsyncMock(return_value=KeyInfo()))
    try:
        await cli._refresh_quota(session)
        assert session.free_calls is None
    finally:
        await session.client.aclose()


async def test_failed_refresh_keeps_last_value_and_waits_for_next_interval(quota_session, monkeypatch):
    session = quota_session
    session.free_calls = 80
    clock = [1000.0]
    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    info = AsyncMock(side_effect=[
        OpenRouterAuthError("temporarily unavailable"),
        KeyInfo.from_api(quota_payload(100)["data"]),
    ])
    monkeypatch.setattr(session.client, "key_info", info)
    try:
        await cli._refresh_quota(session)
        assert session.free_calls == 80
        clock[0] += 899
        await cli._refresh_quota(session)
        assert info.await_count == 1
        clock[0] += 1
        await cli._refresh_quota(session)
        assert session.free_calls == 100
        assert info.await_count == 2
    finally:
        await session.client.aclose()


async def test_each_command_response_forms_one_output_block(quota_session):
    session = quota_session
    try:
        session.renderer.emit("previous output")
        await cli._handle_command(session, "/help")
        await cli._handle_command(session, "/cost")
        output = session.renderer.stream.getvalue()
        assert output.startswith("\nprevious output\n\n" + cli.HELP)
        assert cli.HELP + "\n\n  in=" in output
    finally:
        await session.client.aclose()
