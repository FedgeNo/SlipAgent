"""API key verification and the `/key` command.

Regression coverage for a real incident: `GET /models` is public and does not
authenticate, so validating a new key against it accepted any string and the
command happily overwrote the stored credential with garbage.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.cli import Session, Style, _key_command, _mask_key, _replace_key
from slipagent.config import dotenv_path
from slipagent.openrouter import OpenRouterAuthError, OpenRouterClient
from slipagent.tools import ToolRegistry
from slipagent.types import KeyInfo

GOOD_KEY = "sk-or-v1-" + "a" * 64
BAD_KEY = "show"

KEY_PAYLOAD = {
    "data": {
        "label": "sk-or-v1-aaa...fa7",
        "limit": 0,
        "limit_remaining": 0,
        "usage": 0,
        "is_free_tier": False,
        "free_model_daily_requests": {"used": 88, "limit": 1000, "remaining": 912},
        "expires_at": "2027-10-01T21:43:00.001Z",
    }
}


def client_for(handler: Any, key: str = GOOD_KEY) -> OpenRouterClient:
    return OpenRouterClient(api_key=key, transport=httpx.MockTransport(handler))


def make_session(client: OpenRouterClient, registry: ToolRegistry) -> Session:
    agent = Agent(client=client, registry=registry, model="test/model")
    return Session(
        agent=agent,
        registry=registry,
        client=client,
        renderer=None,  # type: ignore[arg-type]
        workspace=None,  # type: ignore[arg-type]
        api_key=client.api_key,
        base_url=client.base_url,
        http_referer=None,
        app_title="slipagent",
        transport=client._client._transport,
    )


def accepting_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/key"):
        return httpx.Response(200, json=KEY_PAYLOAD)
    return httpx.Response(200, json={"data": []})


def good_key_only(request: httpx.Request) -> httpx.Response:
    """Accepts GOOD_KEY, 401s anything else, on every endpoint."""
    if request.headers.get("Authorization") != f"Bearer {GOOD_KEY}":
        return httpx.Response(
            401, json={"error": {"message": "User not found.", "code": 401}}
        )
    if request.url.path.endswith("/key"):
        return httpx.Response(200, json=KEY_PAYLOAD)
    return httpx.Response(200, json={"data": [{"id": "some/model"}]})


def rejecting_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/key"):
        return httpx.Response(
            401, json={"error": {"message": "User not found.", "code": 401}}
        )
    # /models is public and answers regardless of the token.
    return httpx.Response(200, json={"data": [{"id": "some/model"}]})


# --------------------------------------------------------------------------- #
# key_info
# --------------------------------------------------------------------------- #


async def test_key_info_parses_limits_and_quota() -> None:
    client = client_for(accepting_handler)
    info = await client.key_info()

    assert info.label == "sk-or-v1-aaa...fa7"
    assert info.limit == 0
    assert info.limit_remaining == 0
    assert info.cannot_reach_paid_models is True
    assert info.free_quota is not None
    assert info.free_quota.remaining == 912
    assert info.free_quota.limit == 1000


async def test_key_info_rejects_bad_key() -> None:
    client = client_for(rejecting_handler, BAD_KEY)

    with pytest.raises(OpenRouterAuthError):
        await client.key_info()


async def test_key_info_is_an_authenticated_endpoint() -> None:
    """The regression: /models answers a bad key, /key must not."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Authorization") != f"Bearer {GOOD_KEY}":
            return httpx.Response(401, json={"error": {"message": "nope"}})
        return httpx.Response(200, json=KEY_PAYLOAD)

    good = client_for(handler, GOOD_KEY)
    bad = client_for(handler, BAD_KEY)

    assert (await good.key_info()).limit == 0
    with pytest.raises(OpenRouterAuthError):
        await bad.key_info()


def test_cannot_reach_paid_models_false_when_limited() -> None:
    assert KeyInfo(limit=25.0, is_free_tier=False).cannot_reach_paid_models is False
    assert KeyInfo(limit=None).cannot_reach_paid_models is False
    assert KeyInfo(limit=0, is_free_tier=True).cannot_reach_paid_models is False


# --------------------------------------------------------------------------- #
# _mask_key
# --------------------------------------------------------------------------- #


def test_mask_key_hides_the_middle() -> None:
    masked = _mask_key(GOOD_KEY)

    assert GOOD_KEY not in masked
    assert masked.startswith("sk-or-v1-a")
    assert masked.endswith(GOOD_KEY[-4:])


def test_mask_key_fully_hides_short_values() -> None:
    assert _mask_key("show") == "****"
    assert _mask_key("") == ""


# --------------------------------------------------------------------------- #
# /key command
# --------------------------------------------------------------------------- #


async def test_key_show_reports_limits(capsys: pytest.CaptureFixture[str]) -> None:
    client = client_for(accepting_handler)
    session = make_session(client, ToolRegistry())
    out = io.StringIO()

    await _key_command(session, "show", Style(enabled=False), out)

    text = out.getvalue()
    assert "$0.00 limit" in text
    assert "912/1000 free-model requests left today" in text
    assert "$0 spend limit" in text


async def test_invalid_key_is_rejected_and_original_kept() -> None:
    client = client_for(rejecting_handler)
    session = make_session(client, ToolRegistry())
    session.api_key = GOOD_KEY
    out = io.StringIO()

    await _key_command(session, BAD_KEY, Style(enabled=False), out)

    assert "rejected" in out.getvalue()
    assert session.api_key == GOOD_KEY


async def test_valid_key_is_accepted() -> None:
    client = client_for(good_key_only, BAD_KEY)
    session = make_session(client, ToolRegistry())
    out = io.StringIO()

    await _replace_key(session, Style(enabled=False), out, candidate=GOOD_KEY)

    assert "key accepted" in out.getvalue()
    assert session.api_key == GOOD_KEY


async def test_bad_key_never_reaches_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rejected key must leave the stored credential untouched."""
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENROUTER_API_KEY={GOOD_KEY}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    client = client_for(rejecting_handler)
    session = make_session(client, ToolRegistry())
    session.api_key = GOOD_KEY
    out = io.StringIO()

    await _key_command(session, BAD_KEY, Style(enabled=False), out)

    assert env_file.read_text() == f"OPENROUTER_API_KEY={GOOD_KEY}\n"


async def test_accepted_key_is_not_saved_without_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Validation passing must not imply permission to overwrite .env."""
    env_file = tmp_path / ".env"
    old_key = "sk-or-v1-" + "c" * 64
    env_file.write_text(f"OPENROUTER_API_KEY={old_key}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # non-tty

    client = client_for(good_key_only, old_key)
    session = make_session(client, ToolRegistry())
    session.api_key = old_key
    out = io.StringIO()

    await _key_command(session, GOOD_KEY, Style(enabled=False), out)

    assert "key accepted" in out.getvalue()
    assert "not saved" in out.getvalue()
    # The file still holds the OLD key: swapping in memory is not persistence.
    assert env_file.read_text() == f"OPENROUTER_API_KEY={old_key}\n"


async def test_key_command_refuses_to_prompt_without_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    client = client_for(accepting_handler)
    session = make_session(client, ToolRegistry())
    out = io.StringIO()

    await _key_command(session, "", Style(enabled=False), out)

    # Non-tty: shows status rather than consuming piped input.
    assert "free quota" in out.getvalue()
    assert session.api_key == GOOD_KEY


async def test_stored_key_survives_a_failed_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENROUTER_API_KEY={GOOD_KEY}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    client = client_for(rejecting_handler)
    session = make_session(client, ToolRegistry())
    session.api_key = GOOD_KEY
    out = io.StringIO()

    await _key_command(session, BAD_KEY, Style(enabled=False), out)

    assert session.api_key == GOOD_KEY
    assert GOOD_KEY in env_file.read_text()
    assert dotenv_path(tmp_path) == env_file