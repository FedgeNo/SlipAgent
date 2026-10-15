"""Rotate background compression across configured, non-working providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .api import APIClient
from .capabilities import RequestProfile
from .providers import PROVIDERS, create_client

ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class CompressionUnavailableError(ValueError):
    """No configured provider can run compression independently of the agent."""


class CompressionProviders:
    def __init__(self, selected: Callable[[], str], options: Callable[[], dict[str, dict[str, Any]]]) -> None:
        self.selected = selected
        self.options = options
        self.last = ""
        self.clients: dict[tuple[str, str, str], APIClient] = {}

    async def select(self) -> tuple[APIClient, str, RequestProfile | None, int]:
        options = self.options()
        available = [name for name in PROVIDERS if name != self.selected()
                     and options.get(name, {}).get("api_key", "").strip()]
        if not available:
            raise CompressionUnavailableError("compression needs an API key for a provider other than the selected provider")
        index = (available.index(self.last) + 1) % len(available) if self.last in available else 0
        provider = available[index]
        self.last = provider
        settings = {**options[provider], "on_retry": None}
        key = (provider, settings["api_key"], settings.get("base_url") or "")
        client = self.clients.get(key)
        if client is None:
            client = create_client(provider, **settings)
            self.clients[key] = client
        model = ULTRA_MODEL if provider == "nvidia" else ULTRA_MODEL + ":free"
        profile = await client.model_capabilities(model)
        if profile is None:
            raise CompressionUnavailableError(f"{provider} did not provide capabilities for {model}")
        return client, model, profile, min(8192, profile.max_completion_tokens or 8192)

    async def aclose(self) -> None:
        import asyncio
        clients = list(self.clients.values())
        self.clients.clear()
        await asyncio.gather(*(client.aclose() for client in clients))
