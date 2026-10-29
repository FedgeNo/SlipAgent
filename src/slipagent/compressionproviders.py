"""One provider rotation shared by every background compression attempt.

The selected coding provider serves only working requests; background jobs use
the other configured providers, at most one job per provider at a time.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from .api import APIClient
from .modelrouting import ModelRouting
from .providers import PROVIDERS, create_client
from .types import ModelInfo

COMPRESSION_UNAVAILABLE = (
    "No eligible free models are available for compression. Compression will not work "
    "until a provider with free models is added. Add OpenRouter or NVIDIA."
)


class CompressionUnavailableError(ValueError):
    """No configured free route can accept this compression job."""


class CompressionProviders:
    def __init__(self, models: ModelRouting) -> None:
        self.models = models
        self.last = ""
        self.clients: dict[tuple[str, str, str], APIClient] = {}

    async def background(self) -> list[ModelInfo]:
        """Ranked compression models on every provider except the coding provider."""
        ranked = await self.models.candidates("compression", include_failed=True)
        foreground = self.models.session.client.provider
        return [model for model in ranked if model.provider != foreground]

    async def slots(self) -> int:
        """How many background jobs may run at once: one per eligible provider."""
        return len({model.provider for model in await self.background()})

    async def select(self, excluded: set[str], *, failed_provider: str | None = None,
                     revision: int | None = None, busy: Counter[str] | None = None) -> tuple[APIClient, ModelInfo]:
        """Reserve an idle route, waiting when another job owns every eligible provider."""
        while True:
            ranked = await self.background()
            if revision is not None and revision != self.models.revision:
                excluded.clear()
                failed_provider = None
            session = self.models.session
            if not ranked:
                raise CompressionUnavailableError(COMPRESSION_UNAVAILABLE)
            failures = self.models.failed["compression"] | excluded
            healthy = [model for model in ranked if model.selector not in failures]
            available = healthy or ranked
            alternatives = [model for model in available if model.provider != failed_provider]
            if alternatives:
                available = alternatives
            available = [model for model in available if busy is None or busy[model.provider] <= 0]
            if available:
                break
            released = self._availability()
            released.clear()
            await released.wait()
        # Advance before any network await so concurrent turn/group jobs share
        # the same order. Each provider contributes its best remaining model.
        providers = list(PROVIDERS)
        start = (providers.index(self.last) + 1) % len(providers) if self.last in providers else 0
        order = providers[start:] + providers[:start]
        provider = next(name for name in order if any(model.provider == name for model in available))
        self.last = provider
        model = next(model for model in available if model.provider == provider)
        settings = {**session.compression_options()[provider], "on_retry": None}
        key = (provider, settings["api_key"], settings.get("base_url") or "")
        client = self.clients.get(key)
        if client is None:
            client = create_client(provider, **settings)
            self.clients[key] = client
        if busy is not None:
            busy[provider] += 1
        return client, model

    def _availability(self) -> asyncio.Event:
        # Existing routing owners acquire this synchronization state after reload.
        released: asyncio.Event = self.__dict__.setdefault("_provider_released", asyncio.Event())
        return released

    def release(self, provider: str, busy: Counter[str]) -> None:
        busy[provider] -= 1
        self._availability().set()

    async def aclose(self) -> None:
        clients = list(self.clients.values())
        self.clients.clear()
        await asyncio.gather(*(client.aclose() for client in clients))
