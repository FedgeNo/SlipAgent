"""One provider rotation shared by every background compression attempt."""

from __future__ import annotations

from .api import APIClient
from .modelpricing import price_label
from .modelrouting import ModelRouting
from .providers import PROVIDERS, create_client
from .types import ModelInfo


class CompressionUnavailableError(ValueError):
    """No configured free route can accept this compression job."""


class CompressionProviders:
    def __init__(self, models: ModelRouting) -> None:
        self.models = models
        self.last = ""
        self.clients: dict[tuple[str, str, str], APIClient] = {}
        self.single_provider_notified = False

    async def select(self, excluded: set[str], *, failed_provider: str | None = None,
                     revision: int | None = None) -> tuple[APIClient, ModelInfo]:
        ranked = await self.models.candidates("compression", include_failed=True)
        if revision is not None and revision != self.models.revision:
            excluded.clear()
            failed_provider = None
        session = self.models.session
        working = next((model for model in session._catalog or []
                        if model.provider == session.client.provider and model.id == session.agent.model), None)
        paid = working is not None and price_label(working.pricing).startswith("$")
        if not paid:
            others = [model for model in ranked if model.provider != session.client.provider]
            if others:
                ranked = others
        if not ranked:
            raise CompressionUnavailableError("no eligible ranked free compression models; check provider keys or update SlipAgent")
        if len({model.provider for model in ranked}) == 1 and len(self.models.credentials) == 1:
            if not self.single_provider_notified:
                self.models.notice("One provider is available for coding and compression. "
                                   "Add more provider API keys to spread requests and improve availability.")
                self.single_provider_notified = True
        failures = self.models.failed["compression"] | excluded
        healthy = [model for model in ranked if model.selector not in failures]
        available = healthy or ranked
        alternatives = [model for model in available if model.provider != failed_provider]
        if alternatives:
            available = alternatives
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
        return client, model

    async def aclose(self) -> None:
        import asyncio
        clients = list(self.clients.values())
        self.clients.clear()
        await asyncio.gather(*(client.aclose() for client in clients))
