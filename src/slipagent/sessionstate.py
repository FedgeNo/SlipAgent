"""Live session state and provider selection, separate from terminal dispatch."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .agent import Agent
from .api import APIClient, APIError, APIConfigError
from .config import Config, ConfigError
from .lifecycle import finish_cleanup
from .mcp import MCPManager
from .model_catalog import coding_models
from .providers import active_providers, create_client, provider_class, fetch_catalog
from .resources import SessionResources
from .runtime import RuntimeFrame
from .tools.base import ToolRegistry
from .types import KeyInfo, ModelInfo
from .workspace import Workspace
from .work import SessionWork
from . import startupcache
from .tools.blocking import run_blocking

if TYPE_CHECKING:
    from .cli import Renderer
    from .modelrouting import ModelRouting

CATALOG_REFRESH_SECONDS = 300.0


@dataclass(slots=True)
class Session:
    agent: Agent
    registry: ToolRegistry
    client: APIClient
    renderer: Renderer
    workspace: Workspace
    api_key: str
    base_url: str
    http_referer: str | None
    app_title: str
    # Carried so rebuilt clients (e.g. after `/key`) keep the same transport.
    transport: Any | None = None
    mcp: MCPManager | None = None
    # Free-model calls left on the key, for the status bar. None until the
    # first successful quota fetch.
    free_calls: int | None = None
    quota_checked_at: float | None = None
    _catalog: list[ModelInfo] | None = None
    _input_future: asyncio.Future[str | None] | None = None
    reloader: RuntimeFrame | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    work: SessionWork = field(init=False)

    def __post_init__(self) -> None:
        self.work = SessionWork(self.extensions)

    def configure_compaction(self) -> None:
        from .compressionproviders import CompressionProviders
        compactor = self.agent._compactor()
        if getattr(compactor, "routing", None) is None:
            compactor.routing = CompressionProviders(self.model_routing())

    def model_routing(self) -> ModelRouting:
        from .modelrouting import ModelRouting
        routing: ModelRouting | None = self.extensions.get("model_routing")
        if routing is None:
            routing = ModelRouting(self)
            self.extensions["model_routing"] = routing
            self.registry.services["model_fallback"] = routing.fallback
            self.registry.services["prepare_model_request"] = routing.prepare_working
            self.registry.services["reset_model_routing"] = routing.reset
        return routing

    def compression_options(self) -> dict[str, dict[str, Any]]:
        """Include remembered session credentials and configured provider endpoints."""
        from .providers import PROVIDERS
        result = dict(self.extensions.get("provider_options", {}))
        result[self.client.provider] = {
            "api_key": self.api_key, "base_url": self.base_url, "transport": self.transport,
            "http_referer": self.http_referer, "app_title": self.app_title,
            **self.client.replacement_options(),
        }
        for name in PROVIDERS:
            if name in result or name == self.client.provider:
                continue
            implementation = provider_class(name)
            key = os.environ.get(implementation.key_env, "").strip()
            if key:
                result[name] = {"api_key": key, "base_url": os.environ.get(implementation.base_url_env),
                                "http_referer": self.http_referer, "app_title": self.app_title}
        return result

    @property
    def resources(self) -> SessionResources:
        return self.registry.resources

    def _provider_keys(self) -> dict[str, str]:
        keys = {name: options["api_key"] for name, options in self.extensions.get("provider_options", {}).items()}
        if hasattr(self.client, "provider"):
            keys[self.client.provider] = self.client.api_key
        return keys

    def provider_identity(self, provider: str) -> str:
        """Fingerprint the key and endpoint a provider's cached metadata belongs to."""
        implementation = provider_class(provider)
        if provider == getattr(self.client, "provider", None):
            key, base_url = self.client.api_key, self.client.base_url
        else:
            options = self.extensions.get("provider_options", {}).get(provider, {})
            key = self._provider_keys().get(provider) or os.environ.get(implementation.key_env, "")
            base_url = options.get("base_url") or os.environ.get(implementation.base_url_env) or implementation.default_base_url
        return startupcache.fingerprint(key, (base_url or implementation.default_base_url).rstrip("/"))

    async def use_cached_catalog(self) -> bool:
        """Adopt every active provider's saved catalog without a network request."""
        names = active_providers(keys=self._provider_keys())
        identities = {name: self.provider_identity(name) for name in names}
        cached = await run_blocking(startupcache.cached_catalog, identities) if names else None
        if not cached:
            return False
        self._catalog = cached
        self.extensions["catalog_providers"] = names
        self.extensions["catalog_checked_at"] = time.monotonic()
        self.extensions["catalog_errors"] = []
        return True

    async def catalog(self, refresh: bool = False) -> list[ModelInfo]:
        """Current provider inventory, cached for at most five minutes."""
        from .modelrouting import transient_error
        keys = self._provider_keys()
        names = active_providers(keys=keys)
        # Existing sessions acquire a cache timestamp on their first call after
        # reload. Availability comes only from catalogs; the curated score table
        # keeps historical models without manufacturing selectable endpoints.
        checked_at = self.extensions.setdefault("catalog_checked_at", time.monotonic())
        expired = time.monotonic() - checked_at >= CATALOG_REFRESH_SECONDS
        if self._catalog is None or refresh or expired or self.extensions.get("catalog_providers", names) != names:
            async def catalog_for(provider: str) -> list[ModelInfo]:
                if provider == self.client.provider:
                    return await self.client.list_models()
                implementation = provider_class(provider)
                options = self.extensions.get("provider_options", {}).get(provider, {})
                return await fetch_catalog(provider, api_key=keys.get(provider),
                                           base_url=options.get("base_url", os.environ.get(implementation.base_url_env)),
                                           on_retry=self.renderer.retry_countdown)
            results = await asyncio.gather(*(catalog_for(name) for name in names), return_exceptions=True)
            catalog: list[ModelInfo] = []
            errors: list[str] = []
            for name, result in zip(names, results):
                if isinstance(result, BaseException):
                    if not isinstance(result, (APIError, ConfigError)):
                        raise result
                    errors.append(f"{name}: {result}")
                    if transient_error(result):
                        # Temporary metadata failures supply no evidence of removal.
                        catalog.extend(model for model in self._catalog or [] if model.provider == name)
                else:
                    catalog.extend(result)
            self.extensions["catalog_errors"] = errors
            if not catalog and errors:
                raise APIError("; ".join(errors))
            fetched = [name for name, result in zip(names, results) if not isinstance(result, BaseException)]
            await run_blocking(startupcache.save_catalogs, catalog,
                               {name: self.provider_identity(name) for name in fetched})
            self._catalog = catalog
            self.extensions["catalog_providers"] = names
            self.extensions["catalog_checked_at"] = time.monotonic()
        models = coding_models(self._catalog)
        # Refresh curated values after a score-table reload without fetching
        # catalogs again. Preserve provider-supplied alias metadata for IDs that
        # are absent from our exact identity table.
        from .modelscores import model_score, reviewed_model
        for model in models:
            if reviewed_model(model.provider, model.id):
                model.score = model_score(model.provider, model.id)
        routing = self.extensions.get("model_routing")
        if routing is not None:
            routing.catalog_notice(models)
        return models

    def use_model(self, slug: str) -> None:
        """Switch the model for subsequent steps."""
        self.agent.model = slug.strip()

    async def use_api_key(self, key: str, *, provider: str | None = None,
                          verify: bool = False) -> KeyInfo | None:
        """Install a provider credential, preflighting before changing live state."""
        provider = provider or self.client.provider
        implementation = provider_class(provider)
        options = self.compression_options().get(provider)
        if options is None:
            options = {"base_url": os.environ.get(implementation.base_url_env),
                       "http_referer": self.http_referer, "app_title": self.app_title,
                       "on_retry": self.client.on_retry}
        options = {**options, "api_key": key}
        replacement = create_client(provider, **options)
        try:
            info = await replacement.key_info() if verify else None
            if verify and info is None and getattr(implementation, "catalog_requires_auth", False):
                await replacement.list_models()
            self.agent._compactor().reset()
            await self.agent.wait_for_compaction()
        except BaseException:
            await finish_cleanup(asyncio.create_task(replacement.aclose()))
            raise
        previous = self.client
        options.update(api_key=replacement.api_key, base_url=replacement.base_url)
        self.extensions.setdefault("provider_options", {})[provider] = options
        if provider == previous.provider:
            self.client = replacement
            self.agent.client = replacement
            self.api_key = replacement.api_key
            self.free_calls = None
            self.quota_checked_at = None
        self._catalog = None
        self.model_routing().reset(provider)
        retired = previous if self.client is replacement else replacement
        await finish_cleanup(asyncio.create_task(retired.aclose()))
        return info

    def _provider_client(self, model: ModelInfo) -> APIClient:
        """A new client for another provider, using its session or environment settings."""
        options = self.extensions.get("provider_options", {}).get(model.provider)
        if options is None:
            config = Config.from_env(provider=model.provider, model=model.id)
            options = {"api_key": config.api_key, "base_url": config.base_url,
                       "http_referer": config.http_referer, "app_title": config.app_title,
                       "on_retry": self.renderer.retry_countdown}
        return create_client(model.provider, **options)

    async def refresh_profile(self, model: ModelInfo) -> None:
        """Fetch and save a model's current profile without changing the selection."""
        client = self.client if model.provider == self.client.provider else self._provider_client(model)
        try:
            capabilities = await client.model_capabilities(model.id, refresh=True, store=False)
            if capabilities is None:
                return
            await run_blocking(startupcache.save_profile, model,
                               startupcache.fingerprint(client.api_key, client.base_url), capabilities)
            if client is self.client and model.id == self.agent.model:
                client.cache_capabilities(model.id, capabilities)
        finally:
            if client is not self.client:
                await finish_cleanup(asyncio.create_task(client.aclose()))

    async def select_model(self, model: ModelInfo, *, automatic: bool = False, cached: bool = False) -> None:
        """Preflight the candidate before replacing any active provider state.

        With `cached`, a saved profile for this key and endpoint replaces the
        network capability check; requests still verify the model.
        """
        replacement = self.client if model.provider == self.client.provider else self._provider_client(model)
        try:
            capabilities = None
            fetched = False
            if cached:
                identity = startupcache.fingerprint(replacement.api_key, replacement.base_url)
                capabilities = await run_blocking(startupcache.cached_profile, model, identity)
            if capabilities is None:
                capabilities = await replacement.model_capabilities(model.id, refresh=True, store=False)
                fetched = True
            if capabilities is None:
                raise APIConfigError("The API did not supply capabilities to verify this model's compatibility.")
            await self.agent._context_view(self.registry.specs(), 0, capabilities=capabilities, preview=True,
                                           candidate_model=model.id)
            if replacement is not self.client and self.agent._compactor().routing is None:
                # Embedded callers can still use the working client for summaries.
                self.agent._compactor().reset()
                await self.agent.wait_for_compaction()
        except BaseException:
            if replacement is not self.client:
                await finish_cleanup(asyncio.create_task(replacement.aclose()))
            raise
        replacement.cache_capabilities(model.id, capabilities)
        if fetched:
            await run_blocking(startupcache.save_profile, model,
                               startupcache.fingerprint(replacement.api_key, replacement.base_url), capabilities)
        previous = self.client
        if previous is not replacement:
            self.extensions.setdefault("provider_options", {})[previous.provider] = {
                "api_key": self.api_key, "base_url": self.base_url,
                "http_referer": self.http_referer, "app_title": self.app_title,
                **previous.replacement_options(),
            }
        self.client = replacement
        self.agent.client = replacement
        limit = replacement.max_temperature
        if self.agent.temperature is not None and self.agent.temperature > limit:
            from .markdown import literal
            self.agent.temperature = limit
            self.renderer.emit(literal(f"Temperature lowered to {limit:g}, the {replacement.provider} maximum."))
        self.api_key = replacement.api_key
        self.base_url = replacement.base_url
        self.use_model(model.id)
        if not automatic:
            self.model_routing().enabled = False
        if previous is not replacement:
            self.transport = None
            self.free_calls = None
            self.quota_checked_at = None
            await finish_cleanup(asyncio.create_task(previous.aclose()))
