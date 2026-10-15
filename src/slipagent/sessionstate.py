"""Live session state and provider selection, separate from terminal dispatch."""

from __future__ import annotations

import asyncio
import os
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
from .types import ModelInfo
from .workspace import Workspace
from .work import SessionWork

if TYPE_CHECKING:
    from .cli import Renderer


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
            compactor.routing = CompressionProviders(lambda: self.client.provider, self.compression_options)

    def compression_options(self) -> dict[str, dict[str, Any]]:
        """Include remembered session credentials and configured provider endpoints."""
        from .providers import PROVIDERS
        result = dict(self.extensions.get("provider_options", {}))
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

    async def catalog(self, refresh: bool = False) -> list[ModelInfo]:
        """Combined provider catalogs, refreshed on explicit model selection."""
        keys = {name: options["api_key"] for name, options in self.extensions.get("provider_options", {}).items()}
        keys[self.client.provider] = self.client.api_key
        names = active_providers(keys=keys)
        if self._catalog is None or refresh or self.extensions.get("catalog_providers", names) != names:
            async def catalog_for(provider: str) -> list[ModelInfo]:
                if provider == self.client.provider:
                    return await self.client.list_models()
                implementation = provider_class(provider)
                return await fetch_catalog(provider, base_url=os.environ.get(implementation.base_url_env),
                                           on_retry=self.renderer.retry_countdown)
            results = await asyncio.gather(*(catalog_for(name) for name in names), return_exceptions=True)
            catalog: list[ModelInfo] = []
            errors: list[str] = []
            for name, result in zip(names, results):
                if isinstance(result, BaseException):
                    if not isinstance(result, (APIError, ConfigError)):
                        raise result
                    errors.append(f"{name}: {result}")
                else:
                    catalog.extend(result)
            self.extensions["catalog_errors"] = errors
            if not catalog and errors:
                raise APIError("; ".join(errors))
            self._catalog = catalog
            self.extensions["catalog_providers"] = names
        return coding_models(self._catalog)

    def use_model(self, slug: str) -> None:
        """Switch the model for subsequent steps."""
        self.agent.model = slug.strip()

    async def use_api_key(self, key: str) -> None:
        """Swap in a new API key, rebuilding the transport that holds it."""
        await self.agent.wait_for_compaction()
        replacement = create_client(
            self.client.provider,
            api_key=key,
            base_url=self.base_url,
            http_referer=self.http_referer,
            app_title=self.app_title,
            transport=self.transport,
            **self.client.replacement_options(),
        )
        await self.client.aclose()
        self.client = replacement
        self.agent.client = replacement
        self.api_key = key
        self._catalog = None
        self.free_calls = None
        self.quota_checked_at = None

    async def select_model(self, model: ModelInfo) -> None:
        """Preflight the candidate before replacing any active provider state."""
        replacement = self.client
        if model.provider != self.client.provider:
            options = self.extensions.get("provider_options", {}).get(model.provider)
            if options is None:
                config = Config.from_env(provider=model.provider, model=model.id)
                options = {"api_key": config.api_key, "base_url": config.base_url,
                           "http_referer": config.http_referer, "app_title": config.app_title,
                           "on_retry": self.renderer.retry_countdown}
            replacement = create_client(model.provider, **options)
        try:
            capabilities = await replacement.model_capabilities(model.id, refresh=True, store=False)
            if capabilities is None:
                raise APIConfigError("The API did not supply capabilities to verify this model's compatibility.")
            await self.agent._context_view(self.registry.specs(), 0, capabilities=capabilities, preview=True,
                                           candidate_model=model.id)
            if replacement is not self.client:
                # Pending jobs use the old client; originals survive cancellation.
                self.agent._compactor().reset()
                await self.agent.wait_for_compaction()
        except BaseException:
            if replacement is not self.client:
                await finish_cleanup(asyncio.create_task(replacement.aclose()))
            raise
        replacement.cache_capabilities(model.id, capabilities)
        previous = self.client
        if previous is not replacement:
            self.extensions.setdefault("provider_options", {})[previous.provider] = {
                "api_key": self.api_key, "base_url": self.base_url,
                "http_referer": self.http_referer, "app_title": self.app_title,
                **previous.replacement_options(),
            }
        self.client = replacement
        self.agent.client = replacement
        self.api_key = replacement.api_key
        self.base_url = replacement.base_url
        self.use_model(model.id)
        if previous is not replacement:
            self.transport = None
            self.free_calls = None
            self.quota_checked_at = None
            await finish_cleanup(asyncio.create_task(previous.aclose()))
