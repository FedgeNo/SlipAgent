"""Resolve selectable API modules without creating connections at import time."""

from __future__ import annotations

from importlib import import_module
import os
from collections.abc import Callable, Mapping
from typing import Any

from .api import APIClient, APIConfigError
from .types import ModelInfo

PROVIDERS = {"openrouter": "OpenRouterClient", "nvidia": "NvidiaClient"}


def provider_class(provider: str) -> type[APIClient]:
    """Return a provider implementation by its configured name."""
    symbol = PROVIDERS.get(provider)
    if symbol is None:
        raise APIConfigError(f"Unknown API provider {provider!r}; choose {', '.join(PROVIDERS)}.")
    module = import_module(f".{provider}", package=__package__)
    implementation: type[APIClient] = getattr(module, symbol)
    return implementation


def create_client(provider: str = "openrouter", **options: Any) -> APIClient:
    """Create the selected provider; the caller owns its async lifetime."""
    return provider_class(provider)(**options)


def active_providers(environ: Mapping[str, str] | None = None, *,
                     keys: Mapping[str, str] | None = None) -> list[str]:
    """Activate only providers with a nonblank environment or session key."""
    env = os.environ if environ is None else environ
    overrides = keys or {}
    return [name for name in PROVIDERS
            if overrides.get(name, env.get(provider_class(name).key_env, "")).strip()]


async def fetch_catalog(provider: str, *, base_url: str | None = None,
                        on_retry: Callable[[str, int], None] | None = None) -> list[ModelInfo]:
    """Read a provider's public catalog without requiring or sending a key."""
    async with create_client(provider, api_key="", base_url=base_url, catalog_only=True, on_retry=on_retry) as client:
        return await client.list_models()
