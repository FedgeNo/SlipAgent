"""Resolve selectable API modules without creating connections at import time."""

from __future__ import annotations

from importlib import import_module
import os
from collections.abc import Callable, Mapping
from typing import Any

from .api import APIClient, APIConfigError
from .types import ModelInfo

PROVIDERS = {"openrouter": "OpenRouterClient", "nvidia": "NvidiaClient", "kilo": "KiloClient", "mistral": "MistralClient", "google": "GoogleClient"}


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
                        api_key: str | None = None,
                        on_retry: Callable[[str, int], None] | None = None) -> list[ModelInfo]:
    """Authenticate catalogs only when the provider requires it."""
    implementation = provider_class(provider)
    key = ""
    if getattr(implementation, "catalog_requires_auth", False):
        key = api_key if api_key is not None else os.environ.get(implementation.key_env, "")
        if not key.strip():
            raise APIConfigError(f"Set {implementation.key_env} to list {provider} models.")
    async with create_client(provider, api_key=key, base_url=base_url, catalog_only=True, on_retry=on_retry) as client:
        return await client.list_models()
