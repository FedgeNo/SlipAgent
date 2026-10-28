"""Resolve selectable API modules without creating connections at import time."""

from __future__ import annotations

from importlib import import_module
import base64
import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from .api import APIClient, APIConfigError
from .types import ModelInfo

PROVIDERS = {"openrouter": "OpenRouterClient", "nvidia": "NvidiaClient", "kilo": "KiloClient", "mistral": "MistralClient", "google": "GoogleClient", "groq": "GroqClient", "anthropic": "AnthropicClient"}


KEY_PREFIXES = {"openrouter": ("sk-or-v1-",), "nvidia": ("nvapi-",),
                "mistral": ("mstrl_",), "google": ("AIza", "AQ."), "groq": ("gsk_",), "anthropic": ("sk-ant-",)}


def key_provider(key: str) -> str | None:
    """Identify provider-specific key formats locally; authentication is separate."""
    prefixes = KEY_PREFIXES
    for provider, values in prefixes.items():
        if any(key.startswith(prefix) and len(key) > len(prefix) for prefix in values):
            return provider
    if len(key) > 16384 or not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", key):
        return None
    try:
        header, payload = [json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
                           for part in key.split(".")[:2]]
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    # Kilo's signed user tokens carry kiloUserId and a version. Decoding these
    # identifies the destination; only the provider can verify the signature.
    if (isinstance(header, dict) and header.get("alg") == "HS256"
            and isinstance(payload, dict) and isinstance(payload.get("kiloUserId"), str)
            and payload["kiloUserId"] and type(payload.get("version")) is int):
        return "kilo"
    return None


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
