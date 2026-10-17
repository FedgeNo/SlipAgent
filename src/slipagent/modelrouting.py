"""Session-owned eligibility and failed routes for working and compression models."""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

from .api import APIError, APIConfigError, APIConnectionError, APIResponseError, APITransportError, RETRYABLE_STATUS
from .context import ContextError
from .modelpricing import free_label, model_order
from .modelscores import MODEL_ROUTERS, reviewed_model
from .providers import provider_class
from .types import ModelInfo

if TYPE_CHECKING:
    from .sessionstate import Session


# Transient working-model failures skip the model for this long, not for the session.
WORKING_COOLDOWN = 300.0


def transient_error(error: Exception) -> bool:
    return isinstance(error, (APITransportError, TimeoutError)) or (
        isinstance(error, APIResponseError) and error.status_code in RETRYABLE_STATUS)


def eligible_free(model: ModelInfo) -> bool:
    # Google's free tier is account-specific; a paid key bills these same IDs.
    if (model.provider == "google" and model.pricing.get("free_tier") is True
            and os.environ.get("GEMINI_BILLING_TIER", "free").strip().lower() != "free"):
        return False
    return free_label(model.pricing) is not None


class ModelRouting:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.enabled = False
        self.failed: dict[str, set[str]] = {"working": set(), "compression": set()}
        self.failed_providers: set[str] = set()
        self.credentials: dict[str, tuple[str, str]] = {}
        self.notified: set[str] = set()
        self.catalog_lock = asyncio.Lock()
        self.revision = 0

    def reset(self, provider: str | None = None) -> None:
        """Forget exclusions without changing conversation or completed tool work."""
        self.revision += 1
        for failures in self.failed.values():
            if provider is None:
                failures.clear()
            else:
                failures.difference_update(value for value in tuple(failures) if value.startswith(provider + "::"))
        if provider is None:
            self.failed_providers.clear()
            self.cooling().clear()
        else:
            self.failed_providers.discard(provider)
            for selector in [selector for selector in self.cooling() if selector.startswith(provider + "::")]:
                del self.cooling()[selector]

    def cooling(self) -> dict[str, float]:
        """Working models skipped after a transient failure, with monotonic expiry times."""
        cooling: dict[str, float] = self.__dict__.setdefault("_cooling", {})
        now = time.monotonic()
        for selector in [selector for selector, until in cooling.items() if until <= now]:
            del cooling[selector]
        return cooling

    def notice(self, message: str) -> None:
        from .markdown import literal
        self.session.renderer.emit(literal(message))

    def catalog_notice(self, models: list[ModelInfo]) -> None:
        unknown = {model.selector for model in models if free_label(model.pricing) is not None
                   and model.score is None and model.id not in MODEL_ROUTERS
                   and not reviewed_model(model.provider, model.id)}
        new = unknown - self.notified
        if new:
            self.notified.update(new)
            self.notice("New free models have been released: " + ", ".join(sorted(new))
                        + ". Update SlipAgent for updated coding scores.")

    async def candidates(self, purpose: str, *, include_failed: bool = False) -> list[ModelInfo]:
        options = self.session.compression_options()
        # Match APIClient's effective endpoint. Selecting a provider fills in
        # its default URL and strips trailing slashes; that is not a key change
        # and must not resurrect a route that has just failed.
        credentials = {name: (value["api_key"],
                              (value.get("base_url") or provider_class(name).default_base_url).rstrip("/"))
                       for name, value in options.items() if value.get("api_key", "").strip()}
        for name in self.credentials.keys() | credentials.keys():
            if self.credentials.get(name) != credentials.get(name):
                self.reset(name)
        self.credentials = credentials
        async with self.catalog_lock:
            models = await self.session.catalog()
        self.catalog_notice(models)
        ranked = sorted((model for model in models if model.score is not None
                         and model.provider in credentials and model.provider not in self.failed_providers
                         and eligible_free(model)), key=model_order)
        healthy = [model for model in ranked if model.selector not in self.failed[purpose]]
        if purpose == "working" and not include_failed:
            # Cooling models return once every other route is unavailable.
            cooling = self.cooling()
            healthy = [model for model in healthy if model.selector not in cooling] or healthy
        return ranked if include_failed else healthy

    def reject_working(self, model: ModelInfo, error: Exception) -> None:
        if transient_error(error):
            self.cooling()[model.selector] = time.monotonic() + WORKING_COOLDOWN
        self.reject(model, "working", error)

    def reject(self, model: ModelInfo, purpose: str, error: Exception) -> None:
        if isinstance(error, APIResponseError) and error.status_code == 401:
            self.failed_providers.add(model.provider)
        elif not transient_error(error):
            self.failed[purpose].add(model.selector)

    async def enable(self, *, cached: bool = False) -> None:
        self.reset()
        self.enabled = True
        await self.select_working(cached=cached)

    async def skip(self) -> None:
        """Select the next ranked model; the current one stays excluded until reset.

        With optimal off, this starts it normally from the highest score.
        """
        if not self.enabled:
            await self.enable()
            return
        model = ModelInfo(self.session.agent.model, provider=self.session.client.provider)
        self.failed["working"].add(model.selector)
        self.notice(f"Optimal skipped {model.selector}.")
        await self.select_working()

    async def prepare_working(self) -> None:
        """Keep exhausted or newly ineligible routes stopped between user turns."""
        if self.enabled:
            candidates = await self.candidates("working")
            if not any(model.provider == self.session.client.provider and model.id == self.session.agent.model
                       for model in candidates):
                await self.select_working()

    async def select_working(self, *, cached: bool = False) -> None:
        """Select the best healthy candidate; `cached` uses saved profiles when present."""
        candidates = await self.candidates("working")
        for model in candidates:
            if model.provider in self.failed_providers:
                continue
            try:
                await self.session.select_model(model, automatic=True, cached=cached)
            except APIConnectionError:
                # A failed metadata connection must not blacklist every model.
                raise
            except (APIError, ContextError) as exc:
                self.reject_working(model, exc)
                self.notice(f"Optimal skipped {model.selector}: {exc}")
                continue
            self.notice(f"Optimal selected {model.selector} (estimated coding score {model.score:.1f}/100).")
            return
        raise APIConfigError("Optimal exhausted the eligible ranked free models. "
                             "Use /optimal reset to retry from the top, or /model to select a model.")

    async def fallback(self, error: Exception) -> bool:
        if not self.enabled:
            return False
        if isinstance(error, APIConnectionError):
            self.notice("Connection retries exhausted; keeping the current model. Retry when the connection recovers.")
            return False
        model = ModelInfo(self.session.agent.model, provider=self.session.client.provider)
        self.reject_working(model, error)
        self.notice(f"Optimal fallback from {model.selector}: {error}")
        await self.select_working()
        return True
