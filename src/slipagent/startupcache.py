"""Last known provider catalogs and model profiles, so startup needs no network.

Interactive startup selects a model from these entries and refreshes them in the
background. Each provider entry is keyed by a fingerprint of its API key and base
URL, and entries expire after `MAX_AGE_SECONDS`; requests still verify the model.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .capabilities import ModelCapabilities, RequestProfile
from .config import data_directory, save_private_text
from .types import ModelInfo

CACHE_NAME = "startup-cache.json"
VERSION = 1
MAX_AGE_SECONDS = 7 * 24 * 3600
MAX_PROFILES = 20
_PROFILE_CLASSES: dict[str, type[RequestProfile]] = {
    "RequestProfile": RequestProfile, "ModelCapabilities": ModelCapabilities}


def cache_path() -> Path:
    return data_directory() / CACHE_NAME


def fingerprint(api_key: str, base_url: str) -> str:
    return hashlib.sha256(f"{api_key}\0{base_url}".encode()).hexdigest()[:24]


def _load() -> dict[str, Any]:
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return {}
    return data


def _save(data: dict[str, Any]) -> None:
    data["version"] = VERSION
    try:
        save_private_text(cache_path(), json.dumps(data, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _fresh(entry: Any, identity: str) -> bool:
    return (isinstance(entry, dict) and entry.get("identity") == identity
            and isinstance(entry.get("saved_at"), (int, float))
            and 0 <= time.time() - entry["saved_at"] < MAX_AGE_SECONDS)


def save_catalogs(models: Iterable[ModelInfo], identities: Mapping[str, str]) -> None:
    """Record the catalogs fetched successfully for these providers."""
    data = _load()
    catalogs = data.setdefault("catalogs", {})
    grouped: dict[str, list[dict[str, Any]]] = {provider: [] for provider in identities}
    for model in models:
        if model.provider in grouped:
            grouped[model.provider].append({**asdict(model), "score": None})
    for provider, entries in grouped.items():
        catalogs[provider] = {"identity": identities[provider], "saved_at": time.time(), "models": entries}
    _save(data)


def cached_catalog(identities: Mapping[str, str]) -> list[ModelInfo] | None:
    """Every requested provider's fresh catalog, or None when any is missing."""
    catalogs = _load().get("catalogs", {})
    models: list[ModelInfo] = []
    try:
        for provider, identity in identities.items():
            entry = catalogs.get(provider)
            if not _fresh(entry, identity):
                return None
            models.extend(ModelInfo(**{**model, "score": None}) for model in entry["models"])
    except (TypeError, ValueError):
        return None
    return models


def _encode(value: Any) -> Any:
    if isinstance(value, set):
        return {"set": sorted(value)}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"set"} and isinstance(value["set"], list):
        return set(value["set"])
    return value


def save_profile(model: ModelInfo, identity: str, profile: RequestProfile) -> None:
    kind = type(profile).__name__
    if _PROFILE_CLASSES.get(kind) is not type(profile):
        return
    from . import __version__
    data = _load()
    profiles = data.setdefault("profiles", {})
    # Profiles are rebuilt attribute by attribute, so a build change invalidates them.
    profiles[model.selector] = {"identity": identity, "saved_at": time.time(), "class": kind,
                                "build": __version__,
                                "state": {key: _encode(value) for key, value in vars(profile).items()}}
    for selector in sorted(profiles, key=lambda name: profiles[name].get("saved_at", 0))[:-MAX_PROFILES]:
        del profiles[selector]
    _save(data)


def cached_profile(model: ModelInfo, identity: str) -> RequestProfile | None:
    from . import __version__
    entry = _load().get("profiles", {}).get(model.selector)
    if not _fresh(entry, identity) or not isinstance(entry.get("state"), dict) or entry.get("build") != __version__:
        return None
    kind = _PROFILE_CLASSES.get(entry.get("class"))
    if kind is None:
        return None
    profile = kind.__new__(kind)
    profile.__dict__.update({key: _decode(value) for key, value in entry["state"].items()})
    if not isinstance(getattr(profile, "parameters", None), set) or not isinstance(getattr(profile, "context_length", None), int):
        return None
    return profile
