"""Consistent free-model labels and ordering across catalog views."""

from __future__ import annotations

import math
from typing import Any

from .types import ModelInfo


def free_label(pricing: dict[str, Any]) -> str | None:
    if pricing.get("free_tier") is True:
        return "free tier"
    try:
        if all(float(pricing[name]) == 0 for name in ("prompt", "completion")) and all(
                float(pricing.get(name, 0)) == 0 for name in ("request", "internal_reasoning")):
            return "free"
    except (KeyError, TypeError, ValueError):
        pass
    return None


def price_label(pricing: dict[str, Any]) -> str:
    free = free_label(pricing)
    if free:
        return free
    try:
        prompt, completion = (float(pricing[name]) * 1e6 for name in ("prompt", "completion"))
        if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)):
            return "-"
    except (KeyError, TypeError, ValueError):
        return "-"
    return f"${prompt:.2f}/${completion:.2f}"


def model_order(model: ModelInfo) -> tuple[bool, int, str, str]:
    return (free_label(model.pricing) is None, model.rank or 1_000_000, model.id, model.provider)
