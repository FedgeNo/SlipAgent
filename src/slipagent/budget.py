"""Conservative token calibration for the selected working-model request profile."""

from __future__ import annotations

import json
from typing import Any

from .context import estimate_tokens


class ContextBudget:
    def __init__(self) -> None:
        self.model = ""
        self.profile: Any = None
        self.scale = 1.0
        self.fraction = 1.0

    def select(self, model: str, profile: Any) -> float:
        # Refreshing endpoint metadata invalidates measurements for old routing.
        if self.model != model or self.profile is not profile:
            self.model, self.profile, self.scale = model, profile, 1.0
            self.fraction = 1.0
        return self.scale

    def observe(self, request: str, prompt_tokens: int) -> None:
        if prompt_tokens <= 0 or not request:
            return
        body = json.loads(request)
        estimate = sum(estimate_tokens(json.dumps(message, ensure_ascii=False))
                       for message in body.get("messages", []))
        for key in ("tools", "response_format"):
            if key in body:
                estimate += estimate_tokens(json.dumps(body[key]))
        if estimate:
            # Measurements can tighten the configured limits, never relax them.
            # Each subsequent view is counted anew after selection/compaction.
            self.scale = max(self.scale, prompt_tokens / estimate * 1.1)
