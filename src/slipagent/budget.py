"""Conservative token calibration for the selected working-model request profile."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from .context import estimate_tokens

CALIBRATION_WINDOW = 5


class ContextBudget:
    def __init__(self) -> None:
        self.model = ""
        self.profile: Any = None
        self.scale = 1.0
        self.fraction = 1.0
        self.measurements: deque[float] = deque(maxlen=CALIBRATION_WINDOW)

    def select(self, model: str, profile: Any) -> float:
        # Refreshing endpoint metadata invalidates measurements for old routing.
        if self.model != model or self.profile is not profile:
            self.model, self.profile, self.scale = model, profile, 1.0
            self.fraction = 1.0
            self.measurements = deque(maxlen=CALIBRATION_WINDOW)
        return self.scale

    def observe(self, request: dict[str, Any] | str, prompt_tokens: int) -> None:
        if prompt_tokens <= 0 or not request:
            return
        body = json.loads(request) if isinstance(request, str) else request
        estimate = sum(estimate_tokens(json.dumps(message, ensure_ascii=False))
                       for message in body.get("messages", []))
        for key in ("tools", "response_format"):
            if key in body:
                estimate += estimate_tokens(json.dumps(body[key]))
        if estimate:
            # Older live instances acquire the window at their next observation.
            if not hasattr(self, "measurements"):
                self.measurements = deque(maxlen=CALIBRATION_WINDOW)
            self.measurements.append(prompt_tokens / estimate)
            self.scale = sum(self.measurements) / len(self.measurements) * 1.1
