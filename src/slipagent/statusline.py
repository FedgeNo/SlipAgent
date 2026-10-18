"""Status priority, expiring notices, rotating tips, and bounded text display."""

from __future__ import annotations

import time
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.layout.utils import explode_text_fragments
from wcwidth import iter_graphemes, width as display_width

PULSE_FRAMES = ("🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘")
NOTICE_SECONDS = 30
TIP_SECONDS = 15
NAVIGATION_TIP = "Shift+Tab to select UI component."


class StatusLine:
    """Retain transient display state independently of terminal layout rebuilds."""

    def __init__(self) -> None:
        self.notification = ""
        self.changed_at: float | None = None
        self.retry = ""
        self.retry_spinning = False
        self.command = ""
        self.navigation_until = 0.0
        self.tips_started: float | None = None
        self.marquees: dict[str, tuple[tuple[str, int], float]] = {}

    @classmethod
    def from_legacy(cls, state: dict[str, Any]) -> StatusLine:
        """Move retained fields once when an older terminal accepts this generation."""
        result = cls()
        for old, new in (("_notification", "notification"), ("_status_changed_at", "changed_at"),
                         ("_retry_status", "retry"), ("_command_status", "command"),
                         ("_navigation_tip_until", "navigation_until"), ("_tips_started_at", "tips_started"),
                         ("_marquee_states", "marquees")):
            if old in state:
                setattr(result, new, state.pop(old))
        return result

    def notify(self, text: str) -> None:
        self.notification = " ".join(text.split())
        self.changed_at = time.monotonic() if self.notification else None

    def current_notification(self) -> str:
        if self.notification:
            if self.changed_at is None:
                self.changed_at = time.monotonic()
            elif time.monotonic() - self.changed_at >= NOTICE_SECONDS:
                self.notify("")
        return self.notification

    def navigation(self) -> None:
        self.navigation_until = time.monotonic() + TIP_SECONDS

    def navigation_tip(self) -> str:
        return NAVIGATION_TIP if time.monotonic() < self.navigation_until else ""

    def tip(self, choices: list[str]) -> str:
        now = time.monotonic()
        if self.tips_started is None:
            self.tips_started = now
        return choices[int((now - self.tips_started) // TIP_SECONDS) % len(choices)]

    def notice_style(self, visible: bool) -> str:
        style = "class:status"
        if visible and self.changed_at is not None:
            elapsed = time.monotonic() - self.changed_at
            if 0 <= elapsed < .9 and int(elapsed / .15) % 2 == 0:
                style += " class:status-flash"
        return style

    def activity(self, *, held: bool, working: bool, stopping: bool,
                 concise: bool, frame: str) -> tuple[str, str, bool]:
        """Return style, text, and whether the idle Commands action is available."""
        if held:
            return "class:held", "Agent Held", False
        if self.retry and not stopping:
            # A model that is still working keeps its pulse beside the countdown.
            spinning = getattr(self, "retry_spinning", False)
            return "class:retry", f"{frame} {self.retry}" if spinning else self.retry, False
        if self.command:
            return "class:pulse", self.command, False
        if not working:
            return "class:idle", "Ready", True
        label = "Stopping After This Step" if stopping else "Working" if concise else "Working (esc to interrupt)"
        return "class:pulse", f"{frame} {label}", False

    def marquee(self, fragments: StyleAndTextTuples, columns: int, key: str) -> StyleAndTextTuples:
        """Scroll overflowing fragments while retaining graphemes and link handlers."""
        text = fragment_list_to_text(fragments)
        if columns <= 0 or display_width(text) <= columns:
            self.marquees.pop(key, None)
            return fragments if columns > 0 else []
        identity = text
        for frame in PULSE_FRAMES:
            if identity.startswith(frame):
                identity = identity[len(frame):]
                break
        signature = (identity, columns)
        now = time.monotonic()
        if key not in self.marquees or self.marquees[key][0] != signature:
            self.marquees[key] = (signature, now)
        expanded = explode_text_fragments([*fragments, ("", "   ")])
        clusters = []
        position = 0
        for cluster in iter_graphemes(text + "   "):
            end = position + len(cluster)
            clusters.append((max(0, display_width(cluster)), expanded[position:end]))
            position = end
        elapsed = (now - self.marquees[key][1]) % (1 + len(clusters) * .2)
        offset = int(max(0, elapsed - 1) / .2)
        result: StyleAndTextTuples = []
        remaining = columns
        for width, parts in clusters[offset:] + clusters[:offset]:
            if width > remaining:
                result.append(("", " " * remaining))
                break
            result.extend(parts)
            remaining -= width
            if not remaining:
                break
        return result
