"""Shared selection and geometry for the terminal's main content area."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.layout.containers import AnyContainer

from .focus import FocusTarget
from .uiwindows import TranscriptWindow

if TYPE_CHECKING:
    from .terminal import TerminalUI


class MainArea:
    def __init__(self, terminal: TerminalUI) -> None:
        self.terminal = terminal

    @property
    def inspector_visible(self) -> bool:
        return bool(self.terminal.__dict__.get("_inspection_visible")) and self.terminal.output.get_size().columns >= 60

    @property
    def inspector_width(self) -> int:
        return max(1, self.terminal.output.get_size().columns // 4)

    @staticmethod
    def split_width(columns: int) -> int:
        return columns - max(1, columns // 4) - 1

    @property
    def width(self) -> int:
        columns = self.terminal.output.get_size().columns
        return self.split_width(columns) if self.inspector_visible else max(1, columns - 1)

    def in_inspector(self, column: int) -> bool:
        return self.inspector_visible and column >= self.terminal.output.get_size().columns - self.inspector_width

    @property
    def mode(self) -> str:
        ui = self.terminal
        return "editor" if ui._editor_active else "viewer" if ui._viewer_active else "context" if ui.context_visible else "feed"

    def focus_target(self) -> FocusTarget:
        ui = self.terminal
        targets: dict[str, FocusTarget] = {"editor": ui._editor_input, "viewer": ui._viewer_window,
                                          "context": ui.context_window, "feed": ui.transcript}
        return targets[self.mode]

    def scroll_target(self) -> TranscriptWindow | None:
        target = self.focus_target()
        return target if isinstance(target, TranscriptWindow) else None

    def container(self, context: AnyContainer, feed: AnyContainer) -> AnyContainer:
        ui = self.terminal
        return {"editor": ui._editor_container, "viewer": ui._viewer_container,
                "context": context, "feed": feed}[self.mode]
