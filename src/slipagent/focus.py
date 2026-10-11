"""Clockwise panel focus and restoration, independent of agent activity."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import UIControl
from prompt_toolkit.widgets import TextArea

FocusTarget = TextArea | Window


class FocusCoordinator:
    """Keep focus changes separate from the editor's hold on agent work."""

    def __init__(self, app: Callable[[], Application[None]], prompt: TextArea,
                 editor: Callable[[], TextArea | None], primary: Callable[[], FocusTarget],
                 sidebars: tuple[Window, Window], reveal: Callable[[], None]) -> None:
        self.app = app
        self.prompt = prompt
        self.editor = editor
        self.primary = primary
        self.sidebars = sidebars
        self.reveal = reveal
        self.prompt_active = False

    @staticmethod
    def control(target: FocusTarget) -> UIControl:
        return target.control if isinstance(target, TextArea) else target.content

    def focus(self, target: FocusTarget) -> None:
        self.prompt_active = target is self.prompt
        self.app().layout.focus(target)
        if target in self.sidebars:
            self.reveal()

    def restore_primary(self) -> None:
        self.focus(self.primary())

    def cycle(self) -> None:
        panels: list[FocusTarget] = [self.prompt]
        editor = self.editor()
        if editor is not None:
            panels.append(editor)
        panels.extend(self.sidebars)
        current = self.app().layout.current_control
        index = next((i for i, panel in enumerate(panels) if self.control(panel) is current), -1)
        self.focus(panels[(index + 1) % len(panels)])

    def preserve(self, layout: Layout, previous: UIControl) -> None:
        """Keep accessible sidebar or draft focus when rebuilding presentation."""
        if previous in (window.content for window in self.sidebars):
            if previous in layout.find_all_controls():
                layout.focus(previous)
        elif previous is self.prompt.control and self.prompt_active:
            layout.focus(self.prompt)
