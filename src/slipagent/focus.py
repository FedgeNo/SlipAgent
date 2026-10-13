"""Clockwise panel focus and restoration, independent of agent activity."""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.layout import walk
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
        primary = self.primary()
        if primary is not self.prompt:
            panels.append(primary)
        if self.app().output.get_size().columns >= 60:
            panels.extend(self.sidebars)
        current = self.app().layout.current_control
        index = next((i for i, panel in enumerate(panels) if self.control(panel) is current), -1)
        self.focus(panels[(index + 1) % len(panels)])

    def restore(self, previous: UIControl) -> None:
        """Restore an invoking control only while its region is available."""
        if previous is self.prompt.control:
            self.focus(self.prompt)
            return
        for target in (self.primary(), *self.sidebars):
            if self.control(target) is previous and self.visible(target):
                self.focus(target)
                return
        self.restore_primary()

    def visible(self, target: FocusTarget) -> bool:
        if target in self.sidebars and self.app().output.get_size().columns < 60:
            return False
        control = self.control(target)
        return any(isinstance(window, Window) and window.content is control
                   for window in walk(self.app().layout.container, skip_hidden=True))

    def reconcile(self) -> None:
        current = self.app().layout.current_control
        if current in (window.content for window in self.sidebars):
            window = next(window for window in self.sidebars if window.content is current)
            if not self.visible(window):
                self.restore_primary()

    def preserve(self, layout: Layout, previous: UIControl) -> None:
        """Keep accessible sidebar or draft focus when rebuilding presentation."""
        if previous in (window.content for window in self.sidebars):
            if any(isinstance(window, Window) and window.content is previous
                   for window in walk(layout.container, skip_hidden=True)) and self.app().output.get_size().columns >= 60:
                layout.focus(previous)
            else:
                self.restore_primary()
        elif previous is self.prompt.control and self.prompt_active:
            layout.focus(self.prompt)
