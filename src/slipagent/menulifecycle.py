"""Focus and notification lifetime shared by footer menus and settings."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from prompt_toolkit.layout.containers import AnyContainer, HSplit, to_container
from prompt_toolkit.layout.controls import UIControl

from .focus import FocusCoordinator, FocusTarget


class MenuLifecycle:
    def __init__(self, terminal: Any) -> None:
        self.terminal = terminal

    @contextmanager
    def opened(self, target: FocusTarget, cleanup: Callable[[], None], hint: str = "") -> Iterator[None]:
        ui = self.terminal
        previous = ui.app.layout.current_control
        try:
            if hint:
                ui.notify(hint)
            ui._focus.focus(target)
            ui.app.invalidate()
            yield
        finally:
            cleanup()
            ui.notify("")
            ui._focus.restore(previous)
            ui.app.invalidate()

    def footer(self, title: UIControl, body: AnyContainer, target: Callable[[], FocusTarget], bottom: AnyContainer) -> HSplit:
        panel = FocusCoordinator.panel(title, body, target, style="")
        panel.children.append(to_container(bottom))
        panel.height = lambda: self.terminal._footer_rows()
        panel.style = "class:footer class:menu"
        return panel
