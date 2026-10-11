"""Word-sized action links; list rows supply their own selection hit areas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType


@dataclass(frozen=True)
class LinkHover:
    current: Callable[[], object]
    update: Callable[[object], None]


class ActionLink(ABC):
    def __init__(self, label: str, key: object, hover: LinkHover, *, style: str = "",
                 enabled: Callable[[], bool] = lambda: True) -> None:
        self.label = label
        self.key = key
        self.hover = hover
        self.style = style
        self.enabled = enabled

    @abstractmethod
    def invoke(self) -> None:
        raise NotImplementedError

    def mouse(self, event: MouseEvent) -> object:
        if event.event_type == MouseEventType.MOUSE_MOVE:
            self.hover.update(self.key)
        elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and self.enabled():
            self.invoke()
        return None

    def fragments(self) -> StyleAndTextTuples:
        style = self.style + " class:clickable"
        if self.hover.current() == self.key:
            style += " class:hover"
        return [(style.strip(), self.label, self.mouse)]


class CallbackLink(ActionLink):
    def __init__(self, label: str, key: object, hover: LinkHover,
                 action: Callable[[], object], *, style: str = "",
                 enabled: Callable[[], bool] = lambda: True) -> None:
        super().__init__(label, key, hover, style=style, enabled=enabled)
        self.action = action

    def invoke(self) -> None:
        self.action()


class TaskLink(ActionLink):
    """Schedule asynchronous actions with an explicit application task owner."""

    def __init__(self, label: str, key: object, hover: LinkHover,
                 operation: Callable[[], Coroutine[Any, Any, None]],
                 spawn: Callable[[Coroutine[Any, Any, None]], object], *, style: str = "",
                 enabled: Callable[[], bool] = lambda: True) -> None:
        super().__init__(label, key, hover, style=style, enabled=enabled)
        self.operation = operation
        self.spawn = spawn

    def invoke(self) -> None:
        self.spawn(self.operation())
