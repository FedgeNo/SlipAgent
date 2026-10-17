"""Word-sized action links; list rows supply their own selection hit areas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.application.current import get_app
from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
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
        style = "class:link " + self.style + " class:clickable"
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


def link_key(handler: Callable[..., object]) -> object:
    """Identify freshly rendered links by their action rather than object identity."""
    owner = getattr(handler, "__self__", None)
    return owner.key if isinstance(owner, ActionLink) else handler


def keyboard_content(control: UIControl, content: UIContent) -> UIContent:
    """Highlight only the keyboard-selected action in the focused control."""
    selected = control.__dict__.get("_keyboard_key")
    if selected is None or get_app().layout.current_control is not control:
        return content

    def get_line(index: int) -> StyleAndTextTuples:
        result: StyleAndTextTuples = []
        for fragment in content.get_line(index):
            if len(fragment) == 3 and link_key(fragment[2]) == selected:
                result.append((fragment[0] + " class:keyboard-focus", fragment[1], fragment[2]))
            else:
                result.append(fragment)
        return result

    result = UIContent(get_line=get_line, line_count=content.line_count,
                       cursor_position=content.cursor_position, menu_position=content.menu_position,
                       show_cursor=False)
    for key in ("focus_rows", "is_current"):
        if key in content.__dict__:
            result.__dict__[key] = content.__dict__[key]
    return result


class ActionControl(FormattedTextControl):
    """Expose formatted mouse actions to the shared keyboard focus route."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.update(focusable=True, show_cursor=False)
        super().__init__(*args, **kwargs)

    def create_content(self, width: int, height: int | None) -> UIContent:
        original = super().create_content(width, height)
        # FormattedTextControl normally strips handlers from UIContent; retain
        # them so focus and mouse activation share the exact rendered actions.
        lines = list(split_lines(self._get_formatted_text_cached()))
        content = UIContent(get_line=lambda index: lines[index], line_count=len(lines),
                            cursor_position=original.cursor_position, show_cursor=False)
        selected = self.__dict__.get("_keyboard_point")
        if isinstance(selected, Point):
            content.cursor_position = selected
        return keyboard_content(self, content)
