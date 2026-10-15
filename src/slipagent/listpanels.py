"""Selectable lists with stable keys and shared mouse/keyboard behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from .uiwindows import ListWindow

K = TypeVar("K")


@dataclass(frozen=True)
class ListActions(Generic[K]):
    """Explicit host operations; panels do not depend on a terminal instance."""

    activate: Callable[[K], object]
    invalidate: Callable[[], None]
    focused: Callable[[], bool]
    hover: Callable[[K], bool]
    set_hover: Callable[[K], None]


class ListPanel(ABC, Generic[K]):
    """Own selection and rendering; descendants supply items and activation."""

    def __init__(self, actions: ListActions[K]) -> None:
        self.actions = actions
        self.selected_key: K | None = None
        self.window = ListWindow(FormattedTextControl(
            self.fragments, focusable=True, show_cursor=False,
            get_cursor_position=lambda: Point(0, self.index),
        ))
        self.window._following = False

    @abstractmethod
    def keys(self) -> list[K]:
        raise NotImplementedError

    @abstractmethod
    def label(self, key: K) -> str:
        raise NotImplementedError

    def clickable(self, key: K) -> bool:
        return True

    @property
    def selection(self) -> K | None:
        keys = self.keys()
        return self.selected_key if self.selected_key in keys else keys[0] if keys else None

    @property
    def index(self) -> int:
        keys = self.keys()
        selected = self.selection
        return keys.index(selected) if selected is not None else 0

    def move(self, amount: int) -> None:
        keys = self.keys()
        if not keys:
            self.selected_key = None
            return
        self.selected_key = keys[max(0, min(len(keys) - 1, self.index + amount))]
        self.reveal()
        self.actions.invalidate()

    def reveal(self) -> None:
        self.window.reveal()

    def open_selected(self) -> None:
        key = self.selection
        if key is not None and self.clickable(key):
            self.actions.activate(key)

    def item_style(self, key: K) -> str:
        return "class:menu-item" + (" class:hover" if self.actions.focused() and key == self.selection else "")

    def item_text(self, key: K) -> str:
        return self.label(key)

    def fragments(self) -> StyleAndTextTuples:
        keys = self.keys()
        if not keys:
            return [("class:status", "None yet")]
        fragments: StyleAndTextTuples = []
        for index, key in enumerate(keys):
            if index:
                fragments.append(("", "\n"))
            style = self.item_style(key)
            if self.clickable(key):
                style += " class:clickable" + (" class:hover" if self.actions.hover(key) else "")
                def click(event: MouseEvent, key: K = key) -> object:
                    if event.event_type == MouseEventType.MOUSE_MOVE:
                        self.actions.set_hover(key)
                    elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                        self.selected_key = key
                        self.open_selected()
                    return None
                fragments.append((style, self.item_text(key), click))
            else:
                fragments.append((style, self.item_text(key)))
        return fragments


class TouchedPanel(ListPanel[str]):
    """Project paths retain their insertion order and directory-open policy."""

    def __init__(self, items: Mapping[str, str], root: Callable[[], Path | None],
                 can_open_folder: Callable[[], bool], actions: ListActions[str]) -> None:
        self.items = items
        self.root = root
        self.can_open_folder = can_open_folder
        super().__init__(actions)

    def keys(self) -> list[str]:
        return list(reversed(list(self.items)))

    def label(self, key: str) -> str:
        status = self.items[key]
        root = self.root()
        try:
            if root is not None and (root / key).is_dir():
                status = "dir"
        except OSError:
            pass
        return f"{key} [{status}]"

    def clickable(self, key: str) -> bool:
        root = self.root()
        try:
            directory = root is not None and (root / key).is_dir()
        except OSError:
            directory = False
        return not directory or self.can_open_folder()


class CallsPanel(ListPanel[tuple[int, str]]):
    """Tool invocations are keyed by durable step and call identity."""

    def __init__(self, items: Mapping[tuple[int, str], dict[str, Any]], actions: ListActions[tuple[int, str]]) -> None:
        self.items = items
        super().__init__(actions)

    def keys(self) -> list[tuple[int, str]]:
        return list(reversed(list(self.items)))

    def label(self, key: tuple[int, str]) -> str:
        item = self.items[key]
        return f"{item['tool']} [{item['status']}]"

    def item_style(self, key: tuple[int, str]) -> str:
        status = self.items[key]["status"]
        role = {"running": "call-running", "completed": "call-completed", "error": "call-error"}.get(status)
        return super().item_style(key) + (" class:" + role if role else "")


class ChoicePanel(ListPanel[int]):
    """Modal choices share selection behavior with persistent sidebar lists."""

    def __init__(self, actions: ListActions[int]) -> None:
        self.options: list[tuple[str, str]] = []
        super().__init__(actions)

    def keys(self) -> list[int]:
        return list(range(len(self.options)))

    def label(self, key: int) -> str:
        return self.options[key][1]

    def item_style(self, key: int) -> str:
        return "class:menu-item" + (" class:menu-selected" if key == self.selection else "")

    def item_text(self, key: int) -> str:
        return ("› " if key == self.selection else "  ") + self.label(key)
