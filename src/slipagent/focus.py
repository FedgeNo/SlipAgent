"""Visible region and link focus, independent of agent activity."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.containers import AnyContainer, HSplit
from prompt_toolkit.layout.layout import walk
from prompt_toolkit.layout.controls import UIControl
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.widgets import TextArea
from wcwidth import width as display_width

from .links import ActionControl, ActionLink, link_key
from .uiwindows import TranscriptWindow

FocusStop = tuple[Window, object | None, Point, Callable[[MouseEvent], object] | None]

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

    @staticmethod
    def header(control: UIControl, target: Callable[[], FocusTarget], *, style: str = "") -> Window:
        """Associate a non-editable title with its focusable panel body."""
        window = Window(control, height=1, style=style)
        window.__dict__["_focus_target"] = target
        return window

    @classmethod
    def panel(cls, title: UIControl, body: AnyContainer, target: Callable[[], FocusTarget],
              *, style: str = "class:panel-title") -> HSplit:
        """Build a titled region whose title focuses its retained body."""
        return HSplit([cls.header(title, target, style=style), body])

    def mouse_target_current(self, point: Point) -> bool:
        """Reject hit maps left behind by a panel change before the next redraw."""
        app = self.app()
        screen = app.renderer._last_screen
        if screen is None:
            return False
        live = set(walk(app.layout.container, skip_hidden=True))
        for window, position in screen.visible_windows_to_write_positions.items():
            if (position.xpos <= point.x < position.xpos + position.width
                    and position.ypos <= point.y < position.ypos + position.height):
                if window not in live:
                    return False
                target = window.__dict__.get("_focus_target")
                if target is not None:
                    panel = target()
                    if (panel.window if isinstance(panel, TextArea) else panel) not in live:
                        return False
        return True

    def click(self, point: Point) -> None:
        """Focus before dispatch so native input handlers also place the cursor."""
        if not self.mouse_target_current(point):
            self.app().invalidate()
            return
        screen = self.app().renderer._last_screen
        if screen is None:
            return
        for window, position in reversed(list(screen.visible_windows_to_write_positions.items())):
            if not (position.xpos <= point.x < position.xpos + position.width and
                    position.ypos <= point.y < position.ypos + position.height):
                continue
            target = window.__dict__.get("_focus_target")
            panel = target() if target is not None else window
            control = self.control(panel)
            if not control.is_focusable():
                continue
            self.focus(self.prompt if control is self.prompt.control else panel)
            control.__dict__.pop("_keyboard_key", None)
            control.__dict__.pop("_keyboard_point", None)
            if isinstance(control, ActionControl):
                info = window.render_info
                source = info.visible_line_to_row_col.get(point.y - position.ypos) if info is not None else None
                if source is not None and info is not None:
                    column = source[1]
                    cells = 0
                    for fragment in info.ui_content.get_line(source[0]):
                        end = cells + max(0, display_width(fragment[1]))
                        if cells <= point.x - position.xpos < end and len(fragment) == 3:
                            control.__dict__["_keyboard_key"] = link_key(fragment[2])
                            control.__dict__["_keyboard_point"] = Point(column, source[0])
                            break
                        column += len(fragment[1])
                        cells = end
            self.app().invalidate()
            return

    def restore_prompt(self) -> None:
        if any(window.content is self.prompt.control for window, _, _, _ in self.stops()):
            self.focus(self.prompt)
        else:
            self.restore_primary()

    def stops(self, *, components: bool = False) -> list[FocusStop]:
        """Walk components and indexed feed actions without scanning scrollback."""
        stops: list[FocusStop] = []
        size = self.app().output.get_size()
        for window in walk(self.app().layout.container, skip_hidden=True):
            if not isinstance(window, Window) or not window.content.is_focusable():
                continue
            if window in self.sidebars and size.columns < 60:
                continue
            control = window.content
            actions = isinstance(control, ActionControl)
            if not actions:
                stops.append((window, None, Point(0, 0), None))
            if not actions and not control.__dict__.get("_keyboard_links", False):
                continue
            if components and not actions:
                continue
            info = window.render_info
            width = info.window_width if info is not None else size.columns
            height = info.window_height if info is not None else size.rows
            if width <= 0 or height <= 0:
                continue
            content = control.create_content(width, height)
            rows: Iterable[int]
            if actions:
                rows = info.displayed_lines if info is not None else range(min(content.line_count, height))
            elif "focus_rows" in content.__dict__:
                rows = content.__dict__["focus_rows"]
            elif info is not None:
                rows = info.displayed_lines
            else:
                start = max(0, content.line_count - height) if getattr(window, "_following", False) else window.vertical_scroll
                rows = range(start, min(content.line_count, start + height))
            seen: list[object] = []
            for row in rows:
                column = 0
                cells = 0
                for fragment in content.get_line(row):
                    if len(fragment) == 3 and fragment[1].strip() and (not actions or cells < width):
                        handler = fragment[2]
                        owner = getattr(handler, "__self__", None)
                        key = link_key(handler)
                        if key not in seen and (not isinstance(owner, ActionLink) or owner.enabled()):
                            seen.append(key)
                            stops.append((window, key, Point(column, row), handler))
                    column += len(fragment[1])
                    cells += max(0, display_width(fragment[1]))
        prompt_visible = any(window.content is self.prompt.control for window, _, _, _ in stops)
        if self.editor() is not None or prompt_visible:
            # Keep the draft at one stable position when editor actions replace
            # its footer. Otherwise their changing order can trap traversal.
            stops = [stop for stop in stops if stop[0].content is not self.prompt.control]
            stops.insert(0, (self.prompt.window, None, Point(0, 0), None))
        return stops

    def travel(self, direction: int = 1, *, components: bool = False) -> None:
        stops = self.stops(components=components)
        if not stops:
            return
        current = self.app().layout.current_control
        selected = current.__dict__.get("_keyboard_key")
        if components and current.__dict__.get("_keyboard_links", False):
            selected = None
        index = next((i for i, (window, key, _, _) in enumerate(stops)
                      if window.content is current and key == selected), -1 if direction > 0 else 0)
        window, key, point, _ = stops[(index + direction) % len(stops)]
        # The editor replaces the draft visually until the prompt is focused.
        if window.content is self.prompt.control:
            self.focus(self.prompt)
        else:
            self.focus(window)
        window.content.__dict__.pop("_keyboard_key", None)
        window.content.__dict__.pop("_keyboard_point", None)
        if key is not None:
            window.content.__dict__["_keyboard_key"] = key
            window.content.__dict__["_keyboard_point"] = point
            if window.content.__dict__.get("_keyboard_links", False) and isinstance(window, TranscriptWindow):
                window.reveal_row(point.y)

    def activate(self) -> None:
        current = self.app().layout.current_control
        selected = current.__dict__.get("_keyboard_key")
        for window, key, point, handler in self.stops():
            if window.content is current and key == selected and handler is not None:
                handler(MouseEvent(point, MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
                return

    def restore(self, previous: UIControl) -> None:
        """Restore an invoking control only while its region is available."""
        if previous is self.prompt.control:
            self.focus(self.prompt)
            return
        for target in (self.primary(), *self.sidebars):
            if self.control(target) is previous and self.visible(target):
                self.focus(target)
                return
        for window in walk(self.app().layout.container, skip_hidden=True):
            if isinstance(window, Window) and window.content is previous:
                self.focus(window)
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
        elif isinstance(current, ActionControl):
            if not any(window.content is current and key == current.__dict__.get("_keyboard_key")
                       for window, key, _, _ in self.stops()):
                self.restore_prompt()

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
        elif any(isinstance(window, Window) and window.content is previous
                 for window in walk(layout.container, skip_hidden=True)):
            layout.focus(previous)
