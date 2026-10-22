"""Fixed REPL footer and a scrolling transcript on interactive terminals."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Coroutine
from typing import TextIO
from pathlib import Path
from typing import Any

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, to_filter
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.input import Input
from prompt_toolkit.history import DummyHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import ColorDepth, Output, create_output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from wcwidth import iter_graphemes, width as display_width

from .transcript import TranscriptFile, WrappedTranscript
from .uiwindows import TranscriptScrollbar, TranscriptWindow
from .focus import FocusCoordinator, FocusTarget
from .links import ActionControl, ActionLink, CallbackLink, TaskLink, LinkHover, keyboard_content
from .submissions import Submission, cancelled
from .listpanels import TouchedPanel, CallsPanel, ChoicePanel, ListActions
from .palette import USER_BACKGROUND_COLOR, USER_PROMPT_STYLE
from .markdown import THEMES, MarkdownRenderer, code_blocks, literal
from .clipboard import clipboard_sequence, copy_to_clipboard
from .explorer import folder_opener, open_folder
from .viewerpanels import ViewerDeck, TextViewerPanel, DiffViewerPanel, FileViewerPanel, file_signature
from .inspection import IndexedText, field_snapshot, artifact_snapshot, build_snapshot
from .transcriptlayouts import TranscriptLayouts
from .types import ToolCall
from .diffcomments import attach_reference, diff_reference
from .statusline import NOTICE_SECONDS, PULSE_FRAMES, TIP_SECONDS, StatusLine
from .mainarea import MainArea
from .menulifecycle import MenuLifecycle
from .terminalbindings import build_bindings
from .terminalmouse import route_mouse
from .terminallayout import build_layout, build_style

FOOTER_ROWS = 7


def _key_supplied(text: str) -> bool:
    parts = text.split(None, 1)
    return (len(parts) == 2 and parts[0].lower() == "/key"
            and parts[1].strip().lower() not in {"show", "status"})


class SuggestedPrompt(str):
    """A feed action submitted verbatim, without slash-command dispatch."""


class QueuedSuggestion(Submission, SuggestedPrompt):
    """Retain the suggestion contract while waiting in the submission queue."""


MIN_REDRAW_INTERVAL = 1 / 30
# Animation frames; idle screens redraw only on events and display deadlines.
REFRESH_INTERVAL = .15
FEED_LINK_PREFIXES = ("class:artifact-", "class:file-", "class:directory-", "class:suggestion-")


class TranscriptControl(UIControl):
    """Ask for indexed rows without FormattedTextControl's whole-text scan."""

    def __init__(self, content: Callable[[int], UIContent], mouse: Callable[[MouseEvent], object] | None = None) -> None:
        self._content = content
        self._mouse = mouse

    def mouse_handler(self, mouse_event: MouseEvent) -> object:
        self.__dict__.setdefault("_mouse", None)
        return self._mouse(mouse_event) if self._mouse is not None else NotImplemented

    def create_content(self, width: int, height: int | None) -> UIContent:
        self.__dict__["_keyboard_links"] = self._mouse is not None
        return keyboard_content(self, self._content(width))

    def is_focusable(self) -> bool:
        return self._mouse is not None


class WordWrapInput(Processor):
    """Pad display rows before whole words without changing editable text."""

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        width = max(1, ti.width)
        text = fragment_list_to_text(ti.fragments)
        fragments = explode_text_fragments(ti.fragments)
        words = {match.start(): display_width(match.group()) for match in re.finditer(r"\S+", text)}
        result: StyleAndTextTuples = []
        source_positions: list[int] = []
        display_positions: list[int] = []
        column = 0
        index = 0
        for cluster in iter_graphemes(text):
            word_width = words.get(index, 0)
            if column and word_width <= width and column + word_width > width:
                padding = width - column
                result.append((fragments[index][0], " " * padding))
                display_positions.extend([index] * padding)
                column = 0
            start = len(display_positions)
            source_positions.extend(range(start, start + len(cluster)))
            display_positions.extend(range(index, index + len(cluster)))
            result.extend(fragments[index:index + len(cluster)])
            cluster_width = max(0, display_width(cluster))
            if column + cluster_width > width:
                column = 0
            column = (column + cluster_width) % width
            index += len(cluster)
        source_positions.append(len(display_positions))
        display_positions.append(len(text))
        return Transformation(
            result,
            source_to_display=lambda position: source_positions[min(max(position, 0), len(source_positions) - 1)],
            display_to_source=lambda position: display_positions[min(max(position, 0), len(display_positions) - 1)],
        )


class InputWindow(TranscriptWindow):
    """Allow wheel inspection without changing the draft or its cursor."""

    def scroll(self, rows: int) -> None:
        self.__dict__["_manual_scroll"] = True
        self._following = False
        super().scroll(rows)

    def _scroll(self, ui_content: UIContent, width: int, height: int) -> None:
        buffer = getattr(self.content, "buffer", None)
        state = (buffer.text, buffer.cursor_position, width) if buffer is not None else None
        if self.__dict__.get("_input_state") != state:
            self.__dict__["_manual_scroll"] = False
            self._pending_scroll = 0
        self.__dict__["_input_state"] = state
        if self.__dict__.get("_manual_scroll", False):
            super()._scroll(ui_content, width, height)
        else:
            Window._scroll(self, ui_content, width, height)


# Records per block laid out together while a restored feed fills upward. Small
# enough that the newest output shows within one chunk's layout time.
RESTORE_CHUNK_RECORDS = 16


def _append_record(flow: WrappedTranscript, record: tuple[Any, ...]) -> None:
    if record[0]:
        _, text, first, final, reasoning = record
        flow.append_markdown(text, first=first, final=final, reasoning=reasoning)
    else:
        _, text, first, indents, extra_style = record
        flow.append(text, first=first, continuation_indents=indents, extra_style=extra_style)


def _failure_location(exc: BaseException) -> str:
    """Where an exception came from, naming the plugin file when one is involved."""
    import traceback
    from .plugins import plugin_directory

    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return "before running"
    try:
        plugin_root = str(plugin_directory())
    except Exception:
        plugin_root = ""
    for frame in reversed(frames):
        if plugin_root and frame.filename.startswith(plugin_root):
            return f"in plugin {Path(frame.filename).name} line {frame.lineno}"
    frame = frames[-1]
    return f"at {Path(frame.filename).name} line {frame.lineno}"


class TerminalUI:
    """Persistent buffers/application with replaceable presentation methods.

    Original chunks are stored in a session file; wrapped rows are a disposable
    indexed file with a bounded memory cache. Keep drafts/history/queues alive
    during refresh, and isolate secret prompts from those ordinary buffers.
    """
    _selection_start: Point | None
    _selection_end: Point | None
    _scrollbar: TranscriptScrollbar
    context_window: TranscriptWindow

    def __init__(
        self, status: Callable[[int], str], stream: TextIO, *, color: bool = True,
        input: Input | None = None, output: Output | None = None,
        status_suffix: Callable[[], str] | None = None,
        project_directory: Path | None = None,
    ) -> None:
        self.output = output if output is not None else create_output(stdout=stream)
        self._status = status
        self._status_suffix = status_suffix
        self._project_directory = project_directory
        self._folder_opener = folder_opener() if project_directory is not None else None
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._raw_output = TranscriptFile()
        self._block_starts: set[int] = set()
        self._continuation_indents: dict[int, dict[int, int]] = {}
        self._flow: WrappedTranscript | None = None
        self._wrapped_columns = 0
        self._wrapped_count = 0
        self._line_count = 0
        self._eof = False
        self.working = False
        self.stopping = False
        self.pausing = False
        self._password = False
        self._input_label = "> "
        self.input = TextArea(
            height=5, multiline=False, wrap_lines=True,
            prompt=lambda: self._prompt_label(),
            password=Condition(lambda: self._password), accept_handler=lambda buffer: self._accept(buffer),
            style="class:user",
        )
        self.input.window = InputWindow(self.input.control)
        self.input.window.style = "class:text-area class:user"
        self.input.window.height = Dimension.exact(5)
        self.transcript = TranscriptWindow(TranscriptControl(lambda width: self._transcript_content(width)))
        self._ensure_transcript()
        self.app: Application[None] = Application(
            layout=self._layout(), key_bindings=self._bindings(),
            full_screen=True, input=input, output=self.output,
            mouse_support=True, before_render=lambda app: self._update_title(),
            after_render=lambda app: self._route_mouse_wheel(app),
            # Periodic redraws come from _refresh_loop only while something animates.
            min_redraw_interval=MIN_REDRAW_INTERVAL,
            color_depth=ColorDepth.DEPTH_24_BIT if color else ColorDepth.DEPTH_1_BIT,
            style=self._style(),
        )
        self._title_pulse_bound = True

    def _layout(self) -> Layout:
        return build_layout(self)

    def _ensure_inspection(self) -> None:
        if "_inspection_visible" in self.__dict__:
            if "_touched_items" not in self.__dict__:
                self.__dict__["_touched_items"] = self.__dict__.pop("_touched_files", {})
            for group, target in ((0, self._files_window), (2, self._calls_window)):
                header = self._inspection_panel.children[group].get_children()[0]
                header.__dict__["_focus_target"] = lambda target=target: target
            return
        self._inspection_visible = False
        self._touched_items: dict[str, str] = {}
        self._inspection_calls: dict[tuple[int, str], dict[str, Any]] = {}
        self._touched_panel = TouchedPanel(
            self._touched_items,
            root=lambda: self.__dict__.get("_project_directory"),
            can_open_folder=lambda: self.__dict__.get("_folder_opener") is not None,
            actions=ListActions(activate=lambda key: self.app.create_background_task(self._inspect_file(key)),
            invalidate=lambda: self.app.invalidate(),
            focused=lambda: self.app.layout.current_window is self._files_window,
            hover=lambda key: self.__dict__.get("_hovered_control") == ("inspection", True, key),
            set_hover=lambda key: self.__dict__.__setitem__("_hovered_control", ("inspection", True, key))),
        )
        self._calls_panel = CallsPanel(
            self._inspection_calls,
            actions=ListActions(activate=lambda key: self.app.create_background_task(self._inspect_call(self._inspection_calls[key])),
            invalidate=lambda: self.app.invalidate(),
            focused=lambda: self.app.layout.current_window is self._calls_window,
            hover=lambda key: self.__dict__.get("_hovered_control") == ("inspection", False, key),
            set_hover=lambda key: self.__dict__.__setitem__("_hovered_control", ("inspection", False, key))),
        )
        self._files_window: TranscriptWindow = self._touched_panel.window
        self._calls_window: TranscriptWindow = self._calls_panel.window
        self._inspection_panel: HSplit = HSplit([
            FocusCoordinator.panel(FormattedTextControl("Touched"), self._files_window, lambda: self._files_window, style="class:panel-title"),
            Window(height=1),
            FocusCoordinator.panel(FormattedTextControl("Calls"), self._calls_window, lambda: self._calls_window, style="class:panel-title"),
        ], width=lambda: Dimension.exact(self.main_area.inspector_width), style="class:menu")

    def toggle_inspection(self) -> None:
        self._ensure_inspection()
        focused = self._inspection_focused()
        self._inspection_visible = not self._inspection_visible
        if not self._inspection_visible and focused:
            self._focus.restore_primary()
        if self._inspection_visible and self.output.get_size().columns < 60:
            self.notify("Widen terminal to show Inspect.")
        self.app.invalidate()

    @property
    def main_area(self) -> MainArea:
        if "_main_area" not in self.__dict__:
            self._main_area = MainArea(self)
        return self._main_area

    def _primary_panel(self) -> FocusTarget:
        return self.main_area.focus_target()

    def _inspection_focused(self) -> bool:
        return (self.main_area.inspector_visible
                and self.app.layout.current_window in (self._files_window, self._calls_window))

    def _cycle_panel(self, direction: int = 1, *, components: bool = False) -> None:
        self.status_line.navigation()
        self._focus.travel(direction, components=components)
        self.app.invalidate()

    def _ensure_focus(self) -> None:
        if "_focus" not in self.__dict__:
            self._focus = FocusCoordinator(
                lambda: self.app, self.input,
                lambda: self._editor_input if self._editor_active and self._menu_future is None and not self._options_field_active else None,
                self._primary_panel, (self._files_window, self._calls_window),
                lambda: self._move_inspection(0),
            )
        # Existing focus owners also need the new modal availability predicate.
        self._focus.editor = lambda: self._editor_input if self._editor_active and self._menu_future is None and not self._options_field_active else None

    def _inspection_keys(self, files: bool) -> list[Any]:
        return self._touched_panel.keys() if files else self._calls_panel.keys()

    def _inspection_selection(self, files: bool) -> Any:
        return self._touched_panel.selection if files else self._calls_panel.selection

    def _move_inspection(self, amount: int) -> None:
        if self.app.layout.current_window is self._files_window:
            self._touched_panel.move(amount)
        else:
            self._calls_panel.move(amount)

    def _activate_inspection(self) -> None:
        if self.app.layout.current_window is self._files_window:
            self._touched_panel.open_selected()
        else:
            self._calls_panel.open_selected()

    def _link_hover(self) -> LinkHover:
        return LinkHover(lambda: self.__dict__.get("_hovered_control"),
                         lambda key: self.__dict__.__setitem__("_hovered_control", key))

    def _action_link(self, label: str, key: object, action: Callable[[], object], *,
                     style: str = "", enabled: Callable[[], bool] = lambda: True) -> StyleAndTextTuples:
        return CallbackLink(label, key, self._link_hover(), action, style=style, enabled=enabled).fragments()

    def _task_link(self, label: str, key: object, operation: Callable[[], Coroutine[Any, Any, None]], *,
                   style: str = "", enabled: Callable[[], bool] = lambda: True) -> StyleAndTextTuples:
        return TaskLink(label, key, self._link_hover(), operation, self.app.create_background_task,
                        style=style, enabled=enabled).fragments()

    def _feed_link(self, style: str, text: str) -> ActionLink | None:
        for token in style.split():
            for kind in ("file", "artifact", "directory", "suggestion"):
                prefix = f"class:{kind}-"
                if not token.startswith(prefix):
                    continue
                value = self.__dict__.get(f"_{kind}_links", {}).get(int(token.removeprefix(prefix)))
                key = ("feed", token)
                if kind == "suggestion" and isinstance(value, str):
                    return CallbackLink(text, key, self._link_hover(), lambda: self._submit_suggestion(value),
                                        style=style, enabled=lambda: not self.working and self._menu_future is None)
                operation: Callable[[], Coroutine[Any, Any, None]]
                if kind == "file" and isinstance(value, tuple):
                    operation = lambda: self._inspect_file(str(value[0]), line=value[1])
                elif kind == "artifact" and isinstance(value, int):
                    operation = lambda: self.open_artifact(value)
                elif kind == "directory" and isinstance(value, Path) and self.__dict__.get("_folder_opener") is not None:
                    operation = lambda: self._explore_directory(value)
                else:
                    continue
                return TaskLink(text, key, self._link_hover(), operation, self.app.create_background_task, style=style)
        return None

    def _inspect_link(self) -> StyleAndTextTuples:
        label = "Inspect [Ctrl+O]" if self._right_control_width() > 18 else "Inspect"
        if self._inspection_visible and self.output.get_size().columns >= 60:
            return [*self._action_link("Close Inspector [Ctrl+O]", "inspect", self.toggle_inspection), ("", "  "),
                    *self._task_link("Options", "options", self._options_panel.show), ("", " ")]
        return [*self._action_link(label, "inspect", self.toggle_inspection), ("", " ")]

    def _right_control_width(self) -> int:
        columns = self.output.get_size().columns
        if self._inspection_visible and columns >= 60:
            return 35
        return 30 if columns >= 80 else 18 if columns >= 30 else 0

    def _footer_rows(self) -> int:
        """Leave room for feed output when the terminal is very short."""
        return min(FOOTER_ROWS, max(3, self.output.get_size().rows - 3))

    def _show_scroll_bottom(self) -> bool:
        return not (self.transcript._following or self._viewer_active or self._editor_active or self.context_visible)

    def _scroll_bottom_link(self) -> StyleAndTextTuples:
        return [("", " "), *self._action_link("Scroll to Bottom", "scroll-bottom", self.scroll_to_end), ("", " ")]

    def _ensure_options(self) -> None:
        if "_options_panel" not in self.__dict__:
            from .options import OptionsPanel
            self._options_field_active = False
            self._options_panel = OptionsPanel(self)
            wrap_lines = self._options_panel.values.get("wrap", True)
            self._editor_input.wrap_lines = wrap_lines
            self._editor_input.window.wrap_lines = to_filter(wrap_lines)

    def track_call(self, call: ToolCall, step: int) -> None:
        self._ensure_inspection()
        self._inspection_calls[(step, call.id)] = {"step_id": step, "call_id": call.id,
                                                 "tool": call.name, "arguments": call.arguments, "status": "running"}
        if call.name == "run_command":
            from .shellpaths import shell_file_references

            command = call.arguments.get("command")
            root = self.__dict__.get("_project_directory")
            if isinstance(command, str) and root is not None:
                tracking = shell_file_references(command, root)
                self._inspection_calls[(step, call.id)]["shell_tracking"] = tracking
                for reference in tracking["references"]:
                    shell_target = Path(reference["path"])
                    display = str(shell_target)
                    if shell_target.is_relative_to(root):
                        display = shell_target.relative_to(root).as_posix()
                    previous = self._touched_items.get(display, "")
                    if not previous or previous.startswith("shell "):
                        modes = set(reference["access"].split("/"))
                        if previous:
                            modes.update(previous.removeprefix("shell ").removesuffix(" (inferred)").split("/"))
                        access = "/".join(mode for mode in ("read", "write", "delete") if mode in modes)
                        self._touched_items[display] = f"shell {access} (inferred)"
        path = call.arguments.get("path")
        if path is None and call.name in {"list_dir", "grep", "glob"}:
            path = "."
        if isinstance(path, str) and call.name in {"read_file", "write_file", "edit_file", "list_dir", "grep", "glob"}:
            self._touched_items.setdefault(path, "requested")
        if call.name in {"copy_path", "move_path", "delete_path", "create_directory"}:
            for key in ("path", "source", "destination"):
                target = call.arguments.get(key)
                if isinstance(target, str):
                    self._touched_items.setdefault(target, "requested")
        self.app.invalidate()

    def finish_call(self, call: ToolCall, step: int, result: Any, error: bool) -> None:
        self._ensure_inspection()
        saved = self._inspection_calls.get((step, call.id))
        if saved is not None:
            saved.update(status="error" if error else "completed", result=result)
            if call.name in {"copy_path", "move_path", "delete_path", "create_directory"} and isinstance(result, dict):
                for change in result.get("changes", []):
                    self._touched_items[change["path"]] = change["action"]
            path = call.arguments.get("path")
            if path is None and call.name in {"list_dir", "grep", "glob"}:
                path = "."
            if not error and isinstance(path, str) and path in self._touched_items:
                if call.name == "read_file" and self._touched_items[path].startswith("shell "):
                    self._touched_items[path] = "read"
                self._touched_items[path] = "modified" if call.name not in {"read_file", "list_dir", "grep", "glob"} else self._touched_items.get(path, "read")
                if self._touched_items[path] == "requested":
                    self._touched_items[path] = "read"
            store = self.__dict__.get("_artifact_store")
            if store is not None:
                saved["artifacts"] = store.references(step, call.id)
            from .turnstore import TurnStore
            records = self.__dict__.get("_inspection_store")
            if records is None:
                records = self.__dict__["_inspection_store"] = TurnStore()
            self._inspection_calls[(step, call.id)] = records.write(f"call-{records.sequence + 1}.json", saved)
        self.app.invalidate()

    def has_diff(self, call_id: str, step: int) -> bool:
        store = self.__dict__.get("_artifact_store")
        return store is not None and any(record["kind"] == "diff" for record in store.references(step, call_id))

    def _inspection_entries(self, *, files: bool) -> StyleAndTextTuples:
        return self._touched_panel.fragments() if files else self._calls_panel.fragments()

    async def _inspect_file(self, path: str, *, line: int = 1) -> None:
        root = self.__dict__.get("_project_directory")
        if root is not None and (root / path).is_dir():
            await self._explore_directory(root / path)
            return
        coordinator = self.__dict__.get("_user_edits")
        if coordinator is not None:
            root = self.__dict__.get("_project_directory")
            if root is not None:
                await coordinator.request(root / path, line)
            return
        await self._read_file(path, line=line)

    async def _read_file(self, path: str, *, line: int = 1) -> None:
        request = self._new_viewer_request()
        root = self.__dict__.get("_project_directory")
        if root is None:
            return
        try:
            target = root / path
            before = target.stat()
            text = await build_snapshot(IndexedText.from_path, target)
        except (OSError, UnicodeError, ValueError) as exc:
            self.notify("Could not open text file.")
            self.write(f"{path}: {exc}")
            return
        if request != self._viewer.request:
            text.close()
            return
        self._show_viewer(FileViewerPanel(path, text, target, file_signature(before)), line=line)

    async def _inspect_call(self, call: dict[str, Any]) -> None:
        request = self._new_viewer_request()
        try:
            snapshot = await build_snapshot(field_snapshot, dict(call))
        except (OSError, ValueError, UnicodeError) as exc:
            self.notify("Could not open call record.")
            self.write(f"Call {call['call_id']}: {exc}")
            return
        if request != self._viewer.request:
            snapshot.close()
            return
        self.open_viewer(f"{call['tool']} — {call['call_id']}", snapshot)

    def _new_viewer_request(self) -> int:
        return self._viewer.next_request()

    def _ensure_editor(self) -> None:
        if "_editor_panel" not in self.__dict__:
            from .editorui import EditorPanel
            self._editor_panel = EditorPanel(self)
            self._editor_active = False
            self._editor_input = self._editor_panel.input
            self._editor_container = self._editor_panel.container
        self._editor_panel.ensure_control()
        self._editor_input.window.style = "class:text-area class:editor"

    async def _open_editor_now(self, path: Path, *, line: int = 1) -> None:
        self._ensure_editor()
        self._focus.prompt_active = False
        await self._editor_panel.open(path, line)

    async def _save_editor(self) -> None:
        await self._editor_panel.save()

    async def _close_editor(self) -> None:
        await self._editor_panel.close()

    async def _reload_editor(self) -> None:
        await self._editor_panel.reload()

    def _change_editor_page(self, delta: int) -> None:
        self._editor_panel.change_page(delta)

    def _ensure_viewer(self) -> None:
        if "_viewer" not in self.__dict__:
            if "_viewer_window" in self.__dict__:
                raise RuntimeError("Restart required: viewer ownership changed.")
            self._viewer = ViewerDeck(lambda: self._viewer_header())
        header = self._viewer.container.children[0]
        if isinstance(header, Window):
            if not isinstance(header.content, ActionControl):
                header.content = ActionControl(lambda: self._viewer_header())
            else:
                # Retained controls must resolve the current generation's method.
                header.content.text = lambda: self._viewer_header()
        header.__dict__["_focus_target"] = lambda: self._viewer_window

    @property
    def _viewer_active(self) -> bool:
        return self._viewer.active

    @property
    def _viewer_window(self) -> TranscriptWindow:
        return self._viewer.window

    @property
    def _viewer_container(self) -> HSplit:
        return self._viewer.container

    def _viewer_header(self) -> StyleAndTextTuples:
        text = self._viewer.panel.title + " — Read Only | ↑↓ / Wheel / PgUp / PgDn | Esc: Close"
        panel = self._viewer.panel
        if isinstance(panel, DiffViewerPanel) and panel.reference is not None:
            reference = dict(panel.reference)
            return [("", panel.title + " | "), *self._action_link(
                "Comment on Diff", "comment-on-diff", lambda: self._comment_on_diff(reference)),
                ("", " — Read Only | ↑↓ / Wheel / PgUp / PgDn | Esc: Close")]
        if not self._viewer.panel.changed:
            return [("", text)]
        return [("class:retry", "File changed on disk. "),
                *self._task_link("Reload [Ctrl+R]", "reload-viewer", self._reload_viewer), ("", " | " + text)]

    async def _reload_viewer(self) -> None:
        panel = self._viewer.panel
        if isinstance(panel, FileViewerPanel):
            scroll = self._viewer_window.vertical_scroll
            await self._read_file(str(panel.source))
            self._viewer_window.vertical_scroll = scroll
            self.app.invalidate()

    def open_viewer(self, title: str, text: str | IndexedText, *, diff: bool = False, line: int = 1) -> None:
        self._show_viewer((DiffViewerPanel if diff else TextViewerPanel)(title, text), line=line)

    def _show_viewer(self, panel: TextViewerPanel, *, line: int = 1) -> None:
        if self._editor_active:
            panel.close()
            self.notify("Close the editor before opening a viewer.")
            return
        self._viewer.show(panel, line)
        self._focus.focus(self._viewer_window)
        self.app.invalidate()

    def close_viewer(self) -> None:
        self._viewer.close()
        self._focus.focus(self._menu_window if self._menu_future is not None else self.input)
        self.app.invalidate()

    async def open_artifact(self, record_id: int, *, line: int = 1) -> None:
        request = self._new_viewer_request()
        store = self.__dict__.get("_artifact_store")
        if store is None:
            self.notify("Inspection records unavailable.")
            return
        try:
            path = store.record_path(record_id)
            record = dict(store.records[record_id])
            snapshot = await build_snapshot(artifact_snapshot, path, record)
        except (OSError, ValueError, UnicodeError) as exc:
            self.notify("Could not open inspection record.")
            self.write(f"Inspection record {record_id}: {exc}")
            return
        if request != self._viewer.request:
            snapshot.close()
            return
        self.open_viewer(f"Record {record_id}", snapshot, diff=record["kind"] == "diff", line=line)
        panel = self._viewer.panel
        if self._viewer_active and isinstance(panel, DiffViewerPanel) and panel.control.lines is snapshot:
            panel.reference = diff_reference(record)

    def _comment_on_diff(self, reference: dict[str, Any]) -> None:
        if self._input_label != "> " or self._password or self._eof:
            return
        self.__dict__["_diff_reference"] = dict(reference)
        self._focus.focus(self.input)
        self.app.invalidate()

    def _remove_diff_reference(self) -> None:
        self.__dict__["_diff_reference"] = None
        self.app.invalidate()

    def _prompt_label(self) -> str:
        reference = self.__dict__.get("_diff_reference")
        if reference is not None and self._input_label == "> ":
            return f"[Diff {reference['record_id']}] > "
        return self._input_label

    def write_artifact_reference(self, record: dict[str, object]) -> None:
        identifier = record["record_id"]
        self._ensure_transcript()
        inline = record["kind"] == "diff" and len(self._raw_output) - 1 in self.__dict__.get("_file_links", {})
        if inline:
            self.write_chunk("  ")
        self.__dict__.setdefault("_artifact_links", {})[len(self._raw_output)] = identifier
        for call in self.__dict__.get("_inspection_calls", {}).values():
            if call["call_id"] == record.get("call_id") and call["step_id"] == record.get("step_id"):
                references = call.setdefault("artifacts", [])
                if not any(saved["record_id"] == identifier for saved in references):
                    references.append(record)
        label = "Diff" if record["kind"] == "diff" else "Output"
        text = f"View {label} — Record {identifier}"
        if inline:
            self.write_chunk(text)
        else:
            self.write("    " + text)

    def _artifact_at(self, point: Point) -> int | None:
        if self._flow is None:
            return None
        rows = self._flow.content_rows()
        row = self._flow.get_row(point.y) if 0 <= point.y < len(self._flow.rows) else (
            rows[point.y - len(self._flow.rows)] if 0 <= point.y - len(self._flow.rows) < len(rows) else []
        )
        text = fragment_list_to_text(row)
        left = display_width(text[:len(text) - len(text.lstrip())])
        right = display_width(text.rstrip())
        if not left <= point.x < right:
            return None
        column = 0
        for style, text, *_ in row:
            end = column + display_width(text)
            if column <= point.x < end:
                for token in style.split():
                    if token.startswith("class:artifact-"):
                        identifier = self.__dict__.get("_artifact_links", {}).get(int(token.removeprefix("class:artifact-")))
                        return identifier if isinstance(identifier, int) else None
            column = end
        return None

    def _ensure_menu(self) -> None:
        """Add menu state to live instances without replacing the input buffer."""
        if hasattr(self, "_menu_future"):
            return
        self._menu_future: asyncio.Future[str | None] | None = None
        self._menu_title = ""
        self._choice_panel = ChoicePanel(
            ListActions(activate=lambda index: self._finish_menu(self._menu_options[index][0]),
            invalidate=lambda: self.app.invalidate(),
            focused=lambda: self.app.layout.current_window is self._menu_window,
            hover=lambda index: self.__dict__.get("_hovered_control") == ("menu", index),
            set_hover=lambda index: self.__dict__.__setitem__("_hovered_control", ("menu", index))),
        )
        self._menu_window: TranscriptWindow = self._choice_panel.window
        self._menu_window.height = Dimension.exact(5)

    @property
    def _menu_options(self) -> list[tuple[str, str]]:
        return self._choice_panel.options

    @_menu_options.setter
    def _menu_options(self, options: list[tuple[str, str]]) -> None:
        self._choice_panel.options = options

    @property
    def _menu_index(self) -> int:
        return self._choice_panel.index

    @_menu_index.setter
    def _menu_index(self, index: int) -> None:
        self._choice_panel.selected_key = index

    def _notice_width(self) -> int:
        columns = self.output.get_size().columns
        if self._menu_future is not None or self._options_field_active:
            prefix = self._menu_title if self._menu_future is not None else "Folder Opener"
        else:
            columns -= self._right_control_width()
            if self._show_scroll_bottom():
                columns = max(0, (self.output.get_size().columns - 18) // 2)
            prefix = fragment_list_to_text(self._activity_status())
            if self._editor_active or self._viewer_active:
                prefix += "  ESC to Close Text Editor " if self._editor_active else "  ESC to exit viewer "
        remaining = columns - display_width(prefix) - 2
        return max(1, remaining if remaining > 0 else columns)

    def _activity_width(self) -> int:
        columns = self.output.get_size().columns
        if self._menu_future is not None or self._options_field_active:
            return columns
        return max(0, (columns - 18) // 2) if self._show_scroll_bottom() else max(0, columns - self._right_control_width())

    def _readout_width(self) -> int:
        return max(1, self.output.get_size().columns)

    def _bottom_row(self, *, menu: bool) -> VSplit:
        return VSplit([
            Window(ActionControl(lambda: self._readout(menu=menu)), height=1, style="class:status"),
        ], height=1)

    def _readout(self, *, menu: bool) -> StyleAndTextTuples:
        """Bottom row content; a failing readout callback must not stop the terminal.

        prompt_toolkit redraws this row constantly. An exception here, typically
        from a plugin override of the status line, would otherwise surface as an
        unhandled event-loop exception on every redraw and leave the session stuck.
        """
        try:
            if menu:
                return self._menu_hint()
            if self._options_field_active:
                return self._options_hint()
            return self._status_fragments()
        except Exception as exc:
            return self._failed_readout(exc)

    def _failed_readout(self, exc: BaseException) -> StyleAndTextTuples:
        text = f"Status line failed {_failure_location(exc)}: {type(exc).__name__}: {exc}"
        if text != self.__dict__.get("_readout_failure"):
            self.__dict__["_readout_failure"] = text
            self.write(text + "\nFix the file or disable the plugin with /plugin; the readout returns once it renders again.")
        width = self._readout_width()
        if display_width(text) > width:
            text = text[:max(0, width - 1)] + "…"
        return [("fg:ansired", text)]

    def _options_hint(self) -> StyleAndTextTuples:
        return [*self._action_link("Save", "option-save", lambda: self._options_panel.accept(self._options_panel.field.buffer)),
                ("", " [Enter] | "), *self._action_link("Cancel", "option-cancel", self._options_panel.cancel),
                ("", " [Esc]")]

    def notify(self, text: str) -> None:
        """Show a notice for 30 seconds, until replaced, or until cleared."""
        self.status_line.notify(text)
        self.app.invalidate()

    @property
    def status_line(self) -> StatusLine:
        if "_status_line" not in self.__dict__:
            self._status_line = StatusLine.from_legacy(self.__dict__)
        return self._status_line

    @property
    def _notification(self) -> str:
        return self.status_line.notification

    @property
    def _status_changed_at(self) -> float | None:
        return self.status_line.changed_at

    def _current_notification(self) -> str:
        return self.status_line.current_notification()

    def _notification_text(self, columns: int | None = None) -> str:
        notification = self._current_notification()
        text = self._navigation_tip() or notification
        if not text:
            tips = ["Select and right-click for text options."]
            if self._editor_active:
                tips = ["Ctrl+S saves; Esc closes the editor.", "PgUp/PgDn scrolls with overlap."]
            elif self._viewer_active:
                tips = ["Scroll to inspect; Esc closes the viewer."]
            tips.insert(1, "Shift+Tab to select UI component.")
            if not self.working and self._input_label == "> " and self._menu_future is None and not self._viewer_active and not self._editor_active:
                tips.append("Click Commands to browse commands.")
            if self.__dict__.get("_folder_opener") is not None:
                tips.append("Click directory links to open folders.")
            text = self.status_line.tip(tips)
        if columns is None:
            columns = self._notice_width()
            if notification and display_width(text) > columns:
                columns = self._activity_width()
        return fragment_list_to_text(self._marquee([("", text)], columns, "notice"))

    def _navigation_tip(self) -> str:
        return self.status_line.navigation_tip()

    def _marquee(self, fragments: StyleAndTextTuples, columns: int, key: str) -> StyleAndTextTuples:
        return self.status_line.marquee(fragments, columns, key)

    def _menu_fragments(self) -> StyleAndTextTuples:
        return self._choice_panel.fragments()

    def _menu_hint(self) -> StyleAndTextTuples:
        """Keep access indicators visible even while the menu replaces readouts."""
        callback = getattr(self, "_status_suffix", None)
        suffix = callback() if callback is not None else ""
        remaining = max(0, self._readout_width() - display_width(
            fragment_list_to_text(to_formatted_text(ANSI(suffix)))))
        hint = "↑/↓ or wheel = move | Enter or click = select | Esc = back "
        if len(hint) > remaining:
            hint = "↑↓/wheel move | Enter/click select | Esc"
        if len(hint) > remaining:
            hint = "↑↓/wheel Enter/click Esc"
        if len(hint) > remaining:
            hint = hint[:remaining - 1] + "…" if remaining else ""
        prefix, separator, tail = hint.partition("back")
        if not separator:
            return list(to_formatted_text(ANSI(hint + suffix)))
        return [("", prefix), *self._action_link("back", "menu-back", lambda: self._finish_menu(None)),
                ("", tail), *to_formatted_text(ANSI(suffix))]

    def _menu_heading(self) -> StyleAndTextTuples:
        legend: StyleAndTextTuples = self.__dict__.get("_menu_legend", [])
        title: StyleAndTextTuples = [("class:menu-title", self._menu_title)]
        if not legend:
            return self._activity_notice(title)
        legend = self._marquee(legend, self._activity_width(), "menu-legend")
        if not legend:
            return self._activity_notice(title)
        left = self._activity_notice(title, suffix=legend)
        gap = max(0, self._activity_width() - display_width(fragment_list_to_text(left)))
        return left[:-len(legend)] + [("", " " * gap)] + legend

    async def choose(self, title: str, options: list[tuple[str, str]], *,
                     styles: dict[str, str] | None = None,
                     legend: StyleAndTextTuples | None = None) -> str | None:
        """Choose a value/label pair in the footer; cancellation preserves the draft."""
        self._ensure_menu()
        if self._menu_future is not None:
            raise RuntimeError("A menu is already open.")
        if not options or self._eof:
            return None
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._menu_future = future
        self._menu_title = title
        self._choice_panel.option_styles = styles or {}
        self.__dict__["_menu_legend"] = legend or []
        self._menu_options = [(value, re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", label))
                              for value, label in options]
        self._menu_index = 0
        self._menu_window.vertical_scroll = self._menu_window.vertical_scroll_2 = 0

        def cleanup() -> None:
            self._menu_future = None
            self._menu_options = []
            self._choice_panel.option_styles = {}
            self.__dict__["_menu_legend"] = []

        with MenuLifecycle(self).opened(self._menu_window, cleanup):
            return await future

    def _move_menu(self, rows: int) -> None:
        self._choice_panel.move(rows)

    def _finish_menu(self, value: str | None) -> None:
        if self._menu_future is not None and not self._menu_future.done():
            self._menu_future.set_result(value)

    def insert_command(self, command: str) -> None:
        """Retain the ordinary draft in input history without submitting it."""
        if not _key_supplied(self.input.buffer.text):
            self.input.buffer.append_to_history()
        self.input.buffer.reset(document=Document(command, len(command)))
        self._focus.focus(self.input)
        self.app.invalidate()

    async def _choose_command(self) -> None:
        if self.working or self._menu_future is not None or self._input_label != "> ":
            return
        from .cli import command_menu_options
        selected = await self.choose("Commands", command_menu_options())
        if selected is not None:
            self.insert_command(selected)

    def _commands_mouse(self, event: MouseEvent) -> object:
        if event.event_type == MouseEventType.MOUSE_MOVE:
            self.__dict__["_hovered_control"] = "commands"
        elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
            self.app.create_background_task(self._choose_command())
        return None

    def _status_fragments(self) -> StyleAndTextTuples:
        fragments = to_formatted_text(ANSI(self._status(self._readout_width())))
        if self.__dict__.get("_folder_opener") is None:
            return fragments
        text = fragment_list_to_text(fragments)
        cwd, separator, _ = text.partition("  │  ")
        if not separator or cwd.startswith("model:"):
            return fragments
        remaining = len(cwd)
        result: StyleAndTextTuples = []
        for fragment in fragments:
            style, part = fragment[:2]
            length = min(remaining, len(part))
            if length:
                link_style = style + " class:link class:clickable"
                if self.__dict__.get("_hovered_control") == "folder":
                    link_style += " class:hover"
                result.append((link_style, part[:length], self._folder_mouse))
                remaining -= length
            if length < len(part):
                result.append((style, part[length:]))
        return result

    def _folder_mouse(self, event: MouseEvent) -> object:
        if event.event_type == MouseEventType.MOUSE_MOVE:
            self.__dict__["_hovered_control"] = "folder"
        elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
            self.app.create_background_task(self._explore_project())
        return None

    async def _explore_project(self) -> None:
        directory = self.__dict__.get("_project_directory")
        if directory is not None:
            await self._explore_directory(directory)

    async def _explore_directory(self, directory: Path) -> None:
        opener = self.__dict__.get("_folder_opener")
        if opener is None:
            return
        self.notify("Opening file manager")
        notification_started = self.status_line.changed_at
        opened = await asyncio.to_thread(open_folder, directory, opener)
        if (self.status_line.notification == "Opening file manager"
                and self.status_line.changed_at == notification_started):
            self.notify("")
        if not opened:
            self._folder_opener = None
            self.notify("Could not open folder.")
            self.write(f"Could not open {directory} in the native file explorer.")
            self.app.invalidate()

    def write_directory_reference(self, value: str) -> None:
        root = self.__dict__.get("_project_directory")
        if root is None or self.__dict__.get("_folder_opener") is None:
            return
        try:
            directory = (root / value).resolve()
            exists = directory.is_dir()
        except OSError:
            return
        if not exists:
            return
        self._ensure_transcript()
        links = self.__dict__.setdefault("_directory_links", {})
        links[len(self._raw_output)] = directory
        label = re.sub(r"[\x00-\x1f\x7f]", lambda match: f"\\x{ord(match[0]):02x}", str(directory))
        self.write(f"    {label}")

    def write_file_reference(self, value: str, *, line: int = 1, show_line: bool = True) -> None:
        root = self.__dict__.get("_project_directory")
        if root is None:
            return
        try:
            path = (root / value).resolve()
            if not path.is_file():
                return
        except (OSError, ValueError):
            return
        links = self.__dict__.setdefault("_file_links", {})
        links[len(self._raw_output)] = (path, line)
        label = re.sub(r"[\x00-\x1f\x7f]", lambda match: f"\\x{ord(match[0]):02x}", value)
        location = f":{line}" if show_line else ""
        self.write(f"    {label}{location}")

    def write_suggestion(self, prompt: str) -> None:
        """Display the entire prompt without changing the user's draft."""
        self._ensure_transcript()
        self.__dict__.setdefault("_suggestion_links", {})[len(self._raw_output)] = prompt
        self.write(literal(prompt) + "\n")

    def _submit_suggestion(self, prompt: str) -> None:
        if (self.working or self._viewer_active or self._editor_active
                or self._menu_future is not None or self._input_label != "> " or self._eof):
            return
        # Reserve the run immediately so a double click cannot submit twice.
        self.set_working(True)
        self._submit(prompt, suggested=True)

    def _suggestion_at(self, point: Point) -> str | None:
        if self._flow is None:
            return None
        rows = self._flow.content_rows()
        row = self._flow.get_row(point.y) if 0 <= point.y < len(self._flow.rows) else (
            rows[point.y - len(self._flow.rows)] if 0 <= point.y - len(self._flow.rows) < len(rows) else [])
        text = fragment_list_to_text(row)
        left = display_width(text[:len(text) - len(text.lstrip())])
        right = display_width(text.rstrip())
        if not left <= point.x < right:
            return None
        column = 0
        for style, text, *_ in row:
            end = column + display_width(text)
            if column <= point.x < end:
                for token in style.split():
                    if token.startswith("class:suggestion-"):
                        prompt = self.__dict__.get("_suggestion_links", {}).get(int(token.removeprefix("class:suggestion-")))
                        return prompt if isinstance(prompt, str) else None
            column = end
        return None

    def _file_at(self, point: Point) -> tuple[Path, int] | None:
        if self._flow is None:
            return None
        rows = self._flow.content_rows()
        row = self._flow.get_row(point.y) if 0 <= point.y < len(self._flow.rows) else (
            rows[point.y - len(self._flow.rows)] if 0 <= point.y - len(self._flow.rows) < len(rows) else [])
        text = fragment_list_to_text(row)
        left = display_width(text[:len(text) - len(text.lstrip())])
        right = display_width(text.rstrip())
        if not left <= point.x < right:
            return None
        column = 0
        for style, text, *_ in row:
            end = column + display_width(text)
            if column <= point.x < end:
                for token in style.split():
                    if token.startswith("class:file-"):
                        reference = self.__dict__.get("_file_links", {}).get(int(token.removeprefix("class:file-")))
                        if isinstance(reference, tuple) and len(reference) == 2 and isinstance(reference[0], Path) and isinstance(reference[1], int):
                            return reference[0], reference[1]
            column = end
        return None

    def _directory_at(self, point: Point) -> Path | None:
        if self._flow is None or self.__dict__.get("_folder_opener") is None:
            return None
        rows = self._flow.content_rows()
        row = self._flow.get_row(point.y) if 0 <= point.y < len(self._flow.rows) else (
            rows[point.y - len(self._flow.rows)] if 0 <= point.y - len(self._flow.rows) < len(rows) else []
        )
        text = fragment_list_to_text(row)
        left = display_width(text[:len(text) - len(text.lstrip())])
        right = display_width(text.rstrip())
        if not left <= point.x < right:
            return None
        column = 0
        for style, text, *_ in row:
            end = column + display_width(text)
            if column <= point.x < end:
                for token in style.split():
                    if token.startswith("class:directory-"):
                        directory = self.__dict__.get("_directory_links", {}).get(int(token.removeprefix("class:directory-")))
                        return directory if isinstance(directory, Path) else None
            column = end
        return None

    def _ensure_prompt(self) -> None:
        """Migrate header state; prompt locations come from the transcript file."""
        if hasattr(self, "_pinned_prompt"):
            return
        self._pinned_prompt: tuple[int, int] | None = None
        self._wrapped_columns = 0

    def _pinned_prompt_height(self, columns: int | None = None) -> int:
        size = self.output.get_size()
        if columns is None:
            columns = self.main_area.width
        self._transcript_content(columns)
        assert self._flow is not None
        self._pinned_prompt = self._flow.prompt_at(self.transcript.vertical_scroll)
        if self._pinned_prompt is None:
            return 0
        # One header row, leaving at least one transcript row above the footer.
        available = max(0, size.rows - self._footer_rows() - self._queued_prompt_height() - 1)
        return min(1, available)

    def _pinned_prompt_content(self, width: int) -> UIContent:
        height = self._pinned_prompt_height(width)
        source = self._transcript_content(width)
        start, end = self._pinned_prompt or (0, 0)
        text = fragment_list_to_text(source.get_line(start)).rstrip() if height else ""
        if height and end > start + 1:
            # Reuse bounded, already wrapped rows instead of rereading a long
            # prompt. A first word can wrap onto its own row after the > marker.
            if text.strip() == ">":
                text += " " + fragment_list_to_text(source.get_line(start + 1)).strip()
            prefix = ""
            cells = 0
            for cluster in iter_graphemes(text):
                cells += display_width(cluster)
                if cells > max(0, width - 1):
                    break
                prefix += cluster
            if len(prefix) < len(text) and not text[len(prefix)].isspace():
                boundary = re.search(r"\s+\S*$", prefix)
                if boundary is not None and prefix[:boundary.start()].strip() != ">":
                    prefix = prefix[:boundary.start()]
            text = prefix.rstrip() + "…"

        def get_line(index: int) -> StyleAndTextTuples:
            if not 0 <= index < height:
                return []
            return [("class:user", text)]

        return UIContent(get_line=get_line, line_count=height, show_cursor=False)

    def _ensure_context(self) -> None:
        """Move legacy request text to disk while preserving the context window."""
        from .contextview import ContextControl, ContextView
        if "_context_view" in self.__dict__:
            return
        blocks = self.__dict__.get("_context_blocks", [("", "No model request has been sent yet.")])
        self._context_view = ContextView(blocks)
        self.__dict__.pop("_context_blocks", None)
        self.__dict__.pop("_context_fragments", None)
        self.__dict__.pop("_context_columns", None)
        control = ContextControl(lambda width: self._context_view.content(width))
        if hasattr(self, "context_window"):
            self.context_window.content = control
        else:
            self.context_visible = False
            self.context_window = TranscriptWindow(control)
            self.context_window._following = False

    @property
    def _context_blocks(self) -> Any:
        self._ensure_context()
        return self._context_view

    def set_context(self, request: Any) -> None:
        """Replace the disk-backed snapshot of the actual request."""
        from .contextview import ContextView
        self._ensure_context()
        replacement = ContextView.from_request(request) if isinstance(request, str) else ContextView.from_body(request)
        previous = self._context_view
        self._context_view = replacement
        previous.close()
        self.context_window.seek()
        self.app.invalidate()

    def _scroll_output(self, rows: int) -> None:
        if self._menu_future is not None:
            self._move_menu(rows)
            return
        window = self.main_area.scroll_target()
        if window is not None:
            window.scroll(rows)
        self.app.invalidate()

    def _bindings(self) -> KeyBindings:
        return build_bindings(self)

    def set_skip_handler(self, callback: Callable[[], None] | None) -> None:
        """Offer Ctrl+N while the current request may be skipped."""
        self.__dict__["_skip_callback"] = callback
        self.app.invalidate()

    def set_hold_handler(self, callback: Callable[[], None] | None) -> None:
        """Let the next Esc keep waiting on the current model instead of interrupting."""
        self.__dict__["_hold_callback"] = callback

    def set_interrupt_handler(self, callback: Callable[[], None]) -> None:
        """Interrupt work directly, without submitting or changing the draft."""
        self._interrupt_callback = callback

    def _style(self) -> Style:
        return build_style()

    def refresh(self) -> None:
        """Rebuild presentation while retaining input, transcript, and application."""
        # Build fallible presentation pieces before migrating old storage, so
        # a rejected reload can still run the previous generation's methods.
        focused = self.app.layout.current_control
        layout = self._layout()
        self._focus.preserve(layout, focused)
        bindings = self._bindings()
        style = self._style()
        self._ensure_transcript()
        self._migrate_queued_notices()
        self.app.layout = layout
        self.app.key_bindings = bindings
        self.app.style = style
        if not self.__dict__.get("_title_pulse_bound", False):
            self.app.before_render += lambda app: self._update_title()
            self._title_pulse_bound = True
        self._wrapped_columns = 0
        self.app.min_redraw_interval = MIN_REDRAW_INTERVAL
        self.app.invalidate()

    def _route_mouse_wheel(self, app: Application[None]) -> None:
        route_mouse(self, app)

    def _accept(self, buffer: Buffer) -> bool:
        text = buffer.text
        reference = self.__dict__.get("_diff_reference")
        if (reference is not None and self._input_label == "> " and not self._password
                and text.strip() and not text.lstrip().startswith("/")):
            text = attach_reference(text, reference)
            self._remove_diff_reference()
        if self._input_label == "> " and not self._password:
            self._submit(text)
        else:
            self._lines.put_nowait(text)
        if _key_supplied(buffer.text):
            # prompt_toolkit appends history after the accept handler returns.
            buffer.text = ""
        return False

    def _submit(self, text: str, *, suggested: bool = False) -> None:
        if not text.strip():
            self._lines.put_nowait(text)
            return
        line = (QueuedSuggestion if suggested else Submission)(text, suggested=suggested)
        self.__dict__.setdefault("_submissions", []).append(line)
        self._lines.put_nowait(line)
        self.app.invalidate()

    def consume_submission(self, line: str) -> None:
        submissions = self.__dict__.get("_submissions", [])
        if isinstance(line, Submission):
            submissions[:] = [item for item in submissions if item.identifier != line.identifier]
        self.app.invalidate()

    def _discard_submission(self, line: Submission) -> None:
        if any(item.identifier == line.identifier for item in self.__dict__.get("_submissions", [])):
            line.cancelled = True
            self.consume_submission(line)
            self._focus.restore_prompt()
        else:
            # A redraw can lag behind transfer to the agent's durable queue.
            self._discard_agent_input(line.identifier)

    def _discard_agent_input(self, identifier: str) -> None:
        from .sessions import SessionError
        current = self.__dict__.get("_agent_input_queue")
        try:
            if current is not None and current().discard(identifier):
                self._focus.restore_prompt()
        except SessionError as exc:
            self.notify(f"Could not remove queued input: {exc}")

    def _queued_items(self) -> list[tuple[str, object, Callable[[], object] | None]]:
        self._ensure_queued_prompts()
        current = self.__dict__.get("_agent_input_queue")
        queue = current() if current is not None else None
        entries = queue.snapshot() if queue is not None else []
        result: list[tuple[str, object, Callable[[], object] | None]] = []
        for index, text in enumerate(self._queued_prompts):
            identifier = entries[index][0] if index < len(entries) and entries[index][1] == text else None
            def discard(identifier: str | None = identifier) -> None:
                if identifier is not None:
                    self._discard_agent_input(identifier)
            result.append((text, identifier or ("preview", index), discard if identifier is not None else None))
        for line in self.__dict__.get("_submissions", []):
            label = "/key …" if _key_supplied(line) else str(line)
            def discard_submission(line: Submission = line) -> None:
                self._discard_submission(line)
            result.append((label, line.identifier, discard_submission))
        return result

    def _activity(self) -> StyleAndTextTuples:
        fragments = self._activity_status()
        edits = self.__dict__.get("_user_edits")
        if edits is not None and getattr(edits, "pending", None) is not None:
            return self._activity_notice(self._activity_status(compact=True), suffix=[("", "  "), *self._action_link(
                "Cancel Opening Editor", "cancel-opening-editor", edits.close,
                enabled=lambda: edits.pending is not None), ("", " ")])
        if self._viewer_active or self._editor_active:
            mode = "editor" if self._editor_active else "viewer"
            label = "Close Text Editor" if self._editor_active else "exit viewer"
            link = (self._task_link(label, "exit-editor", self._close_editor) if mode == "editor" else
                    self._action_link(label, "exit-viewer", self.close_viewer))
            if display_width(fragment_list_to_text(fragments)) + len("  ESC to " + label + " ") > self.output.get_size().columns - self._right_control_width():
                fragments = []
            fragments.extend([("", "  ESC to " if fragments else "ESC to "), *link, ("", " ")])
        if self.__dict__.get("_diff_reference") is not None and self._input_label == "> ":
            return self._activity_notice(fragments, suffix=[("", "  "), *self._action_link(
                "Remove Diff Reference", "remove-diff-reference", self._remove_diff_reference), ("", " ")])
        return self._activity_notice(fragments)

    def _activity_notice(self, fragments: StyleAndTextTuples, *, suffix: StyleAndTextTuples | None = None) -> StyleAndTextTuples:
        notification = self._current_notification()
        if self._navigation_tip():
            notification = ""
        suffix = suffix or []
        if suffix:
            suffix = self._marquee(suffix, self._activity_width(), "activity-actions")
        available = max(0, self._activity_width() - display_width(fragment_list_to_text(suffix)))
        if not notification and display_width(fragment_list_to_text(fragments)) >= available:
            return [*self._marquee(fragments, available, "activity"), *suffix]
        remaining = max(0, available - display_width(fragment_list_to_text(fragments)) - 2)
        text = ""
        if notification:
            if not remaining and fragments:
                fragments = (fragments[:1] if self._menu_future is not None or self._options_field_active
                             else self._activity_status(compact=True))
                remaining = max(0, available - display_width(fragment_list_to_text(fragments)) - 2)
            text = self._notification_text(remaining)
        elif self._navigation_tip() or self._menu_future is None and not self._options_field_active:
            text = self._notification_text(remaining)
        if text:
            if display_width(fragment_list_to_text(fragments)) + 2 + display_width(text) > available:
                fragments = []
            style = self.status_line.notice_style(bool(notification))
            if fragments:
                fragments = [*fragments, ("", "  ")]
            fragments = [*fragments, (style, text)]
        return [*self._marquee(fragments, available, "activity"), *suffix]

    def _activity_status(self, *, compact: bool = False) -> StyleAndTextTuples:
        frame = self.__dict__.get("_activity_frame", PULSE_FRAMES[int(time.monotonic() * 4) % len(PULSE_FRAMES)])
        style, text, ready = self.status_line.activity(held=self._editor_active, working=self.working,
            stopping=self.stopping, pausing=self.pausing, concise=self._viewer_active or compact, frame=frame)
        fragments: StyleAndTextTuples = [(style, text)]
        if ready and self._input_label == "> " and not self._viewer_active and not compact:
            fragments.extend([("", "  "), *self._task_link("Commands", "commands", self._choose_command), ("", " ")])
        return fragments

    def set_retry_status(self, text: str, *, spinning: bool = False) -> None:
        """Show transient retry progress without appending a line each second."""
        self.status_line.retry = text
        self.status_line.retry_spinning = spinning
        self.app.invalidate()

    def set_phase(self, text: str) -> None:
        """Show what the running request is doing, such as pacing an upload or waiting for output."""
        self.status_line.set_phase(text)
        self.app.invalidate()

    def set_working(self, working: bool, *, stopping: bool = False, pausing: bool = False) -> None:
        self.working = working
        self.stopping = stopping
        self.pausing = pausing
        if not working:
            self.status_line.retry = ""
        self._update_title()
        self.app.invalidate()

    def set_command_status(self, text: str) -> None:
        """Distinguish settings changes from active model work."""
        self.status_line.command = text
        self.app.invalidate()

    def set_title(self, title: str) -> None:
        """Update the terminal/tab title through the platform's output adapter."""
        self._title = title
        self._update_title()

    def _update_title(self) -> None:
        """Refresh the title animation and schedule open-file change checks."""
        self._focus.reconcile()
        self._viewer.tick(self.app.create_background_task, self.app.invalidate)
        frame = PULSE_FRAMES[int(time.monotonic() * 4) % len(PULSE_FRAMES)]
        self._activity_frame = frame
        title = self.__dict__.get("_title")
        if title is None:
            return
        displayed = f"{frame} {title}" if self.working and not self._editor_active else title
        if displayed != self.__dict__.get("_displayed_title"):
            self.output.set_title(displayed)
            self.output.flush()
            self._displayed_title = displayed

    def clear_transcript(self) -> None:
        """Replace display storage while retaining the input draft and application."""
        self._ensure_transcript()
        self._end_tail_layout()
        self._remove_diff_reference()
        self.clear_queued_prompts()
        archive = TranscriptFile()
        self._close_transcript()
        self._raw_output = archive
        self._block_starts.clear()
        self.__dict__.setdefault("_directory_links", {}).clear()
        self.__dict__.setdefault("_file_links", {}).clear()
        self.__dict__.setdefault("_artifact_links", {}).clear()
        self.__dict__.setdefault("_suggestion_links", {}).clear()
        self.__dict__.setdefault("_touched_items", {}).clear()
        self.__dict__.setdefault("_inspection_calls", {}).clear()
        records = self.__dict__.pop("_inspection_store", None)
        if records is not None:
            records.close()
        if self._viewer.active:
            self.close_viewer()
        self._continuation_indents.clear()
        self._markdown_chunks.clear()
        self._reasoning_chunks.clear()
        self._markdown_finals.clear()
        self._selection_start = self._selection_end = None
        self._selection_dragging = False
        self._flow = None
        self._wrapped_columns = self._wrapped_count = self._line_count = 0
        self._pinned_prompt = None
        self.context_visible = False
        self.transcript.vertical_scroll = self.transcript.vertical_scroll_2 = 0
        self.scroll_to_end()

    def scroll_to_end(self) -> None:
        """Resume following the tail, including after loading saved messages."""
        self.transcript._following = True
        self.transcript._pending_scroll = 0
        self.app.invalidate()

    def write(
        self, text: str, *, continuation_indents: dict[int, int] | None = None,
        user_prompt: bool = False,
    ) -> None:
        """Append a block, optionally aligning wrapped rows of selected source lines."""
        self._ensure_transcript()
        if continuation_indents:
            self._continuation_indents[len(self._raw_output)] = dict(continuation_indents)
        self._block_starts.add(len(self._raw_output))
        self._raw_output.append(text)
        self.app.invalidate()

    def write_chunk(self, text: str, *, first: bool = False) -> None:
        self._ensure_transcript()
        if first or not self._raw_output:
            self._block_starts.add(len(self._raw_output))
        self._raw_output.append(text)
        self.app.invalidate()

    def configure_markdown(self, *, theme: str | None = None, source: bool | None = None) -> None:
        self._ensure_transcript()
        if theme is not None:
            if theme not in THEMES:
                raise ValueError(f"Unknown theme: {theme}")
            self._markdown_theme = theme
        if source is not None:
            self._markdown_source = source
        self._selection_start = self._selection_end = None
        self._selection_dragging = False
        self._wrapped_columns = 0
        self.app.invalidate()

    def write_markdown(self, text: str, *, first: bool = True, final: bool = True, reasoning: bool = False) -> None:
        self._ensure_transcript()
        index = len(self._raw_output)
        self._markdown_chunks.add(index)
        if reasoning:
            self._reasoning_chunks.add(index)
        if final:
            self._markdown_finals.add(index)
        self.write_chunk(text, first=first)

    def finish_markdown(self) -> None:
        self._ensure_transcript()
        if self._raw_output and len(self._raw_output) - 1 in self._markdown_chunks:
            if len(self._raw_output) - 1 not in self._markdown_finals:
                self.write_markdown("", first=False, final=True,
                                    reasoning=len(self._raw_output) - 1 in self._reasoning_chunks)

    def replace_markdown(self, text: str) -> bool:
        """Replace the latest Markdown reply block, when it ends the transcript."""
        self._ensure_transcript()
        end = len(self._raw_output) - 1
        if end < 0 or end not in self._markdown_chunks or end in self._reasoning_chunks:
            return False
        start = end
        while start not in self._block_starts and start > 0 and start - 1 in self._markdown_chunks:
            start -= 1
        for index in range(start, end + 1):
            self._raw_output.clear_record(index)
        self.write_markdown(text, first=True, final=True)
        self._wrapped_columns = 0
        return True

    def record_count(self) -> int:
        """Records written so far; a position for `show_tail_first`."""
        self._ensure_transcript()
        return len(self._raw_output)

    def hold_layout(self, notice: str) -> None:
        """Lay out nothing until `show_tail_first` names the records to show first.

        The notice stays on the status line until the full layout is shown.
        """
        self._end_tail_layout()
        self.__dict__["_tail_pending"] = True
        self.__dict__["_loading_notice"] = notice
        # Writes and placements arrive faster than a redraw costs; coalesce them while loading.
        self.__dict__.setdefault("_redraw_interval_before", self.app.min_redraw_interval)
        self.app.min_redraw_interval = 0.1
        self.notify(notice)

    def show_tail_first(self) -> None:
        """Lay out the newest records first; earlier blocks fill in above them, newest first."""
        self._end_tail_layout()
        layouts = self.__dict__.get("_transcript_layouts")
        if layouts is not None:
            # Held layouts contain nothing; the feed restarts from its end.
            layouts.close()
        if self._raw_output:
            self.__dict__["_tail_from"] = len(self._raw_output)
            self.__dict__["_tail_boundaries"] = sorted(self._block_starts)
        else:
            self._finish_loading()
        self.app.invalidate()

    def _finish_loading(self) -> None:
        notice = self.__dict__.pop("_loading_notice", None)
        if notice is not None and self.status_line.notification == notice:
            self.notify("")
        if "_redraw_interval_before" in self.__dict__:
            self.app.min_redraw_interval = self.__dict__.pop("_redraw_interval_before")

    def _rows_placed_above(self, added: int) -> None:
        """Keep the reader's view and selection where they were when rows join above."""
        if not self.transcript._following:
            self.transcript.vertical_scroll += added
        for name in ("_selection_start", "_selection_end"):
            point = getattr(self, name)
            if point is not None:
                setattr(self, name, Point(x=point.x, y=point.y + added))
        self.app.invalidate()

    def _end_tail_layout(self) -> None:
        self.__dict__.pop("_tail_from", None)
        self.__dict__.pop("_tail_pending", None)
        self.__dict__.pop("_tail_boundaries", None)
        layouts = self.__dict__.get("_transcript_layouts")
        if layouts is not None:
            layouts.drop_incomplete()

    def last_markdown(self) -> str:
        self._ensure_transcript()
        if not self._markdown_chunks:
            return ""
        end = max(self._markdown_chunks)
        start = end
        while start not in self._block_starts and start > 0 and start - 1 in self._markdown_chunks:
            start -= 1
        return "".join(self._raw_output[i] for i in range(start, end + 1))

    def copy_response(self, mode: str = "auto", number: int = 1) -> str:
        source = self.last_markdown()
        if not source:
            raise ValueError("No assistant reply to copy.")
        blocks = code_blocks(source)
        if mode == "markdown" or (mode == "auto" and self._markdown_source):
            text = source
        elif mode == "code" or (mode == "auto" and len(blocks) == 1):
            if not 1 <= number <= len(blocks):
                raise ValueError(f"Code block {number} is unavailable; reply has {len(blocks)} code blocks.")
            text = blocks[number - 1]
        else:
            text = MarkdownRenderer(color=False).plain(source)
        self._copy_text(text)
        return text

    def _copy_text(self, text: str) -> None:
        from prompt_toolkit.clipboard import ClipboardData
        self.app.clipboard.set_data(ClipboardData(text))
        if not copy_to_clipboard(text):
            self.output.write_raw(clipboard_sequence(text))
            self.output.flush()

    def _select_transcript(self, event: MouseEvent) -> object:
        self._ensure_transcript()
        directory = self._directory_at(event.position)
        artifact = self._artifact_at(event.position)
        file = self._file_at(event.position)
        suggestion = self._suggestion_at(event.position)
        if event.button == MouseButton.RIGHT:
            self._selection_dragging = False
            if event.event_type == MouseEventType.MOUSE_UP:
                text = self.selected_text()
                if text:
                    self.app.create_background_task(self._selection_menu(text, file[0] if file else directory))
                else:
                    self.notify("Select text, then right-click for options.")
            return None
        if event.event_type == MouseEventType.MOUSE_MOVE and artifact is not None:
            self.__dict__["_hovered_control"] = ("artifact", artifact)
        if event.event_type == MouseEventType.MOUSE_MOVE and directory is not None:
            self.__dict__["_hovered_control"] = ("directory", directory)
        if event.event_type == MouseEventType.MOUSE_MOVE and file is not None:
            self.__dict__["_hovered_control"] = ("file", file)
        if event.event_type == MouseEventType.MOUSE_MOVE and suggestion is not None:
            self.__dict__["_hovered_control"] = ("suggestion", suggestion)
        selecting_button = event.button in {MouseButton.LEFT, MouseButton.UNKNOWN}
        if event.event_type == MouseEventType.MOUSE_DOWN:
            self._selection_dragging = selecting_button
            self._selection_start = self._selection_end = event.position if selecting_button else None
        elif event.event_type == MouseEventType.MOUSE_MOVE:
            if not selecting_button:
                self._selection_dragging = False
            if not self._selection_dragging:
                return NotImplemented
            self._selection_end = event.position
        elif event.event_type == MouseEventType.MOUSE_UP:
            dragging = self._selection_dragging
            self._selection_dragging = False
            if not dragging:
                return NotImplemented
            if event.button in {MouseButton.RIGHT, MouseButton.MIDDLE}:
                self._selection_start = self._selection_end = None
            else:
                if suggestion is not None and self._selection_start == event.position:
                    self._selection_start = self._selection_end = None
                    self._submit_suggestion(suggestion)
                    return None
                if file is not None and self._selection_start == event.position:
                    self._selection_start = self._selection_end = None
                    self.app.create_background_task(self._inspect_file(str(file[0]), line=file[1]))
                    return None
                if artifact is not None and self._selection_start == event.position:
                    self._selection_start = self._selection_end = None
                    self.app.create_background_task(self.open_artifact(artifact))
                    return None
                if directory is not None and self._selection_start == event.position:
                    self._selection_start = self._selection_end = None
                    self.app.create_background_task(self._explore_directory(directory))
                    return None
                self._selection_end = event.position
        else:
            return NotImplemented
        self.app.invalidate()
        return None

    async def _selection_menu(self, text: str, path: Path | None = None) -> None:
        """Capture the selection before opening the shared action selector."""
        if self._menu_future is not None:
            return
        options = [("copy", "Copy Text"), ("prompt", "Use Selection in Prompt")]
        paths = {}
        if path is not None:
            paths["absolute"] = str(path)
            options.append(("absolute", "Copy Absolute Path"))
            root = self.__dict__.get("_project_directory")
            if root is not None and path.is_relative_to(root):
                paths["relative"] = str(path.relative_to(root))
                options.append(("relative", "Copy Relative Path"))
        action = await self.choose("Selected Text", options)
        if action == "copy":
            self._copy_text(text)
            self.notify("Text copied.")
        elif action == "prompt":
            buffer = self.input.buffer
            separator = "\n" if buffer.text and not buffer.text.endswith("\n") else ""
            buffer.cursor_position = len(buffer.text)
            buffer.insert_text(separator + text)
        elif action in paths:
            self._copy_text(paths[action])
            self.notify("Path copied.")

    def _selection_bounds(self) -> tuple[Point, Point] | None:
        """Convert dragged cell positions to an exclusive range for copy and paint."""
        if self._selection_start is None or self._selection_end is None or self._selection_start == self._selection_end:
            return None
        start, end = sorted([self._selection_start, self._selection_end], key=lambda p: (p.y, p.x))
        return start, Point(end.x + 1, end.y)

    def selected_text(self) -> str:
        bounds = self._selection_bounds()
        if self._flow is None or bounds is None:
            return ""
        start, end = bounds
        pieces = []
        count = len(self._flow.rows) + len(self._flow.content_rows())
        for index in range(max(0, start.y), min(end.y + 1, count)):
            row = self._flow.copy_row(index)
            pieces.append(row.selected(start.x if index == start.y else 0, end.x if index == end.y else None))
        return "".join(pieces)

    def _ensure_queued_prompts(self) -> None:
        """Initialize waiting state without changing the retained transcript."""
        if "_queued_prompts" in self.__dict__:
            return
        notices = self.__dict__.get("_queued_notices", {})
        self._queued_prompts: list[str] = list(notices.values())

    def _migrate_queued_notices(self) -> None:
        """Move legacy entries only after fallible presentation builds succeed."""
        self._ensure_queued_prompts()
        self._ensure_transcript()
        notices = self.__dict__.pop("_queued_notices", {})
        for index in notices:
            # The previous renderer wrote the prompt and its notice as adjacent
            # records. Empty both without changing other records' identities.
            self._raw_output.clear_record(index - 1)
            self._raw_output.clear_record(index)
        self.__dict__.pop("_notice_rows", None)
        if notices:
            self._wrapped_columns = 0
            self._pinned_prompt = None

    def queue_prompt(self, prompt: str) -> None:
        self._ensure_queued_prompts()
        self._queued_prompts.append(prompt)
        self.app.invalidate()

    def clear_queued_prompts(self) -> None:
        self._migrate_queued_notices()
        self._queued_prompts.clear()
        self.app.invalidate()

    def set_queued_prompts(self, prompts: list[str]) -> None:
        self._ensure_queued_prompts()
        self._queued_prompts[:] = prompts
        self.app.invalidate()

    def queued_prompt_sent(self, prompt: str) -> bool:
        """Acknowledge the first matching arrival, including identical prompts."""
        self._ensure_queued_prompts()
        if prompt not in self._queued_prompts:
            return False
        self._queued_prompts.remove(prompt)
        self.app.invalidate()
        return True

    def _queued_prompt_height(self) -> int:
        self._ensure_queued_prompts()
        available = max(0, self.output.get_size().rows - self._footer_rows() - 1)
        return min(len(self._queued_prompts) + len(self.__dict__.get("_submissions", [])), available)

    def _queued_prompt_content(self, width: int) -> UIContent:
        height = self._queued_prompt_height()
        prompts = self._queued_items()[-height:] if height else []
        self.__dict__["_queued_width"] = width

        def get_line(index: int) -> StyleAndTextTuples:
            if not 0 <= index < len(prompts) or width <= 0:
                return []
            prompt, identifier, discard = prompts[index]
            text = "> " + " ".join(literal(prompt).splitlines())
            cancel_label = "Cancel"[:width]
            text_width = max(0, width - len(cancel_label) - 1) if discard is not None else width
            if display_width(text) > text_width:
                prefix = ""
                for cluster in iter_graphemes(text):
                    if display_width(prefix + cluster) > text_width - 1:
                        break
                    prefix += cluster
                text = prefix.rstrip() + "…" if text_width else ""
            result: StyleAndTextTuples = [("class:user", text)]
            if discard is not None:
                result.append(("class:user", " " * max(0, width - display_width(text) - len(cancel_label))))
                result.extend(self._action_link(cancel_label, ("discard-input", identifier), discard))
            return result

        content = UIContent(get_line=get_line, line_count=height, show_cursor=False)
        self.__dict__["_queued_mouse_content"] = content
        return content

    def _queued_prompt_mouse(self, event: MouseEvent) -> object:
        content = self.__dict__.get("_queued_mouse_content")
        if content is None:
            return NotImplemented
        if 0 <= event.position.y < content.line_count:
            column = 0
            for fragment in content.get_line(event.position.y):
                # Window translates terminal cells to source character offsets.
                end = column + len(fragment[1])
                if column <= event.position.x < end and len(fragment) == 3:
                    return fragment[2](event)
                column = end
        return NotImplemented

    def _ensure_transcript(self) -> None:
        """Migrate pre-file-buffer sessions once, without rebuilding the frame."""
        # Older running sessions have no explicit wrap alignment metadata.
        self.__dict__.setdefault("_continuation_indents", {})
        # Existing highlighted selections are inactive after adopting this fix.
        self.__dict__.setdefault("_selection_dragging", False)
        # Explicit migration adds Markdown metadata without treating historical ANSI as Markdown.
        if "_reasoning_chunks" not in self.__dict__:
            self._reasoning_chunks: set[int] = set()
        if "_markdown_chunks" not in self.__dict__:
            self._markdown_chunks: set[int] = set()
            self._markdown_finals: set[int] = set()
            self._markdown_theme = "dark"
            self._markdown_source = False
            self._selection_start = None
            self._selection_end = None
            self.transcript.content = TranscriptControl(lambda width: self._transcript_content(width), self._select_transcript)
            # Retained pre-Markdown flows have no copy sidecar; reflow originals.
            if self._flow is not None and not hasattr(self._flow, "_copy_rows"):
                self._flow.rows.close()
                self._flow = None
                self._wrapped_columns = self._wrapped_count = 0
        original = self.__dict__["_raw_output"]
        if isinstance(original, TranscriptFile):
            return
        archive = TranscriptFile()
        try:
            for block in original:
                archive.append(block)
        except BaseException:
            archive.close()
            raise
        self._raw_output = archive
        self._block_starts = set(range(len(archive)))
        self._flow = None
        self._wrapped_columns = 0
        self._wrapped_count = 0
        # Release the old per-character cache after a successful copy.
        self.__dict__.pop("_transcript", None)
        self.__dict__.pop("_wrapped_spans", None)
        self.transcript.content = TranscriptControl(lambda width: self._transcript_content(width), self._select_transcript)

    def _transcript_content(self, width: int) -> UIContent:
        self._ensure_transcript()
        self._ensure_prompt()
        columns = max(1, width)
        layouts = self.__dict__.get("_transcript_layouts")
        if layouts is None:
            layouts = self._transcript_layouts = TranscriptLayouts()
            if self._flow is not None:
                self._flow.close()
            self._flow = None
        color = self.app.color_depth != ColorDepth.DEPTH_1_BIT
        layouts.configure((id(self._raw_output), self._markdown_theme, color, self._markdown_source),
                          reset=self._wrapped_columns == 0 and self._flow is not None)
        def create(columns: int) -> WrappedTranscript:
            return WrappedTranscript(columns, theme=self._markdown_theme, color=color, source=self._markdown_source)
        total = len(self._raw_output)
        tail_from = self.__dict__.get("_tail_from")
        pending = self.__dict__.get("_tail_pending", False)
        state = layouts.state(columns, create, start=total if pending or tail_from is not None else 0)
        if pending:
            # A restore in progress lays out nothing; redraws must not build the feed top-down.
            state.start = state.count = total
        elif tail_from is not None and state.start > 0:
            # A restored feed never lays out on the event loop: blocks join from the newest down.
            layouts.fill_upward(columns, self.__dict__.get("_tail_boundaries", []), RESTORE_CHUNK_RECORDS, create,
                                self._transcript_record, _append_record, self.app.create_background_task,
                                placed=self._rows_placed_above, done=self.app.invalidate)
        elif tail_from is not None:
            self._end_tail_layout()
            self._finish_loading()
        if self._flow is not state.flow:
            self._selection_start = self._selection_end = None
            self._selection_dragging = False
        self._flow = flow = state.flow
        self._wrapped_columns = columns
        layouts.update(state, total, self._append_transcript_record)
        self._wrapped_count = total
        terminal_columns = self.output.get_size().columns
        full = max(1, terminal_columns - 1)
        narrow = MainArea.split_width(terminal_columns)
        if state.start == 0 and terminal_columns >= 60 and columns in (full, narrow):
            layouts.warm(narrow if columns == full else full, total, create,
                         self._append_transcript_record, self.app.create_background_task)
        preview = flow.content_rows()
        complete = len(flow.rows)
        self._line_count = complete + len(preview)

        def is_current() -> bool:
            return (self._flow is flow and self._wrapped_columns == columns
                    and layouts.states.get(columns) is state)

        def get_line(index: int) -> StyleAndTextTuples:
            # Render info can outlive a replaced or evicted file-backed layout.
            if not is_current():
                return []
            if index < complete:
                fragments = fill_prompt(flow.get_row(index))
                bounds = self._selection_bounds()
                if bounds is not None:
                    start, end = bounds
                    if start.y <= index <= end.y:
                        highlighted: StyleAndTextTuples = []
                        column = 0
                        left = start.x if index == start.y else 0
                        right = end.x if index == end.y else columns
                        for style, text, *_ in fragments:
                            for cluster in iter_graphemes(text):
                                selected = left <= column < right
                                highlighted.append((style + " reverse" if selected else style, cluster))
                                column += max(0, display_width(cluster))
                        return highlighted
                return decorate_links(fragments)
            offset = index - complete
            return decorate_links(fill_prompt(preview[offset])) if offset < len(preview) else []

        def fill_prompt(fragments: StyleAndTextTuples) -> StyleAndTextTuples:
            if any("bg:" + USER_BACKGROUND_COLOR in style.split() for style, *_ in fragments):
                # Display-only padding keeps archived text and copy offsets intact.
                remaining = max(0, columns - display_width(fragment_list_to_text(fragments)))
                return [*fragments, (USER_PROMPT_STYLE, " " * remaining)]
            return fragments

        def decorate_links(fragments: StyleAndTextTuples) -> StyleAndTextTuples:
            result: StyleAndTextTuples = []
            for style, text, *_ in fragments:
                padding_style = " ".join(token for token in style.split() if not token.startswith(FEED_LINK_PREFIXES))
                leading = trailing = ""
                if any(token.startswith(FEED_LINK_PREFIXES) for token in style.split()):
                    leading = text[:len(text) - len(text.lstrip())]
                    text = text[len(leading):]
                    trailing = text[len(text.rstrip()):]
                    text = text[:len(text) - len(trailing)]
                    if leading:
                        result.append((padding_style, leading))
                for token in style.split():
                    if token.startswith("class:suggestion-"):
                        suggestion = self.__dict__.get("_suggestion_links", {}).get(int(token.removeprefix("class:suggestion-")))
                        style += " class:link class:clickable"
                        if self.__dict__.get("_hovered_control") == ("suggestion", suggestion):
                            style += " class:hover"
                    if token.startswith("class:file-"):
                        file = self.__dict__.get("_file_links", {}).get(int(token.removeprefix("class:file-")))
                        style += " class:link class:clickable"
                        if self.__dict__.get("_hovered_control") == ("file", file):
                            style += " class:hover"
                    if token.startswith("class:artifact-"):
                        identifier = self.__dict__.get("_artifact_links", {}).get(int(token.removeprefix("class:artifact-")))
                        style += " class:link class:clickable"
                        if self.__dict__.get("_hovered_control") == ("artifact", identifier):
                            style += " class:hover"
                    if token.startswith("class:directory-") and self.__dict__.get("_folder_opener") is not None:
                        directory = self.__dict__.get("_directory_links", {}).get(int(token.removeprefix("class:directory-")))
                        style += " class:link class:clickable"
                        if self.__dict__.get("_hovered_control") == ("directory", directory):
                            style += " class:hover"
                link = self._feed_link(style, text)
                result.extend(link.fragments() if link is not None else [(style, text)])
                if trailing:
                    result.append((padding_style, trailing))
            return result

        content = UIContent(get_line=get_line, line_count=self._line_count + 1,
                            cursor_position=Point(x=0, y=max(0, self._line_count - 1)), show_cursor=False)
        content.__dict__["focus_rows"] = flow.linked_rows()
        content.__dict__["is_current"] = is_current
        return content

    def _append_transcript_record(self, flow: WrappedTranscript, index: int) -> None:
        _append_record(flow, self._transcript_record(index))

    def _transcript_record(self, index: int) -> tuple[Any, ...]:
        """Everything needed to lay out one record, readable off the event loop."""
        if index in self._markdown_chunks:
            return (True, self._raw_output[index], index in self._block_starts,
                    index in self._markdown_finals, index in self._reasoning_chunks)
        return (False, self._raw_output[index], index in self._block_starts,
                self._continuation_indents.get(index),
                "".join(f" class:{kind}-{index}" for kind in ("directory", "file", "artifact", "suggestion")
                        if index in self.__dict__.get(f"_{kind}_links", {})))

    async def read_line(self) -> str | None:
        while True:
            if self._eof and self._lines.empty():
                return None
            line = await self._lines.get()
            if not cancelled(line):
                return line

    def _end_input(self) -> None:
        options = self.__dict__.get("_options_panel")
        if options is not None:
            options.cancel()
        self._finish_menu(None)
        self._eof = True
        self._lines.put_nowait(None)

    async def ask(self, prompt: str, *, password: bool = False) -> str:
        ordinary_buffer = self.input.buffer
        ordinary_lines = self._lines
        # Replies belong to this question, never to the submission FIFO.
        self._lines = asyncio.Queue()
        if password:
            # A password processor only masks display. A separate buffer and
            # queue also isolate history, drafts, and input already queued.
            self.input.buffer = Buffer(history=DummyHistory(), multiline=False,
                                       accept_handler=lambda buffer: self._accept(buffer))
            self.input.control.buffer = self.input.buffer
        self._input_label = prompt
        self._password = password
        self.app.invalidate()
        try:
            return await self.read_line() or ""
        finally:
            self._lines = ordinary_lines
            if password:
                self.input.buffer.reset()
                self.input.buffer = ordinary_buffer
                self.input.control.buffer = ordinary_buffer
            self._input_label = "> "
            self._password = False
            self.app.invalidate()

    def _refresh_needed(self, last: float, now: float) -> bool:
        """Whether time alone changed what the screen shows since the last tick."""
        status = self.status_line
        pixel = self.__dict__.get("_pixel_mouse")
        if (self.working or status.marquees or (pixel is not None and pixel.enabling)
                or status.changed_at is not None and now - status.changed_at < 1):
            return True
        if (self._viewer_active or self.__dict__.get("_editor_active")) and now - last >= 1:
            return True  # Open-file change checks run from each render.
        deadlines = [status.navigation_until]
        if status.notification and status.changed_at is not None:
            deadlines.append(status.changed_at + NOTICE_SECONDS)
        if status.tips_started is not None:
            deadlines.append(status.tips_started + (now - status.tips_started) // TIP_SECONDS * TIP_SECONDS)
        return any(last < deadline <= now for deadline in deadlines)

    async def _refresh_loop(self) -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(REFRESH_INTERVAL)
            now = time.monotonic()
            if self._refresh_needed(last, now):
                last = now
                self.app.invalidate()

    async def run(self) -> None:
        refresher = asyncio.create_task(self._refresh_loop())
        try:
            await self.app.run_async()
        except EOFError:
            pass
        finally:
            refresher.cancel()
            await asyncio.gather(refresher, return_exceptions=True)
            self._end_input()
            self._close_transcript()

    def _close_transcript(self) -> None:
        pixel_mouse = self.__dict__.pop("_pixel_mouse", None)
        if pixel_mouse is not None:
            pixel_mouse.close()
        panel = self.__dict__.get("_editor_panel")
        if panel is not None:
            panel.dispose()
        coordinator = self.__dict__.get("_user_edits")
        if coordinator is not None:
            coordinator.close()
        self._viewer.close()
        context = self.__dict__.get("_context_view")
        if context is not None:
            context.close()
        records = self.__dict__.pop("_inspection_store", None)
        if records is not None:
            records.close()
        self._raw_output.close()
        layouts = self.__dict__.get("_transcript_layouts")
        if layouts is not None:
            layouts.close()
        elif self._flow is not None:
            self._flow.close()

    def close(self) -> None:
        self._finish_menu(None)
        if self.app.is_running:
            self.app.exit()
        else:
            self._close_transcript()
