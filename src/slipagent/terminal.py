"""Fixed REPL footer and a scrolling transcript on interactive terminals."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
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
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, Dimension, DynamicContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import ColorDepth, Output, create_output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from wcwidth import iter_graphemes, width as display_width, wrap

from .transcript import TranscriptFile, WrappedTranscript
from .palette import FOOTER_BACKGROUND_COLOR, MENU_BACKGROUND_COLOR, MUTED_COLOR, USER_PROMPT_STYLE
from .markdown import THEMES, MarkdownRenderer, code_blocks
from .clipboard import clipboard_sequence, copy_to_clipboard
from .explorer import folder_opener, open_folder
from .viewers import TextViewControl, DiffViewControl
from .inspection import IndexedText, field_snapshot, artifact_snapshot, build_snapshot
from .types import ToolCall

FOOTER_ROWS = 9
PULSE_FRAMES = ("▌", "▛", "▀", "▜", "▐", "▟", "▄", "▙")
MIN_REDRAW_INTERVAL = 1 / 30


class TranscriptControl(UIControl):
    """Ask for indexed rows without FormattedTextControl's whole-text scan."""

    def __init__(self, content: Callable[[int], UIContent], mouse: Callable[[MouseEvent], object] | None = None) -> None:
        self._content = content
        self._mouse = mouse

    def mouse_handler(self, mouse_event: MouseEvent) -> object:
        self.__dict__.setdefault("_mouse", None)
        return self._mouse(mouse_event) if self._mouse is not None else NotImplemented

    def create_content(self, width: int, height: int | None) -> UIContent:
        return self._content(width)


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


class TranscriptWindow(Window):
    """Scroll wrapped output by screen rows without moving input focus."""

    def __init__(self, content: UIControl) -> None:
        super().__init__(content, wrap_lines=True)
        self._following = True
        self._pending_scroll = 0

    def scroll(self, rows: int) -> None:
        self._pending_scroll += rows

    def _scroll(self, ui_content: UIContent, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return

        def line_height(line: int) -> int:
            return ui_content.get_height_for_line(line, width, self.get_line_prefix)

        # The tail can start inside a wrapped line, including lines taller
        # than the entire transcript viewport.
        remaining = height
        bottom = (0, 0)
        for line in range(ui_content.line_count - 1, -1, -1):
            size = line_height(line)
            if size > remaining:
                bottom = (line, size - remaining)
                break
            remaining -= size
            if remaining == 0:
                bottom = (line, 0)
                break

        line, offset = min(
            bottom if self._following else (self.vertical_scroll, self.vertical_scroll_2), bottom,
        )
        offset = min(offset, line_height(line) - 1)
        rows = self._pending_scroll
        self._pending_scroll = 0
        while rows < 0 and (line, offset) > (0, 0):
            if offset:
                offset -= 1
            else:
                line -= 1
                offset = line_height(line) - 1
            rows += 1
        while rows > 0 and (line, offset) < bottom:
            if offset + 1 < line_height(line):
                offset += 1
            else:
                line += 1
                offset = 0
            rows -= 1
        self.vertical_scroll, self.vertical_scroll_2 = min((line, offset), bottom)
        self._following = (self.vertical_scroll, self.vertical_scroll_2) == bottom
        self.horizontal_scroll = 0


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


class TerminalUI:
    """Persistent buffers/application with replaceable presentation methods.

    Original chunks are stored in a session file; wrapped rows are a disposable
    indexed file with a bounded memory cache. Keep drafts/history/queues alive
    during refresh, and isolate secret prompts from those ordinary buffers.
    """
    _selection_start: Point | None
    _selection_end: Point | None

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
        self._password = False
        self._input_label = "> "
        self.input = TextArea(
            height=5, multiline=False, wrap_lines=True,
            prompt=lambda: self._input_label,
            password=Condition(lambda: self._password), accept_handler=lambda buffer: self._accept(buffer),
            style="class:user",
        )
        self.input.window = InputWindow(self.input.control)
        self.input.window.style = "class:text-area class:user"
        self.input.window.height = Dimension.exact(5)
        self.transcript = TranscriptWindow(TranscriptControl(lambda width: self._transcript_content(width)))
        self.app: Application[None] = Application(
            layout=self._layout(), key_bindings=self._bindings(),
            full_screen=True, input=input, output=self.output,
            mouse_support=True, before_render=lambda app: self._update_title(),
            after_render=lambda app: self._route_mouse_wheel(app),
            refresh_interval=.15,
            min_redraw_interval=MIN_REDRAW_INTERVAL,
            color_depth=ColorDepth.DEPTH_24_BIT if color else ColorDepth.DEPTH_1_BIT,
            style=self._style(),
        )
        self._title_pulse_bound = True

    def _layout(self) -> Layout:
        self._ensure_editor()
        self._ensure_options()
        self._ensure_context()
        self._ensure_prompt()
        self._ensure_menu()
        self._ensure_viewer()
        self._ensure_inspection()
        if not isinstance(self.input.window, InputWindow):
            self.input.window = InputWindow(self.input.control)
            self.input.window.style = "class:text-area class:user"
        body_height = Dimension(min=1, preferred=FOOTER_ROWS - 4, max=FOOTER_ROWS - 4)
        self.input.window.height = body_height
        self._menu_window.height = body_height
        self._editor_panel.controls.height = body_height
        self._editor_panel.controls.style = "class:menu"
        self._options_panel.field.window.height = body_height
        # Attach during layout rebuilds so existing sessions gain input wrapping.
        processors = self.input.control.input_processors or []
        if not any(isinstance(processor, WordWrapInput) for processor in processors):
            self.input.control.input_processors = [*processors, WordWrapInput()]
        if self._input_label == "you › ":
            self._input_label = "> "
        activity_row = VSplit([Window(FormattedTextControl(lambda: self._activity()), height=1),
                              Window(FormattedTextControl(self._inspect_link), height=1, width=lambda: self._right_control_width(), align=WindowAlign.RIGHT)])
        scroll_row = VSplit([
            Window(FormattedTextControl(lambda: self._activity()), height=1,
                   width=lambda: max(0, (self.output.get_size().columns - 18) // 2)),
            Window(FormattedTextControl(self._scroll_bottom_link), height=1, width=18),
            Window(FormattedTextControl(self._inspect_link), height=1, align=WindowAlign.RIGHT),
        ])
        footer = HSplit([
            DynamicContainer(lambda: scroll_row if self._show_scroll_bottom() else activity_row),
            DynamicContainer(lambda: self._editor_panel.controls if self._editor_active else self.input),
            self._bottom_row(menu=False),
        ], height=lambda: self._footer_rows() - 2, style="class:footer")
        menu_footer = HSplit([
            Window(FormattedTextControl(lambda: self._menu_title), height=1, style="class:menu-title"),
            self._menu_window,
            self._bottom_row(menu=True),
        ], height=lambda: self._footer_rows() - 2, style="class:footer class:menu")
        setting_footer = HSplit([
            Window(FormattedTextControl("Folder Opener"), height=1, style="class:menu-title"),
            self._options_panel.field,
            self._bottom_row(menu=False),
        ], height=lambda: self._footer_rows() - 2, style="class:footer class:menu")
        context = HSplit([
            Window(FormattedTextControl("Context — latest model request | Ctrl+\\: close"), height=1, style="class:status"),
            self.context_window,
        ])
        # Overlay the prompt so switching between prompts of different heights
        # does not move the scroll position and select a different prompt again.
        pinned_transcript = FloatContainer(self.transcript, floats=[Float(
            top=0, left=0, right=0, height=self._pinned_prompt_height,
            content=Window(TranscriptControl(self._pinned_prompt_content)),
        )])
        transcript = DynamicContainer(lambda: pinned_transcript if self.output.get_size().rows > self._footer_rows() + 1 else self.transcript)
        return Layout(HSplit([
            VSplit([DynamicContainer(lambda: self._editor_container if self._editor_active else self._viewer_container if self._viewer_active else
                                     context if self.context_visible else transcript),
                    ConditionalContainer(self._inspection_panel, Condition(lambda: self._inspection_visible and self.output.get_size().columns >= 60))]),
            Window(height=2),
            DynamicContainer(lambda: menu_footer if self._menu_future is not None else setting_footer if self._options_field_active else footer),
        ]), focused_element=self._menu_window if self._menu_future is not None else self._options_panel.field if self._options_field_active else self._editor_input if self._editor_active else
            self._viewer_window if self._viewer_active else self.input)

    def _ensure_inspection(self) -> None:
        if "_inspection_visible" in self.__dict__:
            if "_touched_items" not in self.__dict__:
                self.__dict__["_touched_items"] = self.__dict__.pop("_touched_files", {})
            return
        self._inspection_visible = False
        self._touched_items: dict[str, str] = {}
        self._inspection_calls: dict[tuple[int, str], dict[str, Any]] = {}
        self._files_window = TranscriptWindow(FormattedTextControl(lambda: self._inspection_entries(files=True)))
        self._calls_window = TranscriptWindow(FormattedTextControl(lambda: self._inspection_entries(files=False)))
        self._files_window._following = self._calls_window._following = False
        self._inspection_panel = HSplit([
            HSplit([Window(FormattedTextControl("Touched"), height=1, style="class:menu-title"),
                    self._files_window]),
            Window(height=1),
            HSplit([Window(FormattedTextControl("Calls"), height=1, style="class:menu-title"),
                    self._calls_window]),
        ], width=lambda: Dimension.exact(max(1, self.output.get_size().columns // 4)), style="class:menu")

    def toggle_inspection(self) -> None:
        self._ensure_inspection()
        self._inspection_visible = not self._inspection_visible
        if self._inspection_visible and self.output.get_size().columns < 60:
            self.notify("Widen terminal to show Inspect.")
        self.app.invalidate()

    def _inspect_link(self) -> StyleAndTextTuples:
        def click(event: MouseEvent) -> object:
            if event.event_type == MouseEventType.MOUSE_MOVE:
                self.__dict__["_hovered_control"] = "inspect"
            elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                self.toggle_inspection()
            return None
        style = "class:clickable" + (" class:hover" if self.__dict__.get("_hovered_control") == "inspect" else "")
        def options(event: MouseEvent) -> object:
            if event.event_type == MouseEventType.MOUSE_MOVE:
                self.__dict__["_hovered_control"] = "options"
            elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                self.app.create_background_task(self._options_panel.show())
            return None
        option_style = "class:clickable" + (" class:hover" if self.__dict__.get("_hovered_control") == "options" else "")
        return [(style, "Inspect [Ctrl+O]" if self._right_control_width() > 18 else "Inspect", click),
                ("", "  "), (option_style, "Options", options), ("", " ")]

    def _right_control_width(self) -> int:
        return 30 if self.output.get_size().columns >= 80 else 18

    def _footer_rows(self) -> int:
        """Leave room for feed output when the terminal is very short."""
        return min(FOOTER_ROWS, max(5, self.output.get_size().rows - 3))

    def _show_scroll_bottom(self) -> bool:
        return not (self.transcript._following or self._viewer_active or self._editor_active or self.context_visible)

    def _scroll_bottom_link(self) -> StyleAndTextTuples:
        def click(event: MouseEvent) -> object:
            if event.event_type == MouseEventType.MOUSE_MOVE:
                self.__dict__["_hovered_control"] = "scroll-bottom"
            elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                self.scroll_to_end()
            return None
        style = "class:clickable"
        if self.__dict__.get("_hovered_control") == "scroll-bottom":
            style += " class:hover"
        return [("", " "), (style, "Scroll to Bottom", click), ("", " ")]

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
                saved["artifacts"] = [record for record in store.records.values() if record["call_id"] == call.id]
        self.app.invalidate()

    def has_diff(self, call_id: str) -> bool:
        store = self.__dict__.get("_artifact_store")
        return store is not None and any(record["kind"] == "diff" and record["call_id"] == call_id for record in store.records.values())

    def _inspection_entries(self, *, files: bool) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        entries: list[tuple[Any, Any]] = list(reversed(list(self._touched_items.items()))) if files else list(reversed(list(self._inspection_calls.items())))
        if not entries:
            return [("class:status", "None yet")]
        for key, value in entries:
            root = self.__dict__.get("_project_directory")
            try:
                directory = files and root is not None and (root / str(key)).is_dir()
            except OSError:
                directory = False
            clickable = not directory or self.__dict__.get("_folder_opener") is not None
            def click(event: MouseEvent, key: Any = key) -> object:
                if event.event_type == MouseEventType.MOUSE_MOVE:
                    self.__dict__["_hovered_control"] = ("inspection", files, key)
                elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                    if files:
                        self.app.create_background_task(self._inspect_file(str(key)))
                    else:
                        call = self._inspection_calls[key]
                        self.app.create_background_task(self._inspect_call(call))
                return None
            label = f"{key} [{value}]" if files else f"{value['tool']} [{value['status']}]"
            style = "class:menu-item"
            if clickable:
                style += " class:clickable" + (" class:hover" if self.__dict__.get("_hovered_control") == ("inspection", files, key) else "")
                fragments.append((style, label, click))
            else:
                fragments.append((style, label))
            fragments.append(("", "\n"))
        return fragments

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
        if request != self.__dict__.get("_viewer_request"):
            text.close()
            return
        self.open_viewer(path, text, line=line)
        self.__dict__["_viewer_source"] = target
        self.__dict__["_viewer_source_stat"] = self._file_signature(before)

    async def _inspect_call(self, call: dict[str, Any]) -> None:
        request = self._new_viewer_request()
        try:
            snapshot = await build_snapshot(field_snapshot, dict(call))
        except (OSError, ValueError, UnicodeError) as exc:
            self.notify("Could not open call record.")
            self.write(f"Call {call['call_id']}: {exc}")
            return
        if request != self.__dict__.get("_viewer_request"):
            snapshot.close()
            return
        self.open_viewer(f"{call['tool']} — {call['call_id']}", snapshot)

    def _new_viewer_request(self) -> int:
        request = int(self.__dict__.get("_viewer_request", 0)) + 1
        self.__dict__["_viewer_request"] = request
        return request

    def _ensure_editor(self) -> None:
        if "_editor_panel" not in self.__dict__:
            from .editorui import EditorPanel
            self._editor_panel = EditorPanel(self)
            self._editor_active = False
            self._editor_input = self._editor_panel.input
            self._editor_container = self._editor_panel.container

    async def _open_editor_now(self, path: Path, *, line: int = 1) -> None:
        self._ensure_editor()
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
        if "_viewer_active" in self.__dict__:
            return
        self._viewer_active = False
        self._viewer_title = ""
        self._viewer_window = TranscriptWindow(TextViewControl(""))
        self._viewer_window._following = False
        self._viewer_container = HSplit([
            Window(FormattedTextControl(self._viewer_header),
                   height=1, style="class:status"),
            self._viewer_window,
        ])

    def _viewer_header(self) -> StyleAndTextTuples:
        text = self._viewer_title + " — Read Only | ↑↓ / Wheel / PgUp / PgDn | Esc: Close"
        if not self.__dict__.get("_viewer_changed"):
            return [("", text)]
        def reload(event: MouseEvent) -> object:
            if event.event_type == MouseEventType.MOUSE_MOVE:
                self.__dict__["_hovered_control"] = "reload-viewer"
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                self.app.create_background_task(self._reload_viewer())
            return None
        return [("class:retry", "File changed on disk. "),
                ("class:clickable" + (" class:hover" if self.__dict__.get("_hovered_control") == "reload-viewer" else ""),
                 "Reload [Ctrl+R]", reload), ("", " | " + text)]

    @staticmethod
    def _file_signature(metadata: Any) -> tuple[int, int, int, int, int]:
        return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)

    async def _check_viewer_file(self, source: Path) -> None:
        try:
            try:
                signature = self._file_signature(await asyncio.to_thread(source.stat))
            except OSError:
                signature = None
            if self.__dict__.get("_viewer_source") == source and signature != self.__dict__.get("_viewer_source_stat"):
                self.__dict__["_viewer_changed"] = True
                self.app.invalidate()
        finally:
            self.__dict__["_viewer_check_pending"] = False

    async def _reload_viewer(self) -> None:
        source = self.__dict__.get("_viewer_source")
        if source is not None:
            scroll = self._viewer_window.vertical_scroll
            await self._read_file(str(source))
            self._viewer_window.vertical_scroll = scroll
            self.app.invalidate()

    def open_viewer(self, title: str, text: str | IndexedText, *, diff: bool = False, line: int = 1) -> None:
        if self.__dict__.get("_editor_active"):
            if isinstance(text, IndexedText):
                text.close()
            self.notify("Close the editor before opening a viewer.")
            return
        self._ensure_viewer()
        self._new_viewer_request()
        old = self._viewer_window.content
        if isinstance(old, TextViewControl):
            old.close()
        self.__dict__.pop("_viewer_source", None)
        self.__dict__["_viewer_changed"] = False
        self._viewer_title = " ".join(title.split())
        self._viewer_window.content = DiffViewControl(text) if diff else TextViewControl(text)
        self._viewer_window.vertical_scroll = text.line_index(line) if isinstance(text, IndexedText) else max(0, line - 1)
        self._viewer_window.vertical_scroll_2 = self._viewer_window._pending_scroll = 0
        self._viewer_window._following = False
        self._viewer_active = True
        self.app.layout.focus(self._viewer_window)
        self.app.invalidate()

    def close_viewer(self) -> None:
        self._new_viewer_request()
        self._viewer_active = False
        self.__dict__.pop("_viewer_source", None)
        content = self._viewer_window.content
        if isinstance(content, TextViewControl):
            content.close()
        self._viewer_window.content = TextViewControl("")
        self.app.layout.focus(self._menu_window if self._menu_future is not None else self.input)
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
        if request != self.__dict__.get("_viewer_request"):
            snapshot.close()
            return
        self.open_viewer(f"Record {record_id}", snapshot, diff=record["kind"] == "diff", line=line)

    def write_artifact_reference(self, record: dict[str, object]) -> None:
        identifier = record["record_id"]
        self._ensure_transcript()
        self.__dict__.setdefault("_artifact_links", {})[len(self._raw_output)] = identifier
        for call in self.__dict__.get("_inspection_calls", {}).values():
            if call["call_id"] == record.get("call_id") and call["step_id"] == record.get("step_id"):
                references = call.setdefault("artifacts", [])
                if not any(saved["record_id"] == identifier for saved in references):
                    references.append(record)
        label = "Diff" if record["kind"] == "diff" else "Output"
        self.write(f"    View {label} — Record {identifier}\n")

    def _artifact_at(self, point: Point) -> int | None:
        if self._flow is None:
            return None
        rows = self._flow.content_rows()
        row = self._flow.get_row(point.y) if 0 <= point.y < len(self._flow.rows) else (
            rows[point.y - len(self._flow.rows)] if 0 <= point.y - len(self._flow.rows) < len(rows) else []
        )
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
        self._menu_options: list[tuple[str, str]] = []
        self._menu_index = 0
        self._menu_window = Window(FormattedTextControl(
            lambda: self._menu_fragments(), focusable=True, show_cursor=False,
            get_cursor_position=lambda: Point(x=0, y=self._menu_index),
        ), height=FOOTER_ROWS - 4, wrap_lines=True)

    def _notice_width(self) -> int:
        columns = self.output.get_size().columns
        return max(1, min(44, columns // 2, columns - 28))

    def _readout_width(self) -> int:
        return max(1, self.output.get_size().columns - self._notice_width() - 1)

    def _bottom_row(self, *, menu: bool) -> VSplit:
        return VSplit([
            Window(FormattedTextControl(lambda: self._menu_hint() if menu else ANSI("Enter: Save | Esc: Cancel") if self._options_field_active else self._status_fragments()),
                   height=1, style="class:status"),
            Window(width=1, height=1),
            self._notification_window(),
        ], height=1)

    def _notification_window(self) -> Window:
        return Window(FormattedTextControl(self._notification_text), height=1,
                      width=lambda: Dimension.exact(self._notice_width()),
                      align=WindowAlign.RIGHT, style="class:status", wrap_lines=False)

    def notify(self, text: str) -> None:
        """Show a notice until replaced or explicitly cleared with an empty string."""
        self.__dict__["_notification"] = " ".join(text.split())
        self.app.invalidate()

    def _notification_text(self) -> str:
        text = self.__dict__.get("_notification", "")
        if not text:
            tips = ["Select and right-click for text options."]
            if self._editor_active:
                tips = ["Ctrl+S saves; Esc closes the editor.", "PgUp/PgDn changes file pages."]
            elif self._viewer_active:
                tips = ["Scroll to inspect; Esc closes the viewer."]
            if not self.working and self._input_label == "> " and self._menu_future is None:
                tips.append("Click Commands to browse commands.")
            if self.__dict__.get("_folder_opener") is not None:
                tips.append("Click directory links to open folders.")
            text = tips[int(time.monotonic() // 15) % len(tips)]
        columns = self._notice_width()
        if display_width(text) <= columns:
            return str(text)
        result = ""
        for cluster in iter_graphemes(text):
            if display_width(result + cluster) > columns - 1:
                break
            result += cluster
        return result + "…"

    def _menu_fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for index, (_, label) in enumerate(self._menu_options):
            if index:
                fragments.append(("", "\n"))
            selected = index == self._menu_index
            def select(event: MouseEvent, index: int = index) -> object:
                if event.event_type == MouseEventType.MOUSE_MOVE:
                    self.__dict__["_hovered_control"] = ("menu", index)
                if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                    self._menu_index = index
                    self._finish_menu(self._menu_options[index][0])
                return None
            style = "class:clickable class:menu-item" + (" class:menu-selected" if selected else "")
            if self.__dict__.get("_hovered_control") == ("menu", index):
                style += " class:hover"
            fragments.append((style,
                              ("› " if selected else "  ") + label, select))
        return fragments

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
        def back(event: MouseEvent) -> object:
            if event.event_type == MouseEventType.MOUSE_MOVE:
                self.__dict__["_hovered_control"] = "menu-back"
            elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                self._finish_menu(None)
            return None
        style = "class:clickable"
        if self.__dict__.get("_hovered_control") == "menu-back":
            style += " class:hover"
        return [("", prefix), (style, "back", back), ("", tail), *to_formatted_text(ANSI(suffix))]

    async def choose(self, title: str, options: list[tuple[str, str]]) -> str | None:
        """Choose a value/label pair in the footer; cancellation preserves the draft."""
        self._ensure_menu()
        if self._menu_future is not None:
            raise RuntimeError("A menu is already open.")
        if not options or self._eof:
            return None
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._menu_future = future
        self._menu_title = title
        self._menu_options = [(value, re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", label))
                              for value, label in options]
        self._menu_index = 0
        self._menu_window.vertical_scroll = self._menu_window.vertical_scroll_2 = 0
        try:
            self.app.layout.focus(self._menu_window)
            self.app.invalidate()
            return await future
        finally:
            self._menu_future = None
            self._menu_options = []
            self.app.layout.focus(self._editor_input if self._editor_active else self._viewer_window if self._viewer_active else self.input)
            self.app.invalidate()

    def _move_menu(self, rows: int) -> None:
        self._menu_index = max(0, min(len(self._menu_options) - 1, self._menu_index + rows))
        self.app.invalidate()

    def _finish_menu(self, value: str | None) -> None:
        if self._menu_future is not None and not self._menu_future.done():
            self._menu_future.set_result(value)

    def insert_command(self, command: str) -> None:
        """Retain the ordinary draft in input history without submitting it."""
        self.input.buffer.append_to_history()
        self.input.buffer.reset(document=Document(command, len(command)))
        self.app.layout.focus(self.input)
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
                link_style = style + " class:clickable"
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
        if not await asyncio.to_thread(open_folder, directory, opener):
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
        self.write(f"    {label}\n")

    def write_file_reference(self, value: str, *, line: int = 1) -> None:
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
        self.write(f"    {label}:{line}\n")

    def _file_at(self, point: Point) -> tuple[Path, int] | None:
        if self._flow is None:
            return None
        rows = self._flow.content_rows()
        row = self._flow.get_row(point.y) if 0 <= point.y < len(self._flow.rows) else (
            rows[point.y - len(self._flow.rows)] if 0 <= point.y - len(self._flow.rows) < len(rows) else [])
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
            columns = size.columns
            if self.__dict__.get("_inspection_visible") and columns >= 60:
                columns -= columns // 4
        self._transcript_content(columns)
        assert self._flow is not None
        self._pinned_prompt = self._flow.prompt_at(self.transcript.vertical_scroll)
        if self._pinned_prompt is None:
            return 0
        # One header row, leaving at least one transcript row above the footer.
        available = max(0, size.rows - self._footer_rows() - 1)
        return min(1, available)

    def _pinned_prompt_content(self, width: int) -> UIContent:
        height = self._pinned_prompt_height(width)
        source = self._transcript_content(width)
        start, end = self._pinned_prompt or (0, 0)
        text = fragment_list_to_text(source.get_line(start)) if height else ""
        if height and end > start + 1:
            # Reuse bounded, already wrapped rows instead of rereading a long
            # prompt. A first word can wrap onto its own row after the > marker.
            if text.strip() == ">":
                text += " " + fragment_list_to_text(source.get_line(start + 1)).lstrip()
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
        """Initialize optional view state for both new and already running terminals."""
        if hasattr(self, "context_window"):
            return
        self.context_visible = False
        self._context_blocks: list[tuple[str, str]] = [("", "No model request has been sent yet.")]
        self._context_columns = 0
        self._context_fragments: StyleAndTextTuples = []
        self.context_window = TranscriptWindow(FormattedTextControl(lambda: self._formatted_context()))
        self.context_window._following = False

    def set_context(self, request: str) -> None:
        """Replace the view with an immutable snapshot of the actual request."""
        self._ensure_context()
        blocks: list[tuple[str, str]] = []
        for field, value in json.loads(request).items():
            if field == "messages":
                blocks.append(("", "Messages — sent order"))
                for index, message in enumerate(value, 1):
                    role = message.get("role", "unknown")
                    style = "class:context-system" if role == "system" else ""
                    content = message.get("content")
                    text = f"[{index:02}] {role.upper()}\n"
                    text += content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
                    extra = {key: item for key, item in message.items() if key not in {"role", "content"}}
                    if extra:
                        text += "\n" + json.dumps(extra, ensure_ascii=False, indent=2)
                    blocks.append((style, text))
            else:
                blocks.append(("", field + ":\n" + json.dumps(value, ensure_ascii=False, indent=2)))
        self._context_blocks = blocks
        self._context_columns = 0
        self.context_window.vertical_scroll = 0
        self.context_window.vertical_scroll_2 = 0
        self.context_window._pending_scroll = 0
        self.context_window._following = False
        self.app.invalidate()

    def _formatted_context(self) -> StyleAndTextTuples:
        columns = max(1, self.output.get_size().columns)
        if columns != self._context_columns:
            self._context_fragments = []
            for index, (style, text) in enumerate(self._context_blocks):
                if index:
                    self._context_fragments.append(("", "\n"))
                for line in text.split("\n"):
                    lines = wrap(line, columns, replace_whitespace=False, break_on_hyphens=False) or [""]
                    self._context_fragments.append((style, "\n".join(lines) + "\n"))
            self._context_columns = columns
        return self._context_fragments

    def _scroll_output(self, rows: int) -> None:
        if self._menu_future is not None:
            self._move_menu(rows)
            return
        window = self._viewer_window if self.__dict__.get("_viewer_active") else self.context_window if self.context_visible else self.transcript
        window.scroll(rows)
        self.app.invalidate()

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        in_menu = Condition(lambda: self._menu_future is not None)
        in_setting = Condition(lambda: self.__dict__.get("_options_field_active", False))
        in_editor = Condition(lambda: self.__dict__.get("_editor_active", False)) & ~in_menu & ~in_setting
        in_viewer = Condition(lambda: self.__dict__.get("_viewer_active", False)) & ~in_menu & ~in_editor & ~in_setting

        @bindings.add("f4", filter=~in_menu & ~in_setting)
        def options_screen(event: KeyPressEvent) -> None:
            self.app.create_background_task(self._options_panel.show())

        @bindings.add("escape", filter=in_setting, eager=True)
        def cancel_setting(event: KeyPressEvent) -> None:
            self._options_panel.cancel()

        @bindings.add("tab", filter=in_editor)
        def editor_tab(event: KeyPressEvent) -> None:
            if not self._editor_panel.busy:
                self._editor_input.buffer.insert_text(" " * self._options_panel.values.get("tab_width", 4))

        @bindings.add("c-z", filter=in_editor)
        @bindings.add("c-y", filter=in_editor)
        def editor_undo(event: KeyPressEvent) -> None:
            if not self._editor_panel.busy:
                if event.key_sequence[-1].key == Keys.ControlZ:
                    self._editor_input.buffer.undo()
                else:
                    self._editor_input.buffer.redo()

        @bindings.add("c-s", filter=in_editor)
        def save_editor(event: KeyPressEvent) -> None:
            self.app.create_background_task(self._save_editor())

        @bindings.add("escape", filter=in_editor, eager=True)
        def close_editor(event: KeyPressEvent) -> None:
            self.app.create_background_task(self._close_editor())

        @bindings.add("c-r", filter=in_editor)
        def reload_editor(event: KeyPressEvent) -> None:
            self.app.create_background_task(self._reload_editor())

        @bindings.add("pageup", filter=in_editor)
        @bindings.add("pagedown", filter=in_editor)
        def page_editor(event: KeyPressEvent) -> None:
            self._change_editor_page(-1 if event.key_sequence[-1].key == Keys.PageUp else 1)

        @bindings.add("c-o")
        def inspect_panel(event: KeyPressEvent) -> None:
            self.toggle_inspection()

        @bindings.add("escape", filter=in_viewer, eager=True)
        def close_view(event: KeyPressEvent) -> None:
            self.close_viewer()

        @bindings.add("c-r", filter=in_viewer)
        def reload_view(event: KeyPressEvent) -> None:
            self.app.create_background_task(self._reload_viewer())

        @bindings.add("up", filter=in_viewer)
        @bindings.add("down", filter=in_viewer)
        @bindings.add("pageup", filter=in_viewer)
        @bindings.add("pagedown", filter=in_viewer)
        def scroll_view(event: KeyPressEvent) -> None:
            key = event.key_sequence[-1].key
            rows = max(1, self.output.get_size().rows - self._footer_rows() - 1) if key in {"pageup", "pagedown"} else 1
            self._scroll_output(-rows if key in {"up", "pageup"} else rows)

        @bindings.add("home", filter=in_viewer)
        @bindings.add("end", filter=in_viewer)
        def edge_view(event: KeyPressEvent) -> None:
            self._viewer_window._following = event.key_sequence[-1].key == Keys.End
            self._viewer_window.vertical_scroll = self._viewer_window.vertical_scroll_2 = 0
            self._viewer_window._pending_scroll = 0
            self.app.invalidate()

        @bindings.add("f2", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting)
        def toggle_markdown(event: KeyPressEvent) -> None:
            self.configure_markdown(source=not self.__dict__.get("_markdown_source", False))

        @bindings.add("f3", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting)
        def copy_response(event: KeyPressEvent) -> None:
            if self.last_markdown():
                self.copy_response()

        @bindings.add("up", filter=in_menu)
        @bindings.add("down", filter=in_menu)
        def move_menu(event: KeyPressEvent) -> None:
            self._move_menu(-1 if event.key_sequence[-1].key == Keys.Up else 1)

        @bindings.add("enter", filter=in_menu, eager=True)
        def select_menu(event: KeyPressEvent) -> None:
            self._finish_menu(self._menu_options[self._menu_index][0])

        @bindings.add("escape", filter=in_menu, eager=True)
        @bindings.add("c-d", filter=in_menu)
        def cancel_menu(event: KeyPressEvent) -> None:
            self._finish_menu(None)

        @bindings.add("escape", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting, eager=True)
        def stop_now(event: KeyPressEvent) -> None:
            callback = getattr(self, "_interrupt_callback", None)
            if callback is not None:
                callback()

        @bindings.add(Keys.Any, filter=in_menu)
        def ignore_menu_text(event: KeyPressEvent) -> None:
            pass

        @bindings.add("c-\\", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting)
        def toggle_context(event: KeyPressEvent) -> None:
            self.context_visible = not self.context_visible
            self.app.invalidate()

        in_context = Condition(lambda: self.context_visible) & ~in_menu & ~in_viewer & ~in_editor & ~in_setting

        @bindings.add("up", filter=in_context)
        @bindings.add("down", filter=in_context)
        @bindings.add("pageup", filter=in_context)
        @bindings.add("pagedown", filter=in_context)
        def scroll_context(event: KeyPressEvent) -> None:
            key = event.key_sequence[-1].key
            rows = max(1, self.output.get_size().rows - self._footer_rows() - 1) if key in {"pageup", "pagedown"} else 1
            self._scroll_output(-rows if key in {"up", "pageup"} else rows)

        @bindings.add("home", filter=in_context)
        def context_top(event: KeyPressEvent) -> None:
            self.context_window._following = False
            self.context_window.vertical_scroll = 0
            self.context_window.vertical_scroll_2 = 0
            self.context_window._pending_scroll = 0
            self.app.invalidate()

        @bindings.add("end", filter=in_context)
        def context_bottom(event: KeyPressEvent) -> None:
            self.context_window._following = True
            self.app.invalidate()

        @bindings.add("<scroll-up>")
        def scroll_up(event: KeyPressEvent) -> None:
            self._scroll_output(-3)

        @bindings.add("<scroll-down>")
        def scroll_down(event: KeyPressEvent) -> None:
            self._scroll_output(3)

        @bindings.add("c-d", filter=~in_menu & ~in_viewer & ~in_editor & ~in_setting)
        def eof(event: KeyPressEvent) -> None:
            if not event.current_buffer.text:
                self._end_input()
            else:
                event.current_buffer.delete()

        return bindings

    def set_interrupt_handler(self, callback: Callable[[], None]) -> None:
        """Interrupt work directly, without submitting or changing the draft."""
        self._interrupt_callback = callback

    def _style(self) -> Style:
        return Style.from_dict({
            "status": MUTED_COLOR, "pulse": "bold ansicyan", "idle": "ansicyan", "user": USER_PROMPT_STYLE,
            "context-system": "#ffff00", "retry": "ansired",
            "footer": "bg:" + FOOTER_BACKGROUND_COLOR,
            "menu": "bg:" + MENU_BACKGROUND_COLOR,
            "menu-title": "bold", "menu-selected": "reverse bold",
            "clickable": "underline", "hover": "reverse",
            "diff-number": MUTED_COLOR,
            "menu-item": "#999999",
            "diff-added": "#66dd88", "diff-removed": "#ff6666",
            "diff-header": "bold #ffdd66", "diff-hunk": "#66ccff",
        })

    def refresh(self) -> None:
        """Rebuild presentation while retaining input, transcript, and application."""
        # Build fallible presentation pieces before migrating old storage, so
        # a rejected reload can still run the previous generation's methods.
        layout = self._layout()
        bindings = self._bindings()
        style = self._style()
        self._ensure_transcript()
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
        original = app.renderer.mouse_handlers

        def mouse_handler(event: MouseEvent) -> object:
            previous_hover = self.__dict__.get("_hovered_control")
            if event.event_type == MouseEventType.MOUSE_MOVE:
                self.__dict__["_hovered_control"] = None
            if event.event_type == MouseEventType.MOUSE_DOWN:
                self._selection_dragging = False
            if event.event_type in (MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN):
                size = self.output.get_size()
                if self._menu_future is not None:
                    self._move_menu(-1 if event.event_type == MouseEventType.SCROLL_UP else 1)
                    return None
                if self._editor_active and event.position.y < size.rows - self._footer_rows() and self._menu_future is None:
                    if not (self._inspection_visible and size.columns >= 60 and event.position.x >= size.columns - size.columns // 4):
                        self._editor_panel.move_cursor(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
                        app.invalidate()
                        return None
                if self._menu_future is None and size.rows - self._footer_rows() + 3 <= event.position.y < size.rows - 1:
                    if self._editor_active:
                        return None
                    window = self.input.window
                    if isinstance(window, InputWindow):
                        window.scroll(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
                    app.invalidate()
                    return None
                if self.__dict__.get("_inspection_visible") and size.columns >= 60 and event.position.x >= size.columns - size.columns // 4 and event.position.y < size.rows - self._footer_rows() and self._menu_future is None:
                    screen = app.renderer._last_screen
                    if screen is not None:
                        for pane in (self._files_window, self._calls_window):
                            position = screen.visible_windows_to_write_positions.get(pane)
                            if position is not None and position.ypos <= event.position.y < position.ypos + position.height:
                                pane.scroll(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
                                break
                    app.invalidate()
                    return None
                self._scroll_output(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
                app.invalidate()
                return None
            if self._menu_future is not None and event.event_type not in (MouseEventType.MOUSE_UP, MouseEventType.MOUSE_MOVE):
                return None  # Keep focus on the menu until selection/cancellation.
            try:
                screen = app.renderer._last_screen
                if screen is not None:
                    for hit_window in (self._menu_window, self._files_window, self._calls_window):
                        position = screen.visible_windows_to_write_positions.get(hit_window)
                        if position is None or not (position.xpos <= event.position.x < position.xpos + position.width
                                                   and position.ypos <= event.position.y < position.ypos + position.height):
                            continue
                        info = hit_window.render_info
                        row = event.position.y - position.ypos
                        source = info.visible_line_to_row_col.get(row) if info is not None else None
                        # Window otherwise clamps blank rows onto the last item.
                        if source is None or info is None or not fragment_list_to_text(info.ui_content.get_line(source[0])):
                            return None
                return original.mouse_handlers[event.position.y][event.position.x](event)
            finally:
                # Releases outside the transcript must also end its drag.
                if event.event_type == MouseEventType.MOUSE_UP:
                    self._selection_dragging = False
                if previous_hover != self.__dict__.get("_hovered_control"):
                    app.invalidate()

        # Wheel events target the input under the pointer, active menu, or output. Clicks retain
        # their normal behavior in the input field when no menu is open.
        handlers = MouseHandlers()
        size = self.output.get_size()
        handlers.set_mouse_handler_for_range(0, size.columns, 0, size.rows, mouse_handler)
        app.renderer.mouse_handlers = handlers

    def _accept(self, buffer: Buffer) -> bool:
        self._lines.put_nowait(buffer.text)
        return False

    def _activity(self) -> StyleAndTextTuples:
        fragments = self._activity_status()
        if self._viewer_active or self._editor_active:
            mode = "editor" if self._editor_active else "viewer"
            def close(event: MouseEvent) -> object:
                if event.event_type == MouseEventType.MOUSE_MOVE:
                    self.__dict__["_hovered_control"] = "exit-" + mode
                elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                    if mode == "editor":
                        self.app.create_background_task(self._close_editor())
                    else:
                        self.close_viewer()
                return None
            style = "class:clickable" + (" class:hover" if self.__dict__.get("_hovered_control") == "exit-" + mode else "")
            if display_width(fragment_list_to_text(fragments)) + len("  ESC to exit viewer ") > self.output.get_size().columns - self._right_control_width():
                fragments = []
            fragments.extend([("", "  ESC to " if fragments else "ESC to "), (style, "exit " + mode, close), ("", " ")])
        return fragments

    def _activity_status(self) -> StyleAndTextTuples:
        if self._editor_active:
            return [("class:idle", "Agent Held")]
        edits = self.__dict__.get("_user_edits")
        if edits is not None and getattr(edits, "pending", None) is not None:
            frame = self.__dict__.get("_activity_frame", PULSE_FRAMES[int(time.monotonic() * 4) % len(PULSE_FRAMES)])
            return [("class:pulse", f"{frame} Opening editor after this turn")]
        retry_status = self.__dict__.get("_retry_status", "")
        if retry_status and not self.stopping:
            return [("class:retry", retry_status)]
        command_status = self.__dict__.get("_command_status", "")
        if command_status:
            return [("class:idle", command_status)]
        if not self.working:
            fragments: StyleAndTextTuples = [("class:idle", "Ready")]
            if self._input_label == "> " and not self._viewer_active:
                style = "class:clickable"
                if self.__dict__.get("_hovered_control") == "commands":
                    style += " class:hover"
                # Window maps clicks past text onto its last character.
                fragments.extend([("", "  "), (style, "Commands", self._commands_mouse), ("", " ")])
            return fragments
        frame = self.__dict__.get("_activity_frame", PULSE_FRAMES[int(time.monotonic() * 4) % len(PULSE_FRAMES)])
        label = "Stopping After This Step" if self.stopping else "Working" if self._viewer_active else "Working (esc to interrupt)"
        return [("class:pulse", f"{frame} {label}")]

    def set_retry_status(self, text: str) -> None:
        """Show transient retry progress without appending a line each second."""
        self.__dict__["_retry_status"] = text
        self.app.invalidate()

    def set_working(self, working: bool, *, stopping: bool = False) -> None:
        self.working = working
        self.stopping = stopping
        if not working:
            self.__dict__["_retry_status"] = ""
        self._update_title()
        self.app.invalidate()

    def set_command_status(self, text: str) -> None:
        """Distinguish settings changes from active model work."""
        self.__dict__["_command_status"] = text
        self.app.invalidate()

    def set_title(self, title: str) -> None:
        """Update the terminal/tab title through the platform's output adapter."""
        self._title = title
        self._update_title()

    def _update_title(self) -> None:
        """Refresh the title animation and schedule open-file change checks."""
        source = self.__dict__.get("_viewer_source")
        now = time.monotonic()
        if source is not None and not self.__dict__.get("_viewer_check_pending") and now >= self.__dict__.get("_viewer_check_at", 0):
            self.__dict__["_viewer_check_at"] = now + 1
            self.__dict__["_viewer_check_pending"] = True
            self.app.create_background_task(self._check_viewer_file(source))
        frame = PULSE_FRAMES[int(time.monotonic() * 4) % len(PULSE_FRAMES)]
        self._activity_frame = frame
        title = self.__dict__.get("_title")
        if title is None:
            return
        displayed = f"{frame} {title}" if self.working else title
        if displayed != self.__dict__.get("_displayed_title"):
            self.output.set_title(displayed)
            self.output.flush()
            self._displayed_title = displayed

    def clear_transcript(self) -> None:
        """Replace display storage while retaining the input draft and application."""
        self._ensure_transcript()
        archive = TranscriptFile()
        self._close_transcript()
        self._raw_output = archive
        self._block_starts.clear()
        self.__dict__.setdefault("_directory_links", {}).clear()
        self.__dict__.setdefault("_file_links", {}).clear()
        self.__dict__.setdefault("_artifact_links", {}).clear()
        self.__dict__.setdefault("_touched_items", {}).clear()
        self.__dict__.setdefault("_inspection_calls", {}).clear()
        if self.__dict__.get("_viewer_active"):
            self.close_viewer()
        self._continuation_indents.clear()
        self._markdown_chunks.clear()
        self._markdown_finals.clear()
        self._selection_start = self._selection_end = None
        self._selection_dragging = False
        self.forget_queued_notices()
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

    def write_markdown(self, text: str, *, first: bool = True, final: bool = True) -> None:
        self._ensure_transcript()
        index = len(self._raw_output)
        self._markdown_chunks.add(index)
        if final:
            self._markdown_finals.add(index)
        self.write_chunk(text, first=first)

    def finish_markdown(self) -> None:
        self._ensure_transcript()
        if self._raw_output and len(self._raw_output) - 1 in self._markdown_chunks:
            if len(self._raw_output) - 1 not in self._markdown_finals:
                self.write_markdown("", first=False, final=True)

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

    def selected_text(self) -> str:
        if self._flow is None or self._selection_start is None or self._selection_end is None:
            return ""
        start, end = sorted([self._selection_start, self._selection_end], key=lambda p: (p.y, p.x))
        pieces = []
        count = len(self._flow.rows) + len(self._flow.content_rows())
        for index in range(max(0, start.y), min(end.y + 1, count)):
            row = self._flow.copy_row(index)
            pieces.append(row.selected(start.x if index == start.y else 0, end.x if index == end.y else None))
        return "".join(pieces)

    def write_queued_notice(self, text: str, prompt: str) -> None:
        """Append a removable full line immediately after a user's prompt."""
        self._ensure_transcript()
        self._queued_notices[len(self._raw_output)] = prompt
        self.write_chunk(text + "\n")

    def forget_queued_notices(self) -> None:
        """Detach old queue receipts when resetting or replacing a session."""
        self._queued_notices: dict[int, str] = {}
        self._notice_rows: dict[int, tuple[int, int]] = {}

    def queued_prompt_sent(self, prompt: str) -> None:
        self._ensure_transcript()
        index = next((i for i, text in self._queued_notices.items() if text == prompt), None)
        if index is None:
            return
        del self._queued_notices[index]
        self._raw_output.clear_record(index)
        span = self._notice_rows.pop(index, None)
        if span is not None and self._flow is not None:
            start, stop = span
            count = stop - start
            self._flow.remove_rows(start, stop)
            self._notice_rows = {i: (left - count, right - count) if left >= stop else (left, right)
                                 for i, (left, right) in self._notice_rows.items()}
            self.transcript.vertical_scroll -= min(count, max(0, self.transcript.vertical_scroll - start))
            self._line_count -= count
            self._pinned_prompt = None
        self.app.invalidate()

    def _ensure_transcript(self) -> None:
        """Migrate pre-file-buffer sessions once, without rebuilding the frame."""
        # Older running sessions have no explicit wrap alignment metadata.
        self.__dict__.setdefault("_continuation_indents", {})
        self.__dict__.setdefault("_queued_notices", {})
        self.__dict__.setdefault("_notice_rows", {})
        # Existing highlighted selections are inactive after adopting this fix.
        self.__dict__.setdefault("_selection_dragging", False)
        # Explicit migration adds Markdown metadata without treating historical ANSI as Markdown.
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
        self.transcript.content = TranscriptControl(lambda width: self._transcript_content(width))

    def _transcript_content(self, width: int) -> UIContent:
        self._ensure_transcript()
        self._ensure_prompt()
        columns = max(1, width)
        if self._flow is None or columns != self._wrapped_columns or not hasattr(self._flow, "prompt_rows"):
            if self._flow is not None:
                self._flow.close()
            self._flow = WrappedTranscript(columns, theme=self._markdown_theme,
                                           color=self.app.color_depth != ColorDepth.DEPTH_1_BIT,
                                           source=self._markdown_source)
            self._notice_rows.clear()
            self._selection_start = self._selection_end = None
            self._selection_dragging = False
            self._wrapped_count = 0
            self._wrapped_columns = columns
        flow = self._flow
        for index in range(self._wrapped_count, len(self._raw_output)):
            start = len(flow.rows)
            if index in self._markdown_chunks:
                flow.append_markdown(self._raw_output[index], first=index in self._block_starts,
                                     final=index in self._markdown_finals)
            else:
                flow.append(self._raw_output[index], first=index in self._block_starts,
                            continuation_indents=self._continuation_indents.get(index),
                            extra_style=(f" class:directory-{index}" if index in self.__dict__.get("_directory_links", {}) else "") +
                                        (f" class:file-{index}" if index in self.__dict__.get("_file_links", {}) else "") +
                                        (f" class:artifact-{index}" if index in self.__dict__.get("_artifact_links", {}) else ""))
            if index in self._queued_notices:
                self._notice_rows[index] = (start, len(flow.rows))
        self._wrapped_count = len(self._raw_output)
        preview = flow.content_rows()
        complete = len(flow.rows)
        self._line_count = complete + len(preview)

        def get_line(index: int) -> StyleAndTextTuples:
            if index < complete:
                fragments = flow.get_row(index)
                if self._selection_start is not None and self._selection_end is not None:
                    start, end = sorted([self._selection_start, self._selection_end], key=lambda p: (p.y, p.x))
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
            return decorate_links(preview[offset]) if offset < len(preview) else []

        def decorate_links(fragments: StyleAndTextTuples) -> StyleAndTextTuples:
            result: StyleAndTextTuples = []
            for style, text, *_ in fragments:
                for token in style.split():
                    if token.startswith("class:file-"):
                        file = self.__dict__.get("_file_links", {}).get(int(token.removeprefix("class:file-")))
                        style += " class:clickable"
                        if self.__dict__.get("_hovered_control") == ("file", file):
                            style += " class:hover"
                    if token.startswith("class:artifact-"):
                        identifier = self.__dict__.get("_artifact_links", {}).get(int(token.removeprefix("class:artifact-")))
                        style += " class:clickable"
                        if self.__dict__.get("_hovered_control") == ("artifact", identifier):
                            style += " class:hover"
                    if token.startswith("class:directory-") and self.__dict__.get("_folder_opener") is not None:
                        directory = self.__dict__.get("_directory_links", {}).get(int(token.removeprefix("class:directory-")))
                        style += " class:clickable"
                        if self.__dict__.get("_hovered_control") == ("directory", directory):
                            style += " class:hover"
                result.append((style, text))
            return result

        return UIContent(get_line=get_line, line_count=self._line_count + 1,
                         cursor_position=Point(x=0, y=max(0, self._line_count - 1)), show_cursor=False)

    async def read_line(self) -> str | None:
        if self._eof and self._lines.empty():
            return None
        return await self._lines.get()

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
        if password:
            # A password processor only masks display. A separate buffer and
            # queue also isolate history, drafts, and input already queued.
            self.input.buffer = Buffer(history=DummyHistory(), multiline=False,
                                       accept_handler=lambda buffer: self._accept(buffer))
            self.input.control.buffer = self.input.buffer
            self._lines = asyncio.Queue()
        self._input_label = prompt
        self._password = password
        self.app.invalidate()
        try:
            return await self.read_line() or ""
        finally:
            if password:
                self.input.buffer.reset()
                self.input.buffer = ordinary_buffer
                self.input.control.buffer = ordinary_buffer
                self._lines = ordinary_lines
            self._input_label = "> "
            self._password = False
            self.app.invalidate()

    async def run(self) -> None:
        try:
            await self.app.run_async()
        except EOFError:
            pass
        finally:
            self._end_input()
            self._close_transcript()

    def _close_transcript(self) -> None:
        panel = self.__dict__.get("_editor_panel")
        if panel is not None:
            panel.dispose()
        coordinator = self.__dict__.get("_user_edits")
        if coordinator is not None:
            coordinator.close()
        self._new_viewer_request()
        viewer = self.__dict__.get("_viewer_window")
        if viewer is not None and isinstance(viewer.content, TextViewControl):
            viewer.content.close()
        self._raw_output.close()
        if self._flow is not None:
            self._flow.close()

    def close(self) -> None:
        self._finish_menu(None)
        if self.app.is_running:
            self.app.exit()
        else:
            self._close_transcript()
