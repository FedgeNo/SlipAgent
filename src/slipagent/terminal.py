"""Fixed REPL footer and a scrolling transcript on interactive terminals."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from typing import TextIO

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.input import Input
from prompt_toolkit.history import DummyHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Dimension, DynamicContainer, Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output import ColorDepth, Output, create_output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from wcwidth import iter_graphemes, width as display_width, wrap

from .transcript import TranscriptFile, WrappedTranscript
from .palette import MUTED_COLOR, USER_COLOR

FOOTER_ROWS = 6
PULSE_FRAMES = ("·", "•", "●", "•")
MIN_REDRAW_INTERVAL = 1 / 30


class TranscriptControl(UIControl):
    """Ask for indexed rows without FormattedTextControl's whole-text scan."""

    def __init__(self, content: Callable[[int], UIContent]) -> None:
        self._content = content

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


class TerminalUI:
    """Persistent buffers/application with replaceable presentation methods.

    Original chunks are stored in a session file; wrapped rows are a disposable
    indexed file with a bounded memory cache. Keep drafts/history/queues alive
    during refresh, and isolate secret prompts from those ordinary buffers.
    """
    def __init__(
        self, status: Callable[[int], str], stream: TextIO, *, color: bool = True,
        input: Input | None = None, output: Output | None = None,
        status_suffix: Callable[[], str] | None = None,
    ) -> None:
        self.output = output if output is not None else create_output(stdout=stream)
        self._status = status
        self._status_suffix = status_suffix
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
            height=3, multiline=False, wrap_lines=True,
            prompt=lambda: self._input_label,
            password=Condition(lambda: self._password), accept_handler=lambda buffer: self._accept(buffer),
            style="class:user",
        )
        self.input.window.height = Dimension.exact(3)
        self.transcript = TranscriptWindow(TranscriptControl(lambda width: self._transcript_content(width)))
        self.app: Application[None] = Application(
            layout=self._layout(), key_bindings=self._bindings(),
            full_screen=True, input=input, output=self.output,
            mouse_support=True, after_render=lambda app: self._route_mouse_wheel(app),
            refresh_interval=.15,
            min_redraw_interval=MIN_REDRAW_INTERVAL,
            color_depth=ColorDepth.DEPTH_24_BIT if color else ColorDepth.DEPTH_1_BIT,
            style=self._style(),
        )

    def _layout(self) -> Layout:
        self._ensure_context()
        self._ensure_prompt()
        self._ensure_menu()
        # Attach during layout rebuilds so existing sessions gain input wrapping.
        processors = self.input.control.input_processors or []
        if not any(isinstance(processor, WordWrapInput) for processor in processors):
            self.input.control.input_processors = [*processors, WordWrapInput()]
        if self._input_label == "you › ":
            self._input_label = "> "
        footer = HSplit([
            Window(FormattedTextControl(lambda: self._activity()), height=1),
            self.input,
            Window(height=1),
            Window(FormattedTextControl(lambda: ANSI(self._status(self.output.get_size().columns))),
                   height=1, style="class:status"),
        ], height=FOOTER_ROWS)
        menu_footer = HSplit([
            Window(FormattedTextControl(lambda: self._menu_title), height=1, style="class:menu-title"),
            self._menu_window,
            Window(FormattedTextControl(lambda: self._menu_hint()),
                   height=1, style="class:status"),
        ], height=FOOTER_ROWS)
        context = HSplit([
            Window(FormattedTextControl("Context — latest model request | Ctrl+\\: close"), height=1, style="class:status"),
            self.context_window,
        ])
        # Overlay the prompt so switching between prompts of different heights
        # does not move the scroll position and select a different prompt again.
        transcript = FloatContainer(self.transcript, floats=[Float(
            top=0, left=0, right=0, height=self._pinned_prompt_height,
            content=Window(TranscriptControl(self._pinned_prompt_content)),
        )])
        return Layout(HSplit([
            DynamicContainer(lambda: context if self.context_visible else transcript),
            Window(height=1),
            DynamicContainer(lambda: menu_footer if self._menu_future is not None else footer),
        ]), focused_element=self._menu_window if self._menu_future is not None else self.input)

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
        ), height=FOOTER_ROWS - 2, wrap_lines=True)

    def _menu_fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for index, (_, label) in enumerate(self._menu_options):
            if index:
                fragments.append(("", "\n"))
            selected = index == self._menu_index
            fragments.append(("class:menu-selected" if selected else "",
                              ("› " if selected else "  ") + label))
        return fragments

    def _menu_hint(self) -> ANSI:
        """Keep access indicators visible even while the menu replaces readouts."""
        callback = getattr(self, "_status_suffix", None)
        suffix = callback() if callback is not None else ""
        remaining = max(0, self.output.get_size().columns - display_width(
            fragment_list_to_text(to_formatted_text(ANSI(suffix)))))
        hint = "↑/↓ = move | Enter = select | Esc = back"
        if len(hint) > remaining:
            hint = hint[:remaining - 1] + "…" if remaining else ""
        return ANSI(hint + suffix)

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
            self.app.layout.focus(self.input)
            self.app.invalidate()

    def _move_menu(self, rows: int) -> None:
        self._menu_index = max(0, min(len(self._menu_options) - 1, self._menu_index + rows))
        self.app.invalidate()

    def _finish_menu(self, value: str | None) -> None:
        if self._menu_future is not None and not self._menu_future.done():
            self._menu_future.set_result(value)

    def _ensure_prompt(self) -> None:
        """Migrate header state; prompt locations come from the transcript file."""
        if hasattr(self, "_pinned_prompt"):
            return
        chunk = getattr(self, "_prompt_chunk", None)
        pinned = getattr(self, "_prompt_pinned", False)
        self._pinned_prompt: tuple[int, int] | None = None
        self._prompt_pending = chunk is not None and not pinned
        self._wrapped_columns = 0

    def _pinned_prompt_height(self, columns: int | None = None) -> int:
        size = self.output.get_size()
        self._transcript_content(size.columns if columns is None else columns)
        assert self._flow is not None
        selected = self._flow.prompt_at(self.transcript.vertical_scroll)
        if self._prompt_pending:
            # New input releases the old header; scrolling explicitly resumes
            # browsing, otherwise wait for the newest prompt to reach the top.
            if selected is not None and selected == self._flow.prompt_at(self._line_count):
                self._prompt_pending = False
            else:
                selected = None
        self._pinned_prompt = selected
        if self._pinned_prompt is None:
            return 0
        # One header row, leaving at least one transcript row above the footer.
        available = max(0, size.rows - FOOTER_ROWS - 2)
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
        window = self.context_window if self.context_visible else self.transcript
        if not self.context_visible and rows:
            self._prompt_pending = False
        window.scroll(rows)
        self.app.invalidate()

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        in_menu = Condition(lambda: self._menu_future is not None)

        @bindings.add("up", filter=in_menu)
        @bindings.add("down", filter=in_menu)
        def move_menu(event: KeyPressEvent) -> None:
            self._move_menu(-1 if event.key_sequence[-1].key == Keys.Up else 1)

        @bindings.add("enter", filter=in_menu, eager=True)
        def select_menu(event: KeyPressEvent) -> None:
            self._finish_menu(self._menu_options[self._menu_index][0])

        @bindings.add("escape", filter=in_menu, eager=True)
        @bindings.add("c-c", filter=in_menu)
        @bindings.add("c-d", filter=in_menu)
        def cancel_menu(event: KeyPressEvent) -> None:
            self._finish_menu(None)

        @bindings.add("escape", filter=~in_menu, eager=True)
        def stop_now(event: KeyPressEvent) -> None:
            callback = getattr(self, "_interrupt_callback", None)
            if callback is not None:
                callback()

        @bindings.add(Keys.Any, filter=in_menu)
        def ignore_menu_text(event: KeyPressEvent) -> None:
            pass

        @bindings.add("c-\\", filter=~in_menu)
        def toggle_context(event: KeyPressEvent) -> None:
            self.context_visible = not self.context_visible
            self.app.invalidate()

        in_context = Condition(lambda: self.context_visible) & ~in_menu

        @bindings.add("up", filter=in_context)
        @bindings.add("down", filter=in_context)
        @bindings.add("pageup", filter=in_context)
        @bindings.add("pagedown", filter=in_context)
        def scroll_context(event: KeyPressEvent) -> None:
            key = event.key_sequence[-1].key
            rows = max(1, self.output.get_size().rows - FOOTER_ROWS - 2) if key in {"pageup", "pagedown"} else 1
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

        @bindings.add("c-d", filter=~in_menu)
        def eof(event: KeyPressEvent) -> None:
            if not event.current_buffer.text:
                self._end_input()
            else:
                event.current_buffer.delete()

        @bindings.add("c-c", filter=~in_menu)
        def interrupt(event: KeyPressEvent) -> None:
            event.current_buffer.reset()
            if self.working:
                self._lines.put_nowait("/stop")
            else:
                self._end_input()

        return bindings

    def set_interrupt_handler(self, callback: Callable[[], None]) -> None:
        """Interrupt work directly, without submitting or changing the draft."""
        self._interrupt_callback = callback

    def _style(self) -> Style:
        return Style.from_dict({
            "status": MUTED_COLOR, "pulse": "bold ansicyan", "idle": MUTED_COLOR, "user": USER_COLOR,
            "context-system": "#ffff00",
            "menu-title": "bold", "menu-selected": "reverse bold",
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
        self._wrapped_columns = 0
        self.app.min_redraw_interval = MIN_REDRAW_INTERVAL
        self.app.invalidate()

    def _route_mouse_wheel(self, app: Application[None]) -> None:
        original = app.renderer.mouse_handlers

        def mouse_handler(event: MouseEvent) -> object:
            if event.event_type in (MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN):
                self._scroll_output(-3 if event.event_type == MouseEventType.SCROLL_UP else 3)
                app.invalidate()
                return None
            if self._menu_future is not None:
                return None  # Keep focus on the menu until selection/cancellation.
            return original.mouse_handlers[event.position.y][event.position.x](event)

        # Wheel events target the active menu or output view. Clicks retain
        # their normal behavior in the input field when no menu is open.
        handlers = MouseHandlers()
        size = self.output.get_size()
        handlers.set_mouse_handler_for_range(0, size.columns, 0, size.rows, mouse_handler)
        app.renderer.mouse_handlers = handlers

    def _accept(self, buffer: Buffer) -> bool:
        self._lines.put_nowait(buffer.text)
        return False

    def _activity(self) -> StyleAndTextTuples:
        if not self.working:
            return [("class:idle", "Ready")]
        frame = PULSE_FRAMES[int(time.monotonic() * 4) % len(PULSE_FRAMES)]
        label = "Stopping After This Step" if self.stopping else "Working (esc to interrupt)"
        return [("class:pulse", f"{frame} {label}")]

    def set_working(self, working: bool, *, stopping: bool = False) -> None:
        self.working = working
        self.stopping = stopping
        self.app.invalidate()

    def set_title(self, title: str) -> None:
        """Update the terminal/tab title through the platform's output adapter."""
        self.output.set_title(title)
        self.output.flush()

    def clear_transcript(self) -> None:
        """Replace display storage while retaining the input draft and application."""
        self._ensure_transcript()
        archive = TranscriptFile()
        self._close_transcript()
        self._raw_output = archive
        self._block_starts.clear()
        self._continuation_indents.clear()
        self.forget_queued_notices()
        self._flow = None
        self._wrapped_columns = self._wrapped_count = self._line_count = 0
        self._pinned_prompt = None
        self._prompt_pending = False
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
        if user_prompt:
            self._ensure_prompt()
            self._pinned_prompt = None
            self._prompt_pending = True
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
            self._flow = WrappedTranscript(columns)
            self._notice_rows.clear()
            self._wrapped_count = 0
            self._wrapped_columns = columns
        flow = self._flow
        for index in range(self._wrapped_count, len(self._raw_output)):
            start = len(flow.rows)
            flow.append(
                self._raw_output[index], first=index in self._block_starts,
                continuation_indents=self._continuation_indents.get(index),
            )
            if index in self._queued_notices:
                self._notice_rows[index] = (start, len(flow.rows))
        self._wrapped_count = len(self._raw_output)
        preview = flow.content_rows()
        complete = len(flow.rows)
        self._line_count = complete + len(preview)

        def get_line(index: int) -> StyleAndTextTuples:
            if index < complete:
                return flow.get_row(index)
            offset = index - complete
            return preview[offset] if offset < len(preview) else []

        return UIContent(get_line=get_line, line_count=self._line_count + 1,
                         cursor_position=Point(x=0, y=max(0, self._line_count - 1)), show_cursor=False)

    async def read_line(self) -> str | None:
        if self._eof and self._lines.empty():
            return None
        return await self._lines.get()

    def _end_input(self) -> None:
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
        self._raw_output.close()
        if self._flow is not None:
            self._flow.close()

    def close(self) -> None:
        self._finish_menu(None)
        if self.app.is_running:
            self.app.exit()
        else:
            self._close_transcript()
