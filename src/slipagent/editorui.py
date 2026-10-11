"""Built-in editor presentation, independent of agent execution ownership."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import TextArea
from wcwidth import iter_graphemes, width as display_width

from .editor import FileEditor, PAGE_SEGMENTS, signature
from .editorcontrol import EditorControl, EditorWindow


class EditorPanel:
    def __init__(self, terminal: Any) -> None:
        self.terminal = terminal
        self.file: FileEditor | None = None
        self.page = 0
        self.busy = False
        self.saving_task: asyncio.Task[Any] | None = None
        self.saved_pages: dict[int, str] = {}
        self.input = TextArea(multiline=True, wrap_lines=True,
                              read_only=Condition(lambda: self.busy))
        self.ensure_control()
        self.container = HSplit([Window(FormattedTextControl(self.header), height=1, style="class:status class:footer"), self.input])
        self.controls = Window(FormattedTextControl(self.actions), height=5)

    def ensure_control(self) -> None:
        if hasattr(self, "container"):
            header = self.container.children[0]
            if isinstance(header, Window):
                header.style = "class:status class:footer"
        if not isinstance(self.input.control, EditorControl):
            self.input.control = EditorControl(self, self.input.control)
            self.input.window.content = self.input.control
        if not isinstance(self.input.window, EditorWindow):
            previous = self.input.window
            self.input.window = EditorWindow(self.input.control, style=previous.style,
                                             wrap_lines=previous.wrap_lines, height=previous.height)
            self.input.window.vertical_scroll = previous.vertical_scroll
            self.input.window.vertical_scroll_2 = previous.vertical_scroll_2
            if hasattr(self, "container"):
                self.container.children[1] = self.input.window

    def actions(self) -> StyleAndTextTuples:
        changed = self.file is not None and signature(self.file.path) != self.file.loaded
        title = "Editor Actions — File changed on disk." if changed else "Editor Actions"
        result: StyleAndTextTuples = [("class:menu-title", title + "\n")]
        groups = [
            [("Save File [Ctrl+S]", lambda: self.terminal.app.create_background_task(self.save())),
             ("Reload File [Ctrl+R]", lambda: self.terminal.app.create_background_task(self.reload()))],
            [("Page Up [PgUp]", lambda: self.move_screen(-1)),
             ("Page Down [PgDn]", lambda: self.move_screen(1))],
            [("Undo [Ctrl+Z]", self.input.buffer.undo), ("Redo [Ctrl+Y]", self.input.buffer.redo)],
            [("Editor Options [F4]", lambda: self.terminal.app.create_background_task(self.terminal._options_panel.show()))],
        ]
        for group in groups:
            for index, (label, action) in enumerate(group):
                if index:
                    result.append(("", "  |  "))
                result.extend(self.terminal._action_link(
                    label, ("editor-action", label), action, style="class:menu-item",
                    enabled=lambda: not self.busy,
                ))
            result.append(("", " \n"))
        return result

    def header(self) -> StyleAndTextTuples:
        if self.file is None:
            return []
        columns = self.terminal.output.get_size().columns
        if self.terminal.__dict__.get("_inspection_visible") and columns >= 60:
            columns -= columns // 4
        text = str(self.file.path)
        if display_width(text) > columns:
            clusters = list(iter_graphemes(text))
            left = ""
            right = ""
            left_space = max(0, (columns - 1) // 2)
            right_space = max(0, columns - 1 - left_space)
            for cluster in clusters:
                if display_width(left + cluster) > left_space:
                    break
                left += cluster
            for cluster in reversed(clusters):
                if display_width(cluster + right) > right_space:
                    break
                right = cluster + right
            text = left + "…" + right if columns > 0 else ""
        return [("", text)]

    async def open(self, path: Path, line: int) -> None:
        if self.file is not None:
            return
        self.busy = True
        worker = asyncio.create_task(asyncio.to_thread(FileEditor, path))
        try:
            try:
                file = await asyncio.shield(worker)
            except asyncio.CancelledError:
                from .lifecycle import finish_cleanup
                loaded = await finish_cleanup(worker)
                loaded.close()
                raise
            self.file = file
            self.saved_pages = {}
            self.page = file.original.line_index(line) // PAGE_SEGMENTS
            self.input.buffer.reset(document=Document(file.page_text(self.page)))
            segment = file.original.line_index(line) - self.page * PAGE_SEGMENTS
            position = sum(len(file.original.segment(self.page * PAGE_SEGMENTS + n)) for n in range(segment))
            self.input.buffer.cursor_position = position
            self.terminal._editor_active = True
            self.terminal._focus.focus(self.input)
            self.terminal.notify("Agent held while editor is open.")
        except (OSError, UnicodeError, ValueError) as exc:
            self.terminal.notify("Could not open editor.")
            self.terminal.write(f"{path}: {exc}")
        finally:
            self.busy = False
            self.terminal.app.invalidate()

    def flush(self) -> None:
        if self.file is not None:
            self.file.set_page(self.page, self.input.text)

    def dirty(self) -> bool:
        assert self.file is not None
        current = {page: hashlib.sha256(self.file.page_text(page).encode()).hexdigest() for page in self.file.pages}
        if self.input.text == self.file.original_page_text(self.page):
            current.pop(self.page, None)
        else:
            current[self.page] = hashlib.sha256(self.input.text.encode()).hexdigest()
        return current != self.saved_pages

    def change_page(self, delta: int) -> bool:
        if self.busy or self.file is None:
            return False
        page = max(0, min(self.file.page_count - 1, self.page + delta))
        if page == self.page:
            return False
        try:
            self.flush()
        except (OSError, UnicodeError) as exc:
            self.terminal.notify("Could not change page; edits retained.")
            self.terminal.write(str(exc))
            return False
        previous = self.page
        control = self.input.control
        shift = 0
        if isinstance(control, EditorControl):
            if page == previous + 1:
                shift = -len(control.neighbor_lines(previous - 1))
            elif page == previous - 1:
                shift = len(control.neighbor_lines(page - 1))
        self.page = page
        self.input.buffer.reset(document=Document(self.file.page_text(self.page)))
        self.input.window.vertical_scroll = max(0, self.input.window.vertical_scroll + shift) if abs(page - previous) == 1 else 0
        self.terminal.app.invalidate()
        return True

    def move_cursor(self, rows: int) -> None:
        """Move by displayed rows across disk-backed section boundaries."""
        if self.busy or self.file is None:
            return
        control = self.input.control
        if isinstance(control, EditorControl):
            control.move_visual(rows)
        self.terminal.app.invalidate()

    def move_horizontal(self, direction: int) -> None:
        if self.busy or self.file is None:
            return
        buffer = self.input.buffer
        position = buffer.cursor_position
        if direction < 0 and position == 0:
            if self.change_page(-1):
                buffer.cursor_position = max(0, len(buffer.text) - 1)
        elif direction > 0 and position >= len(buffer.text) - int(buffer.text.endswith("\n")):
            offset = max(0, position - len(buffer.text) + 1)
            if self.change_page(1):
                buffer.cursor_position = min(offset, len(buffer.text))
            else:
                buffer.cursor_position = min(position + 1, len(buffer.text))
        else:
            buffer.cursor_position = max(0, min(len(buffer.text), position + direction))
        self.terminal.app.invalidate()

    def move_screen(self, direction: int) -> None:
        info = self.input.window.render_info
        height = info.window_height if info is not None else max(1, self.terminal.output.get_size().rows - self.terminal._footer_rows() - 1)
        rows = direction * max(1, height - 1)
        self.move_cursor(rows)
        if isinstance(self.input.window, EditorWindow):
            self.input.window.scroll_rows(rows)

    def move_to_end(self, *, end: bool) -> None:
        if self.busy or self.file is None:
            return
        self.change_page(self.file.page_count if end else -self.file.page_count)
        if self.page == (self.file.page_count - 1 if end else 0):
            self.input.buffer.cursor_position = len(self.input.text) if end else 0
        self.terminal.app.invalidate()

    async def save(self) -> bool:
        if self.busy or self.file is None:
            return False
        self.busy = True
        self.saving_task = asyncio.current_task()
        published = False
        try:
            self.flush()
            overwrite = False
            observed = signature(self.file.path)
            if observed != self.file.loaded:
                action = await self.terminal.choose("File Changed on Disk", [("cancel", "Cancel Save"), ("reload", "Reload From Disk"), ("overwrite", "Overwrite Disk File")])
                if action == "reload":
                    self.busy = False
                    await self.reload()
                    return False
                if action != "overwrite":
                    return False
                overwrite = True
            # Produce the notification before publishing; diff failures leave disk alone.
            file = self.file
            def publish() -> str:
                diff = file.diff()
                file.save(overwrite=overwrite, confirmed_signature=observed)
                return diff
            worker = asyncio.create_task(asyncio.to_thread(publish))
            cancellation = None
            try:
                diff = await asyncio.shield(worker)
            except asyncio.CancelledError as exc:
                from .lifecycle import finish_cleanup
                cancellation = exc
                diff = await finish_cleanup(worker)
            published = True
            self.saved_pages = {page: hashlib.sha256(self.file.page_text(page).encode()).hexdigest() for page in self.file.pages}
            coordinator = self.terminal.__dict__.get("_user_edits")
            if coordinator is not None:
                coordinator.record(self.file.path, diff, self.file.edit_id)
            self.terminal.notify("File saved.")
            if cancellation is not None:
                raise cancellation
            return True
        except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
            if published:
                self.terminal.notify("File saved; recording failed.")
                self.terminal.write(f"File was saved, but its edit notification failed: {exc}")
                return True
            self.terminal.notify("Save failed; edits retained.")
            self.terminal.write(f"Could not save file: {exc}")
            return False
        finally:
            self.saving_task = None
            self.busy = False
            self.terminal._focus.focus(self.input)
            self.terminal.app.invalidate()

    async def reload(self) -> None:
        if self.busy or self.file is None:
            return
        if self.dirty():
            action = await self.terminal.choose("Reload Discards Unsaved Edits", [("cancel", "Keep Editing"), ("reload", "Discard Unsaved Edits and Reload")])
            if action != "reload":
                self.terminal._focus.focus(self.input)
                return
        old = self.file
        line = old.original.line_number(self.page * PAGE_SEGMENTS) + self.input.buffer.document.cursor_position_row
        for page in old.pages:
            if page < self.page:
                line += old.page_text(page).count("\n") - old.original_page_text(page).count("\n")
        self.file = None
        await self.open(old.path, line)
        if getattr(self, "file") is None:
            self.file = old
            self.terminal._focus.focus(self.input)
        else:
            old.close()

    async def close(self, *, release: bool = True) -> bool:
        if self.busy or self.file is None:
            return False
        if self.dirty():
            action = await self.terminal.choose("Unsaved Changes", [("save", "Save"), ("discard", "Don't Save"), ("cancel", "Cancel")])
            if action == "save":
                if not await self.save():
                    return False
            elif action != "discard":
                self.terminal._focus.focus(self.input)
                return False
        self.dispose()
        self.terminal._focus.focus(self.terminal.input)
        coordinator = self.terminal.__dict__.get("_user_edits")
        if release and coordinator is not None:
            coordinator.close()
        if release:
            self.terminal.notify("Editor closed.")
        self.terminal.app.invalidate()
        return True

    def dispose(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        self.terminal._editor_active = False

    async def shutdown(self) -> None:
        from .lifecycle import finish_cleanup
        task = self.saving_task
        try:
            if task is not None and task is not asyncio.current_task():
                self.terminal._finish_menu(None)
                async def settle() -> None:
                    await asyncio.gather(task, return_exceptions=True)
                await finish_cleanup(asyncio.create_task(settle()))
        finally:
            self.dispose()
