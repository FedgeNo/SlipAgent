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
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType, MouseButton
from prompt_toolkit.widgets import TextArea

from .editor import FileEditor, PAGE_SEGMENTS, signature


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
        self.container = HSplit([Window(FormattedTextControl(self.header), height=1, style="class:status"), self.input])
        self.controls = Window(FormattedTextControl(self.actions), height=5)

    def actions(self) -> StyleAndTextTuples:
        result: StyleAndTextTuples = [("class:menu-title", "Editor Actions\n")]
        groups = [
            [("Save File [Ctrl+S]", lambda: self.terminal.app.create_background_task(self.save())),
             ("Reload File [Ctrl+R]", lambda: self.terminal.app.create_background_task(self.reload()))],
            [("Previous Page [PgUp]", lambda: self.change_page(-1)),
             ("Next Page [PgDn]", lambda: self.change_page(1))],
            [("Undo [Ctrl+Z]", self.input.buffer.undo), ("Redo [Ctrl+Y]", self.input.buffer.redo)],
            [("Editor Options [F4]", lambda: self.terminal.app.create_background_task(self.terminal._options_panel.show()))],
        ]
        for group in groups:
            for index, (label, action) in enumerate(group):
                if index:
                    result.append(("", "  |  "))
                def click(event: MouseEvent, action: Any = action, label: str = label) -> object:
                    if event.event_type == MouseEventType.MOUSE_MOVE:
                        self.terminal.__dict__["_hovered_control"] = ("editor-action", label)
                    elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and not self.busy:
                        action()
                    return None
                style = "class:menu-item class:clickable"
                if self.terminal.__dict__.get("_hovered_control") == ("editor-action", label):
                    style += " class:hover"
                result.append((style, label, click))
            result.append(("", " \n"))
        return result

    def header(self) -> StyleAndTextTuples:
        if self.file is None:
            return []
        changed = signature(self.file.path) != self.file.loaded
        text = f"{self.file.path.name} — Editor | Page {self.page + 1}/{self.file.page_count} | Ctrl+S: Save | PgUp/PgDn: Pages"
        if changed:
            text = "File changed on disk. " + text
        def reload(event: MouseEvent) -> object:
            if event.event_type == MouseEventType.MOUSE_MOVE:
                self.terminal.__dict__["_hovered_control"] = "reload-editor"
            elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                self.terminal.app.create_background_task(self.reload())
            return None
        style = "class:clickable"
        if self.terminal.__dict__.get("_hovered_control") == "reload-editor":
            style += " class:hover"
        return [("class:retry" if changed else "", text + " | "), (style, "Reload [Ctrl+R]", reload), ("", " ")]

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
            self.terminal.app.layout.focus(self.input)
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
        self.page = page
        self.input.buffer.reset(document=Document(self.file.page_text(self.page)))
        self.terminal.app.invalidate()
        return True

    def move_cursor(self, rows: int) -> None:
        """Cross disk-backed page boundaries without losing local edits."""
        if self.busy or self.file is None:
            return
        buffer = self.input.buffer
        column = buffer.document.cursor_position_col
        direction = -1 if rows < 0 else 1
        for _ in range(abs(rows)):
            row = buffer.document.cursor_position_row
            last = buffer.document.line_count - 1
            # A page-ending newline belongs to the next page's first line.
            if buffer.text.endswith("\n") and self.page < self.file.page_count - 1:
                last -= 1
            target = row + direction
            if direction > 0 and target > last:
                if not self.change_page(1):
                    break
                target = 0
            elif target < 0:
                if not self.change_page(-1):
                    break
                target = buffer.document.line_count - 1
                if buffer.text.endswith("\n"):
                    target -= 1
                target = max(0, target)
            buffer.cursor_position = buffer.document.translate_row_col_to_index(target, column)
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
            self.terminal.app.layout.focus(self.input)
            self.terminal.app.invalidate()

    async def reload(self) -> None:
        if self.busy or self.file is None:
            return
        if self.dirty():
            action = await self.terminal.choose("Reload Discards Unsaved Edits", [("cancel", "Keep Editing"), ("reload", "Discard Unsaved Edits and Reload")])
            if action != "reload":
                self.terminal.app.layout.focus(self.input)
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
            self.terminal.app.layout.focus(self.input)
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
                self.terminal.app.layout.focus(self.input)
                return False
        self.dispose()
        self.terminal.app.layout.focus(self.terminal.input)
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
