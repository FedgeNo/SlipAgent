"""Document undo checkpoints backed by the editor's private immutable pages."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from wcwidth import iter_graphemes, grapheme_boundary_before


def grapheme_edge(text: str, position: int, direction: int) -> int:
    if direction < 0:
        return int(grapheme_boundary_before(text, max(0, position)))
    if position >= len(text):
        return len(text)
    start = int(grapheme_boundary_before(text, position + 1))
    return start + len(next(iter_graphemes(text, start=start)))


class EditorBuffer(Buffer):
    def __init__(self, panel: Any, previous: Buffer) -> None:
        self.panel = panel
        super().__init__(document=previous.document, multiline=True, read_only=previous.read_only)
        self.document_undo: list[Path] = []
        self.document_redo: list[Path] = []
        self.on_text_changed += lambda buffer: self.document_redo.clear()
        # A compatible reload adopts the current page's existing undo stack.
        if panel.file is not None:
            current = self.state()
            for text, position in previous._undo_stack:
                panel.file.set_page(panel.page, text)
                self.document_undo.append(self.store({**self.state(flush=False), "cursor": position}))
            for text, position in previous._redo_stack:
                panel.file.set_page(panel.page, text)
                self.document_redo.append(self.store({**self.state(flush=False), "cursor": position}))
            panel.file.pages = {int(page): Path(path) for page, path in current["pages"].items()}

    def _set_cursor_position(self, value: int) -> bool:
        value = max(0, min(value, len(self.text)))
        if value < len(self.text):
            start = int(grapheme_boundary_before(self.text, value + 1))
            if start < value:
                value = grapheme_edge(self.text, start, 1) if value > self.cursor_position else start
        return super()._set_cursor_position(value)

    def delete_before_cursor(self, count: int = 1) -> str:
        start = max(0, self.cursor_position - count)
        if start < len(self.text):
            start = int(grapheme_boundary_before(self.text, start + 1))
        return super().delete_before_cursor(self.cursor_position - start)

    def delete(self, count: int = 1) -> str:
        end = min(len(self.text), self.cursor_position + count)
        if count > 0:
            end = grapheme_edge(self.text, end - 1, 1)
        return super().delete(end - self.cursor_position)

    def clear_document_history(self) -> None:
        self.document_undo.clear()
        self.document_redo.clear()

    def state(self, *, flush: bool = True) -> dict[str, Any]:
        if flush:
            self.panel.flush()
        return {"page": self.panel.page, "cursor": self.cursor_position,
                "pages": {str(page): str(path) for page, path in self.panel.file.pages.items()}}

    def store(self, state: dict[str, Any], path: Path | None = None) -> Path:
        if path is None:
            with tempfile.NamedTemporaryFile(dir=self.panel.file.overlay.name, delete=False) as target:
                path = Path(target.name)
        path.write_text(json.dumps(state), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    def save_to_undo_stack(self, clear_redo_stack: bool = True) -> None:
        if self.panel.file is None or getattr(self.panel, "read_only", False):
            return
        state = self.state()
        if self.document_undo and self.load(self.document_undo[-1])["pages"] == state["pages"]:
            self.store(state, self.document_undo[-1])
        else:
            self.document_undo.append(self.store(state))

    def restore_state(self, state: dict[str, Any]) -> None:
        self.panel.file.pages = {int(page): Path(path) for page, path in state["pages"].items()}
        self.panel.page = state["page"]
        text = self.panel.file.page_text(self.panel.page)
        self.reset(document=Document(text, min(len(text), state["cursor"])))
        self.panel.input.window.vertical_scroll = self.panel.input.window.vertical_scroll_2 = 0
        self.panel.terminal.app.invalidate()

    def undo(self) -> None:
        if self.panel.file is None or getattr(self.panel, "read_only", False) or self.panel.busy:
            return
        current = self.state()
        while self.document_undo:
            state = self.load(self.document_undo.pop())
            if state["pages"] != current["pages"]:
                self.document_redo.append(self.store(current))
                self.restore_state(state)
                break

    def redo(self) -> None:
        if self.panel.file is None or getattr(self.panel, "read_only", False) or self.panel.busy or not self.document_redo:
            return
        self.document_undo.append(self.store(self.state()))
        self.restore_state(self.load(self.document_redo.pop()))
