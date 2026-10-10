"""Read-only text and diff controls with lazy per-line formatting."""

from __future__ import annotations

import re
import struct
import tempfile
from collections.abc import Sequence
from .inspection import IndexedText

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import UIContent, UIControl


class TextViewControl(UIControl):
    def __init__(self, text: str | IndexedText) -> None:
        if isinstance(text, str):
            snapshot = IndexedText()
            try:
                for start in range(0, len(text), 4096):
                    snapshot.text.write(text[start:start + 4096])
                text = snapshot.finish()
            except BaseException:
                snapshot.close()
                raise
        self.lines: Sequence[str] = text

    def close(self) -> None:
        if isinstance(self.lines, IndexedText):
            self.lines.close()

    def is_focusable(self) -> bool:
        return True

    def line_style(self, line: str) -> str:
        return ""

    def get_line(self, index: int) -> StyleAndTextTuples:
        line = self.lines[index]
        literal = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", lambda match: f"\\x{ord(match[0]):02x}", line)
        return [(self.line_style(line), literal)]

    def create_content(self, width: int, height: int | None) -> UIContent:
        return UIContent(get_line=self.get_line, line_count=len(self.lines),
                         cursor_position=Point(x=0, y=0), show_cursor=False)


class DiffViewControl(TextViewControl):
    def __init__(self, text: str | IndexedText) -> None:
        super().__init__(text)
        self._numbers = tempfile.TemporaryFile(mode="w+b")
        self._number_count = 0
        self._old = self._new = 0
        self._in_hunk = False
        self._old_left = self._new_left = 0
        self._continued = False

    def close(self) -> None:
        self._numbers.close()
        super().close()

    def get_line(self, index: int) -> StyleAndTextTuples:
        # Build a disk-backed gutter lazily; even enormous diffs stay bounded.
        record = struct.Struct("QQ")
        while self._number_count <= index:
            number = self._number_count
            line = self.lines[number]
            old = new = 0
            if not self._continued:
                match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    self._old = int(match[1])
                    self._new = int(match[3])
                    self._old_left = int(match[2]) if match[2] is not None else 1
                    self._new_left = int(match[4]) if match[4] is not None else 1
                    self._in_hunk = True
                elif self._in_hunk and line.startswith((" ", "+", "-")):
                    if not line.startswith("+"):
                        old = self._old
                        self._old += 1
                        self._old_left -= 1
                    if not line.startswith("-"):
                        new = self._new
                        self._new += 1
                        self._new_left -= 1
                    self._in_hunk = self._old_left > 0 or self._new_left > 0
                elif line.startswith(("diff ", "--- ", "+++ ")):
                    self._in_hunk = False
            self._continued = isinstance(self.lines, IndexedText) and self.lines.continues_line(number)
            self._numbers.seek(number * record.size)
            self._numbers.write(record.pack(old, new))
            self._number_count += 1
        self._numbers.seek(index * record.size)
        old, new = record.unpack(self._numbers.read(record.size))
        if not old and not new and not self.line_style(self.lines[index]):
            return super().get_line(index)
        gutter = f"{str(old) if old else '':>6} {str(new) if new else '':>6} │ "
        content = super().get_line(index)
        if old or new:
            style = "" if old and new else "class:diff-removed" if old else "class:diff-added"
            content = [(style, text) for _, text, *_ in content]
        return [("class:diff-number", gutter), *content]

    def line_style(self, line: str) -> str:
        if line.startswith(("--- ", "+++ ", "diff ", "index ")):
            return "class:diff-header"
        if line.startswith("@@"):
            return "class:diff-hunk"
        if line.startswith("+"):
            return "class:diff-added"
        if line.startswith("-"):
            return "class:diff-removed"
        return ""
