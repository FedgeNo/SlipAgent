"""Read-only text and diff controls with lazy per-line formatting."""

from __future__ import annotations

import re

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import UIContent, UIControl


class TextViewControl(UIControl):
    def __init__(self, text: str) -> None:
        self.lines = text.splitlines() or [""]

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
