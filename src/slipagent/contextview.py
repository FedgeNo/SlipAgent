"""Disk-backed request inspection with a bounded cache of displayed rows."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import UIContent, UIControl
from wcwidth import wrap

from .transcript import TranscriptFile
from .sources import json_chunks
from .data_text import MessageSections, RenderedContent, RenderedJSON


class ContextControl(UIControl):
    """Focusable request rows without whole-document formatted-text caches."""

    def __init__(self, content: Callable[[int], UIContent]) -> None:
        self.content = content

    def create_content(self, width: int, height: int) -> UIContent:
        return self.content(width)

    def is_focusable(self) -> bool:
        return True


class ContextView:
    def __init__(self, blocks: Iterable[tuple[str, Any]]) -> None:
        self.blocks = TranscriptFile()
        self.rows = TranscriptFile()
        self.columns = 0
        self.cache: OrderedDict[int, StyleAndTextTuples] = OrderedDict()
        try:
            for style, text in blocks:
                self.blocks.append_chunks(json_chunks([style, text]))
        except BaseException:
            self.close()
            raise

    def __iter__(self) -> Iterator[tuple[str, str]]:
        for block in self.blocks:
            style, text = json.loads(block)
            yield style, text

    @classmethod
    def from_body(cls, body: Any) -> ContextView:
        def blocks() -> Iterator[tuple[str, Any]]:
            for field, value in body.index.items():
                if field != "messages":
                    yield "", RenderedContent(MessageSections([field + ":\n", RenderedJSON(value)]))
                    continue
                yield "", "Messages — sent order"
                for index, message in enumerate(value, 1):
                    role = message.get("role", "unknown")
                    parts = [f"[{index:02}] {role.upper()}\n", message.get("content") or "null"]
                    extra = {key: item for key, item in message.items() if key not in {"role", "content"}}
                    if extra:
                        parts.extend(["\n", RenderedJSON(extra)])
                    yield "class:context-system" if role == "system" else "", RenderedContent(MessageSections(parts))
        return cls(blocks())

    @classmethod
    def from_request(cls, request: str) -> ContextView:
        def blocks() -> Iterator[tuple[str, str]]:
            for field, value in json.loads(request).items():
                if field != "messages":
                    yield "", field + ":\n" + json.dumps(value, ensure_ascii=False, indent=2)
                    continue
                yield "", "Messages — sent order"
                for index, message in enumerate(value, 1):
                    role = message.get("role", "unknown")
                    content = message.get("content")
                    text = f"[{index:02}] {role.upper()}\n"
                    text += content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
                    extra = {key: item for key, item in message.items() if key not in {"role", "content"}}
                    if extra:
                        text += "\n" + json.dumps(extra, ensure_ascii=False, indent=2)
                    yield "class:context-system" if role == "system" else "", text
        return cls(blocks())

    def content(self, columns: int) -> UIContent:
        columns = max(1, columns)
        if columns != self.columns:
            rows = TranscriptFile()
            try:
                for index, (style, text) in enumerate(self):
                    if index:
                        rows.append(json.dumps(["", ""]))
                    for line in text.split("\n"):
                        for part in wrap(line, columns, replace_whitespace=False, break_on_hyphens=False) or [""]:
                            rows.append(json.dumps([style, part], ensure_ascii=False))
                rows.append(json.dumps(["", ""]))
            except BaseException:
                rows.close()
                raise
            self.rows.close()
            self.rows = rows
            self.columns = columns
            self.cache.clear()
        return UIContent(get_line=self.row, line_count=len(self.rows), show_cursor=False)

    def row(self, index: int) -> StyleAndTextTuples:
        if index not in self.cache:
            style, text = json.loads(self.rows[index])
            self.cache[index] = [(style, text)]
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def close(self) -> None:
        self.blocks.close()
        self.rows.close()
        self.cache.clear()
