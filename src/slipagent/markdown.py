"""Markdown tokens to terminal fragments, with independent source/copy text."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from typing import TextIO

from markdown_it import MarkdownIt
from markdown_it.token import Token
from prompt_toolkit.formatted_text import StyleAndTextTuples
from pygments.lexers import get_lexer_by_name
from pygments.token import Token as SyntaxToken
from pygments.util import ClassNotFound
from wcwidth import iter_graphemes, width as display_width


# Foreground roles use separated hues; body and strong deliberately share a color.
THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "body": "#e6e6e6", "heading": "#ffffff", "code": "#8be9fd",
        "code-bg": "#20252d", "inline-bg": "#26343e", "link": "#82aaff",
        "quote": "#c4a7e7", "marker": "#8fbc8f", "keyword": "#c4a7e7",
        "string": "#a6e3a1", "number": "#f9c784", "comment": "#a6adb8",
        "name": "#8be9fd", "operator": "#f38ba8", "syntax": "#e6e6e6",
    },
    "light": {
        "body": "#404040", "heading": "#101010", "code": "#005f73",
        "code-bg": "#edf0f4", "inline-bg": "#dce8ec", "link": "#204db5",
        "quote": "#7030a0", "marker": "#306c35", "keyword": "#7030a0",
        "string": "#266b31", "number": "#934900", "comment": "#596373",
        "name": "#005f73", "operator": "#a12648", "syntax": "#282828",
    },
    "ironbow": {
        "body": "#e6e6e6", "heading": "#ffffff", "code": "#ffd166",
        "code-bg": "#22162e", "inline-bg": "#38223d", "link": "#c5adff",
        "quote": "#f5a6d8", "marker": "#ffb86b", "keyword": "#df9aff",
        "string": "#ffd166", "number": "#ffb86b", "comment": "#b5a5bc",
        "name": "#ff99c8", "operator": "#ff8e80", "syntax": "#fff0cf",
    },
    "monochrome": {},
}


def literal(text: str) -> str:
    """Untrusted text cannot inject terminal controls through Markdown."""
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", lambda m: f"\\x{ord(m[0]):02x}", text)


@dataclass
class DisplayRow:
    fragments: StyleAndTextTuples
    copy_text: str
    # Each visible character boundary maps into copy_text, excluding decoration.
    positions: list[int] = field(default_factory=list)

    def selected(self, start: int, stop: int | None = None) -> str:
        text = "".join(part[1] for part in self.fragments)
        def offset(column: int) -> int:
            cells = chars = 0
            for cluster in iter_graphemes(text):
                if cells >= column:
                    break
                cells += max(0, display_width(cluster))
                chars += len(cluster)
            return self.positions[min(chars, len(self.positions) - 1)] if self.positions else min(chars, len(self.copy_text))
        left = offset(start)
        right = len(self.copy_text) if stop is None else offset(stop)
        return self.copy_text[left:right]


def parser() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": False, "typographer": False}).enable(["table", "strikethrough"])


class MarkdownRenderer:
    """No HTML or Pygments formatter is involved in terminal rendering."""

    def __init__(self, theme: str = "dark", *, color: bool = True) -> None:
        self.theme = theme
        self.color = color
        self.md = parser()

    def style(self, role: str, *attributes: str) -> str:
        colors = THEMES[self.theme] if self.color else {}
        parts = list(attributes)
        if role in colors:
            parts.append(colors[role])
        if role in {"code", "syntax", "keyword", "string", "number", "comment", "name", "operator"} and "code-bg" in colors:
            parts.append("bg:" + colors["code-bg"])
        if role == "inline" and "code" in colors:
            parts.extend([colors["code"], "bg:" + colors["inline-bg"]])
        return " ".join(parts)

    def inline(self, tokens: list[Token], *, heading: bool = False) -> StyleAndTextTuples:
        result: StyleAndTextTuples = []
        stack: list[str] = []
        links: list[str] = []
        for token in tokens:
            kind = token.type
            if kind in {"strong_open", "em_open", "s_open"}:
                stack.append({"strong_open": "bold", "em_open": "italic", "s_open": "strike"}[kind])
            elif kind in {"strong_close", "em_close", "s_close"}:
                if stack:
                    stack.pop()
            elif kind == "link_open":
                links.append(str(token.attrGet("href") or ""))
            elif kind == "link_close":
                destination = links.pop() if links else ""
                result.append((self.style("link", "underline"), f" ({literal(destination)})"))
            elif kind == "image":
                result.append((self.style("link"), f"{literal(token.content)} ({literal(str(token.attrGet('src') or ''))})"))
            else:
                role = "heading" if heading else "body"
                attrs = ["bold"] if heading else list(stack)
                if links:
                    role = "link"
                    attrs.append("underline")
                if kind == "code_inline":
                    role = "inline"
                value = "\n" if kind in {"hardbreak", "softbreak"} else token.content
                result.append((self.style(role, *attrs), literal(value)))
        return result

    def rows(self, fragments: StyleAndTextTuples, columns: int, *, prefix: str = "",
             code: bool = False, padding: bool = True) -> list[DisplayRow]:
        """Wrap graphemes; copy maps exclude prefixes, continuation marks and padding."""
        columns = max(1, columns)
        result: list[DisplayRow] = []
        line: StyleAndTextTuples = []
        original = ""
        positions: list[int] = []
        cells = 0
        continued = False

        def begin() -> None:
            nonlocal cells
            marker = "↪ " if code and continued and columns > 2 else " " * display_width(prefix) if continued else prefix
            if display_width(marker) >= columns:
                marker = ""
            if marker:
                line.append((self.style("comment") if code else self.style("marker"), marker))
                positions.extend([0] * len(marker))
                cells = display_width(marker)

        def flush(separator: str = "") -> None:
            nonlocal line, original, positions, cells, continued
            if code and padding and cells < columns:
                pad_text = " " * (columns - cells)
                line.append((self.style("syntax"), pad_text))
                positions.extend([len(original)] * len(pad_text))
            positions.append(len(original))
            result.append(DisplayRow(line, original + separator, positions))
            line, original, positions, cells = [], "", [], 0
            continued = True
            begin()

        begin()
        for style, value, *_ in fragments:
            # Word boundaries keep short prose words together; code preserves columns.
            words = [value] if code else re.split(r"(\s+)", value)
            for word in words:
                size = display_width(word) if "\n" not in word else 0
                room = max(1, columns - display_width(prefix))
                if not code and word.strip() and size <= room and cells and cells + size > columns:
                    flush()
                for cluster in iter_graphemes(word):
                    if cluster == "\n":
                        flush("\n")
                        continued = False
                        line, positions, cells = [], [], 0
                        begin()
                        continue
                    shown = " " * min(8 - cells % 8, max(1, columns - cells)) if cluster == "\t" else literal(cluster)
                    width = max(0, display_width(shown))
                    if cells + width > columns and original:
                        flush()
                    positions.extend([len(original)] * len(shown))
                    original += cluster
                    line.append((style, shown))
                    cells += width
        if original or not result:
            flush()
        return result

    def code(self, text: str, language: str) -> StyleAndTextTuples:
        if not language or language.lower() in {"text", "plaintext", "plain"}:
            return [(self.style("syntax"), text)]
        try:
            lexer = get_lexer_by_name(language)
        except ClassNotFound:
            return [(self.style("syntax"), text)]
        result: StyleAndTextTuples = []
        # Unprocessed tokens avoid Pygments' newline/tab normalization.
        for _, kind, value in lexer.get_tokens_unprocessed(text):
            role = "syntax"
            for parent, candidate in [(SyntaxToken.Comment, "comment"), (SyntaxToken.Keyword, "keyword"),
                                      (SyntaxToken.Literal.String, "string"), (SyntaxToken.Literal.Number, "number"),
                                      (SyntaxToken.Name, "name"), (SyntaxToken.Operator, "operator")]:
                if kind[:len(parent)] == parent:
                    role = candidate
                    break
            result.append((self.style(role), value))
        return result

    def table(self, tokens: list[Token], columns: int) -> list[DisplayRow]:
        cells: list[list[StyleAndTextTuples]] = []
        row: list[StyleAndTextTuples] = []
        for token in tokens:
            if token.type == "tr_open":
                row = []
            elif token.type == "inline":
                row.append(self.inline(token.children or [], heading=not cells))
            elif token.type == "tr_close":
                cells.append(row)
        if not cells:
            return []
        count = len(cells[0])
        widths = [max(display_width("".join(s[1] for s in r[i])) for r in cells if i < len(r)) for i in range(count)]
        result: list[DisplayRow] = []
        if sum(widths) + max(0, count - 1) * 3 <= columns:
            for r in cells:
                fragments: StyleAndTextTuples = []
                for i, cell in enumerate(r):
                    if i:
                        fragments.append((self.style("marker"), " │ "))
                    fragments.extend(cell)
                    fragments.append(("", " " * (widths[i] - display_width("".join(s[1] for s in cell)))))
                text = "".join(s[1] for s in fragments).rstrip()
                result.append(DisplayRow(fragments, text + "\n"))
        else:
            headers = ["".join(s[1] for s in cell) for cell in cells[0]]
            for r in cells[1:]:
                if result:
                    result.append(DisplayRow([], "\n"))
                for i, cell in enumerate(r):
                    fragments = [(self.style("body", "bold"), headers[i] + ": "), *cell]
                    rows = self.rows(fragments, columns)
                    rows[-1].copy_text += "\n"
                    result.extend(rows)
            if not result:
                result = self.rows(self.inline([Token("text", "", 0, content=" | ".join(headers))]), columns)
        return result

    def render(self, source: str, columns: int, *, tokens: list[Token] | None = None,
               pad_code: bool = True) -> list[DisplayRow]:
        tokens = self.md.parse(source) if tokens is None else tokens
        rows: list[DisplayRow] = []
        lists: list[tuple[str, int]] = []
        quote = 0
        heading = False
        item_marker = ""
        source_lines = source.splitlines(keepends=True)
        previous_end = 0
        index = 0
        while index < len(tokens):
            token = tokens[index]
            kind = token.type
            if token.map and token.level == 0:
                if token.map[0] > previous_end and rows:
                    rows.append(DisplayRow([], "\n"))
                previous_end = token.map[1]
            if kind in {"bullet_list_open", "ordered_list_open"}:
                lists.append((token.markup if kind == "bullet_list_open" else "ordered", int(token.attrGet("start") or 1)))
            elif kind in {"bullet_list_close", "ordered_list_close"}:
                lists.pop()
            elif kind == "list_item_open":
                marker, number = lists[-1]
                item_marker = f"{number}. " if marker == "ordered" else "- "
                lists[-1] = (marker, number + 1)
            elif kind == "list_item_close":
                item_marker = ""
            elif kind == "blockquote_open":
                quote += 1
            elif kind == "blockquote_close":
                quote -= 1
            elif kind == "heading_open":
                heading = True
            elif kind == "heading_close":
                heading = False
            elif kind == "inline":
                fragments = self.inline(token.children or [], heading=heading)
                prefix = "> " * quote + "  " * max(0, len(lists) - 1) + item_marker
                if quote:
                    fragments = [(style.replace(self.style("body"), self.style("quote"))
                                  if self.style("body") else style, text) for style, text, *_ in fragments]
                block = self.rows(fragments, columns, prefix=prefix)
                # Prose copy includes meaningful list/quote markers, not wrap padding.
                if prefix:
                    block[0].copy_text = prefix + block[0].copy_text
                    block[0].positions = [min(i, len(prefix)) if i < len(prefix) else p + len(prefix)
                                          for i, p in enumerate(block[0].positions)]
                block[-1].copy_text += "\n"
                rows.extend(block)
                item_marker = ""
            elif kind in {"fence", "code_block"}:
                body = token.content
                if kind == "fence" and token.level == 0 and token.map:
                    start, end = token.map
                    body_lines = source_lines[start + 1:end]
                    if body_lines and re.match(r"^ {0,3}" + re.escape(token.markup[0]) + "{" + str(len(token.markup)) + r",}\s*$", body_lines[-1]):
                        body_lines = body_lines[:-1]
                    body = "".join(body_lines)
                language = token.info.split()[0] if token.info.strip() else ""
                block = self.rows(self.code(body, language), columns, code=True, padding=pad_code)
                rows.extend(block)
            elif kind == "table_open":
                end = index + 1
                while end < len(tokens) and tokens[end].type != "table_close":
                    end += 1
                rows.extend(self.table(tokens[index:end + 1], columns))
                index = end
            elif kind == "hr":
                rows.append(DisplayRow([(self.style("marker"), "─" * min(columns, 24))], "---\n"))
            index += 1
        return rows

    def plain(self, source: str) -> str:
        return "".join(row.copy_text for row in self.render(source, 1_000_000, pad_code=False)).rstrip("\n")


class MarkdownStream:
    """Retain originals on disk and reparse only a bounded mutable block preview."""

    PREVIEW_LIMIT = 65536

    def __init__(self, renderer: MarkdownRenderer) -> None:
        self.renderer = renderer
        self.source: TextIO = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="")
        self.pending = ""
        self.dirty = False
        self.literal_preview = False

    def append(self, text: str) -> None:
        self.source.seek(0, 2)
        self.source.write(text)
        self.pending += text
        self.dirty = True

    def original(self) -> str:
        self.source.seek(0)
        return self.source.read()

    def update(self, columns: int, *, final: bool = False) -> tuple[list[DisplayRow], int]:
        self.dirty = False
        if final:
            return self.renderer.render(self.original(), columns), 0
        if self.literal_preview or len(self.pending) > self.PREVIEW_LIMIT:
            self.literal_preview = True
            cut = self.pending.rfind("\n") + 1
            if not cut and len(self.pending) > self.PREVIEW_LIMIT:
                cut = len(self.pending) - self.PREVIEW_LIMIT // 2
            prefix, self.pending = self.pending[:cut], self.pending[cut:]
            stable = self.renderer.rows([(self.renderer.style("body"), literal(prefix))], columns) if prefix else []
            return stable + self.renderer.rows([(self.renderer.style("body"), literal(self.pending))], columns), len(stable)
        tokens = self.renderer.md.parse(self.pending)
        starts = [token.map[0] for token in tokens if token.level == 0 and token.map]
        if len(starts) < 2:
            return self.renderer.render(self.pending, columns, tokens=tokens), 0
        cut = sum(len(line) for line in self.pending.splitlines(keepends=True)[:starts[-1]])
        prefix, tail = self.pending[:cut], self.pending[cut:]
        # Keep separator blanks with the committed prefix.
        stable = self.renderer.render(prefix, columns)
        if prefix.endswith("\n\n") and stable:
            stable.append(DisplayRow([], "\n"))
        self.pending = tail
        return stable + self.renderer.render(tail, columns), len(stable)

    def close(self) -> None:
        self.source.close()


def code_blocks(source: str) -> list[str]:
    """Extract code blocks; preserve exact source for top-level fenced bodies."""
    renderer = MarkdownRenderer(color=False)
    result: list[str] = []
    lines = source.splitlines(keepends=True)
    for token in renderer.md.parse(source):
        if token.type not in {"fence", "code_block"}:
            continue
        if token.type == "fence" and token.map and token.level == 0:
            start, end = token.map
            body = lines[start + 1:end]
            if body and re.match(r"^ {0,3}" + re.escape(token.markup[0]) + "{" + str(len(token.markup)) + r",}\s*$", body[-1]):
                body = body[:-1]
            result.append("".join(body))
        else:
            result.append(token.content)
    return result
