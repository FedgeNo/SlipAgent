"""Markdown tokens to terminal fragments, with independent source/copy text."""

from __future__ import annotations

import re
import tempfile
from bisect import bisect_right
from itertools import accumulate
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TextIO

from markdown_it import MarkdownIt
from markdown_it.token import Token
from prompt_toolkit.formatted_text import StyleAndTextTuples
from pygments.lexers import get_lexer_by_name
from pygments.token import Token as SyntaxToken
from pygments.util import ClassNotFound
from wcwidth import iter_graphemes, width as display_width
from .palette import (
    THOUGHT_COLOR, THOUGHT_CODE_BACKGROUND_COLOR, TEXT_COLOR, LINK_COLOR,
    TOOL_COLOR, SUCCESS_COLOR, WARNING_COLOR, ERROR_COLOR, MUTED_COLOR,
    CODE_BACKGROUND_COLOR, INLINE_BACKGROUND_COLOR,
)


# Foreground roles use separated hues; body and strong deliberately share a color.
THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "body": TEXT_COLOR, "heading": TEXT_COLOR, "code": LINK_COLOR,
        "code-bg": CODE_BACKGROUND_COLOR, "inline-bg": INLINE_BACKGROUND_COLOR, "link": LINK_COLOR,
        "quote": TOOL_COLOR, "marker": SUCCESS_COLOR, "keyword": TOOL_COLOR,
        "string": SUCCESS_COLOR, "number": WARNING_COLOR, "comment": MUTED_COLOR,
        "name": LINK_COLOR, "operator": ERROR_COLOR, "syntax": TEXT_COLOR,
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


CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
WHITESPACE_RUNS = re.compile(r"(\s+)")
PLAIN_TEXT = re.compile(r"[\n -~]*")
CODE_LINES = re.compile(r"(\n)")


def literal(text: str) -> str:
    """Untrusted text cannot inject terminal controls through Markdown."""
    if CONTROL_CHARACTERS.search(text) is None:
        return text
    return CONTROL_CHARACTERS.sub(lambda m: f"\\x{ord(m[0]):02x}", text)


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
            attrs = (["bold"] if heading else []) + list(stack)
            if kind in {"strong_open", "em_open", "s_open"}:
                stack.append({"strong_open": "bold", "em_open": "italic", "s_open": "strike"}[kind])
            elif kind in {"strong_close", "em_close", "s_close"}:
                if stack:
                    stack.pop()
            elif kind == "link_open":
                links.append(str(token.attrGet("href") or ""))
            elif kind == "link_close":
                destination = links.pop() if links else ""
                result.append((self.style("link", *attrs, "underline"), f" ({literal(destination)})"))
            elif kind == "image":
                result.append((self.style("link", *attrs), f"{literal(token.content)} ({literal(str(token.attrGet('src') or ''))})"))
            else:
                role = "heading" if heading else "body"
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
        room = max(1, columns - display_width(prefix))
        wrap = not code
        result: list[DisplayRow] = []
        line: StyleAndTextTuples = []
        # Consecutive clusters in one style are kept as a run; a tuple per
        # character would dominate the cost of long output.
        run: list[str] = []
        run_style = ""
        original = ""
        positions: list[int] = []
        # Characters copied one-for-one map to consecutive offsets, so they
        # are held as one open range and written out once per row.
        range_start = range_stop = 0
        cells = 0
        continued = False

        def settle_run() -> None:
            if run:
                line.append((run_style, "".join(run)))
                run.clear()

        def close_range() -> None:
            nonlocal range_start, range_stop
            if range_stop > range_start:
                positions.extend(range(range_start, range_stop))
            range_start = range_stop = 0

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
            settle_run()
            close_range()
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
        for item in fragments:
            style, value = item[0], item[1]
            if style != run_style:
                settle_run()
                run_style = style
            # Word boundaries keep short prose words together; code preserves columns.
            # With a capturing split, even entries are words and odd ones whitespace.
            # Code splits at newlines only, which never divides a grapheme unless a CR is present.
            words = (CODE_LINES.split(value) if "\r" not in value else [value]) if code else WHITESPACE_RUNS.split(value)
            # Printable ASCII (newlines allowed) is one cell per character and needs
            # no escaping or grapheme segmentation; everything else takes the general path.
            plain = value.isascii() and (value.isprintable() or PLAIN_TEXT.fullmatch(value) is not None)
            single_line = plain and "\n" not in value
            count = len(words)
            offsets: list[int] | None = None
            resume = 0
            for index, word in enumerate(words):
                if index < resume or not word:
                    continue
                is_word = not index & 1
                if single_line:
                    word_plain = fast = True
                    size = len(word)
                else:
                    word_plain = plain or (word.isascii() and (word.isprintable() or PLAIN_TEXT.fullmatch(word) is not None))
                    if word_plain:
                        fast = "\n" not in word
                        size = len(word) if fast else 0
                    else:
                        fast = False
                        size = 0 if "\n" in word else display_width(word)
                if wrap and is_word and size <= room and cells and cells + size > columns:
                    flush()
                if fast:
                    if cells + size <= columns and not (wrap and not is_word and continued and not original and result):
                        # The word fits on this row, and so do any following words that
                        # still fit; a run of them is placed as one slice of the text.
                        text = word
                        if single_line and index + 1 < count:
                            if offsets is None:
                                offsets = list(accumulate(map(len, words), initial=0))
                            end = bisect_right(offsets, offsets[index] + columns - cells, index + 1) - 1
                            if end > index + 1:
                                text = value[offsets[index]:offsets[end]]
                                size = len(text)
                                resume = end
                        start = len(original)
                        if range_stop == start:
                            range_stop += size
                        else:
                            close_range()
                            range_start, range_stop = start, start + size
                        original += text
                        run.append(text)
                        cells += size
                        continue
                    clusters: Iterable[str] = word
                elif word_plain:
                    clusters = word
                else:
                    clusters = iter_graphemes(word)
                for cluster in clusters:
                    if cluster == "\n":
                        flush("\n")
                        continued = False
                        line, positions, cells = [], [], 0
                        begin()
                        continue
                    if word_plain:
                        shown, width = cluster, 1
                    else:
                        shown = " " * min(8 - cells % 8, max(1, columns - cells)) if cluster == "\t" else literal(cluster)
                        width = max(0, display_width(shown))
                    if cells + width > columns and original:
                        flush()
                    if wrap and cluster.isspace() and continued and not original and result:
                        # Keep the separator in copied prose, off the display row.
                        result[-1].copy_text += cluster
                        continue
                    start = len(original)
                    if len(shown) == 1:
                        if range_stop == start:
                            range_stop += 1
                        else:
                            close_range()
                            range_start, range_stop = start, start + 1
                    else:
                        close_range()
                        positions.extend([start] * len(shown))
                    original += cluster
                    run.append(shown)
                    cells += width
        if original or not result:
            flush()
        else:
            settle_run()
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


class ReasoningMarkdownRenderer(MarkdownRenderer):
    """Render reasoning with the same structure and a neutral grey palette."""

    def style(self, role: str, *attributes: str) -> str:
        parts = list(attributes)
        if self.color:
            parts.append(THOUGHT_COLOR)
            if role == "inline":
                parts.append("bg:" + THOUGHT_CODE_BACKGROUND_COLOR)
        return " ".join(parts)


class MarkdownStream:
    """Retain originals on disk and reparse only a bounded mutable block preview."""

    PREVIEW_LIMIT = 65536

    def __init__(self, renderer: MarkdownRenderer) -> None:
        self.renderer = renderer
        self.source: TextIO = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="")
        self.pending = ""
        self.dirty = False
        self.literal_preview = False
        # (fence character, fence length, language) while a code block is open.
        self.fence: tuple[str, int, str] | None = None

    def append(self, text: str) -> None:
        self.source.seek(0, 2)
        self.source.write(text)
        self.pending += text
        self.dirty = True

    def original(self) -> str:
        self.source.seek(0)
        return self.source.read()

    def _open_fence(self) -> bool:
        """Enter streaming code mode when pending starts an unclosed fence."""
        lines = self.pending.splitlines(keepends=True)
        start = 0
        while start < len(lines) and not lines[start].strip():
            start += 1
        opening = re.match(r"^ {0,3}(`{3,}|~{3,})([^\n]*)\n$", lines[start]) if start < len(lines) else None
        if opening is None or opening[1][0] == "`" and "`" in opening[2]:
            return False
        marker = opening[1]
        closing = re.compile(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*\r?\n?$")
        if any(closing.match(line) for line in lines[start + 1:]):
            return False
        self.fence = marker[0], len(marker), opening[2].split()[0] if opening[2].strip() else ""
        self.pending = "".join(lines[start + 1:])
        return True

    def _fence_update(self, columns: int) -> tuple[list[DisplayRow], int]:
        """Render only newly completed code lines; the final render rehighlights all."""
        assert self.fence is not None
        character, length, language = self.fence
        closing = re.compile(r"^ {0,3}" + re.escape(character) + "{" + str(length) + r",}[ \t]*\r?\n$")
        lines = self.pending.splitlines(keepends=True)
        complete = 0
        closed = False
        while complete < len(lines) and lines[complete].endswith("\n"):
            if closing.match(lines[complete]):
                closed = True
                break
            complete += 1
        body = "".join(lines[:complete])
        stable = self.renderer.rows(self.renderer.code(body, language), columns, code=True) if body else []
        if closed:
            self.fence = None
            self.pending = "".join(lines[complete + 1:])
            rows, committed = self.update(columns)
            return stable + rows, len(stable) + committed
        self.pending = "".join(lines[complete:])
        tail = self.renderer.rows(self.renderer.code(self.pending, language), columns, code=True) if self.pending else []
        return stable + tail, len(stable)

    def update(self, columns: int, *, final: bool = False) -> tuple[list[DisplayRow], int]:
        self.dirty = False
        if final:
            self.fence = None
            return self.renderer.render(self.original(), columns), 0
        if self.fence is not None:
            return self._fence_update(columns)
        if not self.literal_preview and self._open_fence():
            return self._fence_update(columns)
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
        if self._open_fence():
            rows, committed = self._fence_update(columns)
            return stable + rows, len(stable) + committed
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
