"""Session transcript storage and incremental, styled word wrapping.

Original chunks and completed display rows live in private temporary files.
Only row offsets, a small read cache, and the unfinished wrapping tail stay in
memory. A width change rebuilds display rows from the original chunks.
"""

from __future__ import annotations

import json
import re
import tempfile
from array import array
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import overload

from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from wcwidth import width as display_width, wrap
from .palette import USER_BACKGROUND_COLOR, USER_TEXT_COLOR
from .markdown import DisplayRow, MarkdownRenderer, MarkdownStream


class TranscriptFile(Sequence[str]):
    """Indexed UTF-8 records; embedded newlines need no special treatment."""

    def __init__(self) -> None:
        self._file = tempfile.TemporaryFile(mode="w+b")
        self._offsets = array("Q", [0])

    def append(self, text: str) -> None:
        data = text.encode("utf-8", errors="surrogatepass")
        self._file.seek(self._offsets[-1])
        self._file.write(data)
        self._offsets.append(self._file.tell())
        if hasattr(self, "_ends"):
            self._ends.append(self._file.tell())

    def _editable_index(self) -> array[int]:
        # Existing live files have contiguous offsets. Split their end offsets
        # only when removing a notice, leaving file contents in place.
        if not hasattr(self, "_ends"):
            self._ends = self._offsets[1:]
        return self._ends

    def clear_record(self, index: int) -> None:
        """Empty one record without changing other records' indices."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._editable_index()[index] = self._offsets[index]

    def remove(self, start: int, stop: int) -> None:
        """Remove indexed records without reading or rewriting their neighbors."""
        if not 0 <= start <= stop <= len(self):
            raise IndexError((start, stop))
        del self._editable_index()[start:stop]
        del self._offsets[start:stop]  # Keep the final file-end offset for appends.

    def __len__(self) -> int:
        return len(self._offsets) - 1

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._file.seek(self._offsets[index])
        end = self._ends[index] if hasattr(self, "_ends") else self._offsets[index + 1]
        return self._file.read(end - self._offsets[index]).decode(
            "utf-8", errors="surrogatepass",
        )

    def close(self) -> None:
        self._file.close()


class AnsiStream(ANSI):
    """Adapt prompt-toolkit's ANSI decoder to successive network chunks.

    Keeping its coroutine alive preserves SGR state and incomplete escapes.
    Drain its output after each feed so it never retains old character tuples.
    This small adapter is covered by split-escape/color tests against the pinned
    prompt-toolkit 3.x dependency; wrapping uses only public wcwidth functions.
    """

    def __init__(self) -> None:
        super().__init__("")
        self._parser = self._parse_corot()
        next(self._parser)

    def feed(self, text: str) -> Iterator[tuple[str, str]]:
        for char in text:
            self._parser.send(char)
            for style, value, *_ in self._formatted_text:
                # Transcript output is text, not a route for raw terminal
                # control sequences to escape into the interactive footer.
                if "[ZeroWidthEscape]" not in style:
                    yield style, value
            self._formatted_text.clear()


def compact_fragments(characters: StyleAndTextTuples) -> StyleAndTextTuples:
    """Store style runs instead of one persistent Python tuple per character."""
    result: StyleAndTextTuples = []
    pieces: list[str] = []
    previous = ""
    for style, text, *_ in characters:
        if pieces and style != previous:
            result.append((previous, "".join(pieces)))
            pieces = []
        previous = style
        pieces.append(text)
    if pieces:
        result.append((previous, "".join(pieces)))
    return result


class WrappedTranscript:
    """Incrementally formatted rows with a revisable wrapping tail.

    A growing word can move off the previous row, and a later code point can
    extend the final grapheme. Keeping two rows uncommitted allows both without
    revisiting the rest of a long paragraph. Indented unfinished words can need
    additional rows, still bounded by the width of one word. Newlines finalize
    the whole tail.
    """

    CACHE_ROWS = 256

    def __init__(self, columns: int, *, theme: str = "dark", color: bool = True, source: bool = False) -> None:
        self.columns = max(1, columns)
        self.rows = TranscriptFile()
        self.prompt_rows: list[tuple[int, int]] = []
        self._prompt_line = False
        self._cache: OrderedDict[int, StyleAndTextTuples] = OrderedDict()
        self._ansi = AnsiStream()
        self._pending: StyleAndTextTuples = []
        self._preview: list[StyleAndTextTuples] | None = None
        self._indent = ""
        self._indent_known = False
        self._continued = False
        self._consumed_width = 0
        self._space_run = 0
        self.started = False
        self._markdown_renderer = MarkdownRenderer(theme, color=color)
        self._markdown_source = source
        self._markdown: MarkdownStream | None = None
        self._markdown_start = self._markdown_tail = 0
        self._copy_rows = TranscriptFile()
        self._copy_index: dict[int, int] = {}

    def append_markdown(self, text: str, *, first: bool, final: bool = False) -> None:
        if first or self._markdown is None:
            self.finish_markdown()
            if self.started:
                self._settle(final=True)
            self.started = False
            self._markdown = MarkdownStream(self._markdown_renderer)
            self._markdown_start = self._markdown_tail = len(self.rows)
        self._markdown.append(text)
        if final:
            self.finish_markdown()

    def _flush_markdown(self, *, final: bool = False) -> None:
        stream = self._markdown
        if stream is None or not (stream.dirty or final):
            return
        start = self._markdown_start if final else self._markdown_tail
        self.remove_rows(start, len(self.rows))
        if self._markdown_source:
            text = stream.original() if final else stream.pending
            rendered = self._markdown_renderer.rows([("", text)], self.columns)
            committed = 0
        else:
            rendered, committed = stream.update(self.columns, final=final)
        for row in rendered:
            self._copy_index[len(self.rows)] = len(self._copy_rows)
            self._copy_rows.append(json.dumps([row.copy_text, row.positions], ensure_ascii=False))
            self._save(compact_fragments(row.fragments))
        self._markdown_tail = start + committed
        stream.dirty = False

    def finish_markdown(self) -> None:
        if self._markdown is not None:
            self._flush_markdown(final=True)
            self._markdown.close()
            self._markdown = None

    def copy_row(self, index: int) -> DisplayRow:
        fragments = self.get_row(index)
        record = self._copy_index.get(index)
        if record is not None:
            text, positions = json.loads(self._copy_rows[record])
            return DisplayRow(fragments, text, positions)
        return DisplayRow(fragments, "".join(item[1] for item in fragments) + "\n")

    def _save(self, row: StyleAndTextTuples) -> None:
        self.rows.append(json.dumps(row, ensure_ascii=False))

    def _starts_prompt(self) -> bool:
        if not self._pending or self._pending[0][1] != ">":
            return False
        styles = set(self._pending[0][0].lower().split())
        # Recognize both historical greens and the explicit white-on-green pair.
        return bool({"#66ff66", "#00ff00"}.intersection(styles)
                    or USER_TEXT_COLOR in styles
                    and {"bg:" + USER_BACKGROUND_COLOR, "bg:#005000"}.intersection(styles))

    def prompt_at(self, row: int) -> tuple[int, int] | None:
        """Nearest preceding green > line, including all its wrapped rows."""
        tail = self.content_rows()
        complete = len(self.rows)
        if not self._continued and self._starts_prompt() and complete <= row:
            return complete, complete + len(tail)
        index = bisect_right(self.prompt_rows, row, key=lambda span: span[0]) - 1
        if index < 0:
            return None
        start, end = self.prompt_rows[index]
        if self._continued and self._prompt_line and index == len(self.prompt_rows) - 1:
            end += len(tail)
        return start, end

    def _render(self) -> tuple[list[StyleAndTextTuples], list[int]]:
        text = "".join(item[1] for item in self._pending)
        if not self._indent_known:
            leading = text[:len(text) - len(text.lstrip(" \t"))]
            self._indent = leading if len(leading) < self.columns else ""
            self._indent_known = bool(text.strip(" \t")) or len(leading) >= self.columns
        prefix = self._indent if self._continued else ""
        if display_width(prefix + text) <= self.columns:
            lines = [prefix + text]
        else:
            lines = wrap(text, self.columns, initial_indent=prefix,
                         subsequent_indent=self._indent, replace_whitespace=False,
                         break_on_hyphens=False, propagate_sgr=False) or [""]
        rows: list[StyleAndTextTuples] = []
        starts: list[int] = []
        position = 0
        for index, line in enumerate(lines):
            indent = prefix if index == 0 else self._indent
            body = line[len(indent):] if line.startswith(indent) else line
            start = text.find(body, position)
            if start < 0:
                raise ValueError("Wrapped transcript row does not match its source text")
            starts.append(start)
            style = self._pending[start][0] if start < len(self._pending) else ""
            row: StyleAndTextTuples = [(style, indent)] if indent and line.startswith(indent) else []
            row.extend(self._pending[start:start + len(body)])
            rows.append(compact_fragments(row))
            position = start + len(body)
        return rows, starts

    def _settle(self, *, final: bool = False) -> None:
        rows, starts = self._render()
        count = len(rows) if final else max(0, len(rows) - 2)
        if count and not final:
            text = "".join(item[1] for item in self._pending)
            last_word = re.search(r"\S+$", text)
            if last_word is not None and display_width(last_word.group()) <= self.columns:
                # A short unfinished word may later become a long word and
                # fill the preceding row. Deep indentation can spread that
                # word across more than two rows before it reaches this point.
                word_row = next((i for i, start in enumerate(starts) if start >= last_word.start()), len(rows) - 1)
                count = min(count, max(0, word_row - 1))
        if count and not self._continued:
            self._prompt_line = self._starts_prompt()
            if self._prompt_line:
                self.prompt_rows.append((len(self.rows), len(self.rows)))
        for row in rows[:count]:
            self._save(row)
        if count and self._prompt_line:
            self.prompt_rows[-1] = (self.prompt_rows[-1][0], len(self.rows))
        if final:
            self._prompt_line = False
            self._pending.clear()
            self._indent = ""
            self._indent_known = False
            self._continued = False
            self._consumed_width = 0
            self._space_run = 0
        elif count:
            consumed = starts[count]
            self._consumed_width += display_width("".join(item[1] for item in self._pending[:consumed]))
            del self._pending[:consumed]
            self._continued = True
        self._preview = None

    def _override_indent(self, columns: int | None) -> None:
        # On a terminal narrower than the label, retain normal wrapping so
        # indentation cannot consume all the available space.
        if columns is not None and 0 <= columns < self.columns:
            self._indent = " " * columns
            self._indent_known = True

    def append(
        self, text: str, *, first: bool,
        continuation_indents: dict[int, int] | None = None,
    ) -> None:
        """Append text; indent overrides use zero-based source lines in this chunk."""
        self.finish_markdown()
        if first:
            if self.started:
                self._settle(final=True)
            self._ansi = AnsiStream()
        self.started = True
        self._preview = None
        indents = continuation_indents or {}
        line_number = 0
        self._override_indent(indents.get(line_number))
        # Bounded feed intervals prevent a large tool result from becoming one
        # enormous intermediate styled-character list.
        interval = max(64, self.columns * 4)
        received = 0
        for style, value in self._ansi.feed(text):
            for char in value:
                if char == "\n":
                    self._settle(final=True)
                    line_number += 1
                    self._override_indent(indents.get(line_number))
                elif char == "\t":
                    column = self._consumed_width + display_width("".join(item[1] for item in self._pending))
                    for _ in range(8 - column % 8):
                        self._space(style)
                elif char == " ":
                    self._space(style)
                else:
                    self._space_run = 0
                    self._pending.append((style, char))
                received += 1
                if received >= interval:
                    self._settle()
                    received = 0

    def _space(self, style: str) -> None:
        self._space_run += 1
        # A gap wider than the entire row is dropped at a word-wrap boundary.
        # Its excess spaces need only contribute to tab alignment. Originals
        # remain in the raw file for a future resize to any larger width.
        if self._space_run <= self.columns + 1:
            self._pending.append((style, " "))
        else:
            self._consumed_width += 1

    def content_rows(self) -> list[StyleAndTextTuples]:
        self._flush_markdown()
        if self._markdown is not None:
            return []
        if self._preview is None:
            self._settle()
            self._preview = self._render()[0] if self.started else []
        return self._preview

    def get_row(self, index: int) -> StyleAndTextTuples:
        if index not in self._cache:
            row = json.loads(self.rows[index])
            self._cache[index] = [(style, text) for style, text in row]
            if len(self._cache) > self.CACHE_ROWS:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]

    def remove_rows(self, start: int, stop: int) -> None:
        """Retract complete notice rows, preserving the unfinished output tail."""
        count = stop - start
        if not count:
            return
        self.rows.remove(start, stop)
        self._cache = OrderedDict(
            (index if index < start else index - count, row)
            for index, row in self._cache.items() if index < start or index >= stop
        )
        def shifted(index: int) -> int:
            return index - min(count, max(0, index - start))
        self.prompt_rows = [(shifted(left), shifted(right)) for left, right in self.prompt_rows
                            if left < start or left >= stop]
        self._copy_index = {shifted(index): record for index, record in self._copy_index.items()
                            if index < start or index >= stop}

    def close(self) -> None:
        if self._markdown is not None:
            self._markdown.close()
        self._copy_rows.close()
        self.rows.close()
        self._cache.clear()
