"""Bounded-memory text snapshots and streaming inspection-record decoding."""

from __future__ import annotations

import json
import re
import struct
import tempfile
import asyncio
from collections.abc import Iterator, Sequence, Callable
from pathlib import Path
from threading import RLock
from typing import Any, TextIO, overload

CHUNK_SIZE = 4096
_OFFSET = struct.Struct("QQ")
_LINE = struct.Struct("Q")


async def build_snapshot(operation: Callable[..., IndexedText], *args: Any) -> IndexedText:
    from .lifecycle import finish_cleanup
    worker = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        snapshot = await finish_cleanup(worker)
        snapshot.close()
        raise


class IndexedText(Sequence[str]):
    """Private snapshot with disk-backed indexes and bounded display segments.

    Very long source lines use multiple display segments. The logical-line
    index keeps source line links accurate without loading those lines whole.
    """

    def __init__(self) -> None:
        self._snapshot_lock = RLock()
        self.text = tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="\n", errors="surrogatepass")
        self._index = tempfile.TemporaryFile(mode="w+b")
        self.logical = tempfile.TemporaryFile(mode="w+b")
        self._count = 0
        self.logical_count = 0

    @property
    def _read_lock(self) -> RLock:
        # Existing snapshots survive reloads. Concurrent first reads must share
        # one lock, including across the UI and background editor-save worker.
        if "_snapshot_lock" not in self.__dict__:
            self.__dict__.setdefault("_snapshot_lock", RLock())
        return self._snapshot_lock

    def finish(self) -> IndexedText:
        self.text.flush()
        self.text.seek(0)
        start_of_line = True
        while True:
            offset = self.text.tell()
            part = self.text.readline(CHUNK_SIZE)
            if not part:
                break
            if start_of_line:
                self.logical.write(_LINE.pack(self._count))
                self.logical_count += 1
            self._index.write(_OFFSET.pack(offset, len(part)))
            self._count += 1
            start_of_line = part.endswith("\n")
        if not self._count:
            self._index.write(_OFFSET.pack(0, 0))
            self._count = 1
        return self

    @classmethod
    def from_path(cls, path: Path) -> IndexedText:
        result = cls()
        try:
            with path.open(encoding="utf-8") as source:
                while chunk := source.read(CHUNK_SIZE):
                    result.text.write(chunk)
            return result.finish()
        except BaseException:
            result.close()
            raise

    def line_index(self, line: int) -> int:
        if not self.logical_count:
            return 0
        with self._read_lock:
            self.logical.seek(max(0, min(line - 1, self.logical_count - 1)) * _LINE.size)
            return int(_LINE.unpack(self.logical.read(_LINE.size))[0])

    def line_number(self, index: int) -> int:
        """Return the logical line containing a display segment using the disk index."""
        with self._read_lock:
            low, high = 0, self.logical_count
            while low < high:
                middle = (low + high) // 2
                self.logical.seek(middle * _LINE.size)
                start = _LINE.unpack(self.logical.read(_LINE.size))[0]
                if start <= index:
                    low = middle + 1
                else:
                    high = middle
            return max(1, low)

    def __len__(self) -> int:
        return self._count

    def continues_line(self, index: int) -> bool:
        """Whether a display segment continues into the next segment."""
        text = self.segment(index)
        return bool(text and not text.endswith("\n") and index + 1 < len(self))

    def segment(self, index: int) -> str:
        """Read one bounded segment, preserving its original line ending."""
        with self._read_lock:
            self._index.seek(index * _OFFSET.size)
            offset, length = _OFFSET.unpack(self._index.read(_OFFSET.size))
            self.text.seek(offset)
            return self.text.read(length)

    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> list[str]: ...
    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[n] for n in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.segment(index).removesuffix("\n").removesuffix("\r")

    def close(self) -> None:
        with self._read_lock:
            self.text.close()
            self._index.close()
            self.logical.close()


def write_fields(target: TextIO, value: Any, indent: int = 0) -> None:
    """Format already-decoded call data without building another full string."""
    pad = " " * indent
    if isinstance(value, (dict, list)) and value:
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for n, (key, item) in enumerate(items):
            if n:
                target.write("\n\n")
            target.write(pad + (str(key) + ":" if isinstance(value, dict) else f"[{key}]") + "\n")
            write_fields(target, item, indent + 2)
    elif isinstance(value, str):
        target.write(pad)
        for start in range(0, len(value), CHUNK_SIZE):
            target.write(value[start:start + CHUNK_SIZE].replace("\n", "\n" + pad))
    else:
        target.write(pad + json.dumps(value, ensure_ascii=False))


def field_snapshot(value: Any) -> IndexedText:
    result = IndexedText()
    try:
        write_fields(result.text, value)
        return result.finish()
    except BaseException:
        result.close()
        raise


class JSONRecordReader:
    """Stream the flat artifact schema, including arbitrarily large strings."""

    def __init__(self, source: TextIO) -> None:
        self.source = source
        self.buffer = ""
        self.position = 0

    def peek(self) -> str:
        if self.position == len(self.buffer):
            self.buffer = self.source.read(CHUNK_SIZE)
            self.position = 0
        return self.buffer[self.position:self.position + 1]

    def take(self) -> str:
        result = self.peek()
        self.position += bool(result)
        return result

    def whitespace(self) -> None:
        while self.peek() and self.peek() in " \r\n\t":
            self.take()

    def expect(self, value: str) -> None:
        if self.take() != value:
            raise ValueError("Invalid inspection-record JSON.")

    def string(self) -> Iterator[str]:
        self.expect('"')
        while True:
            if not self.peek():
                raise ValueError("Unterminated inspection-record string.")
            match = re.search(r'["\\\x00-\x1f]', self.buffer[self.position:])
            end = self.position + match.start() if match else len(self.buffer)
            if end > self.position:
                yield self.buffer[self.position:end]
                self.position = end
            if not match:
                continue
            char = self.take()
            if char == '"':
                return
            if char != "\\":
                raise ValueError("Unescaped control character in inspection record.")
            escape = self.take()
            if escape == "u":
                digits = "".join(self.take() for _ in range(4))
                if re.fullmatch(r"[0-9a-fA-F]{4}", digits) is None:
                    raise ValueError("Incomplete Unicode escape.")
                code = int(digits, 16)
                if 0xD800 <= code <= 0xDBFF:
                    self.expect("\\")
                    self.expect("u")
                    low_digits = "".join(self.take() for _ in range(4))
                    if re.fullmatch(r"[0-9a-fA-F]{4}", low_digits) is None:
                        raise ValueError("Invalid Unicode surrogate pair.")
                    low = int(low_digits, 16)
                    if not 0xDC00 <= low <= 0xDFFF:
                        raise ValueError("Invalid Unicode surrogate pair.")
                    code = 0x10000 + ((code - 0xD800) << 10) + low - 0xDC00
                yield chr(code)
            else:
                escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
                if escape not in escapes:
                    raise ValueError("Invalid inspection-record escape.")
                yield escapes[escape]

    def fields(self, targets: dict[str, TextIO]) -> dict[str, bool]:
        self.whitespace()
        self.expect("{")
        seen: set[str] = set()
        string_fields: dict[str, bool] = {}
        while True:
            self.whitespace()
            key_parts = []
            size = 0
            for part in self.string():
                size += len(part)
                if size > 32:
                    raise ValueError("Unknown inspection-record field.")
                key_parts.append(part)
            key = "".join(key_parts)
            if key not in targets or key in seen:
                raise ValueError("Unknown or duplicate inspection-record field.")
            seen.add(key)
            self.whitespace()
            self.expect(":")
            self.whitespace()
            if self.peek() == '"':
                string_fields[key] = True
                if key == "step_id":
                    raise ValueError("Expected integer or null step ID.")
                for part in self.string():
                    targets[key].write(part)
            elif key in {"step_id", "call_id"}:
                string_fields[key] = False
                token = ""
                while self.peek() and self.peek() not in ",} \r\n\t":
                    token += self.take()
                    if len(token) > 32:
                        raise ValueError("Invalid inspection-record metadata.")
                parsed = json.loads(token)
                if parsed is not None and (key != "step_id" or type(parsed) is not int):
                    raise ValueError("Invalid inspection-record metadata.")
                targets[key].write(token)
            else:
                raise ValueError("Expected string inspection-record field.")
            self.whitespace()
            separator = self.take()
            if separator == "}":
                break
            if separator != ",":
                raise ValueError("Invalid inspection-record separator.")
        self.whitespace()
        if self.peek() or seen != targets.keys():
            raise ValueError("Incomplete or trailing inspection-record JSON.")
        return string_fields


def artifact_snapshot(path: Path, metadata: dict[str, Any]) -> IndexedText:
    from contextlib import ExitStack
    import shutil

    result = IndexedText()
    try:
        with ExitStack() as stack:
            fields: dict[str, TextIO] = {key: stack.enter_context(tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="", errors="surrogatepass"))
                      for key in ("command", "output", "error", "call_id", "step_id", "kind")}
            with path.open(encoding="utf-8") as source:
                string_fields = JSONRecordReader(source).fields(fields)
            for key in ("command", "call_id", "step_id", "kind"):
                fields[key].seek(0)
                expected = metadata[key]
                if string_fields[key] != isinstance(expected, str):
                    raise ValueError("Inspection-record metadata type mismatch.")
                # Compare metadata in bounded slices rather than copying a large command.
                expected = expected if isinstance(expected, str) else json.dumps(expected)
                for start in range(0, len(expected), CHUNK_SIZE):
                    if fields[key].read(CHUNK_SIZE) != expected[start:start + CHUNK_SIZE]:
                        raise ValueError("Inspection-record metadata mismatch.")
                if fields[key].read(1):
                    raise ValueError("Inspection-record metadata mismatch.")
            for key in ("command", "output"):
                fields[key].seek(0)
                if key == "output":
                    result.text.write("\n\n")
                shutil.copyfileobj(fields[key], result.text, CHUNK_SIZE)
            fields["error"].seek(0)
            first = fields["error"].read(CHUNK_SIZE)
            if first:
                result.text.write("\n\n--- stderr / retention errors ---\n" + first)
                shutil.copyfileobj(fields["error"], result.text, CHUNK_SIZE)
        return result.finish()
    except BaseException:
        result.close()
        raise
