"""Bounded readers and serializers for immutable, file-backed request values."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNK_CHARS = 16384


def spool_directory() -> Path:
    """Use application data storage because system temporary space may be RAM-backed."""
    from .config import data_directory
    directory = data_directory() / "spool"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


class TextSource:
    def chunks(self) -> Iterator[str]:
        raise NotImplementedError

    def read(self) -> str:
        return "".join(self.chunks())


@dataclass(frozen=True)
class FileText(TextSource):
    path: Path
    offset: int
    size: int
    characters: int
    quoted: bool = False

    def chunks(self) -> Iterator[str]:
        with self.path.open("rb") as binary:
            binary.seek(self.offset)
            if self.quoted:
                import io
                from .inspection import JSONRecordReader
                with io.TextIOWrapper(binary, encoding="utf-8") as source:
                    yield from JSONRecordReader(source).string()
            else:
                import codecs
                decoder = codecs.getincrementaldecoder("utf-8")()
                remaining = self.size
                while remaining:
                    chunk = binary.read(min(65536, remaining))
                    if not chunk:
                        raise OSError(f"Incomplete request source: {self.path}")
                    remaining -= len(chunk)
                    yield decoder.decode(chunk)
                final = decoder.decode(b"", final=True)
                if final:
                    yield final

    def read(self) -> str:
        return "".join(self.chunks())

    def __len__(self) -> int:
        return self.characters


def text_chunks(value: str | TextSource) -> Iterator[str]:
    if isinstance(value, TextSource):
        yield from value.chunks()
    else:
        for start in range(0, len(value), CHUNK_CHARS):
            yield value[start:start + CHUNK_CHARS]


def json_key(key: Any, *, ensure_ascii: bool = False, allow_nan: bool = True) -> str:
    if not isinstance(key, (str, int, float, bool, type(None))):
        raise TypeError("JSON object keys must be strings, numbers, booleans, or None")
    if not isinstance(key, str):
        key = json.dumps(key, allow_nan=allow_nan)
    return json.dumps(key, ensure_ascii=ensure_ascii)


def json_chunks(value: Any, *, indent: int | None = 2, level: int = 0,
                ensure_ascii: bool = False, allow_nan: bool = True) -> Iterator[str]:
    """Match JSON text formatting while bounding string escaping allocations."""
    if isinstance(value, (str, TextSource)):
        yield '"'
        for chunk in text_chunks(value):
            yield json.dumps(chunk, ensure_ascii=ensure_ascii)[1:-1]
        yield '"'
    elif isinstance(value, (dict, list, tuple)):
        mapping = isinstance(value, dict)
        yield "{" if mapping else "["
        for index, (key, item) in enumerate(value.items() if isinstance(value, dict) else enumerate(value)):
            if index:
                yield ", " if indent is None else ","
            if indent is not None:
                yield "\n" + " " * ((level + 1) * indent)
            if mapping:
                yield json_key(key, ensure_ascii=ensure_ascii, allow_nan=allow_nan) + ": "
            yield from json_chunks(item, indent=indent, level=level + 1, ensure_ascii=ensure_ascii, allow_nan=allow_nan)
        if value and indent is not None:
            yield "\n" + " " * (level * indent)
        yield "}" if mapping else "]"
    else:
        yield json.dumps(value, ensure_ascii=ensure_ascii, allow_nan=allow_nan)


def materialize(value: Any) -> Any:
    """Load full values only for explicit retrieval and legacy interfaces."""
    if isinstance(value, TextSource):
        return value.read()
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [materialize(item) for item in value]
    return value


def write_indexed_json(path: Path, value: Any) -> Any:
    """Write one JSON record and return its small tree of file-range references."""
    with path.open("xb") as target:
        def write(item: Any) -> Any:
            if isinstance(item, (str, TextSource)):
                start = target.tell()
                characters = 0
                target.write(b'"')
                for chunk in text_chunks(item):
                    characters += len(chunk)
                    target.write(json.dumps(chunk, ensure_ascii=False)[1:-1].encode("utf-8"))
                target.write(b'"')
                size = target.tell() - start
                reference = FileText(path, start, size, characters, True)
                target.flush()
                return reference if characters >= 512 else reference.read()
            if isinstance(item, dict):
                target.write(b"{")
                result = {}
                for index, (key, child) in enumerate(item.items()):
                    if index:
                        target.write(b", ")
                    target.write((json_key(key, allow_nan=False) + ": ").encode("utf-8"))
                    result[key] = write(child)
                target.write(b"}")
                return result
            if isinstance(item, (list, tuple)):
                target.write(b"[")
                children = []
                for index, child in enumerate(item):
                    if index:
                        target.write(b", ")
                    children.append(write(child))
                target.write(b"]")
                return children
            target.write(json.dumps(item, allow_nan=False).encode("utf-8"))
            return item
        try:
            return write(value)
        except BaseException:
            target.close()
            path.unlink(missing_ok=True)
            raise
