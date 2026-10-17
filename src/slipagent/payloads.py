"""Version-two inspection records share immutable UTF-8 command payloads."""

from __future__ import annotations

import bisect
import codecs
import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from .fileio import open_regular

CHUNK_BYTES = 65536


def sync_directory(directory: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def payload_path(directory: Path, digest: str) -> Path:
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Invalid command payload identifier")
    parent = directory / "payloads"
    path = parent / digest
    if parent.is_symlink() or path.is_symlink():
        raise OSError("Command payload was replaced by a symbolic link")
    return path


def copy_payload(directory: Path, source: Path | None) -> tuple[dict[str, Any], bool]:
    if source is None:
        return {"sha256": "", "bytes": 0, "characters": 0}, False
    parent = directory / "payloads"
    if parent.is_symlink():
        raise OSError("Command payload directory was replaced by a symbolic link")
    parent.mkdir(mode=0o700, exist_ok=True)
    sync_directory(directory)
    descriptor, temporary = tempfile.mkstemp(prefix=".payload-", dir=parent)
    digest = hashlib.sha256()
    size = characters = 0
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with os.fdopen(descriptor, "wb") as target, open_regular(source) as original:
            while block := original.read(CHUNK_BYTES):
                size += len(block)
                characters += len(decoder.decode(block))
                digest.update(block)
                target.write(block)
            characters += len(decoder.decode(b"", final=True))
            target.flush()
            os.fsync(target.fileno())
        data: dict[str, Any] = {"sha256": digest.hexdigest(), "bytes": size, "characters": characters}
        destination = payload_path(directory, data["sha256"])
        if destination.exists():
            stream_payload(directory, data, None)
            return data, False
        os.replace(temporary, destination)
        sync_directory(parent)
        return data, True
    finally:
        Path(temporary).unlink(missing_ok=True)


# Verified payloads by file identity, with (byte, character) offsets at block boundaries.
_verified: dict[tuple[str, int, int, int], list[tuple[int, int]]] = {}
VERIFIED_LIMIT = 64


def _payload_file(directory: Path, descriptor: dict[str, Any]) -> Path | None:
    if (not isinstance(descriptor, dict) or set(descriptor) != {"sha256", "bytes", "characters"}
            or any(type(descriptor[key]) is not int or descriptor[key] < 0 for key in ("bytes", "characters"))):
        raise ValueError("Invalid command payload metadata")
    digest = descriptor["sha256"]
    if digest == "" and descriptor["bytes"] == descriptor["characters"] == 0:
        return None
    return payload_path(directory, digest)


def _verify(source: BinaryIO, descriptor: dict[str, Any], target: TextIO | None) -> list[tuple[int, int]]:
    checksum = hashlib.sha256()
    size = characters = 0
    marks = [(0, 0)]
    decoder = codecs.getincrementaldecoder("utf-8")()
    while raw := source.read(CHUNK_BYTES):
        checksum.update(raw)
        size += len(raw)
        text = decoder.decode(raw)
        characters += len(text)
        marks.append((size - len(decoder.getstate()[0]), characters))
        if target is not None:
            target.write(text)
    text = decoder.decode(b"", final=True)
    characters += len(text)
    if target is not None:
        target.write(text)
    if size != descriptor["bytes"] or characters != descriptor["characters"] or checksum.hexdigest() != descriptor["sha256"]:
        raise ValueError("Command payload integrity check failed")
    return marks


def stream_payload(directory: Path, descriptor: dict[str, Any], target: TextIO | None) -> None:
    path = _payload_file(directory, descriptor)
    if path is not None:
        with open_regular(path) as source:
            _verify(source, descriptor, target)


def payload_page(directory: Path, descriptor: dict[str, Any], offset: int, limit: int) -> str:
    """Return characters `[offset, offset + limit)`, verifying each payload file once."""
    path = _payload_file(directory, descriptor)
    if path is None:
        return ""
    with open_regular(path) as source:
        stat = os.fstat(source.fileno())
        key = (descriptor["sha256"], stat.st_dev, stat.st_ino, stat.st_mtime_ns)
        marks = _verified.pop(key, None)
        if marks is None:
            marks = _verify(source, descriptor, None)
        _verified[key] = marks
        while len(_verified) > VERIFIED_LIMIT:
            del _verified[next(iter(_verified))]
        byte, position = marks[bisect.bisect_right([mark[1] for mark in marks], offset) - 1]
        source.seek(byte)
        decoder = codecs.getincrementaldecoder("utf-8")()
        pieces = []
        end = offset + limit
        while position < end:
            raw = source.read(CHUNK_BYTES)
            text = decoder.decode(raw, final=not raw)
            left, right = max(offset - position, 0), min(len(text), end - position)
            if right > left:
                pieces.append(text[left:right])
            position += len(text)
            if not raw:
                break
    return "".join(pieces)


class SmallField(io.StringIO):
    def write(self, text: str) -> int:
        if self.tell() + len(text) > 65536:
            raise ValueError("Inspection payload reference is too large")
        return super().write(text)


def record_fields(path: Path, metadata: dict[str, Any], targets: dict[str, TextIO], *,
                  payloads: bool = True) -> dict[str, bool]:
    """Validate a record; version-two payloads stream into targets unless `payloads` is false."""
    from .inspection import JSONRecordReader

    version = metadata.get("format_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported inspection-record format version")
    if version == 1:
        with io.TextIOWrapper(open_regular(path), encoding="utf-8", newline="") as source:
            return JSONRecordReader(source).fields(targets)
    fields = {name: SmallField() for name in ("output", "error", "error_suffix", "format_version")}
    with io.TextIOWrapper(open_regular(path), encoding="utf-8", newline="") as source:
        types = JSONRecordReader(source).fields({**targets, **fields})
    if types.pop("format_version") or fields["format_version"].getvalue() != "2":
        raise ValueError("Inspection-record version mismatch")
    if not types.pop("error_suffix") or fields["error_suffix"].getvalue() != metadata.get("error_suffix"):
        raise ValueError("Inspection-record error suffix mismatch")
    references = metadata.get("payloads")
    if not isinstance(references, dict) or set(references) != {"output", "error"}:
        raise ValueError("Invalid inspection-record payload references")
    for field in ("output", "error"):
        descriptor = references[field]
        if not isinstance(descriptor, dict) or not types[field] or fields[field].getvalue() != descriptor.get("sha256"):
            raise ValueError("Inspection-record payload reference mismatch")
        if payloads:
            stream_payload(path.parent, descriptor, targets[field])
    if payloads:
        targets["error"].write(fields["error_suffix"].getvalue())
    return types
