"""Version-two inspection records share immutable UTF-8 command payloads."""

from __future__ import annotations

import codecs
import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any, TextIO

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


def stream_payload(directory: Path, descriptor: dict[str, Any], target: TextIO | None) -> None:
    if (not isinstance(descriptor, dict) or set(descriptor) != {"sha256", "bytes", "characters"}
            or any(type(descriptor[key]) is not int or descriptor[key] < 0 for key in ("bytes", "characters"))):
        raise ValueError("Invalid command payload metadata")
    digest = descriptor["sha256"]
    if digest == "" and descriptor["bytes"] == descriptor["characters"] == 0:
        return
    path = payload_path(directory, digest)
    checksum = hashlib.sha256()
    size = characters = 0
    decoder = codecs.getincrementaldecoder("utf-8")()
    with open_regular(path) as source:
        while raw := source.read(CHUNK_BYTES):
            checksum.update(raw)
            size += len(raw)
            text = decoder.decode(raw)
            characters += len(text)
            if target is not None:
                target.write(text)
        text = decoder.decode(b"", final=True)
        characters += len(text)
        if target is not None:
            target.write(text)
    if size != descriptor["bytes"] or characters != descriptor["characters"] or checksum.hexdigest() != digest:
        raise ValueError("Command payload integrity check failed")


class SmallField(io.StringIO):
    def write(self, text: str) -> int:
        if self.tell() + len(text) > 65536:
            raise ValueError("Inspection payload reference is too large")
        return super().write(text)


def record_fields(path: Path, metadata: dict[str, Any], targets: dict[str, TextIO]) -> dict[str, bool]:
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
    payloads = metadata.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != {"output", "error"}:
        raise ValueError("Invalid inspection-record payload references")
    for field in ("output", "error"):
        descriptor = payloads[field]
        if not isinstance(descriptor, dict) or not types[field] or fields[field].getvalue() != descriptor.get("sha256"):
            raise ValueError("Inspection-record payload reference mismatch")
        stream_payload(path.parent, descriptor, targets[field])
    targets["error"].write(fields["error_suffix"].getvalue())
    return types
