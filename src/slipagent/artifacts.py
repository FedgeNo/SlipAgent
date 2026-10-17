"""Private immutable inspection records addressed by session-owned numbers."""

from __future__ import annotations

import json
import io
import os
import shutil
import tempfile
import time
import threading
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from .prompts import load_prompt
from .tools.base import Tool, ToolResult, current_invocation
from .tools.blocking import run_blocking

MAX_ARTIFACT_PAGE_CHARS = 12_000


class ArtifactFieldPage(io.TextIOBase):
    """Collect one decoded string slice while validating the enclosing record."""

    def __init__(self, offset: int, limit: int, expected: str | None = None) -> None:
        self.offset, self.limit, self.expected = offset, limit, expected
        self.characters = 0
        self.pieces: list[str] = []

    def write(self, text: str) -> int:
        start = self.characters
        self.characters += len(text)
        if self.expected is not None and self.expected[start:self.characters] != text:
            raise ValueError("Inspection-record metadata mismatch.")
        left, right = max(start, self.offset), min(self.characters, self.offset + self.limit)
        if right > left:
            self.pieces.append(text[left - start:right - start])
        return len(text)


class ArtifactStore:
    def __init__(self, quota_bytes: int) -> None:
        self.quota_bytes = quota_bytes
        self.records: dict[int, dict[str, Any]] = {}
        self.used_bytes = 0
        self._directory: Path | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._lock = threading.RLock()

    @property
    def directory(self) -> Path:
        if self._directory is None:
            if self._temporary is None:
                self._temporary = tempfile.TemporaryDirectory(prefix="slipagent-artifacts-")
            return Path(self._temporary.name).resolve()
        return self._directory

    def use_directory(self, directory: Path, *, source: Path | None = None) -> None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if source is not None and source.exists() and source != directory:
            shutil.copytree(source, directory, dirs_exist_ok=True, symlinks=True)
        records = []
        index = directory / "index.json"
        if index.exists():
            records = json.loads(index.read_text(encoding="utf-8"))
        restored = {}
        for record in records:
            identifier = record["record_id"]
            version = record.get("format_version", 1)
            suffix = ".json" if version == 1 else ".v2.json"
            if (type(version) is not int or version not in (1, 2) or type(identifier) is not int
                    or identifier < 1 or record["filename"] != f"{identifier}{suffix}"):
                raise ValueError("Invalid inspection-record index")
            restored[identifier] = record
        self._directory = directory
        self.__dict__.pop("_encoded", None)
        self.records = restored
        self.used_bytes = sum(record["bytes"] for record in restored.values())

    def save(self, command: str, content: str | Iterator[str], *, kind: str, step_id: int | None = None,
             call_id: str | None = None, error: str | Iterator[str] = "") -> dict[str, Any]:
        with self._lock:
            return self._save(command, content, kind=kind, step_id=step_id, call_id=call_id, error=error)

    def _save(self, command: str, content: str | Iterator[str], *, kind: str, step_id: int | None,
              call_id: str | None, error: str | Iterator[str], payloads: dict[str, Any] | None = None,
              error_suffix: str = "", replace_id: int | None = None) -> dict[str, Any]:
        invocation = current_invocation.get()
        if invocation is not None and step_id is None and call_id is None:
            step_id, call_id = invocation
        identifier = replace_id if replace_id is not None else time.time_ns() // 1000
        old = self.records.get(replace_id) if replace_id is not None else None
        payload_bytes = sum(item["bytes"] for item in payloads.values()) if payloads is not None else 0
        baseline = self.used_bytes - (old["bytes"] if old else 0) + payload_bytes
        while True:
            if replace_id is None and identifier in self.records:
                identifier += 1
                continue
            path = self.directory / f"{identifier}{'.v2' if payloads is not None else ''}.json"
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError:
                if replace_id is not None:
                    raise OSError("Inspection migration destination already exists; original record was retained")
                identifier += 1
        size = 0
        try:
            with os.fdopen(fd, "wb") as target:
                def write(text: str) -> None:
                    nonlocal size
                    encoded = text.encode("utf-8")
                    if baseline + size + len(encoded) > self.quota_bytes:
                        raise OSError("Inspection-record storage quota reached; record was not saved.")
                    target.write(encoded)
                    size += len(encoded)
                write("{")
                values = {"command": command, "output": content, "error": error,
                          "call_id": call_id, "step_id": step_id, "kind": kind}
                if payloads is not None:
                    values.update(format_version=2, error_suffix=error_suffix)
                for n, (key, value) in enumerate(values.items()):
                    write(("," if n else "") + "\n  " + json.dumps(key) + ": ")
                    if value is None or isinstance(value, int):
                        write(json.dumps(value))
                    else:
                        write('"')
                        chunks = (value[start:start + 4096] for start in range(0, len(value), 4096)) if isinstance(value, str) else value
                        for chunk in chunks:
                            write(json.dumps(chunk, ensure_ascii=False)[1:-1])
                        write('"')
                write("\n}")
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        record = {"record_id": identifier, "filename": path.name, "kind": kind,
                  "command": command, "step_id": step_id, "call_id": call_id,
                  "bytes": size + payload_bytes}
        if payloads is not None:
            record.update(format_version=2, payloads=payloads, error_suffix=error_suffix, record_bytes=size)
        proposed = {**self.records, identifier: record}
        fd, name = tempfile.mkstemp(prefix=".index-", dir=self.directory)
        published = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as index_target:
                index_target.write(self._encoded_index(proposed))
                index_target.flush()
                os.fsync(index_target.fileno())
            os.replace(name, self.directory / "index.json")
            published = True
            self.records = proposed
            self.used_bytes = baseline + size
            if payloads is not None:
                from .payloads import sync_directory
                sync_directory(self.directory)
        except BaseException:
            if not published:
                path.unlink(missing_ok=True)
            raise
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return dict(record)

    def _encoded_index(self, records: dict[int, dict[str, Any]]) -> str:
        """Encode the index, reusing each unchanged record's JSON text."""
        encoded = self.__dict__.setdefault("_encoded", {})
        parts = []
        for identifier, record in records.items():
            cached = encoded.get(identifier)
            if cached is None or cached[0] is not record:
                cached = encoded[identifier] = (record, json.dumps(record, ensure_ascii=False))
            parts.append(cached[1])
        return "[" + ", ".join(parts) + "]"

    def save_command(self, command: str, streams: dict[str, Path | None], *, kind: str,
                     step_id: int | None, call_id: str | None, error_suffix: str,
                     replace_id: int | None = None) -> dict[str, Any]:
        from .payloads import copy_payload, payload_path

        with self._lock:
            if replace_id is not None:
                self._compare_command(replace_id, command, streams, kind, step_id, call_id, error_suffix)
            created = []
            payloads = {}
            try:
                for field, name in (("output", "stdout"), ("error", "stderr")):
                    descriptor, new = copy_payload(self.directory, streams[name])
                    payloads[field] = descriptor
                    if new:
                        created.append(payload_path(self.directory, descriptor["sha256"]))
                if replace_id is not None:
                    orphan = self.directory / f"{replace_id}.v2.json"
                    if orphan.exists():
                        expected = {**self.records[replace_id], "format_version": 2,
                                    "payloads": payloads, "error_suffix": error_suffix}
                        self._compare_command(replace_id, command, streams, kind, step_id, call_id, error_suffix,
                                              record_override=expected, path_override=orphan)
                        orphan.unlink()
                return self._save(command, payloads["output"]["sha256"], kind=kind, step_id=step_id,
                                  call_id=call_id, error=payloads["error"]["sha256"], payloads=payloads,
                                  error_suffix=error_suffix, replace_id=replace_id)
            except BaseException:
                for path in created:
                    if not any(item.get("sha256") == path.name for record in self.records.values()
                               for item in record.get("payloads", {}).values()):
                        path.unlink(missing_ok=True)
                raise

    def _compare_command(self, record_id: int, command: str, streams: dict[str, Path | None],
                         kind: str, step_id: int | None, call_id: str | None, error_suffix: str, *,
                         record_override: dict[str, Any] | None = None, path_override: Path | None = None) -> None:
        from contextlib import ExitStack
        from .fileio import open_regular
        from .payloads import record_fields
        from typing import TextIO, cast

        class CompareField(io.TextIOBase):
            def __init__(self, source: Any, suffix: str = "") -> None:
                self.source = source
                self.suffix = io.StringIO(suffix)

            def write(self, text: str) -> int:
                expected = self.source.read(len(text))
                expected += self.suffix.read(len(text) - len(expected))
                if text != expected:
                    raise ValueError("Legacy inspection record differs from command log; both copies were retained")
                return len(text)

            def complete(self) -> None:
                if self.source.read(1) or self.suffix.read(1):
                    raise ValueError("Legacy inspection record is shorter than command log; both copies were retained")

        record = self.records[record_id] if record_override is None else record_override
        if any(record[key] != value for key, value in dict(command=command, kind=kind, step_id=step_id, call_id=call_id).items()):
            raise ValueError("Legacy command provenance mismatch; both copies were retained")
        with ExitStack() as stack:
            targets = {}
            for field, name in (("output", "stdout"), ("error", "stderr")):
                path = streams[name]
                source = (stack.enter_context(io.TextIOWrapper(open_regular(path), encoding="utf-8", newline=""))
                          if path is not None else io.StringIO())
                targets[field] = CompareField(source, error_suffix if field == "error" else "")
            for field in ("command", "kind", "step_id", "call_id"):
                expected = record[field] if isinstance(record[field], str) else json.dumps(record[field])
                targets[field] = CompareField(io.StringIO(expected))
            path = self.record_path(record_id) if path_override is None else path_override
            types = record_fields(path, record, {key: cast(TextIO, value) for key, value in targets.items()})
            for field in ("command", "kind", "step_id", "call_id"):
                if types[field] != isinstance(record[field], str):
                    raise ValueError("Legacy command metadata type mismatch")
            for target in targets.values():
                target.complete()

    def legacy_command_copy(self, record: dict[str, Any]) -> Path | None:
        """Recover cleanup after the version-two index committed before a crash."""
        from .payloads import payload_path
        old = self.directory / f"{record['record_id']}.json"
        if not old.exists():
            return None
        streams = {name: payload_path(self.directory, record["payloads"][field]["sha256"])
                   if record["payloads"][field]["sha256"] else None
                   for name, field in (("stdout", "output"), ("stderr", "error"))}
        self._compare_command(record["record_id"], record["command"], streams, record["kind"], record["step_id"],
                              record["call_id"], record["error_suffix"],
                              record_override={**record, "format_version": 1}, path_override=old)
        return old

    def read(self, record_id: int) -> dict[str, Any]:
        path = self.record_path(record_id)
        record = self.records[record_id]
        if record.get("format_version") == 2:
            import sys
            return {**record, "output": self.read_page(record_id, "output", 0, sys.maxsize)["output"],
                    "error": self.read_page(record_id, "error", 0, sys.maxsize)["error"]}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"command", "output", "error", "call_id", "step_id", "kind"} or any(
            not isinstance(data[key], str) for key in ("command", "output", "error", "kind")
        ) or any(data[key] != record[key] for key in ("command", "call_id", "step_id", "kind")):
            raise ValueError("Invalid inspection record content.")
        return {**record, **data}

    def record_path(self, record_id: int) -> Path:
        if type(record_id) is not int or record_id not in self.records:
            raise ValueError("Unknown inspection record in this session.")
        record = self.records[record_id]
        path = self.directory / str(record["filename"])
        if path.is_symlink():
            raise OSError("Inspection record was replaced by a symbolic link.")
        return path

    def read_page(self, record_id: int, field: str, offset: int, limit: int) -> dict[str, Any]:
        from .payloads import record_fields
        from typing import TextIO, cast

        path = self.record_path(record_id)
        record = self.records[record_id]
        fields = {}
        for key in ("command", "output", "error", "call_id", "step_id", "kind"):
            expected = None
            if key in {"command", "call_id", "step_id", "kind"}:
                expected = record[key] if isinstance(record[key], str) else json.dumps(record[key])
            fields[key] = ArtifactFieldPage(offset if key == field else 0, limit if key == field else 0, expected)
        version_two = record.get("format_version") == 2
        string_fields = record_fields(path, record, {key: cast(TextIO, target) for key, target in fields.items()},
                                      payloads=not version_two)
        for key in ("command", "call_id", "step_id", "kind"):
            expected = fields[key].expected
            if (string_fields[key] != isinstance(record[key], str)
                    or expected is None or fields[key].characters != len(expected)):
                raise ValueError("Inspection-record metadata mismatch.")
        if version_two and field in ("output", "error"):
            # Shared payloads are verified once, then paged by seeking.
            from .payloads import payload_page
            descriptor = record["payloads"][field]
            suffix = record["error_suffix"] if field == "error" else ""
            total = descriptor["characters"] + len(suffix)
            text = payload_page(self.directory, descriptor, offset, limit)
            start = max(offset - descriptor["characters"], 0)
            text += suffix[start:start + limit - len(text)]
        else:
            selected = fields[field]
            text, total = "".join(selected.pieces), selected.characters
        next_offset = offset + len(text)
        return {**self._page_metadata(record), field: text, "field": field, "offset": offset,
                "next_offset": next_offset if next_offset < total else None,
                "eof": next_offset >= total, "total_characters": total,
                "offset_unit": "characters"}

    @staticmethod
    def _page_metadata(record: dict[str, Any]) -> dict[str, Any]:
        return {**record, "command": record["command"][:500],
                "command_truncated": len(record["command"]) > 500}

    def listing_page(self, offset: int, limit: int) -> dict[str, Any]:
        from itertools import islice

        records = [self._page_metadata(record) for record in islice(self.records.values(), offset, offset + limit)]
        next_offset = offset + len(records)
        total = len(self.records)
        return {"records": records, "offset": offset, "next_offset": next_offset if next_offset < total else None,
                "eof": next_offset >= total, "total_records": total, "offset_unit": "records"}

    def references(self, step_id: int, call_id: str) -> list[dict[str, Any]]:
        return [dict(record) for record in self.records.values()
                if record["step_id"] == step_id and record["call_id"] == call_id]

    def clear(self) -> None:
        self.records.clear()
        self.__dict__.pop("_encoded", None)
        self.used_bytes = 0
        self._directory = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    async def aclose(self) -> None:
        self.clear()


class ReadArtifactTool(Tool):
    parameter_prompts = "tools/read-artifact-parameters.json"
    name = "read_artifact"
    description_prompt = "tools/read-artifact.md"
    description = load_prompt(description_prompt)
    progress_exempt = True
    parameters = {"type": "object", "properties": {
        "record_id": {"type": "integer", "minimum": 1, "description": ""},
        "field": {"type": "string", "enum": ["output", "error", "command"], "description": ""},
        "offset": {"type": "integer", "minimum": 0, "description": ""},
        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ARTIFACT_PAGE_CHARS, "description": ""}}}

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    async def run(self, record_id: int | None = None, *, field: str = "output",
                  offset: int = 0, limit: int | None = None) -> ToolResult:
        try:
            if field not in {"output", "error", "command"} or type(offset) is not int or offset < 0:
                raise ValueError("Use output, error, or command and a nonnegative character/record offset.")
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("limit must be a positive integer")
            if record_id is None:
                return ToolResult.ok(self.store.listing_page(offset, min(limit or 20, 100)))
            return ToolResult.ok(await run_blocking(self.store.read_page, record_id, field, offset,
                                                     min(limit or 4000, MAX_ARTIFACT_PAGE_CHARS)))
        except (OSError, ValueError, UnicodeError) as exc:
            return ToolResult.error(str(exc))
