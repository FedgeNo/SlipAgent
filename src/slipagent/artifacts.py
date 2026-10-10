"""Private immutable inspection records addressed by session-owned numbers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import threading
from pathlib import Path
from typing import Any

from .prompts import load_prompt
from .tools.base import Tool, ToolResult, current_invocation
from .tools.blocking import run_blocking


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
            return Path(self._temporary.name)
        return self._directory

    def use_directory(self, directory: Path, *, source: Path | None = None) -> None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if source is not None and source.exists() and source != directory:
            shutil.copytree(source, directory, dirs_exist_ok=True)
        records = []
        index = directory / "index.json"
        if index.exists():
            records = json.loads(index.read_text(encoding="utf-8"))
        restored = {}
        for record in records:
            identifier = record["record_id"]
            if type(identifier) is not int or identifier < 1 or record["filename"] != f"{identifier}.json":
                raise ValueError("Invalid inspection-record index")
            restored[identifier] = record
        self._directory = directory
        self.records = restored
        self.used_bytes = sum(record["bytes"] for record in restored.values())

    def save(self, command: str, content: str, *, kind: str, step_id: int | None = None,
             call_id: str | None = None, error: str = "") -> dict[str, Any]:
        with self._lock:
            return self._save(command, content, kind=kind, step_id=step_id, call_id=call_id, error=error)

    def _save(self, command: str, content: str, *, kind: str, step_id: int | None,
              call_id: str | None, error: str) -> dict[str, Any]:
        invocation = current_invocation.get()
        if invocation is not None and step_id is None and call_id is None:
            step_id, call_id = invocation
        data = json.dumps({"command": command, "output": content, "error": error,
                           "call_id": call_id, "step_id": step_id, "kind": kind},
                          ensure_ascii=False, indent=2).encode("utf-8")
        if self.used_bytes + len(data) > self.quota_bytes:
            raise OSError("Inspection-record storage quota reached; record was not saved.")
        identifier = time.time_ns() // 1000
        while True:
            path = self.directory / f"{identifier}.json"
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError:
                identifier += 1
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        record = {"record_id": identifier, "filename": path.name, "kind": kind,
                  "command": command, "step_id": step_id, "call_id": call_id,
                  "bytes": len(data)}
        proposed = {**self.records, identifier: record}
        fd, name = tempfile.mkstemp(prefix=".index-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump(list(proposed.values()), target, ensure_ascii=False)
                target.flush()
                os.fsync(target.fileno())
            os.replace(name, self.directory / "index.json")
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.records = proposed
        self.used_bytes += len(data)
        return dict(record)

    def read(self, record_id: int) -> dict[str, Any]:
        if type(record_id) is not int or record_id not in self.records:
            raise ValueError("Unknown inspection record in this session.")
        record = self.records[record_id]
        path = self.directory / record["filename"]
        if path.is_symlink():
            raise OSError("Inspection record was replaced by a symbolic link.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"command", "output", "error", "call_id", "step_id", "kind"} or any(
            not isinstance(data[key], str) for key in ("command", "output", "error", "kind")
        ) or any(data[key] != record[key] for key in ("command", "call_id", "step_id", "kind")):
            raise ValueError("Invalid inspection record content.")
        return {**record, **data}

    def references(self, step_id: int, call_id: str) -> list[dict[str, Any]]:
        return [dict(record) for record in self.records.values()
                if record["step_id"] == step_id and record["call_id"] == call_id]

    def clear(self) -> None:
        self.records.clear()
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
    parameters = {"type": "object", "properties": {"record_id": {"type": "integer", "minimum": 1, "description": ""}}}

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    async def run(self, record_id: int | None = None) -> ToolResult:
        try:
            if record_id is None:
                return ToolResult.ok({"records": list(self.store.records.values())})
            return ToolResult.ok(await run_blocking(self.store.read, record_id))
        except (OSError, ValueError, UnicodeError) as exc:
            return ToolResult.error(str(exc))
