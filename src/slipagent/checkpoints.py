"""Private, durable file-edit checkpoints; conversation history stays intact."""

from __future__ import annotations

import hashlib
import difflib
import json
import os
import shutil
import stat
import tempfile
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from .workspace import Workspace, WorkspaceError
from .fileio import open_regular, read_regular

current_checkpoint: ContextVar[FileCheckpoints | None] = ContextVar("slipagent_checkpoint", default=None)
FileState = tuple[str | None, int | None]
CHUNK_BYTES = 65536
PREVIEW_BYTES = 200000


class CheckpointError(OSError):
    """A checkpoint cannot be saved or safely restored."""


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_stream(source: BinaryIO, output: BinaryIO | None = None) -> str:
    digest = hashlib.sha256()
    while block := source.read(CHUNK_BYTES):
        digest.update(block)
        if output is not None:
            output.write(block)
    return digest.hexdigest()


def _file_state(path: Path) -> FileState:
    if not path.exists():
        return None, None
    with open_regular(path) as source:
        mode = stat.S_IMODE(os.fstat(source.fileno()).st_mode)
        return _hash_stream(source), mode


def _publish(temporary: str, path: Path) -> None:
    os.replace(temporary, path)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        _publish(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class FileCheckpoints:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.temporary = tempfile.TemporaryDirectory(prefix="slipagent-checkpoints-")
        self.directory = Path(self.temporary.name)
        self.batches: list[dict[str, Any]] = []
        self.active: dict[str, Any] | None = None

    def clear(self) -> None:
        self.directory = Path(self.temporary.name)
        self.batches = []
        self.active = None
        shutil.rmtree(self.directory, ignore_errors=True)
        self.directory.mkdir(mode=0o700)

    def use_directory(self, directory: Path, *, source: Path | None = None) -> None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if source is not None and source.exists() and source != directory:
            shutil.copytree(source, directory, dirs_exist_ok=True)
        index = directory / "index.json"
        try:
            batches = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
            if not isinstance(batches, list) or any(not self._valid_batch(batch) for batch in batches):
                raise ValueError("invalid checkpoint index")
        except (OSError, ValueError) as exc:
            raise CheckpointError(f"Cannot load file checkpoints: {exc}") from exc
        self.directory, self.batches, self.active = directory, batches, None

    @staticmethod
    def _valid_batch(batch: Any) -> bool:
        if (not isinstance(batch, dict) or not isinstance(batch.get("files"), list)
                or not isinstance(batch.get("id"), str) or type(batch.get("step_id")) is not int
                or not isinstance(batch.get("created"), str) or type(batch.get("rewound")) is not bool):
            return False
        for entry in batch["files"]:
            if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                    or not {"path", "before", "after", "mode", "status"} <= entry.keys()
                    or not Path(entry["path"]).is_absolute()
                    or not isinstance(entry.get("status"), str)
                    or entry["status"] not in {"pending", "done", "failed"}):
                return False
            for key in ("before", "after"):
                value = entry.get(key)
                if value is None and key == "before":
                    continue
                if not isinstance(value, str) or not _valid_hash(value):
                    return False
            for key in ("mode", "after_mode"):
                mode = entry.get(key)
                if mode is not None and (type(mode) is not int or not 0 <= mode <= 0o7777):
                    return False
        return True

    def _persist(self) -> None:
        _save(self.directory / "index.json", json.dumps(self.batches, ensure_ascii=False).encode("utf-8"))

    def begin(self, step_id: int) -> None:
        self.active = {"id": uuid.uuid4().hex, "step_id": step_id,
                       "created": datetime.now(timezone.utc).isoformat(), "files": [], "rewound": False}

    def finish(self) -> None:
        self.active = None

    def _backup(self, target: Path) -> FileState:
        directory = self.directory / "blobs"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as output, open_regular(target) as source:
                mode = stat.S_IMODE(os.fstat(source.fileno()).st_mode)
                before = _hash_stream(source, output)
                output.flush()
                os.fsync(output.fileno())
            _publish(name, directory / before)
            return before, mode
        finally:
            Path(name).unlink(missing_ok=True)

    def prepare(self, target: Path, after: bytes) -> dict[str, Any] | None:
        if self.active is None:
            return None
        before: str | None = None
        mode: int | None = None
        if target.exists():
            before, mode = self._backup(target)
        entry = {"path": str(target), "before": before, "after": _hash(after),
                 "mode": mode, "status": "pending"}
        self.active["files"].append(entry)
        if not any(batch is self.active for batch in self.batches):
            self.batches.append(self.active)
        # Backups and intended hashes reach durable storage before publishing edits.
        try:
            self._persist()
        except OSError:
            self.active["files"].remove(entry)
            if not self.active["files"]:
                self.batches.remove(self.active)
            raise
        return entry

    def complete(self, entry: dict[str, Any] | None) -> None:
        if entry is not None:
            entry["status"] = "done"
            entry["after_mode"] = stat.S_IMODE(Path(entry["path"]).stat().st_mode)
            self._persist()

    def abort(self, entry: dict[str, Any] | None) -> None:
        if entry is not None:
            target = Path(entry["path"])
            try:
                actual, _ = _file_state(target)
                entry["status"] = "failed" if actual == entry["before"] else "pending"
            except OSError:
                entry["status"] = "pending"
            self._persist()

    def listing(self) -> list[dict[str, Any]]:
        return [batch for batch in reversed(self.batches)
                if not batch["rewound"] and any(entry["status"] != "failed" for entry in batch["files"])]

    def _selected(self, checkpoint_id: str) -> list[dict[str, Any]]:
        remaining = list(reversed(self.listing()))
        for index, batch in enumerate(remaining):
            if batch["id"] == checkpoint_id:
                return remaining[index:]
        raise CheckpointError("Unknown or already restored file checkpoint.")

    def _backup_path(self, digest: str) -> Path:
        if not isinstance(digest, str) or not _valid_hash(digest):
            raise CheckpointError("Invalid checkpoint backup hash. No files restored.")
        return (self.directory / "blobs" / digest).resolve(strict=True)

    def preview(self, checkpoint_id: str) -> str:
        from .tools.editing import split_lines

        selected, originals, planned = self._plan(checkpoint_id)
        names = [self.workspace.relative(path) for path in planned]
        diffs: list[str] = []
        remaining = 12000
        for path, (before, _) in planned.items():
            if planned[path] == originals[path]:
                continue
            after = originals[path][0]
            name = self.workspace.relative(path)
            before_path = self._backup_path(before) if before is not None else None
            after_path = path if after is not None else None
            size = sum(source.stat().st_size for source in (before_path, after_path) if source is not None)
            if size > PREVIEW_BYTES:
                diff = f"{name}: diff omitted for large contents; original bytes are retained\n"
            else:
                before_data = read_regular(before_path, PREVIEW_BYTES) if before_path is not None else b""
                after_data = read_regular(after_path, PREVIEW_BYTES) if after_path is not None else b""
                if before is not None and _hash(before_data) != before:
                    raise CheckpointError("Checkpoint backup is damaged. No files restored.")
                if after is not None and _hash(after_data) != after:
                    raise CheckpointError(f"File changed during rewind preview: {path}. No files restored.")
                if b"\0" in before_data or b"\0" in after_data:
                    diff = f"{name}: binary contents restored\n"
                else:
                    diff = "".join(difflib.unified_diff(
                        [line + "\n" for line in split_lines(after_data.decode("utf-8", errors="replace"))],
                        [line + "\n" for line in split_lines(before_data.decode("utf-8", errors="replace"))],
                        fromfile=name + " (current)", tofile=name + " (restored)",
                    ))
                    if not diff and before_data != after_data:
                        diff = f"{name}: byte-level changes in line endings or encoding\n"
            if len(diff) > remaining:
                diffs.append(diff[:remaining] + "\n… diff preview abbreviated\n")
                break
            diffs.append(diff)
            remaining -= len(diff)
        return (f"Restore files to before step {selected[0]['step_id']} ({len(selected)} edit batch(es)):\n"
                + "\n".join("  " + name for name in names)
                + "\n\n" + "\n".join(diffs)
                + "\nConversation history is retained. Shell commands, MCP actions, and external edits are not undone.")

    def _plan(self, checkpoint_id: str) -> tuple[list[dict[str, Any]], dict[Path, FileState], dict[Path, FileState]]:
        selected = self._selected(checkpoint_id)
        originals: dict[Path, FileState] = {}
        planned: dict[Path, FileState] = {}
        final: dict[Path, FileState] = {}
        for batch in selected:
            for entry in batch["files"]:
                if entry["status"] != "failed":
                    final.setdefault(Path(entry["path"]), (entry["before"], entry["mode"]))
        restored: set[Path] = set()
        for batch in reversed(selected):
            for entry in reversed(batch["files"]):
                if entry["status"] == "failed":
                    continue
                target = Path(entry["path"])
                try:
                    resolved = self.workspace.resolve(target)
                except WorkspaceError as exc:
                    raise CheckpointError(f"Cannot restore {target}: {exc}. No files restored.") from exc
                if resolved != target or target.is_symlink():
                    raise CheckpointError(f"Path changed since the edit: {target}. No files restored.")
                if target not in originals:
                    originals[target] = _file_state(target)
                    planned[target] = originals[target]
                    if originals[target] == final[target]:
                        restored.add(target)
                if target in restored:
                    continue
                actual, mode = planned[target]
                if actual == entry["before"] and mode == entry["mode"]:
                    continue  # Unpublished writes and already-restored entries need no write.
                expected_mode = entry.get("after_mode", entry["mode"])
                if actual != entry["after"] or expected_mode is not None and mode != expected_mode:
                    raise CheckpointError(f"File changed since the edit: {target}. No files restored.")
                before = entry["before"]
                if before is not None:
                    if _file_state(self._backup_path(before))[0] != before:
                        raise CheckpointError("Checkpoint backup is damaged. No files restored.")
                    planned[target] = before, entry["mode"]
                else:
                    planned[target] = None, None
        return selected, originals, planned

    def rewind(self, checkpoint_id: str) -> list[str]:
        from .tools.files import _atomic_write_stream

        selected, originals, planned = self._plan(checkpoint_id)
        restored_names: list[str] = []
        try:
            for target, (digest, mode) in planned.items():
                if planned[target] == originals[target]:
                    continue

                def check_current() -> None:
                    if _file_state(target) != originals[target] or self.workspace.resolve(target) != target:
                        raise CheckpointError(f"File changed during rewind: {target}")

                check_current()
                if digest is None:
                    target.unlink(missing_ok=True)
                else:
                    blob = self._backup_path(digest)

                    def copy(output: BinaryIO) -> None:
                        with open_regular(blob) as source:
                            if _hash_stream(source, output) != digest:
                                raise CheckpointError("Checkpoint backup is damaged")

                    # Verify the bytes copied into the temporary file, then
                    # recheck the destination immediately before publishing it.
                    _atomic_write_stream(self.workspace, target, copy, before_replace=check_current)
                    if mode is not None:
                        target.chmod(mode)
                restored_names.append(self.workspace.relative(target))
        except (OSError, WorkspaceError) as exc:
            raise CheckpointError(f"Rewind interrupted: {exc}. Restored files: {', '.join(restored_names) or '(none)'}. Inspect before retrying.") from exc
        for batch in selected:
            batch["rewound"] = True
        try:
            self._persist()
        except OSError as exc:
            raise CheckpointError(f"Files restored ({', '.join(restored_names)}), but the checkpoint index could not be saved: {exc}") from exc
        return restored_names

    async def aclose(self) -> None:
        self.temporary.cleanup()


def _valid_hash(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
