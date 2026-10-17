"""Private request snapshots and attempt outcomes, separate from conversation memory."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REQUEST_LOG_QUOTA = 32 * 1024 * 1024


class RequestDiagnostics:
    """One journal's sidecar; a storage failure preserves files and stops logging.

    This archive never participates in context selection. Pending records are
    deliberately durable before transport, since a crash may prevent settlement.
    """
    def __init__(self, quota: int = DEFAULT_REQUEST_LOG_QUOTA) -> None:
        self.quota = quota
        self._rolling_quota = True
        self.used = 0
        self.sequence = 0
        self.path: Path | None = None
        self.temporary: tempfile.TemporaryDirectory[str] | None = None
        self.error = ""

    @property
    def directory(self) -> Path:
        if self.path is None:
            self.temporary = tempfile.TemporaryDirectory(prefix="slipagent-requests-")
            self.path = Path(self.temporary.name).resolve()
        return self.path

    def use_directory(self, directory: Path) -> None:
        self.clear()
        self.path = directory
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            for path in directory.iterdir():
                if path.is_file():
                    self.used += path.stat().st_size
                # Reserve existing attempt names even if their contents are
                # incomplete; resuming must never replace earlier records.
                if path.suffix == ".json" and path.stem.isascii() and path.stem.isdecimal():
                    self.sequence = max(self.sequence, int(path.stem))
        except OSError as exc:
            self.error = str(exc)

    def clear(self) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()
        self.path, self.temporary = None, None
        self.used, self.sequence = 0, 0
        self.error = ""

    def _migrate_quota(self) -> None:
        # Live reload retains the old owner, including a latched quota error.
        if "_rolling_quota" not in self.__dict__:
            if self.__dict__["quota"] is None:
                self.quota = DEFAULT_REQUEST_LOG_QUOTA
            quota_error = f"Request diagnostic quota of {self.quota} bytes reached; existing records are preserved"
            if self.error == quota_error:
                self.error = ""
            self._rolling_quota = True

    def _discard(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return
        path.unlink()
        self.used -= size

    def _trim(self) -> None:
        """Roll whole attempts forward only after the newest write is durable."""
        if self.used <= self.quota:
            return
        records: list[tuple[Path, str | None]] = []
        references: Counter[str] = Counter()
        files = list(self.directory.iterdir())
        for path in files:
            if path.suffix != ".json" or not path.stem.isascii() or not path.stem.isdecimal():
                continue
            try:
                record = json.loads(path.read_text())
            except ValueError:
                record = {}
            snapshot = record.get("request") if isinstance(record, dict) else None
            if not isinstance(snapshot, str) or re.fullmatch(r"[0-9a-f]{64}\.json\.gz", snapshot) is None:
                snapshot = None
            if snapshot is not None:
                references[snapshot] += 1
            records.append((path, snapshot))
        # Interrupted writes may leave an unreferenced snapshot behind.
        for path in files:
            if re.fullmatch(r"[0-9a-f]{64}\.json\.gz", path.name) and path.name not in references:
                self._discard(path)
        for path, snapshot in sorted(records, key=lambda item: int(item[0].stem)):
            if self.used <= self.quota:
                break
            # Even an attempt larger than the target keeps its complete files.
            if int(path.stem) == self.sequence:
                continue
            self._discard(path)
            self._discard(self.directory / f"{path.stem}-response.json.gz")
            if snapshot is not None:
                references[snapshot] -= 1
                if not references[snapshot]:
                    self._discard(self.directory / snapshot)

    def _write(self, path: Path, data: bytes) -> None:
        previous = path.stat().st_size if path.exists() else 0
        fd, temporary = tempfile.mkstemp(prefix=".request-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
            self.used += len(data) - previous
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def begin(self, request: Any, *, step: int, step_id: int) -> int | None:
        self._migrate_quota()
        if self.error:
            return None
        try:
            from .requestbody import RequestBody
            if isinstance(request, RequestBody):
                chunks = request.chunks
            else:
                encoded = (request if isinstance(request, str) else json.dumps(request, ensure_ascii=False)).encode("utf-8")
                chunks = lambda: iter([encoded])
            digest = hashlib.sha256()
            for chunk in chunks():
                digest.update(chunk)
            snapshot = self.directory / (digest.hexdigest() + ".json.gz")
            if not snapshot.exists():
                fd, temporary = tempfile.mkstemp(prefix=".request-", dir=self.directory)
                try:
                    with os.fdopen(fd, "wb") as target:
                        with gzip.GzipFile(fileobj=target, mode="wb", mtime=0, filename="") as compressed:
                            for chunk in chunks():
                                compressed.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                    size = os.path.getsize(temporary)
                    os.replace(temporary, snapshot)
                    self.used += size
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            self.sequence += 1
            record = {"version": 1, "attempt": self.sequence, "step": step, "step_id": step_id, "request": snapshot.name,
                      "created": datetime.now(timezone.utc).isoformat(), "outcome": "pending"}
            self._write(self.directory / f"{self.sequence:08d}.json", json.dumps(record).encode())
            self._trim()
            return self.sequence
        except OSError as exc:
            self.error = str(exc)
            return None

    def finish(self, attempt: int | None, outcome: str, *, detail: str = "", response: str = "",
               usage: dict[str, Any] | None = None) -> None:
        self._migrate_quota()
        if attempt is None or self.error:
            return
        try:
            path = self.directory / f"{attempt:08d}.json"
            record = json.loads(path.read_text())
            record.update(outcome=outcome, detail=detail, response_excerpt=response,
                          response_characters=len(response), usage=usage)
            self._write(path, json.dumps(record, ensure_ascii=False).encode("utf-8"))
            self._trim()
        except (OSError, ValueError) as exc:
            self.error = str(exc)

    def response(self, attempt: int | None, body: dict[str, Any]) -> None:
        """Save the complete provider response before normalization."""
        self._migrate_quota()
        if attempt is None or self.error:
            return
        try:
            path = self.directory / f"{attempt:08d}.json"
            record = json.loads(path.read_text())
            name = f"{attempt:08d}-response.json.gz"
            self._write(self.directory / name, gzip.compress(json.dumps(body, ensure_ascii=False).encode("utf-8"), mtime=0))
            record["provider_response"] = name
            self._write(path, json.dumps(record, ensure_ascii=False).encode("utf-8"))
            self._trim()
        except (OSError, ValueError) as exc:
            self.error = str(exc)

    def listing(self) -> list[dict[str, Any]]:
        return [json.loads((self.directory / f"{index:08d}.json").read_text())
                for index in range(max(1, self.sequence - 19), self.sequence + 1)
                if (self.directory / f"{index:08d}.json").is_file()]

    def read(self, attempt: int) -> str:
        if not 1 <= attempt <= self.sequence:
            raise ValueError("Unknown request attempt")
        path = self.directory / f"{attempt:08d}.json"
        if not path.is_file():
            raise ValueError("Request attempt is no longer retained in the rolling log")
        record = json.loads(path.read_text())
        request = gzip.decompress((self.directory / record["request"]).read_bytes()).decode("utf-8")
        result = json.dumps(record, indent=2, ensure_ascii=False) + "\n\n" + request
        if record.get("provider_response"):
            result += "\n\nProvider response (complete; stream deltas joined):\n"
            result += gzip.decompress((self.directory / record["provider_response"]).read_bytes()).decode("utf-8")
        return result

    async def aclose(self) -> None:
        self.clear()
