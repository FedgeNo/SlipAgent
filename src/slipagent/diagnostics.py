"""Private request snapshots and attempt outcomes, separate from conversation memory."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
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
        self.used = 0
        self.sequence = 0
        self.path: Path | None = None
        self.temporary: tempfile.TemporaryDirectory[str] | None = None
        self.error = ""

    @property
    def directory(self) -> Path:
        if self.path is None:
            self.temporary = tempfile.TemporaryDirectory(prefix="slipagent-requests-")
            self.path = Path(self.temporary.name)
        return self.path

    def use_directory(self, directory: Path) -> None:
        self.clear()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = directory

    def clear(self) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()
        self.path, self.temporary = None, None
        self.used, self.sequence = 0, 0
        self.error = ""

    def _write(self, path: Path, data: bytes) -> None:
        previous = path.stat().st_size if path.exists() else 0
        if self.used + len(data) - previous > self.quota:
            raise OSError(f"Request diagnostic quota of {self.quota} bytes reached; existing records are preserved")
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

    def begin(self, request: str, *, step: int, post: int) -> int | None:
        if self.error:
            return None
        try:
            digest = hashlib.sha256(request.encode("utf-8")).hexdigest()
            snapshot = self.directory / (digest + ".json.gz")
            if not snapshot.exists():
                self._write(snapshot, gzip.compress(request.encode("utf-8"), mtime=0))
            self.sequence += 1
            record = {"version": 1, "attempt": self.sequence, "step": step, "post": post, "request": snapshot.name,
                      "created": datetime.now(timezone.utc).isoformat(), "outcome": "pending"}
            self._write(self.directory / f"{self.sequence:08d}.json", json.dumps(record).encode())
            return self.sequence
        except OSError as exc:
            self.error = str(exc)
            return None

    def finish(self, attempt: int | None, outcome: str, *, detail: str = "", response: str = "",
               usage: dict[str, Any] | None = None) -> None:
        if attempt is None or self.error:
            return
        try:
            path = self.directory / f"{attempt:08d}.json"
            record = json.loads(path.read_text())
            record.update(outcome=outcome, detail=detail[:4000], response_excerpt=response[:16000],
                          response_characters=len(response), usage=usage)
            self._write(path, json.dumps(record, ensure_ascii=False).encode("utf-8"))
        except (OSError, ValueError) as exc:
            self.error = str(exc)

    def listing(self) -> list[dict[str, Any]]:
        return [json.loads((self.directory / f"{index:08d}.json").read_text())
                for index in range(max(1, self.sequence - 19), self.sequence + 1)
                if (self.directory / f"{index:08d}.json").is_file()]

    def read(self, attempt: int) -> str:
        if not 1 <= attempt <= self.sequence:
            raise ValueError("Unknown request attempt")
        record = json.loads((self.directory / f"{attempt:08d}.json").read_text())
        request = gzip.decompress((self.directory / record["request"]).read_bytes()).decode("utf-8")
        return json.dumps(record, indent=2, ensure_ascii=False) + "\n\n" + request

    async def aclose(self) -> None:
        self.clear()
