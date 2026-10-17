"""Incremental private capture of HTTP response bodies before decoding."""

from __future__ import annotations

import asyncio
import gzip
import os
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx

# Unfinished streams become inspectable within this many seconds.
FLUSH_INTERVAL = 1.0


class ResponseRecorder:
    def __init__(self, path: Path, on_error: Callable[[OSError], None], on_close: Callable[[], None]) -> None:
        self.target = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
        try:
            self.compressed = gzip.GzipFile(fileobj=self.target, mode="wb", mtime=0, filename="")
        except OSError:
            self.target.close()
            raise
        self.on_error = on_error
        self.on_close = on_close
        self.failed = False
        self.closed = False
        self.flushed_at = time.monotonic()
        self.timer: asyncio.TimerHandle | None = None

    def write(self, chunk: bytes) -> None:
        if self.failed or self.closed:
            return
        try:
            self.compressed.write(chunk)
        except OSError as exc:
            self.failed = True
            self.on_error(exc)
            return
        # Keep an unfinished stream inspectable while the provider stalls,
        # without a compression sync flush per network chunk.
        delay = self.flushed_at + FLUSH_INTERVAL - time.monotonic()
        if delay <= 0:
            self.flush()
        elif self.timer is None:
            try:
                self.timer = asyncio.get_running_loop().call_later(delay, self.flush)
            except RuntimeError:
                self.flush()

    def flush(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        if self.failed or self.closed:
            return
        try:
            self.compressed.flush()
            self.target.flush()
        except OSError as exc:
            self.failed = True
            self.on_error(exc)
        self.flushed_at = time.monotonic()

    def close(self) -> None:
        if self.closed:
            return
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        self.closed = True
        try:
            self.compressed.close()
            self.target.flush()
            os.fsync(self.target.fileno())
        except OSError as exc:
            self.on_error(exc)
        finally:
            try:
                self.target.close()
            except OSError as exc:
                self.on_error(exc)
            self.on_close()


class CapturedResponseStream(httpx.AsyncByteStream):
    def __init__(self, source: httpx.AsyncByteStream, recorder: ResponseRecorder) -> None:
        self.source = source
        self.recorder = recorder

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.source:
            self.recorder.write(chunk)
            yield chunk

    async def aclose(self) -> None:
        try:
            await self.source.aclose()
        finally:
            self.recorder.close()
