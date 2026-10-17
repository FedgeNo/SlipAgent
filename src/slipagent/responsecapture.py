"""Incremental private capture of HTTP response bodies before decoding."""

from __future__ import annotations

import gzip
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx


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

    def write(self, chunk: bytes) -> None:
        if self.failed or self.closed:
            return
        try:
            self.compressed.write(chunk)
            # Keep an unfinished stream inspectable while the provider stalls.
            self.compressed.flush()
            self.target.flush()
        except OSError as exc:
            self.failed = True
            self.on_error(exc)

    def close(self) -> None:
        if self.closed:
            return
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
