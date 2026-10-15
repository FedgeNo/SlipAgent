"""Cancellable pacing for file-backed provider request uploads."""

from asyncio import Lock, sleep
from collections.abc import AsyncIterator
from typing import Any

from .requestbody import RequestBody


class UploadPacer:
    def __init__(self, tokens_per_minute: int) -> None:
        self.tokens_per_minute = tokens_per_minute
        self.lock = Lock()

    async def wait(self, tokens: float) -> None:
        # Concurrent uploads share the allowance; idle time never builds a burst.
        async with self.lock:
            await sleep(tokens * 60 / self.tokens_per_minute)


class PacedRequestBody(RequestBody):
    def __init__(self, payload: dict[str, Any], pacer: UploadPacer) -> None:
        super().__init__(payload)
        self.pacer = pacer

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks():
            for offset in range(0, len(chunk), 4096):
                part = chunk[offset:offset + 4096]
                await self.pacer.wait(self.estimate * len(part) / self.size)
                yield part
