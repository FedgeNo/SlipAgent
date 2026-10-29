"""Cancellable pacing for file-backed provider request uploads."""

from asyncio import Lock, sleep
from collections.abc import AsyncIterator, Callable
from typing import Any

from .requestbody import RequestBody


class UploadPacer:
    def __init__(self, tokens_per_minute: int) -> None:
        self.tokens_per_minute = tokens_per_minute
        self.lock = Lock()

    async def wait(self, tokens: float, on_queue: Callable[[], None] | None = None) -> None:
        # Concurrent uploads share the allowance; idle time never builds a burst.
        if on_queue is not None and self.lock.locked():
            on_queue()
        async with self.lock:
            await sleep(tokens * 60 / self.tokens_per_minute)


class PacedRequestBody(RequestBody):
    def __init__(self, payload: dict[str, Any], pacer: UploadPacer) -> None:
        super().__init__(payload)
        self.pacer = pacer
        # Receives a status sentence before each paced part when a caller wants progress.
        self.on_wait: Callable[[str], None] | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        left = self.estimate * 60 / self.pacer.tokens_per_minute
        for chunk in self.chunks():
            for offset in range(0, len(chunk), 4096):
                part = chunk[offset:offset + 4096]
                tokens = self.estimate * len(part) / self.size
                on_wait = self.on_wait
                if on_wait is None:
                    await self.pacer.wait(tokens)
                else:
                    on_wait(f"Pacing upload to the provider's token limit, about {max(1, round(left))}s left")
                    await self.pacer.wait(tokens, lambda: on_wait("Waiting for another upload to finish"))
                left -= tokens * 60 / self.pacer.tokens_per_minute
                yield part
