"""Upload pacing preserves request bytes, retries, and cancellation."""

import asyncio
import json

import httpx
import pytest

from slipagent.api import RetryPolicy
from slipagent.google import GoogleClient
from slipagent.openrouter import OpenRouterClient
from slipagent.types import Message
from slipagent.upload import PacedRequestBody, UploadPacer


async def test_large_body_and_replay_are_paced_without_changing_bytes(monkeypatch):
    waits = []
    async def sleep(seconds):
        waits.append(seconds)
    monkeypatch.setattr("slipagent.upload.sleep", sleep)
    body = PacedRequestBody({"messages": [{"role": "user", "content": "x" * 900000}]}, UploadPacer(200000))
    assert body.estimate > 250000
    original = b"".join(body.chunks())
    for _ in range(2):
        waits.clear()
        parts = [chunk async for chunk in body]
        assert b"".join(parts) == original
        assert max(map(len, parts)) <= 4096
        assert len(waits) == len(parts)
        assert sum(waits) == pytest.approx(body.estimate * 60 / 200000)
        assert sum(waits) > 75


async def test_cancelling_wait_releases_shared_pacer(monkeypatch):
    entered = asyncio.Event()
    async def sleep(seconds):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr("slipagent.upload.sleep", sleep)
    pacer = UploadPacer(200000)
    waiting = asyncio.create_task(pacer.wait(100))
    await entered.wait()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not pacer.lock.locked()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("retry", [False, True])
async def test_google_transport_uploads_are_paced(stream, retry, monkeypatch):
    waits = []
    async def sleep(seconds):
        waits.append(seconds)
    monkeypatch.setattr("slipagent.upload.sleep", sleep)
    bodies = []
    attempts = []
    def handle(request):
        assert waits
        attempts.append(request.content)
        if retry and len(attempts) == 1:
            return httpx.Response(503, json={"error": {"message": "Busy"}})
        body = json.loads(request.content)
        assert body["messages"][0]["content"] == "Hello"
        if stream:
            chunk = {"choices": [{"delta": {"content": "Done"}, "finish_reason": "stop"}]}
            return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"},
                                                     "finish_reason": "stop"}]})
    async with GoogleClient("dummy", transport=httpx.MockTransport(handle),
                            retry=RetryPolicy(initial_backoff=0, max_backoff=0)) as client:
        for _ in range(2):
            await client.chat(model="gemini-3.6-flash", messages=[Message.user("Hello")],
                              on_request_body=bodies.append, on_delta=(lambda *args: None) if stream else None)
    assert bodies[0].pacer is bodies[1].pacer
    tokens = sum(b.estimate for b in bodies) + (bodies[0].estimate if retry else 0)
    assert sum(waits) == pytest.approx(tokens * 60 / 200000)
    assert len(attempts) == 2 + int(retry)


async def test_concurrent_uploads_share_the_rate(monkeypatch):
    active = 0
    maximum = 0
    waits = []
    async def sleep(seconds):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        waits.append(seconds)
        await asyncio.sleep(0)
        active -= 1
    monkeypatch.setattr("slipagent.upload.sleep", sleep)
    pacer = UploadPacer(200000)
    bodies = [PacedRequestBody({"messages": [{"content": "x" * 20000}]}, pacer) for _ in range(2)]
    async def consume(body):
        return b"".join([chunk async for chunk in body])
    results = await asyncio.gather(*(consume(body) for body in bodies))
    assert maximum == 1
    assert sum(waits) == pytest.approx(sum(b.estimate for b in bodies) * 60 / 200000)
    assert results == [b"".join(body.chunks()) for body in bodies]


async def test_other_providers_keep_unpaced_uploads():
    async with OpenRouterClient("dummy") as client:
        assert not isinstance(client.request_body({"messages": []}), PacedRequestBody)
