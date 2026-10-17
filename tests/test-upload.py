"""Upload pacing preserves request bytes, retries, and cancellation."""

import asyncio
import json

import httpx
import pytest

from slipagent.api import APIConfigError, RetryPolicy
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


@pytest.mark.parametrize("tier,model,paced", [("free", "gemini-3.6-flash", True),
    ("free", "gemini-paid-only", False), ("paid", "gemini-3.6-flash", False),
    ("paid", "gemini-paid-only", False), ("free", "gemma-4-31b-it", True),
    ("paid", "gemma-4-31b-it", True), ("paid", "gemma-4-26b-a4b-it", True)])
async def test_google_pacing_only_applies_to_free_requests(tier, model, paced, monkeypatch):
    monkeypatch.setenv("GEMINI_BILLING_TIER", tier)
    waits = []
    async def sleep(seconds):
        waits.append(seconds)
    monkeypatch.setattr("slipagent.upload.sleep", sleep)
    async with GoogleClient("dummy") as client:
        body = client.request_body({"model": model, "messages": [{"role": "user", "content": "Hello"}]})
        assert isinstance(body, PacedRequestBody) == paced
        assert b"".join([chunk async for chunk in body]) == b"".join(body.chunks())
        assert bool(waits) == paced


async def test_invalid_google_billing_tier_is_reported(monkeypatch):
    monkeypatch.setenv("GEMINI_BILLING_TIER", "paidd")
    async with GoogleClient("dummy") as client:
        with pytest.raises(APIConfigError, match="GEMINI_BILLING_TIER"):
            client.request_body({"model": "gemini-3.6-flash", "messages": []})


async def test_pacing_waits_do_not_count_toward_the_first_output_deadline():
    def handle(request):
        chunk = {"choices": [{"delta": {"content": "Done"}, "finish_reason": "stop"}]}
        return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})
    async with GoogleClient("dummy", transport=httpx.MockTransport(handle)) as client:
        client._upload_pacer = UploadPacer(60000)
        loop = asyncio.get_running_loop()
        start = loop.time()
        result = await client.chat(model="gemini-3.6-flash", messages=[Message.user("x " * 2000)],
                                   on_delta=lambda *args: None, single_attempt=True, first_output_timeout=0.3)
        assert result.text == "Done"
        assert loop.time() - start > 0.6


@pytest.mark.parametrize("paced", [False, True])
async def test_skipping_the_first_output_wait_ends_the_request(paced):
    from slipagent.api import APIResponseError

    async def handle(request):
        await request.aread()
        await asyncio.Event().wait()

    deadlines = []
    client_type = GoogleClient if paced else OpenRouterClient
    async with client_type("dummy", transport=httpx.MockTransport(handle)) as client:
        if paced:
            client._upload_pacer = UploadPacer(30000)

        def waiting(deadline):
            deadlines.append(deadline)
            if deadline is not None and len(deadlines) == 1:
                # Paced bodies suspend the deadline; the skip applies when pacing yields.
                asyncio.get_running_loop().call_later(0.05, deadline.skip)

        with pytest.raises(APIResponseError, match="Skipped by the user"):
            await asyncio.wait_for(client.chat(
                model="gemini-3.6-flash" if paced else "test", messages=[Message.user("x " * 2000)],
                on_delta=lambda *args: None, single_attempt=True, first_output_timeout=60,
                on_first_output=waiting), 10)
    assert deadlines[0] is not None and deadlines[0].skipped and deadlines[-1] is None
