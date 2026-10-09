"""Three silent retries followed by visible, bounded, interruptible recovery."""

import asyncio
import io
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from slipagent.api import APIResponseError, RetryPolicy
from slipagent.cli import Renderer, Style
from slipagent.openrouter import OpenRouterClient
from slipagent.types import Message
from slipagent.agent import Agent, STOP_NOTICE
from slipagent.capabilities import RequestProfile
from slipagent.tools.base import ToolRegistry


@pytest.mark.parametrize("stream, json_error", [(False, False), (False, True), (True, False), (True, True)])
async def test_three_silent_retries_then_countdown_and_eventual_success(monkeypatch, stream, json_error):
    requests, notices, sleeps = [], [], []
    def handler(request):
        requests.append(request)
        if len(requests) < 6:
            return httpx.Response(200 if json_error else 503, json={"error": {"message": "overloaded", "code": 503}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]})
    async def sleep(delay):
        sleeps.append(delay)
    monkeypatch.setattr("slipagent.api.asyncio.sleep", sleep)
    monkeypatch.setattr("slipagent.api.random.uniform", lambda low, high: high)
    def notice(error, seconds):
        notices.append((len(requests), error, seconds))
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handler), on_retry=notice) as client:
        reply = await client.chat(model="test", messages=[Message.user("hello")],
                                  on_delta=(lambda *args: None) if stream else None)
    assert reply.text == "done" and len(requests) == 6
    assert sleeps[:3] == [1, 2, 4]
    assert notices[0][0] == 4 and notices[0][2] == 8
    assert [seconds for attempt, error, seconds in notices if attempt == 4] == list(range(8, -1, -1))
    assert notices[-1] == (5, "", 0)


async def test_default_recovery_gives_up_before_interval_exceeds_one_minute(monkeypatch):
    requests, waits = [], []
    def handler(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})
    async def sleep(delay):
        waits.append(delay)
    monkeypatch.setattr("slipagent.api.asyncio.sleep", sleep)
    monkeypatch.setattr("slipagent.api.random.uniform", lambda low, high: high)
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(APIResponseError, match="overloaded"):
            await client.chat(model="test", messages=[Message.user("hello")])
    assert len(requests) == 7
    assert waits == [1, 2, 4, 8, 16, 32]
    assert all(delay <= 60 for delay in waits)


async def test_stop_interrupts_visible_countdown_and_clears_it():
    notices = []
    stop = asyncio.Event()
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        def notice(error, seconds):
            notices.append((error, seconds))
            stop.set()
        assert not await asyncio.wait_for(client.wait_retry(3, APIResponseError("overloaded", 503), stop=stop, on_retry=notice), .5)
    assert notices[0][1] > 0 and notices[-1] == ("", 0)


@pytest.mark.parametrize("kind", ["seconds", "date"])
async def test_retry_after_cannot_schedule_more_than_one_minute(kind):
    retry_after = "61" if kind == "seconds" else format_datetime(datetime.now(timezone.utc) + timedelta(minutes=5))
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        assert client.retry_delay(0, retry_after) is None
        assert client.retry_delay(0, "60") == 60


async def test_cancel_clears_visible_countdown():
    notices = []
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        task = asyncio.create_task(client.wait_retry(3, APIResponseError("overloaded", 503),
                                                    on_retry=lambda *args: notices.append(args)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert notices[-1] == ("", 0)


def test_renderer_displays_error_once_and_updates_terminal_countdown():
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, False)
    renderer.emit = lambda text, **kwargs: sink.write(text)
    updates = []
    renderer.terminal = type("Terminal", (), {"set_retry_status": lambda self, text: updates.append(text)})()
    for seconds in (8, 7, 6):
        renderer.retry_countdown("NVIDIA overloaded", seconds)
    renderer.retry_countdown("", 0)
    assert sink.getvalue().count("NVIDIA overloaded") == 1
    assert updates == [f"Retrying in {seconds}s (Esc to interrupt)" for seconds in (8, 7, 6)] + [""]


async def test_agent_retries_silently_then_stop_interrupts_countdown(monkeypatch):
    requests, events = [], []
    def handler(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handler)) as client:
        client.cache_capabilities("test", RequestProfile(parameters={"tools"}, context_length=1_000_000))
        monkeypatch.setattr(client, "_backoff", lambda attempt, retry_after=None: 0 if attempt < 3 else 8)
        agent = Agent(client, ToolRegistry(), "test")
        def on_event(event):
            if event.kind == "retry_wait":
                events.append((len(requests), event.text, event.step))
                agent.request_stop()
        agent.on_event = on_event
        assert await asyncio.wait_for(agent.run("work"), 1) == STOP_NOTICE
        await agent.registry.aclose()
    assert len(requests) == 4
    assert events[0][0] == 4 and events[0][2] == 8
    assert events[-1][1:] == ("", 0)
