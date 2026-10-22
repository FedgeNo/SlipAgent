"""The status line names what a running request is doing."""

import asyncio
import json

import httpx

from slipagent.openrouter import OpenRouterClient
from slipagent.statusline import StatusLine
from slipagent.types import Message
from slipagent.upload import PacedRequestBody, UploadPacer


def stream(*deltas):
    events = [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
    events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    text = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


async def test_phases_follow_a_request_and_clear_at_the_end():
    phases = []
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(
            lambda request: stream({"reasoning": "hmm"}, {"content": "Done"}))) as client:
        await client.chat(model="test", messages=[Message.user("work")], on_delta=lambda *args: None,
                          single_attempt=True, first_output_timeout=60, on_phase=phases.append)
    assert phases == ["Sending request to openrouter", "Waiting for first output", "Thinking", "Receiving reply", ""]


async def test_tool_call_output_clears_the_waiting_phase():
    phases = []
    call = {"tool_calls": [{"index": 0, "id": "call", "function": {"name": "probe", "arguments": "{}"}}]}
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(lambda request: stream(call))) as client:
        await client.chat(model="test", messages=[Message.user("work")], on_delta=lambda *args: None,
                          single_attempt=True, first_output_timeout=60, on_phase=phases.append)
    assert phases[:2] == ["Sending request to openrouter", "Waiting for first output"] and phases[-2:] == ["", ""]


async def test_pacing_reports_remaining_time_and_queueing(monkeypatch):
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("slipagent.upload.sleep", sleep)
    pacer = UploadPacer(200000)
    body = PacedRequestBody({"messages": [{"content": "x" * 20000}]}, pacer)
    messages = []
    body.on_wait = messages.append
    [part async for part in body]
    assert messages and all(message.startswith("Pacing upload") for message in messages)
    left = [int(message.split("about ")[1].split("s")[0]) for message in messages]
    assert left == sorted(left, reverse=True)

    queued = []
    async with pacer.lock:
        task = asyncio.create_task(pacer.wait(1, lambda: queued.append("queued")))
        await asyncio.sleep(0)
    await task
    assert queued == ["queued"]


def test_status_line_shows_the_phase_with_elapsed_time():
    status = StatusLine()
    assert status.activity(held=False, working=True, stopping=False, concise=False, frame="*")[1] == "* Working (esc to interrupt)"
    status.set_phase("Waiting for first output")
    text = status.activity(held=False, working=True, stopping=False, concise=False, frame="*")[1]
    assert text.startswith("* Working: Waiting for first output (0s, esc to interrupt)")
    assert status.activity(held=False, working=True, stopping=False, concise=True, frame="*")[1].startswith("* Waiting for first output (")
    status.retry = "Retrying in 5s"
    assert status.activity(held=False, working=True, stopping=False, concise=False, frame="*")[1] == "Retrying in 5s"
    status.set_phase("")
    status.retry = ""
    assert status.activity(held=False, working=True, stopping=False, concise=False, frame="*")[1] == "* Working (esc to interrupt)"
