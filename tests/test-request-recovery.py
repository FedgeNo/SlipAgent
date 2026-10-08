
from slipagent.types import content_text
import json

import httpx
import pytest

from slipagent.agent import Agent, STOP_NOTICE
from slipagent.capabilities import ModelCapabilities
from slipagent.openrouter import OpenRouterClient, OpenRouterError, RetryPolicy
from slipagent.tools.base import ToolRegistry
from test_agent import RecordingTool, summary_response


def packet(delta, finish=None):
    return ("data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()


class Stream(httpx.AsyncByteStream):
    def __init__(self, broken=False, calls=False):
        self.broken, self.calls = broken, calls

    async def __aiter__(self):
        if self.broken:
            yield packet({"reasoning": "abandoned thought"})
            yield packet({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "record", "arguments": '{"value":'}}]})
            raise httpx.ReadError("connection dropped")
        if self.calls:
            yield packet({"content": "Recording the result."})
            yield packet({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "record", "arguments": '{"value":"once"}'}}]}, "tool_calls")
        else:
            yield packet({"content": "done"}, "stop")
        yield b"data: [DONE]\n\n"


def configured_client(handle, retry):
    client = OpenRouterClient("test", transport=httpx.MockTransport(handle), retry=retry)
    client._capabilities = {"test": ModelCapabilities({}, [{
        "tag": "provider", "supported_parameters": ["tools"], "context_length": 1_000_000,
    }])}
    return client


async def test_partial_tool_stream_retries_once_without_replaying_thoughts_or_tools():
    requests, events = [], []
    tool = RecordingTool()

    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=Stream(broken=len(requests) == 1, calls=len(requests) == 2))

    async with configured_client(handle, RetryPolicy(2, 0, 0)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test", on_event=events.append)
        assert await agent.run("work") == "done"
        await agent.wait_for_compaction()
        await agent.registry.aclose()
    assert len(requests) == 3
    assert requests[0] == requests[1]
    assert tool.seen == [{"value": "once"}]
    assert not any(event.kind == "retry" for event in events)
    assert all("abandoned thought" not in json.dumps(request) for request in requests)
    assert all("abandoned thought" not in (content_text(message.content)) for message in agent.messages)


async def test_stop_during_backoff_prevents_another_request(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "busy"}})

    async with configured_client(handle, RetryPolicy(3, 30, 30)) as client:
        agent = Agent(client, ToolRegistry(), "test")
        def backoff(attempt, retry_after=None):
            agent.request_stop()
            return 30
        monkeypatch.setattr(client, "_backoff", backoff)
        assert await agent.run("work") == STOP_NOTICE
        await agent.registry.aclose()
    assert len(requests) == 1


async def test_transient_errors_remain_silent_until_three_retries_are_exhausted():
    requests, events = [], []
    def handle(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "busy"}})
    async with configured_client(handle, RetryPolicy(3, 0, 0)) as client:
        agent = Agent(client, ToolRegistry(), "test", on_event=events.append)
        with pytest.raises(OpenRouterError, match="busy"):
            await agent.run("work")
        await agent.registry.aclose()
    assert len(requests) == 4
    assert not any(event.kind in {"retry", "warning"} for event in events)


@pytest.mark.parametrize("kind", ["reasoning", "content"])
async def test_repetitive_stream_closes_without_retrying_or_executing_partial_calls(kind):
    requests, seen, closed = [], [], []
    phrase = "I need to inspect the same information once more to decide whether this request is complete. "

    class RepetitiveStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield packet({"tool_calls": [{"index": 0, "id": "one", "function": {
                "name": "record", "arguments": '{"value":"must not execute"}',
            }}]})
            for _ in range(50):
                seen.append(True)
                yield packet({kind: phrase})
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            closed.append(True)

    def handle(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=RepetitiveStream())

    tool = RecordingTool()
    async with configured_client(handle, RetryPolicy(3, 0, 0)) as client:
        agent = Agent(client, ToolRegistry([tool]), "test")
        assert "sustained repetition" in await agent.run("inspect")
        assert agent.stopped
        await agent.registry.aclose()
    assert len(requests) == 1 and len(seen) < 50 and closed
    assert not tool.seen
    assert len(agent.messages) == 1


@pytest.mark.parametrize("status, retries, steps, expected", [(503, 2, 20, 3), (503, 8, 2, 2), (402, 2, 20, 1)])
async def test_request_limits_include_every_transport_attempt(status, retries, steps, expected):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "unavailable"}})

    async with configured_client(handle, RetryPolicy(retries, 0, 0)) as client:
        agent = Agent(client, ToolRegistry(), "test", max_steps=steps)
        try:
            await agent.run("work")
        except OpenRouterError:
            pass
        finally:
            await agent.registry.aclose()
    assert len(requests) == expected
