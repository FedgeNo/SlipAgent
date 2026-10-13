"""Failed model calls are numbered, durable turns rather than invisible retries."""

import asyncio
import json

import httpx
import pytest

from data_text_reader import current_input, request_input
from slipagent.agent import Agent
from slipagent.api import APITransportError
from slipagent.capabilities import ModelCapabilities
from slipagent.context import ContextError, message_tokens
from slipagent.openrouter import OpenRouterClient, RetryPolicy
from slipagent.requestfailures import retry_state
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, content_text, decode_json_content
from test_agent import RecordingTool, summary_response


def packet(delta=None, finish=None):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}) + "\n\n"


def response(*events):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          text="".join(events) + "data: [DONE]\n\n")


def client_for(handle, retries=3):
    client = OpenRouterClient("test", transport=httpx.MockTransport(handle), retry=RetryPolicy(retries, 0, 0))
    client._capabilities["test"] = ModelCapabilities({}, [{
        "tag": "test", "context_length": 1_000_000, "supported_parameters": ["tools"],
    }])
    return client


async def test_provider_error_turn_reaches_model_before_it_decides_whether_to_retry(tmp_path):
    requests = []
    tool = RecordingTool()

    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        if len(requests) == 1:
            return response(packet({"content": "Writing now."}), packet({"tool_calls": [{
                "index": 0, "id": "write", "function": {"name": "record", "arguments": '{"value":'},
            }]}), packet(finish="error"))
        payload = request_input(body["messages"])
        failed = payload["history"][-1]
        assert failed["step_id"] == 1 and failed["agent_response"] is None
        assert failed["tool_calls"] == []
        failure = failed["request_failure"]
        assert failure["outcome"] == "request_error" and failure["tools_executed"] is False
        assert "record" in failure["response_excerpt"]
        retry = current_input(body["messages"])["request_retry"]
        assert retry == {"consecutive_failures": 1, "total_failed_attempts": 1,
                         "failed_step_ids": [1], "last_failed_step": 1}
        assert "Respect instructions to stop" in body["messages"][0]["content"]
        assert failed["user_prompt"] == ["Write the file. If it fails, do not retry."]
        return response(packet({"content": "The attempt failed, so I stopped without retrying the write."}, "stop"))

    async with client_for(handle) as client:
        agent = Agent(client, ToolRegistry([tool]), "test")
        journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
        agent.registry.resources.journal = journal
        assert "stopped" in await agent.run("Write the file. If it fails, do not retry.")
        await agent.wait_for_compaction()
        assert len(agent.history.steps) == 2 and not tool.seen
        loaded = journal.load(journal.session_id)
        assert loaded["history"].steps[0].request_failure == agent.history.steps[0].request_failure
        assert loaded["history"].steps[0].record()["request_failure"]["consecutive_failures"] == 1
        original = decode_json_content((await agent.registry.invoke("recall_history", {"step_id": 1})).content)
        assert original["content"]["request_failure"]["tools_executed"] is False
        assert original["content"]["response"] is None
        await agent.registry.aclose()
    assert len(requests) == 2


async def test_consecutive_and_session_counts_survive_multiple_retries_and_reset_after_success():
    requests, events = [], []

    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        if len(requests) in (1, 2, 4):
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return response(packet({"content": "Done."}, "stop"))

    async with client_for(handle) as client:
        agent = Agent(client, ToolRegistry(), "test", on_event=events.append)
        assert await agent.run("First") == "Done."
        assert await agent.run("Second") == "Done."
        await agent.wait_for_compaction()
        assert [step.id for step in agent.history.steps] == list(range(1, 6))
        failures = [step.request_failure for step in agent.history.steps if hasattr(step, "request_failure")]
        assert [f["consecutive_failures"] for f in failures] == [1, 2, 1]
        assert [f["total_failed_attempts"] for f in failures] == [1, 2, 3]
        assert [current_input(body["messages"])["step_id"] for body in requests] == [1, 2, 3, 4, 5]
        assert current_input(requests[2]["messages"])["request_retry"]["failed_step_ids"] == [1, 2]
        assert current_input(requests[4]["messages"])["request_retry"]["total_failed_attempts"] == 3
        assert [event.step for event in events if event.kind == "step_end"] == [1, 2, 3, 4, 5]
        agent.reset()
        assert not agent.history.steps and retry_state(agent.history.steps) is None
        await agent.registry.aclose()


async def test_last_failed_turn_is_durable_when_retries_stop_and_resume_sees_its_error(tmp_path):
    async with client_for(lambda _: response(packet({"content": "Partial plan"}), packet(finish="error")), retries=0) as client:
        agent = Agent(client, ToolRegistry(), "test")
        journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
        agent.registry.resources.journal = journal
        with pytest.raises(APITransportError):
            await agent.run("Try once")
        data = journal.load(journal.session_id)
        restored = Agent(client, ToolRegistry(), "test")
        journal.restore(restored, data, fork=True)
        view = await restored._context_view(restored.registry.specs(), 2)
        assert request_input(view)["history"][-1]["request_failure"]["tools_executed"] is False
        assert current_input(view)["request_retry"]["consecutive_failures"] == 1
        assert journal.load(journal.session_id)["history"].steps[0].request_failure["outcome"] == "request_error"
        await restored.registry.aclose()
        await agent.registry.aclose()


async def test_rejected_tool_arguments_become_failed_turns_without_executing_them():
    tool = RecordingTool()
    def handle(request):
        return response(packet({"tool_calls": [{"index": 0, "id": "bad", "function": {
            "name": "record", "arguments": '{"value":',
        }}]}, "tool_calls"))
    async with client_for(handle) as client:
        agent = Agent(client, ToolRegistry([tool]), "test")
        with pytest.raises(ContextError, match="3 consecutive"):
            await agent.run("Record it")
        assert len(agent.history.steps) == 3 and not tool.seen
        assert all(step.request_failure["outcome"] == "rejected" for step in agent.history.steps)
        assert all(not step.tool_calls for step in agent.history.steps)
        await agent.registry.aclose()


async def test_cancelled_response_is_recorded_without_creating_user_messages():
    started = asyncio.Event()
    class Paused(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield packet({"content": "Partial"}).encode()
            started.set()
            await asyncio.Event().wait()
    async with client_for(lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Paused())) as client:
        agent = Agent(client, ToolRegistry(), "test")
        work = asyncio.create_task(agent.run("Original request"))
        await started.wait()
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert len(agent.history.steps) == 1
        assert agent.history.steps[0].request_failure["outcome"] == "cancelled"
        assert [m.content for m in agent.messages if m.role == "user"] == ["Original request"]
        await agent.registry.aclose()


@pytest.mark.parametrize("result_size", [10, 100_000])
async def test_resume_after_failure_keeps_unacknowledged_tool_results_and_failure(tmp_path, result_size):
    requests = []
    result = "EXACT TOOL RESULT " * result_size
    tool = RecordingTool(result)

    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        if len(requests) == 1:
            return response(packet({"tool_calls": [{"index": 0, "id": "read", "function": {
                "name": "record", "arguments": '{"value":"A"}',
            }}]}, "tool_calls"))
        return response(packet(finish="error"))

    async with client_for(handle, retries=0) as client:
        agent = Agent(client, ToolRegistry([tool]), "test")
        journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
        agent.registry.resources.journal = journal
        with pytest.raises(APITransportError):
            await agent.run("Read the result")
        await agent.wait_for_compaction()
        data = journal.load(journal.session_id)
        restored = Agent(client, ToolRegistry(), "test")
        journal.restore(restored, data, fork=True)
        view = await restored.history.view(restored.messages, [], keep_steps=1,
                                           context_length=20_000, max_output=1000)
        records = request_input(view)["history"]
        assert [record["step_id"] for record in records] == [1, 2]
        assert records[-1]["request_failure"]["tools_executed"] is False
        if result_size == 10:
            assert records[0]["tool_calls"][0]["result"] == result
        else:
            assert records[0]["representation"] == "excerpt"
        assert message_tokens(view) <= int(20_000 * .85) - 1000
        assert restored.history.steps[0].tool_results[0].content["content"] == result
        assert tool.seen == [{"value": "A"}]
        await restored.registry.aclose()
        await agent.registry.aclose()
