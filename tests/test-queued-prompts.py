"""Queue receipts follow requests, and retract only their own display notices."""

import pytest

from slipagent.context import ContextError
from test_agent import build_agent, completion, call, context_body


async def test_waiting_prompts_are_sent_together_before_new_step():
    agent, client = build_agent([
        completion(tool_calls=[call(value="first")]), completion("done"),
    ])
    events = []
    async def boundary():
        if len(client.calls) == 1:
            agent.enqueue("first correction")
            agent.enqueue("second correction")
    agent.on_boundary = boundary
    agent.on_event = events.append
    await agent.run("start")
    users = [context_body(m.content) for m in client.calls[1]["messages"] if m.role == "user"]
    assert users == ["start", "first correction", "second correction"]
    assert [e.text for e in events if e.kind == "user_message_sent"] == users[1:]


async def test_receipts_exclude_prompts_arriving_during_the_request():
    agent, client = build_agent([completion("done"), completion("followup")])
    events = []
    agent.on_event = events.append
    original = client.chat
    async def chat(**kwargs):
        if not client.calls:
            assert [e.text for e in events if e.kind == "user_message_sent"] == ["same", "same"]
            agent.enqueue("same")
        return await original(**kwargs)
    client.chat = chat
    agent.enqueue("same")
    agent.enqueue("same")
    await agent.run("")
    assert agent.pending == ["same"]
    assert len([e for e in events if e.kind == "user_message_sent"]) == 2
    await agent.run("")
    assert len([e for e in events if e.kind == "user_message_sent"]) == 3


async def test_failed_context_keeps_notice_until_a_request_is_sent(monkeypatch):
    agent, client = build_agent([completion("done")])
    events = []
    agent.on_event = events.append
    agent.enqueue("queued instruction")
    original = type(agent)._context_view
    async def fail(*args, **kwargs):
        raise ContextError("input too large")
    monkeypatch.setattr(type(agent), "_context_view", fail)
    with pytest.raises(ContextError):
        await agent.run("start")
    assert not client.calls
    assert not any(e.kind == "user_message_sent" for e in events)
    monkeypatch.setattr(type(agent), "_context_view", original)
    await agent.run("continue")
    assert [e.text for e in events if e.kind == "user_message_sent"] == ["queued instruction"]


async def test_prompts_arriving_during_context_preparation_join_the_same_request(monkeypatch):
    agent, client = build_agent([completion("done")])
    original = type(agent)._context_view
    async def prepare(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        if not any(m.content == "late correction" for m in self.messages):
            self.enqueue("late correction")
            self.enqueue("another correction")
        return result
    monkeypatch.setattr(type(agent), "_context_view", prepare)
    await agent.run("start")
    users = [context_body(m.content) for m in client.calls[0]["messages"] if m.role == "user"]
    assert users == ["start", "late correction", "another correction"]
    assert len(client.calls) == 1


async def test_stop_after_tools_does_not_acknowledge_unsent_prompts():
    agent, client = build_agent([completion(tool_calls=[call(value="first")]), completion("done")])
    events = []
    def event_handler(event):
        events.append(event)
        if event.kind == "tool_start":
            agent.enqueue("queued correction")
            agent.request_stop()
    agent.on_event = event_handler
    await agent.run("start")
    assert len(client.calls) == 1
    assert not any(e.kind == "user_message_sent" for e in events)
    await agent.run("continue")
    assert [e.text for e in events if e.kind == "user_message_sent"] == ["queued correction"]


async def test_unsent_receipts_survive_session_restore(workspace, tmp_path, monkeypatch):
    from importlib import import_module
    attach = import_module("test-sessions").attach
    agent, _ = build_agent([])
    journal = attach(agent, workspace, tmp_path)
    agent.enqueue("unsent correction")
    async def fail(*args, **kwargs):
        raise ContextError("cannot prepare")
    with monkeypatch.context() as patch:
        patch.setattr(type(agent), "_context_view", fail)
        with pytest.raises(ContextError):
            await agent.run("start")
    data = journal.load(journal.session_id)
    restored, client = build_agent([completion("done")])
    journal.restore(restored, data)
    events = []
    restored.on_event = events.append
    assert list(restored.queued_messages.values()) == ["unsent correction"]
    await restored.run("continue")
    assert [e.text for e in events if e.kind == "user_message_sent"] == ["unsent correction"]
    users = [context_body(m.content) for m in client.calls[0]["messages"] if m.role == "user"]
    assert users == ["unsent correction", "start", "continue"]


async def test_openrouter_receipt_matches_the_submitted_payload():
    import json
    import httpx
    from slipagent.agent import Agent
    from slipagent.capabilities import ModelCapabilities
    from slipagent.openrouter import OpenRouterClient
    from slipagent.tools.base import ToolRegistry
    from test_agent import summary_response
    events, requests = [], []
    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        assert [e.text for e in events if e.kind == "user_message_sent"] == ["first", "second"]
        agent.enqueue("next request")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(
            'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
            'data: [DONE]\n\n'
        ))
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        client._capabilities = {"test": ModelCapabilities({}, [{
            "tag": "provider", "supported_parameters": ["tools"], "context_length": 1_000_000,
        }])}
        agent = Agent(client, ToolRegistry(), "test", on_event=events.append)
        agent.enqueue("first")
        agent.enqueue("second")
        await agent.run("")
        await agent.wait_for_compaction()
        await agent.registry.aclose()
    users = [context_body(m["content"]) for m in requests[0]["messages"] if m["role"] == "user"]
    assert users == ["first", "second"]
    assert agent.pending == ["next request"]
    assert len([e for e in events if e.kind == "user_message_sent"]) == 2
