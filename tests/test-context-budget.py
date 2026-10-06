"""Measured input costs and provider overflow recovery preserve original history."""

import json
from importlib import import_module

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.budget import ContextBudget
from slipagent.openrouter import OpenRouterClient, OpenRouterContextError, _build_error, _StreamCompletion
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message

Router = import_module("test-native-tools").Router


def test_calibration_averages_recent_measurements_and_metadata_refresh_invalidates_it():
    budget, profile = ContextBudget(), object()
    request = json.dumps({"messages": [{"role": "user", "content": "hello"}]})
    assert budget.select("model", profile) == 1
    budget.observe(request, 1000)
    measured = budget.scale
    budget.fraction = .65
    assert measured > 1
    budget.observe(request, 1)
    assert budget.scale == pytest.approx(measured * 1001 / 2000)
    assert budget.select("model", profile) == budget.scale
    assert budget.select("model", object()) == 1
    assert budget.fraction == 1
    assert list(budget.measurements) == []
    budget.observe(request, 1000)
    assert budget.select("other", profile) == 1


def test_old_measurements_age_out_and_live_instances_migrate():
    budget = ContextBudget()
    request = json.dumps({"messages": [{"role": "user", "content": "hello"}]})
    budget.observe(request, 1000)
    high = budget.scale
    for _ in range(5):
        budget.observe(request, 100)
    assert budget.scale == pytest.approx(high / 10)
    assert len(budget.measurements) == 5
    del budget.measurements
    budget.observe(request, 200)
    assert len(budget.measurements) == 1
    assert budget.scale == pytest.approx(high / 5)


def test_missing_usage_does_not_replace_calibration():
    budget = ContextBudget()
    request = json.dumps({"messages": [{"role": "user", "content": "hello"}]})
    budget.observe(request, 100)
    before = budget.scale
    budget.observe(request, 0)
    budget.observe("", 100)
    assert budget.scale == before and len(budget.measurements) == 1


@pytest.mark.parametrize("status, message, overflow", [
    (400, "Maximum context length is 8192 tokens, requested 12000", True),
    (400, "context_length_exceeded", True),
    (413, "prompt_too_long", True),
    (400, "max_tokens exceeds the maximum allowed output tokens", False),
    (404, "No endpoints can handle the requested parameters", False),
    (413, "Request body too large", False),
])
def test_overflow_classification_is_specific(status, message, overflow):
    error = _build_error(httpx.Response(status, json={"error": {"message": message}}))
    assert isinstance(error, OpenRouterContextError) is overflow


def test_streamed_context_error_uses_the_same_recovery_path():
    with pytest.raises(OpenRouterContextError):
        _StreamCompletion(lambda *_: None).add(json.dumps({"error": {
            "code": 400, "message": "Maximum context length exceeded",
        }}))


async def test_provider_overflow_retries_smaller_history_without_losing_originals():
    router = Router({"test/model": ["tools"]}, [{"role": "assistant", "content": "Done"}])
    rejected = []
    def handle(request):
        if request.method == "POST" and not rejected:
            rejected.append(json.loads(request.content))
            return httpx.Response(400, json={"error": {"message": "context_length_exceeded"}})
        return router.handle(request)
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, ToolRegistry(), "test/model")
        originals = [m for i in range(9) for m in (Message.user(f"old {i}"), Message.assistant(str(i) * 240000))]
        agent.extend(originals)
        assert await agent.run("current request") == "Done"
        await agent.wait_for_compaction()
        assert len(json.dumps(router.requests[0])) < len(json.dumps(rejected[0]))
        assert "current request" in json.dumps(router.requests[0])
        assert agent.messages[:len(originals)] == originals
        assert len(agent.history.steps) == 10
        assert agent._budget().fraction == .65


async def test_unshrinkable_overflow_is_not_sent_twice():
    router = Router({"test/model": ["tools"]}, [])
    rejected = []
    def handle(request):
        if request.method == "POST":
            rejected.append(request)
            return httpx.Response(400, json={"error": {"message": "context_length_exceeded"}})
        return router.handle(request)
    async with OpenRouterClient("test", transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, ToolRegistry(), "test/model")
        with pytest.raises(Exception, match="without removable history"):
            await agent.run("keep this exact prompt")
        assert len(rejected) == 1
        assert agent.messages[-1].content == "keep this exact prompt"
