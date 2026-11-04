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


@pytest.mark.parametrize('native', [False, True])
async def test_provider_receives_originals_without_unselected_summaries(native, free_compaction):
    from data_text_reader import request_input

    parameters = ['max_tokens', *(['tools'] if native else ['response_format'])]
    reply = {'role': 'assistant', 'content': json.dumps({'response': 'Done', 'tool_calls': []})}
    router = Router({'test/model': parameters}, [reply])
    async with OpenRouterClient('test', transport=httpx.MockTransport(router.handle)) as client:
        agent = Agent(client, ToolRegistry(), 'test/model', context_tokens=20000, max_tokens=1000)
        free_compaction(agent, client)
        answers = [f'EXACT ANSWER {index}' for index in range(8)]
        agent.extend([message for index, answer in enumerate(answers)
                      for message in (Message.user(f'Question {index}'), Message.assistant(answer))])
        for step in agent.history.steps:
            step.summary = 'UNSELECTED SUMMARY ' * 300
        assert await agent.run('Current request') == 'Done'
        records = request_input(router.requests[0]['messages'])['history']
        assert len(router.requests) == 1
        assert [record['agent_response'] for record in records] == answers
        assert all(record['representation'] == 'full' and 'compressed_summary' not in record for record in records)
        assert 'UNSELECTED SUMMARY' not in json.dumps(router.requests[0])
        await agent.registry.aclose()


def test_calibration_averages_recent_measurements_and_metadata_refresh_invalidates_it():
    budget, profile = ContextBudget(), object()
    request = json.dumps({"messages": [{"role": "user", "content": "hello"}]})
    assert budget.select("model", profile) == 1
    budget.observe(request, 1000)
    measured = budget.scale
    assert measured > 1
    budget.observe(request, 1)
    assert budget.scale == pytest.approx(measured * 1001 / 2000)
    assert budget.select("model", profile) == budget.scale
    assert budget.select("model", object()) == 1
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


async def test_provider_overflow_is_reported_without_retry_or_reducing_future_history():
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
        with pytest.raises(OpenRouterContextError):
            await agent.run("current request")
        await agent.wait_for_compaction()
        assert len(rejected) == 1
        assert not router.requests
        assert agent.messages[:len(originals)] == originals
        assert len(agent.history.steps) == 10
        assert agent.history.steps[9].request_failure["outcome"] == "context_overflow"
        # A live object may still carry the retired recovery setting after reload.
        agent._budget().__dict__['fraction'] = .1
        assert await agent.run("current request") == "Done"
        assert len(router.requests) == 1
        assert len(json.dumps(router.requests[0])) >= len(json.dumps(rejected[0])) * .95
        assert agent.messages[:len(originals)] == originals


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
        with pytest.raises(OpenRouterContextError):
            await agent.run("keep this exact prompt")
        assert len(rejected) == 1
        assert [m.content for m in agent.messages if m.role == "user"] == ["keep this exact prompt"]
        assert agent.history.steps[-1].request_failure["outcome"] == "context_overflow"


async def test_background_notification_metadata_does_not_repeat_command_documents():
    from test_agent import StubClient
    from slipagent.types import content_text

    command = "python - <<'PY'\n" + 'DOCUMENT PAYLOAD ' * 10000
    job = {'job_id': 'job-1', 'log_id': 'log-1', 'state': 'complete', 'streams': {},
           'step_id': 1, 'call_id': 'command-1', 'command': command,
           'artifacts': [{'record_id': 1, 'command': command, 'step_id': 1, 'call_id': 'command-1'}]}
    agent = Agent(StubClient([]), ToolRegistry(), 'test/model', context_tokens=30000)
    agent.extend([Message.user('Continue')])
    agent.registry.context_notes['background_commands'] = {'completed': [job]}
    context = await agent._context_view([], 1)
    system = content_text(context[0].content)
    assert 'job-1' in system and 'command-1' in system and "python - <<'PY'" in system
    assert 'DOCUMENT PAYLOAD' not in system
    assert agent.registry.context_notes['background_commands']['completed'][0]['command'] == command
    assert agent.registry.services['background_snapshot'][0]['command'] == command
    await agent.registry.aclose()
