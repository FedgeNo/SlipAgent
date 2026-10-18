"""Tuning changes actual requests and preserves evidence across turns."""

from slipagent.types import content_text
import json

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.api import APIClient
from slipagent.capabilities import RequestProfile
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolCall
from test_agent import StubClient, RecordingTool, completion, context_records


@pytest.mark.parametrize('parameters,expected', [
    ({'temperature', 'top_p', 'top_k', 'min_p'}, {'top_p': .95, 'top_k': 20, 'min_p': 0}),
    ({'temperature'}, {}),
    (set(), {}),
])
async def test_sampling_filters_capabilities_and_preserves_overrides(parameters, expected):
    bodies = []
    def transport(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'Done'}}]})
    async with APIClient('dummy', base_url='https://stub.invalid', transport=httpx.MockTransport(transport)) as client:
        profile = RequestProfile(parameters=parameters, context_length=32000)
        await client.chat(model='qwen/qwen3-4b', messages=[Message.user('hello')], request_profile=profile)
        assert {k:v for k,v in bodies[0].items() if k in {'temperature','top_p','top_k','min_p'}} == expected
        await client.chat(model='qwen/qwen3-4b', messages=[Message.user('hello')], request_profile=profile, temperature=.8)
        assert bodies[1].get('temperature') == (.8 if 'temperature' in parameters else None)


@pytest.mark.parametrize('model', [
    'dots-studio/dots-3-note-preview:free', 'nvidia/nemotron-3.5-lightning',
    'nvidia/nemotron-3.5-lightning-30b-a3b', 'other/model',
])
async def test_models_omit_temperature_until_explicitly_set(model):
    bodies = []

    def transport(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'Done'}}]})

    async with APIClient('dummy', base_url='https://stub.invalid', transport=httpx.MockTransport(transport)) as client:
        profile = RequestProfile(parameters={'temperature', 'top_p'}, context_length=32000)
        await client.chat(model=model, messages=[Message.user('hello')], request_profile=profile)
        await client.chat(model=model, messages=[Message.user('hello')], request_profile=profile, temperature=0)
    assert 'temperature' not in bodies[0]
    assert bodies[1]['temperature'] == 0


async def test_plan_is_supplied_on_next_turn_restored_and_cleared():
    plan = {'objective': 'Fix parsing', 'stages': [{'task': 'Locate parser', 'status': 'in_progress', 'evidence': ''}]}
    client = StubClient([completion(tool_calls=[ToolCall('p', 'update_plan', plan)]), completion('Done')])
    registry = ToolRegistry()
    agent = Agent(client, registry, 'stub', planning=True)
    try:
        await agent.run('Fix parsing')
        assert 'objective: text (11 characters):\n    Fix parsing' in content_text(client.calls[1]['messages'][0].content)
        assert 'Last updated: step 1' in content_text(client.calls[1]['messages'][0].content)
        restored = Agent(StubClient([]), ToolRegistry(), 'stub', planning=True)
        try:
            restored.extend(agent.messages)
            view = await restored._context_view(restored.registry.specs(), 1)
            assert 'objective: text (11 characters):\n    Fix parsing' in content_text(view[0].content)
            assert 'Last updated: step 1' in content_text(view[0].content)
            restored.set_planning(False)
            view = await restored._context_view(restored.registry.specs(), 1)
            assert '# Working Plan' not in content_text(view[0].content)
            restored.set_planning(True)
            view = await restored._context_view(restored.registry.specs(), 1)
            assert 'objective: text (11 characters):\n    Fix parsing' in content_text(view[0].content)
            restored.reset()
            view = await restored._context_view(restored.registry.specs(), 1)
            assert 'objective: text (11 characters):\n    Fix parsing' not in content_text(view[0].content)
            assert '# Working Plan' not in content_text(view[0].content)
        finally:
            await restored.registry.aclose()
    finally:
        await registry.aclose()


async def test_plan_date_tracks_last_successful_update_not_latest_turn():
    plan = {'objective': 'Fix parsing', 'stages': [{'task': 'Locate parser', 'status': 'pending', 'evidence': ''}]}
    client = StubClient([
        completion('Ready'),
        completion(tool_calls=[ToolCall('p1', 'update_plan', plan)]), completion('Plan saved'),
        completion(tool_calls=[ToolCall('bad', 'update_plan', {**plan, 'stages': []})]), completion('Invalid update'),
        completion(tool_calls=[ToolCall('p2', 'update_plan', plan)]), completion('Plan updated'),
    ])
    agent = Agent(client, ToolRegistry(), 'stub')
    try:
        await agent.run('Start')
        await agent.run('Make a plan')
        await agent.run('Try an invalid update')
        view = await agent._context_view(agent.registry.specs(), 1)
        assert 'Last updated: step 2' in content_text(view[0].content)
        await agent.run('Update the plan')
        view = await agent._context_view(agent.registry.specs(), 1)
        assert 'Last updated: step 6' in content_text(view[0].content)
    finally:
        await agent.registry.aclose()


async def test_allowlist_hides_and_rejects_unexposed_calls():
    tool = RecordingTool()
    client = StubClient([completion(tool_calls=[ToolCall('r', 'record', {'value':'no'})]), completion('Done')])
    registry = ToolRegistry([tool])
    agent = Agent(client, registry, 'stub', exposed_tools=())
    try:
        await agent.run('Inspect only')
        assert [spec.name for spec in client.calls[0]['tools']] == ['recall_history', 'update_plan']
        assert not tool.seen
        assert any(m.role == 'tool' and m.content['status'] == 'error' for m in agent.messages)
    finally:
        await registry.aclose()


@pytest.mark.parametrize('depth', [0, 1, 5, 25])
async def test_reasoning_depth_preserves_archived_originals(depth):
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, 'stub', reasoning_history_steps=depth)
    try:
        for n in range(7):
            reply = Message.assistant('Done')
            reply.reasoning = f'thought-{n}'
            agent.messages.extend([Message.user(f'Question {n}'), reply])
        view = await agent._context_view(registry.specs(), 1)
        records = context_records(view)
        assert sum('reasoning' in record for record in records) == min(depth, 7)
        assert all(step.messages[-1].reasoning for step in agent.history.steps)
    finally:
        await registry.aclose()


async def test_plan_objective_has_leeway_beyond_its_advertised_length():
    from slipagent.tools.planning import UpdatePlanTool
    stages = [{"task": "Inspect", "status": "pending", "evidence": ""}]
    tool = UpdatePlanTool()
    assert not (await tool.run(objective="x" * 1000, stages=stages)).is_error
    refused = await tool.run(objective="x" * 1001, stages=stages)
    assert refused.is_error and "at most 500 characters" in str(refused.content)
