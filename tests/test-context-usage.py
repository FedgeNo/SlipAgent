"""Context readouts receive calibrated estimates and per-request provider usage."""

import json
import math
from importlib import import_module

import httpx

from slipagent.agent import Agent
from slipagent.openrouter import OpenRouterClient
from slipagent.tools.base import ToolRegistry
from test_agent import RecordingTool

native = import_module('test-native-tools')


async def test_context_usage_matches_the_budget_and_latest_provider_request():
    router = native.Router({'test/model': ['tools']}, [
        native.wire('Inspecting.', [native.native_call()]), native.wire(),
    ])
    measured = iter([1000, 1200])

    def handle(request):
        response = router.handle(request)
        if request.method == 'POST':
            payload = response.json()
            payload['usage'] = {'prompt_tokens': next(measured), 'completion_tokens': 1}
            return httpx.Response(200, json=payload)
        return response

    events = []
    registry = ToolRegistry([RecordingTool()])
    async with OpenRouterClient('test', transport=httpx.MockTransport(handle)) as client:
        agent = Agent(client, registry, 'test/model', on_event=events.append)
        profile = await client.model_capabilities('test/model')
        budget = agent._budget()
        budget.select('test/model', profile)
        budget.scale = .8
        try:
            assert await agent.run('Inspect the project.') == 'Done.'
            reports = [json.loads(event.text) for event in events if event.kind == 'context_usage']
            bodies = [event.request_body for event in events if event.kind == 'context']
            assert reports[0] == {'model': 'test/model', 'provider': 'openrouter',
                                  'tokens': math.ceil(bodies[0].estimate * .8), 'estimated': True}
            assert [report['estimated'] for report in reports] == [True, False, True, False]
            assert [report['tokens'] for report in reports if not report['estimated']] == [1000, 1200]
            assert agent.usage.prompt_tokens == 2200
        finally:
            await registry.aclose()
