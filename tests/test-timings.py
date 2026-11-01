"""Local phase counters preserve timing, cancellation, and execution boundaries."""

import asyncio
import io
import json
from types import SimpleNamespace

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.capabilities import RequestProfile
from slipagent.commands import BY_NAME, CommandContext
from slipagent.cli import Style
from slipagent.diagnostics import RequestDiagnostics
from slipagent.openrouter import OpenRouterClient
from slipagent.timings import PhaseTimings, timing_log
from slipagent.tools.base import Tool, ToolRegistry, ToolResult
from slipagent.types import ToolCall
from test_agent import StubClient, completion


def test_bounded_records_active_elapsed_reset_and_disable():
    now = [0.0]
    timings = PhaseTimings(limit=2, clock=lambda: now[0])
    ticket = timings.start('context_selection', 1)
    now[0] = 2
    assert timings.snapshot()[0]['seconds'] == 2
    assert timings.snapshot()[0]['outcome'] == 'running'
    timings.finish(ticket)
    for _ in range(2):
        timings.record('tool_execution', 1, 1, 'local-call')
    assert len(timings.snapshot()) == 2
    ticket = timings.start('provider_response', 2)
    timings.clear()
    timings.finish(ticket)
    assert timings.snapshot() == []
    timings.set_enabled(False)
    with timings.measure('provider_response', 2):
        now[0] += 5
    assert timings.snapshot() == []


async def test_cancelled_phase_retains_elapsed_and_status():
    now = [0.0]
    timings = PhaseTimings(clock=lambda: now[0])
    with pytest.raises(asyncio.CancelledError):
        with timings.measure('provider_response', 3):
            now[0] = 4
            raise asyncio.CancelledError
    assert timings.snapshot() == [{'phase': 'provider_response', 'step': 3, 'call_id': '',
                                    'seconds': 4, 'outcome': 'cancelled'}]


async def test_serial_tool_wait_is_separate_from_execution():
    now = [0.0]

    class Read(Tool):
        name = 'timed_read'
        description = 'Read fixture'

        async def run(self):
            now[0] += 2
            return ToolResult.ok('Read')

    registry = ToolRegistry([Read()])
    registry.services['phase_timings'] = PhaseTimings(clock=lambda: now[0])
    calls = [ToolCall('first', 'timed_read', {}), ToolCall('second', 'timed_read', {})]
    agent = Agent(StubClient([completion('Reading', calls), completion('Done')]), registry, 'test')
    assert await agent.run('Read both') == 'Done'
    rows = timing_log(registry).snapshot()
    waits = {row['call_id']: row['seconds'] for row in rows if row['phase'] == 'tool_wait'}
    execution = {row['call_id']: row['seconds'] for row in rows if row['phase'] == 'tool_execution'}
    assert waits == {'first': 0, 'second': 2}
    assert execution == {'first': 2, 'second': 2}
    await registry.aclose()


async def test_provider_phases_are_saved_only_in_diagnostics(tmp_path, monkeypatch):
    now = [0.0]
    registry = ToolRegistry()
    registry.services['phase_timings'] = PhaseTimings(clock=lambda: now[0])
    diagnostics = RequestDiagnostics()
    diagnostics.use_directory(tmp_path / 'requests')
    registry.resources.diagnostics = diagnostics
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        now[0] += 3
        return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'Done'}, 'finish_reason': 'stop'}]})

    async with OpenRouterClient('dummy-test-key', transport=httpx.MockTransport(reply)) as client:
        client.cache_capabilities('test', RequestProfile(parameters={'tools', 'max_tokens'}, context_length=1000000))
        build = client._build_request_body

        async def prepare(payload):
            result = await build(payload)
            now[0] += .5
            return result

        monkeypatch.setattr(client, '_build_request_body', prepare)
        agent = Agent(client, registry, 'test')
        assert await agent.run('Private fixture request') == 'Done'
    rows = diagnostics.listing()[-1]['timings']
    phases = {row['phase']: row['seconds'] for row in rows}
    assert phases['request_preparation'] == .5
    assert phases['provider_response'] == 3
    assert 'context_selection' in phases
    assert 'phase_timings' not in sent[0]
    assert all(set(row) == {'phase', 'step', 'call_id', 'seconds', 'outcome'} for row in rows)
    await registry.aclose()


async def test_timing_command_can_run_during_work_and_select_turn():
    registry = ToolRegistry()
    timing_log(registry).record('tool_execution', 1.25, 2, 'call-2')
    out = io.StringIO()
    context = CommandContext(SimpleNamespace(registry=registry), Style(False), out)
    command = BY_NAME['timings']
    assert command.background('') and not command.changes_session('clear')
    await command.run(context, '2')
    assert 'step 2' in out.getvalue() and '1.250s' in out.getvalue() and 'call-2' in out.getvalue()
    await command.run(context, 'off')
    assert not timing_log(registry).enabled
    await command.run(context, 'clear')
    assert timing_log(registry).snapshot() == []
