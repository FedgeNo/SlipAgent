"""Temperature controls use the selected endpoints' advertised capabilities."""

import io
import json

import httpx
import pytest

from test_agent import summary_response

from slipagent.agent import Agent
from slipagent.cli import Renderer, Session, Style, _handle_command, _model_command
from slipagent.openrouter import OpenRouterClient
from slipagent.tools.base import ToolRegistry
from slipagent.workspace import Workspace
from test_agent import task_record


def transport(requests, *, broken_metadata=False):
    parameters = {
        "adjustable": ["tools", "response_format", "temperature"],
        "fixed": ["tools", "response_format"],
    }

    def handle(request):
        if request.method == "GET":
            if broken_metadata:
                return httpx.Response(400, json={"error": {"message": "Metadata unavailable"}})
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [
                    {"id": model, "supported_parameters": params} for model, params in parameters.items()
                ]})
            model = request.url.path.split("/models/", 1)[1].removesuffix("/endpoints")
            return httpx.Response(200, json={"data": {"endpoints": [
                {"tag": "provider", "supported_parameters": parameters[model], "context_length": 1000000}
            ]}})
        summary = summary_response(json.loads(request.content))
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(json.loads(request.content))
        record = {
            "task": task_record(requests[-1]["messages"]),
            "response": "Done", "tool_calls": [], "previous_tool_responses_compressed": "",
            "user_prompt_compressed": "Do the task", "agent_response_compressed": "Done",
        }
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": record["response"],
        }, "finish_reason": "stop"}]})

    return httpx.MockTransport(handle)


def session_for(client, tmp_path, model="adjustable"):
    agent = Agent(client, ToolRegistry(), model)
    renderer = Renderer(Style(True), io.StringIO(), False)
    return Session(agent, agent.registry, client, renderer, Workspace(tmp_path),
                   "test", client.base_url, None, "test")


async def test_default_and_command_change_actual_requests(tmp_path):
    requests = []
    async with OpenRouterClient("test", transport=transport(requests)) as client:
        session = session_for(client, tmp_path)
        await session.agent.run("First task")
        await _handle_command(session, "/temperature")
        assert "temperature: 1 (default)" in session.renderer.stream.getvalue()
        await _handle_command(session, "/temperature 0")
        await session.agent.run("Second task")
        await _handle_command(session, "/temperature 0.1")
        await session.agent.run("Third task")
    assert [request["temperature"] for request in requests] == [1.0, 0, 0.1]


async def test_model_switch_omits_unsupported_temperature_and_retains_setting(tmp_path):
    requests = []
    async with OpenRouterClient("test", transport=transport(requests)) as client:
        session = session_for(client, tmp_path)
        await _handle_command(session, "/temperature 0.1")
        await _model_command(session, "fixed", session.renderer.style, io.StringIO())
        await _handle_command(session, "/temperature 0.5")
        assert session.agent.temperature == 0.1
        assert "temperature control is unavailable" in session.renderer.stream.getvalue()
        assert "\x1b[38;2;255;102;102m" in session.renderer.stream.getvalue()
        await session.agent.run("First task")
        await _model_command(session, "adjustable", session.renderer.style, io.StringIO())
        await session.agent.run("Second task")
    assert "temperature" not in requests[0]
    assert requests[1]["temperature"] == 0.1


async def test_default_does_not_add_unsupported_parameter(tmp_path):
    requests = []
    async with OpenRouterClient("test", transport=transport(requests)) as client:
        session = session_for(client, tmp_path, "fixed")
        await session.agent.run("Task")
        await _handle_command(session, "/temperature")
        assert "provider default is used" in session.renderer.stream.getvalue()
    assert "temperature" not in requests[0]


@pytest.mark.parametrize("argument", ["nan", "inf", "-0.1", "2.1", "invalid", "0.2 extra"])
async def test_invalid_values_leave_setting_unchanged(tmp_path, argument):
    async with OpenRouterClient("test", transport=transport([])) as client:
        session = session_for(client, tmp_path)
        session.agent.temperature = 0.1
        await _handle_command(session, f"/temperature {argument}")
        assert session.agent.temperature == 0.1
        assert "must be a number from 0 to 2" in session.renderer.stream.getvalue()


async def test_metadata_failure_leaves_setting_unchanged(tmp_path):
    async with OpenRouterClient("test", transport=transport([], broken_metadata=True)) as client:
        session = session_for(client, tmp_path)
        session.agent.temperature = 0.1
        await _handle_command(session, "/temperature 0.5")
        assert session.agent.temperature == 0.1
        assert "could not verify temperature support" in session.renderer.stream.getvalue()


async def test_temperature_changes_wait_for_idle_but_readout_stays_available(tmp_path):
    async with OpenRouterClient("test", transport=transport([])) as client:
        session = session_for(client, tmp_path)
        session.agent.running = True
        await _handle_command(session, "/temperature 0.5")
        assert session.agent.temperature is None
        assert "Queued /temperature" in session.renderer.stream.getvalue()
        assert session.extensions["deferred_commands"] == ["/temperature 0.5"]
        await _handle_command(session, "/temperature")
        assert "temperature: 1 (default)" in session.renderer.stream.getvalue()
