"""Deferred definitions reach both request carriers without restricting valid calls."""

import json
from importlib import import_module

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.openrouter import OpenRouterClient
from slipagent.tools import build_default_registry
from slipagent.tools.base import ToolRegistry
from slipagent.tools.discovery import CORE_TOOLS, LoadToolsTool, tool_catalog
from slipagent.types import Message, ToolCall
from slipagent.workspace import Workspace
from test_agent import RecordingTool

Router = import_module("test-native-tools").Router


class OptionalTool(RecordingTool):
    name = "optional_record"
    description = "Record an optional value.\n\nA distinctive long contract that should be deferred."


def registry():
    services = {}
    return ToolRegistry([LoadToolsTool(services), OptionalTool()], services=services)


def reply(text, calls, native):
    if native:
        return {"role": "assistant", "content": text, "tool_calls": [call.to_api() for call in calls]}
    return {"role": "assistant", "content": json.dumps({"response": text,
            "tool_calls": [call.to_record() for call in calls]})}


@pytest.mark.parametrize("native", [False, True])
async def test_loading_changes_prompt_and_api_definitions_on_the_next_turn(native):
    router = Router({"test/model": ["tools"] if native else ["response_format"]}, [
        reply("Load the optional tool.", [ToolCall("load", "load_tools", {"names": ["optional_record"]})], native),
        reply("Record it.", [ToolCall("use", "optional_record", {"value": "data"})], native),
        reply("Done.", [], native),
        reply("Still available.", [], native),
    ])
    tools = registry()
    try:
        async with OpenRouterClient("dummy", transport=httpx.MockTransport(router.handle)) as client:
            agent = Agent(client, tools, "test/model")
            assert await agent.run("Use the optional tool.") == "Done."
            await agent.wait_for_compaction()
            assert await agent.run("What is available?") == "Still available."
            await agent.wait_for_compaction()
        first = router.requests[0]
        system = first["messages"][0]["content"]
        assert "optional_record" in system
        assert "A distinctive long contract" not in system
        assert "# Additional Tool Catalog" in system
        assert "## Core Tools\n\n```json\n{\n  \"load_tools\"" in system
        assert "## Additional Loaded Tools\n\n```json\n{}" in system
        assert "| `optional_record` | Record an optional value. |" in system
        for request in router.requests[1:]:
            assert "A distinctive long contract" in request["messages"][0]["content"]
        if native:
            assert {tool["function"]["name"] for tool in first["tools"]} == {"load_tools"}
            assert all("optional_record" in {tool["function"]["name"] for tool in request["tools"]}
                       for request in router.requests[1:])
        else:
            assert all("tools" not in request for request in router.requests)
        assert tools.get("optional_record").seen == [{"value": "data"}]
        result = next(message.content["content"] for message in agent.messages
                      if message.role == "tool" and message.content["tool"] == "load_tools")
        assert isinstance(result["definitions"][0], dict)
        assert result["definitions"][0]["function"]["parameters"]["properties"]["value"]["type"] == "string"
    finally:
        await tools.aclose()


async def test_default_registry_starts_with_only_core_definitions(tmp_path):
    tools = build_default_registry(Workspace(tmp_path))
    try:
        agent = Agent(None, tools, "stub")
        visible = agent._visible_specs(tools.specs())
        assert {spec.name for spec in visible} == CORE_TOOLS
        catalog = tool_catalog(tools)
        assert "write_file" in {entry["name"] for entry in catalog.deferred()}
        assert "recall_history" in {entry["name"] for entry in catalog.deferred()}
        assert "update_plan" in {entry["name"] for entry in catalog.deferred()}
    finally:
        await tools.aclose()


async def test_load_is_idempotent_and_preserves_valid_names_alongside_errors():
    tools = registry()
    try:
        agent = Agent(None, tools, "stub")
        agent._visible_specs(tools.specs())
        first = await tools.invoke("load_tools", {"names": ["optional_record", "missing", "optional_record"]})
        assert first.is_error
        assert first.content["loaded"] == ["optional_record"]
        assert list(first.content["errors"]) == ["missing"]
        again = await tools.invoke("load_tools", {"names": ["optional_record"]})
        assert not again.is_error
        assert again.content["definitions"] == first.content["definitions"]
        assert tools.get("optional_record").seen == []
        assert "optional_record" in {spec.name for spec in agent._visible_specs(tools.specs())}
    finally:
        await tools.aclose()


async def test_valid_deferred_calls_still_execute_and_retain_their_definition():
    router = Router({"test/model": ["tools"]}, [
        reply("Record it.", [ToolCall("use", "optional_record", {"value": "data"})], True),
        reply("Done.", [], True),
    ])
    tools = registry()
    try:
        async with OpenRouterClient("dummy", transport=httpx.MockTransport(router.handle)) as client:
            agent = Agent(client, tools, "test/model")
            assert await agent.run("Record this value.") == "Done."
            await agent.wait_for_compaction()
        assert tools.get("optional_record").seen == [{"value": "data"}]
        assert {tool["function"]["name"] for tool in router.requests[0]["tools"]} == {"load_tools"}
        assert "optional_record" in {tool["function"]["name"] for tool in router.requests[1]["tools"]}
    finally:
        await tools.aclose()


async def test_loader_cannot_expand_an_explicit_tool_allowlist():
    tools = registry()
    try:
        agent = Agent(None, tools, "stub", exposed_tools=())
        agent._visible_specs(tools.specs())
        assert "optional_record" not in {entry["name"] for entry in tool_catalog(tools).deferred()}
        result = await tools.invoke("load_tools", {"names": ["optional_record"]})
        assert result.is_error
        assert result.content["loaded"] == []
        assert "optional_record" not in {spec.name for spec in agent._visible_specs(tools.specs())}
    finally:
        await tools.aclose()


async def test_reset_clears_loading_and_saved_history_restores_it():
    tools = registry()
    try:
        agent = Agent(None, tools, "stub")
        agent._visible_specs(tools.specs())
        result = await tools.invoke("load_tools", {"names": ["optional_record"]})
        messages = [Message.user("Load it."),
                    Message.assistant("Loading.", [ToolCall("load", "load_tools", {"names": ["optional_record"]})]),
                    Message.tool_result("load", {"tool": "load_tools", "status": "success", "content": result.content})]
        agent.messages.extend(messages)
        assert "optional_record" in {spec.name for spec in agent._visible_specs(tools.specs())}
        agent.reset()
        assert "optional_record" not in {spec.name for spec in agent._visible_specs(tools.specs())}
        agent.extend([Message.from_api(message.to_record()) for message in messages])
        assert "optional_record" in {spec.name for spec in agent._visible_specs(tools.specs())}
        # Replacement tool instances borrow the session's retained catalog.
        tools.replace([LoadToolsTool(tools.services), OptionalTool()])
        assert "optional_record" in {spec.name for spec in agent._visible_specs(tools.specs())}
    finally:
        await tools.aclose()
