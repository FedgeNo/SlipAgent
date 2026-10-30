"""Tool definitions and execution share the current request's allowlist."""

import json
import re
from importlib import import_module

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.mcp import MCPManager
from slipagent.openrouter import OpenRouterClient
from slipagent.tools import build_default_registry
from slipagent.tools.base import ToolRegistry
from slipagent.tools.discovery import CORE_TOOLS, UNAVAILABLE_BATCH, LoadToolsTool, ToolCatalog, tool_catalog
from slipagent.types import Message, ToolCall
from slipagent.workspace import Workspace
from data_text_reader import current_input, request_input
from test_agent import RecordingTool, StubClient, completion, context_records
from test_mcp import stub_spec

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
            assert {tool["function"]["name"] for tool in first["tools"]} == {"load_tools", "update_plan"}
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


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mode", ["legacy", "modern"])
@pytest.mark.parametrize("allowlist", [None, (), ("server__echo",)])
async def test_mcp_guidance_precedes_loading_and_respects_tool_allowlist(tmp_path, native, mode, allowlist):
    router = Router({"test/model": ["tools"] if native else ["response_format"]},
                    [reply("Done.", [], native) for _ in range(3)])
    services = {}
    tools = ToolRegistry([LoadToolsTool(services)], services=services)
    manager = MCPManager(tmp_path, registry=tools)
    try:
        state = await manager.connect("server", stub_spec(name="server", mode=mode))
        assert state.status == "connected", state.detail
        async with OpenRouterClient("dummy", transport=httpx.MockTransport(router.handle)) as client:
            agent = Agent(client, tools, "test/model", exposed_tools=allowlist)
            assert await agent.run("Inspect the available tools.") == "Done."
            await agent.wait_for_compaction()
            agent.reset()
            assert await agent.run("Inspect the available tools again.") == "Done."
            await agent.wait_for_compaction()
            await manager.disconnect("server")
            if allowlist is not None:
                agent.exposed_tools = ()
            assert await agent.run("Inspect the disconnected server.") == "Done."
            await agent.wait_for_compaction()

        for request in router.requests[:2]:
            system = request["messages"][0]["content"]
            assert ("Use echo before boom." in system) == (allowlist != ())
            assert ("| `server__echo` |" in system) == (allowlist != ())
            assert ("| `server__boom` |" in system) == (allowlist is None)
            if native:
                names = {tool["function"]["name"] for tool in request["tools"]}
                assert not names.intersection({"server__echo", "server__boom"})
        assert "Use echo before boom." not in router.requests[2]["messages"][0]["content"]
    finally:
        await manager.aclose()
        await tools.aclose()


async def test_default_registry_starts_with_only_core_definitions(tmp_path):
    tools = build_default_registry(Workspace(tmp_path))
    try:
        agent = Agent(None, tools, "stub")
        visible = agent._visible_specs(tools.specs())
        assert {spec.name for spec in visible} == CORE_TOOLS
        assert "write_file" in {spec.name for spec in visible}
        catalog = tool_catalog(tools)
        assert "write_file" not in {entry["name"] for entry in catalog.deferred()}
        assert "recall_history" in {entry["name"] for entry in catalog.deferred()}
        assert "update_plan" not in {entry["name"] for entry in catalog.deferred()}
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


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("allowlist", [None, ("read_file", "optional_record"), ("read_file",)])
@pytest.mark.parametrize("unavailable", ["optional_record", "missing"])
async def test_unavailable_call_rejects_the_whole_batch(monkeypatch, native, allowlist, unavailable):
    calls = [ToolCall("before", "read_file", {"value": "before"}),
             ToolCall("load", "load_tools", {"names": ["optional_record"]}),
             ToolCall("unavailable", unavailable, {"value": "data"}),
             ToolCall("after", "read_file", {"value": "after"})]
    router = Router({"test/model": ["tools"] if native else ["response_format"]}, [
        reply("Run the batch.", calls, native), reply("Done.", [], native),
    ])
    tools = registry()
    loads = []
    loader = tools.get("load_tools")
    original_load = loader.run

    async def load(names):
        loads.append(names)
        return await original_load(names)

    monkeypatch.setattr(loader, "run", load)
    core = RecordingTool()
    core.name = "read_file"
    tools.register(core)
    try:
        async with OpenRouterClient("dummy", transport=httpx.MockTransport(router.handle)) as client:
            agent = Agent(client, tools, "test/model", exposed_tools=allowlist)
            assert await agent.run("Record this value.") == "Done."
            await agent.wait_for_compaction()
        assert core.seen == []
        assert loads == []
        assert tools.get("optional_record").seen == []
        observations = [m for m in agent.messages if m.role == "tool"]
        assistant_calls = [call for message in agent.messages if message.role == "assistant"
                           for call in message.tool_calls or []]
        local_ids = [call.id for call in assistant_calls]
        assert [call.name for call in assistant_calls] == [call.name for call in calls]
        assert len(set(local_ids)) == len(calls)
        assert all(re.fullmatch(r"call_[0-9a-f]{32}", call_id) for call_id in local_ids)
        assert not set(local_ids).intersection(call.id for call in calls)
        assert [m.tool_call_id for m in observations] == local_ids
        assert all(m.content["status"] == "error" for m in observations)
        assert all("No tools were executed" in m.content["content"] for m in observations)
        assert all(unavailable in m.content["content"] for m in observations)
        if unavailable == "optional_record" and allowlist != ("read_file",):
            assert all("load_tools" in m.content["content"] for m in observations)
        if native:
            assert {tool["function"]["name"] for tool in router.requests[0]["tools"]} == {"load_tools", "read_file", "update_plan"}
        retry_messages = router.requests[1]["messages"]
        retry_input = request_input(retry_messages)
        assert retry_input["errors"] == []
        returned = [result for record in context_records(retry_messages) for result in record.get("tool_results", [])]
        assert len(returned) == len(calls)
        assert all("No tools were executed" in result["content"] for result in returned)
        assert "No tools were executed" not in retry_messages[0]["content"]
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


async def test_rejected_batch_preserves_files_and_creates_no_checkpoints(tmp_path):
    path = tmp_path / "existing.txt"
    path.write_text("original")
    tools = build_default_registry(Workspace(tmp_path))
    tools.register(OptionalTool())
    client = StubClient([
        completion(tool_calls=[
            ToolCall("edit", "edit_file", {"path": "existing.txt", "old_string": "original", "new_string": "changed"}),
            ToolCall("write", "write_file", {"path": "new.txt", "content": "new"}),
            ToolCall("optional", "optional_record", {}),
        ]),
        completion("Correct the batch first."),
    ])
    try:
        agent = Agent(client, tools, "test")
        assert await agent.run("Edit the files.") == "Correct the batch first."
        await agent.wait_for_compaction()
        assert path.read_text() == "original"
        assert not (tmp_path / "new.txt").exists()
        assert agent._checkpoints().listing() == []
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


@pytest.mark.parametrize("rejected_first", [False, True])
def test_restoring_tool_use_scopes_rejected_call_ids_to_their_batch(rejected_first):
    accepted = [
        Message.assistant(None, [ToolCall("call_1", "optional_record", {"value": "saved"})]),
        Message.tool_result("call_1", {"tool": "optional_record", "status": "success", "content": "saved"}),
    ]
    rejected = [
        Message.assistant(None, [ToolCall("call_1", "unloaded_record", {})]),
        Message.tool_result("call_1", {"tool": "unloaded_record", "status": "error",
                                      "content": UNAVAILABLE_BATCH + "on this request: unloaded_record"}),
    ]
    unloaded = OptionalTool()
    unloaded.name = "unloaded_record"
    catalog = ToolCatalog()
    messages = rejected + accepted if rejected_first else accepted + rejected
    selected = catalog.select([OptionalTool().spec, unloaded.spec], messages)
    assert selected == {"optional_record"}


def test_archived_history_restores_loading_without_reading_result_text(monkeypatch):
    from slipagent.context import ConversationHistory
    from slipagent.sources import FileText
    from slipagent.turnstore import TurnStore

    unloaded = OptionalTool()
    unloaded.name = "unloaded_record"
    messages = [
        Message.user("start"),
        Message.assistant(None, [ToolCall("call_1", "unloaded_record", {})]),
        # Long enough to be archived as a file reference rather than inline text.
        Message.tool_result("call_1", {"tool": "unloaded_record", "call_id": "call_1", "status": "error",
                                      "content": UNAVAILABLE_BATCH + "x" * 2000}),
        Message.assistant(None, [ToolCall("call_2", "load_tools", {"names": ["optional_record"]})]),
        Message.tool_result("call_2", {"tool": "load_tools", "call_id": "call_2", "status": "success",
                                      "content": {"loaded": ["optional_record"], "definitions": [], "errors": {}}}),
        Message.assistant(None, [ToolCall("call_3", "read_file", {"path": "a"})]),
        Message.tool_result("call_3", {"tool": "read_file", "call_id": "call_3", "status": "success",
                                      "content": "y" * 5000}),
    ]
    specs = [OptionalTool().spec, unloaded.spec]
    expected = ToolCatalog().select(specs, messages)
    history = ConversationHistory()
    history.sync(messages)
    history.store = TurnStore()
    try:
        for step in history.steps:
            history.store.archive(step, messages)
        reads = []
        original = FileText.chunks
        monkeypatch.setattr(FileText, "chunks", lambda self: (reads.append(self.characters), original(self))[1])
        assert ToolCatalog().select(specs, messages) == expected == {"optional_record"}
        # Only the rejected result's prefix is read; ordinary archived results stay on disk.
        assert reads == [len(UNAVAILABLE_BATCH) + 2000]
    finally:
        history.clear()
