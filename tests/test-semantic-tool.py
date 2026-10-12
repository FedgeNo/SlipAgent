"""Deferred discovery, local retrieval, watcher updates, and owned teardown."""

import asyncio
from pathlib import Path
import subprocess
import threading
import time
import sys

import pytest

from slipagent.agent import Agent
from slipagent.semantic.service import SemanticSearch
from slipagent.tools import build_default_registry
from slipagent.tools.discovery import tool_catalog
from slipagent.tools.semantic import SemanticSearchTool
from slipagent.workspace import Workspace
from slipagent.types import ToolCall
from test_agent import StubClient, completion


def test_stable_semantic_modules_do_not_load_inference_libraries():
    from slipagent.runtime import CORE_MODULES
    from test_cli_e2e import cli_environment

    modules = sorted("slipagent." + name for name in CORE_MODULES if name.startswith("semantic"))
    script = ("import importlib, sys\n"
              f"for name in {modules!r}: importlib.import_module(name)\n"
              "assert not {'numpy', 'onnxruntime', 'tokenizers', 'usearch'} & sys.modules.keys()\n")
    subprocess.run([sys.executable, "-c", script], env=cli_environment(), check=True,
                   capture_output=True, text=True, timeout=10)


class Encoder:
    identity = "semantic-tool-fixture-v1"
    threads = 1

    def __init__(self):
        self.documents = []
        self.thread_ids = []

    def dim(self):
        return 3

    def embed(self, text):
        self.thread_ids.append(threading.get_ident())
        return [1, 0, 0]

    def embed_batch(self, texts):
        self.documents.extend(texts)
        return [self.embed(text) for text in texts]


async def test_calling_unloaded_semantic_tool_executes_nothing(workspace):
    tools = build_default_registry(workspace)
    client = StubClient([
        completion("Search now.", [ToolCall("find", "semantic_search", {"query": "some behavior"})]),
        completion("Use ordinary search instead."),
    ])
    try:
        agent = Agent(client, tools, "stub")
        await agent.run("Find behavior in this project.")
        await agent.wait_for_compaction()
        assert "semantic_search" not in tools.services
        results = [message.content for message in agent.messages if message.role == "tool"]
        assert results and all(result["status"] == "error" for result in results)
    finally:
        await tools.aclose()


async def test_discovery_loads_definition_without_starting_model_or_watcher(workspace):
    tools = build_default_registry(workspace)
    try:
        agent = Agent(None, tools, "stub")
        assert "semantic_search" not in {spec.name for spec in agent._visible_specs(tools.specs())}
        catalog = tool_catalog(tools)
        purpose = next(item for item in catalog.deferred() if item["name"] == "semantic_search")
        assert "repository map" in purpose["purpose"]
        assert "semantic_search" not in tools.services
        loaded = await tools.invoke("load_tools", {"names": ["semantic_search"]})
        assert not loaded.is_error
        assert loaded.content["definitions"][0]["function"]["parameters"]["required"] == ["query"]
        assert "semantic_search" in {spec.name for spec in agent._visible_specs(tools.specs())}
        assert "semantic_search" not in tools.services
    finally:
        await tools.aclose()


async def test_search_returns_relative_complete_excerpts_and_keeps_one_worker(workspace):
    source = "def copy_to_clipboard(value):\n    return value\n"
    (workspace.root / "clipboard.py").write_text(source)
    encoder = Encoder()
    service = SemanticSearch(workspace, encoder_factory=lambda: encoder)
    tools = build_default_registry(workspace)
    tools.services["semantic_search"] = service
    try:
        result = await tools.invoke("semantic_search", {"query": "copy value to clipboard"})
        assert not result.is_error, result.text()
        assert result.content["matches"][0] == {
            "path": "clipboard.py", "start_line": 1, "end_line": 2, "text": source, "score": 0.25,
        }
        assert result.content["indexed_files"] == 1
        worker = service._thread
        second = await tools.invoke("semantic_search", {"query": "clipboard"})
        assert not second.is_error
        assert service._thread is worker
        assert set(encoder.thread_ids) == {worker.ident}
        assert service.database.is_relative_to(service.state_directory / "semantic")
        candidate = build_default_registry(workspace, services=tools.services)
        assert candidate.services["semantic_search"] is service
        await candidate.aclose()
        assert not service._closed
    finally:
        await tools.aclose()
    assert service._thread is None
    assert service._closed


async def test_native_watcher_updates_edits_renames_and_deletions(workspace):
    path = workspace.root / "old.py"
    path.write_text("old_identifier\n")
    service = SemanticSearch(workspace, encoder_factory=Encoder)
    tool = SemanticSearchTool(workspace, {"semantic_search": service})
    try:
        assert not (await tool.run("old_identifier")).is_error
        path.write_text("new_identifier\n")
        renamed = workspace.root / "new.py"
        path.rename(renamed)
        deadline = time.monotonic() + 5
        while True:
            result = await tool.run("new_identifier")
            if result.content["matches"] and result.content["matches"][0]["path"] == "new.py":
                break
            assert time.monotonic() < deadline
            await asyncio.sleep(0.05)
        assert result.content["matches"][0]["text"] == "new_identifier\n"
        renamed.unlink()
        while (await tool.run("new_identifier")).content["matches"]:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.05)
    finally:
        await service.aclose()


async def test_persistent_index_reuses_unchanged_vectors_and_reconciles_while_stopped(workspace):
    path = workspace.root / "app.py"
    path.write_text("original_identifier\n")
    first_encoder = Encoder()
    first = SemanticSearch(workspace, encoder_factory=lambda: first_encoder)
    try:
        await asyncio.to_thread(first.lookup, "original", 5)
    finally:
        await first.aclose()
    second_encoder = Encoder()
    second = SemanticSearch(workspace, encoder_factory=lambda: second_encoder)
    try:
        await asyncio.to_thread(second.lookup, "original", 5)
        assert second_encoder.documents == []
    finally:
        await second.aclose()
    path.write_text("updated_identifier\n")
    third_encoder = Encoder()
    third = SemanticSearch(workspace, encoder_factory=lambda: third_encoder)
    try:
        result = await asyncio.to_thread(third.lookup, "updated", 5)
        assert result["matches"][0]["text"] == "updated_identifier\n"
        assert len(third_encoder.documents) == 1
    finally:
        await third.aclose()


async def test_hidden_ignored_dependencies_and_external_symlinks_never_embed(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = Workspace(root)
    subprocess.run(["git", "init", "-q", str(workspace.root)], check=True)
    (workspace.root / ".gitignore").write_text("ignored.py\nignored-dir/\n")
    (workspace.root / "visible.py").write_text("public_code\n")
    for name in (".hidden.py", "ignored.py", "ignored-dir/private.py", "node_modules/dependency.py"):
        path = workspace.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private_fixture\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "external.py"
    secret.write_text("outside_fixture\n")
    (workspace.root / "link.py").symlink_to(secret)
    encoder = Encoder()
    service = SemanticSearch(workspace, encoder_factory=lambda: encoder)
    try:
        result = await asyncio.to_thread(service.lookup, "public_code", 5)
        assert [row["path"] for row in result["matches"]] == ["visible.py"]
        assert result["indexed_files"] == 1
        assert encoder.documents == ["File: visible.py\n\npublic_code\n"]
    finally:
        await service.aclose()
        secret.unlink()
        outside.rmdir()


@pytest.mark.parametrize("arguments", [{"query": ""}, {"query": "   "}, {"query": "test", "limit": 0},
                                       {"query": "test", "limit": 11}, {"query": "test", "path": ".."}])
async def test_invalid_arguments_do_not_start_resources(workspace, arguments):
    services = {}
    result = await SemanticSearchTool(workspace, services).invoke(arguments)
    assert result.is_error
    assert services == {}


async def test_model_failure_is_a_tool_error_and_can_retry(workspace):
    count = 0

    def factory():
        nonlocal count
        count += 1
        if count == 1:
            raise FileNotFoundError("fixture model unavailable")
        return Encoder()

    service = SemanticSearch(workspace, encoder_factory=factory)
    tool = SemanticSearchTool(workspace, {"semantic_search": service})
    try:
        result = await asyncio.wait_for(tool.run("behavior"), 5)
        assert result.is_error and "fixture model unavailable" in result.text()
        assert "grep" in result.text()
        again = await asyncio.wait_for(tool.run("behavior"), 5)
        assert not again.is_error
        assert count == 2
    finally:
        await service.aclose()


async def test_cancellation_drains_initial_index_and_allows_later_restart(workspace):
    (workspace.root / "app.py").write_text("fixture_identifier\n")
    entered = threading.Event()
    release = threading.Event()

    class SlowEncoder(Encoder):
        def embed_batch(self, texts):
            entered.set()
            assert release.wait(5)
            return super().embed_batch(texts)

    service = SemanticSearch(workspace, encoder_factory=SlowEncoder)
    tool = SemanticSearchTool(workspace, {"semantic_search": service})
    task = asyncio.create_task(tool.run("fixture_identifier"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert service._thread is None
        assert not (await asyncio.wait_for(tool.run("fixture_identifier"), 5)).is_error
    finally:
        release.set()
        await service.aclose()


async def test_separate_workspaces_have_separate_indexes(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    left = SemanticSearch(Workspace(a), encoder_factory=Encoder)
    right = SemanticSearch(Workspace(b), encoder_factory=Encoder)
    assert left.database != right.database
    await left.aclose()
    await right.aclose()
    assert left._thread is right._thread is None
