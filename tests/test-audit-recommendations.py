"""Regression coverage for the approved remaining audit recommendations."""

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slipagent.agent import Agent
from slipagent.artifacts import ArtifactStore
from slipagent.capabilities import RequestProfile
from slipagent.checkpoints import FileCheckpoints, current_checkpoint
from slipagent.checks import BatchCheckReport, check_edit_batch_report
from slipagent.compaction import StepCompactor
from slipagent.context import ContextError, ConversationHistory
from slipagent.mcp import MCPClient, MCPError, ServerSpec
from slipagent.sessions import SessionError, SessionJournal
from slipagent.sources import materialize
from slipagent.toolbatch import ToolBatch
from slipagent.tools.base import ToolRegistry, ToolResult, validate_arguments
from slipagent.tools.files import WriteFileTool, _atomic_write
from slipagent.tools.git import GitDiffTool, GitStatusTool
from slipagent.tools.paths import DeletePathTool, MovePathTool
from slipagent.turnstore import TurnStore
from slipagent.types import Message, ToolCall, content_text
from test_agent import RecordingTool, StubClient


@pytest.mark.parametrize("kind", ["clean", "process"])
@pytest.mark.parametrize("tool_type", [GitStatusTool, GitDiffTool])
async def test_git_reads_refuse_executable_filters(workspace, kind, tool_type):
    root = workspace.root
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitattributes").write_text("*.txt filter=audit\n")
    (root / "file.txt").write_text("untracked")
    marker = root / "filter-ran"
    subprocess.run(["git", "-C", str(root), "config", f"filter.audit.{kind}",
                    f"touch {marker}"], check=True)
    result = await tool_type(workspace).run()
    assert result.is_error
    assert "helper" in content_text(result.content)
    assert not marker.exists()


def test_damaged_journal_does_not_hide_valid_sessions(tmp_path):
    journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    agent.messages.append(Message.user("saved question"))
    journal.begin(agent)
    valid = journal.session_id
    broken = "a" * 32
    (journal.directory / (broken + ".jsonl")).write_text('{"type":')
    entries = {item["id"]: item for item in journal.listing()}
    assert entries[valid]["prompt"] == "saved question"
    assert "Damaged session" in journal.label(entries[broken])
    journal.session_id = ""
    assert journal.load("latest")["id"] == valid


@pytest.mark.parametrize("broken_store", ["artifact_store", "file_checkpoints"])
def test_restore_preflight_preserves_open_session(tmp_path, workspace, broken_store):
    registry = ToolRegistry(services={"workspace": workspace,
        "artifact_store": ArtifactStore(1_000_000), "file_checkpoints": FileCheckpoints(workspace)})
    agent = Agent(StubClient([]), registry, "test")
    journal = SessionJournal(str(workspace.root), directory=tmp_path / "state")
    agent.messages.append(Message.user("saved"))
    journal.begin(agent)
    saved = journal.load(journal.session_id)
    agent.messages.append(Message.user("keep my current work"))
    before = list(agent.messages)
    owner = registry.services[broken_store]
    directory = owner.directory
    (directory / "index.json").write_text("invalid json")
    with pytest.raises(SessionError, match="open session was preserved"):
        journal.restore(agent, saved)
    assert agent.messages == before
    assert owner.directory == directory


def test_torn_journal_tracks_multibyte_boundary(tmp_path, monkeypatch):
    journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    agent.messages.append(Message.user("🌙 é"))
    journal.begin(agent)
    path = journal.path
    complete = path.stat().st_size
    with path.open("ab") as target:
        target.write(b'{"type":')
    data = journal.load(journal.session_id)
    assert data["torn"] and data["complete_bytes"] == complete
    # Resuming may stream records, but must not read the whole journal to truncate.
    original = type(path).open

    class NoRead:
        def __init__(self, source):
            self.source = source
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.source.__exit__(*args)
        def truncate(self, offset):
            assert offset == complete
            return self.source.truncate(offset)

    def guarded_open(target, mode="r", *args, **kwargs):
        source = original(target, mode, *args, **kwargs)
        return NoRead(source) if target == path and mode == "r+b" else source

    monkeypatch.setattr(type(path), "open", guarded_open)
    journal.restore(agent, data)
    assert journal.load(journal.session_id)["messages"][-1].content == "🌙 é"


@pytest.mark.parametrize("schema,value", [
    ({"type": "array", "minItems": 1}, []),
    ({"type": "array", "maxItems": 1}, [1, 2]),
    ({"type": "string", "minLength": 1}, ""),
    ({"type": "string", "maxLength": 1}, "ab"),
    ({"type": "string", "pattern": "^a$"}, "b"),
    ({"anyOf": [{"type": "object", "required": ["a"]},
                {"type": "object", "required": ["b"]}]}, {}),
])
def test_advertised_constraints_are_enforced(schema, value):
    assert validate_arguments({"properties": {"value": schema}}, {"value": value})


async def test_integral_floats_are_normalized_before_tool_binding():
    from test_tool_base import EchoTool
    tool = EchoTool()
    result = await tool.invoke({"text": "hello", "times": 2.0})
    assert not result.is_error
    assert type(tool.seen[0]["times"]) is int


@pytest.mark.parametrize("value", [1, "text"])
def test_union_properties_keep_each_branches_constraints(value):
    schema = {"anyOf": [
        {"properties": {"value": {"type": "integer"}}, "required": ["value"]},
        {"properties": {"value": {"type": "string"}}, "required": ["value"]},
    ]}
    assert validate_arguments(schema, {"value": value}) is None


async def test_result_recording_failure_pairs_every_call_without_reexecution(monkeypatch):
    calls = [ToolCall("first", "record", {"value": "one"}),
             ToolCall("second", "record", {"value": "two"})]
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    messages = [Message.assistant("", calls)]
    saves = 0

    def persist():
        nonlocal saves
        saves += 1
        if saves == 1:
            raise OSError("disk full")

    batch = ToolBatch(registry, 1, messages, emit=lambda event: None, persist=persist,
                      archive_interrupted=AsyncMock(), checkpoints=None, allowed_tools=None)
    with pytest.raises(OSError, match="disk full"):
        await batch.run(calls)
    results = [message for message in messages if message.role == "tool"]
    assert [message.tool_call_id for message in results] == ["first", "second"]
    assert all(message.content["status"] == "error" for message in results)
    assert "saving its result failed" in content_text(results[0].content)
    assert "not run" in content_text(results[1].content)
    assert len(tool.seen) == 1


async def test_checks_use_explicit_observation_when_other_messages_are_appended(monkeypatch):
    from slipagent import toolbatch
    calls = [ToolCall("first", "record", {"value": "one"})]
    messages = [Message.assistant("", calls)]
    registry = ToolRegistry([RecordingTool()])
    monkeypatch.setattr(toolbatch, "check_edit_batch_report", AsyncMock(return_value=BatchCheckReport("PASSED", "first")))
    batch = ToolBatch(registry, 1, messages, emit=lambda event: None,
                      persist=lambda: messages.append(Message.user("queued input")),
                      archive_interrupted=AsyncMock(), checkpoints=None, allowed_tools=None)
    await batch.run(calls)
    assert messages[-1].content == "queued input"
    assert "harness_checks" in messages[1].content["content"] or "PASSED" in content_text(messages[1].content)


async def test_unattached_check_report_is_a_warning(monkeypatch):
    from slipagent import toolbatch
    monkeypatch.setattr(toolbatch, "check_edit_batch_report", AsyncMock(return_value=BatchCheckReport("check unavailable")))
    events = []
    batch = ToolBatch(ToolRegistry(), 1, [], emit=events.append, persist=lambda: None,
                      archive_interrupted=AsyncMock(), checkpoints=None, allowed_tools=None)
    assert await batch.run([]) == []
    assert events[-1].text == "check unavailable"


@pytest.mark.parametrize("action", ["move", "delete"])
async def test_checks_follow_final_batch_paths(workspace, monkeypatch, action):
    from slipagent import checks
    writer = WriteFileTool(workspace)
    mover = MovePathTool(workspace)
    deleter = DeletePathTool(workspace)
    registry = ToolRegistry([writer, mover, deleter], services={
        "project_environment": SimpleNamespace(workspace=workspace)})
    path = workspace.root / "old.py"
    path.write_text("x = 1")
    edit = ToolCall("edit", writer.name, {"path": "old.py", "content": "x = 1"})
    if action == "move":
        path.rename(workspace.root / "new.py")
        result = ToolResult.ok({"changes": [{"path": "old.py", "action": "moved_from"},
                                            {"path": "new.py", "action": "moved_to"}]})
        tool = mover
    else:
        path.unlink()
        result = ToolResult.ok({"changes": [{"path": "old.py", "action": "deleted"}]})
        tool = deleter
    from slipagent import environment
    monkeypatch.setattr(environment, "load_project_options", lambda workspace: {
        "python_syntax": False, "checks": [{"name": "audit", "argv": ["check", "{files}"]}]})
    monkeypatch.setattr(checks, "_automatic_checks", lambda paths, root: [])
    spawn = AsyncMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(checks.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(checks, "capture_process", AsyncMock(return_value=("", "", False)))
    report = await check_edit_batch_report(registry, [(edit, ToolResult.ok("edited")),
        (ToolCall("path", tool.name, {}), result)], 1)
    assert report.call_id == "edit"
    if action == "move":
        assert str(workspace.root / "new.py") in spawn.call_args.args
        assert str(path) not in spawn.call_args.args
    else:
        spawn.assert_not_called()
        assert "SKIPPED" in report.text


async def test_failed_write_reports_only_directories_it_created(workspace, monkeypatch):
    from slipagent.tools import files
    (workspace.root / "existing").mkdir()
    monkeypatch.setattr(files.os, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("failed publication")))
    result = await WriteFileTool(workspace).run("existing/new/deep/file.txt", "text")
    assert result.is_error
    effects = content_text(result.content).split("Created directories: ", 1)[1]
    assert "existing/new" in effects and "existing/new/deep" in effects
    assert effects != "existing"
    assert (workspace.root / "existing/new/deep").is_dir()


async def test_rewind_retry_handles_multiple_edits_to_an_already_restored_file(workspace):
    first, second = workspace.root / "first.txt", workspace.root / "second.txt"
    first.write_text("original first")
    second.write_text("original second")
    store = FileCheckpoints(workspace)
    try:
        store.begin(1)
        token = current_checkpoint.set(store)
        try:
            for path, text in [(first, "intermediate"), (first, "final"), (second, "edited")]:
                _atomic_write(workspace, path, text)
        finally:
            current_checkpoint.reset(token)
            store.finish()
        # This is the on-disk state after one destination was restored and the
        # rewind failed before publishing the next destination or its index.
        first.write_text("original first")
        store.rewind(store.listing()[0]["id"])
        assert first.read_text() == "original first"
        assert second.read_text() == "original second"
        assert not store.listing()
    finally:
        await store.aclose()


def test_late_metadata_does_not_duplicate_original_observations(tmp_path):
    history = ConversationHistory()
    calls = [ToolCall("read", "read_file", {"path": "large.txt"})]
    messages = [Message.user("inspect"), Message.assistant("", calls), Message.tool_result("read", "payload " * 10_000)]
    history.sync(messages)
    step = history.steps[0]
    store = TurnStore(tmp_path)
    try:
        store.archive(step, messages)
        raw = (tmp_path / "00000001.raw.json").read_bytes()
        old_view = step.context_messages()[0]
        large_pins = {path for path in tmp_path.rglob("*") if path.is_file() and path.stat().st_size >= len(raw)}
        for record in range(5):
            store.attach_artifact(step, "read", {"record_id": record, "kind": "output"})
        assert (tmp_path / "00000001.raw.json").read_bytes() == raw
        assert large_pins == {path for path in tmp_path.rglob("*") if path.is_file() and path.stat().st_size >= len(raw)}
        metadata = json.loads((tmp_path / "00000001.metadata.json").read_text())
        assert metadata["original"] == "00000001.raw.json"
        assert len(metadata["artifacts"]) == 5
        assert "artifacts" not in materialize(old_view.content)
        assert materialize(step.context_messages()[0].content)["tool_results"][0]["content"] == "payload " * 10_000
        step.request = "remaining input"
        step.messages[0] = Message.user("remaining input")
        store.refresh_input(step)
        assert step.request == "remaining input"
        assert materialize(step.context_messages()[0].content)["user_prompts"] == ["remaining input"]
        assert (tmp_path / "00000001.raw.json").read_bytes() == raw
        assert large_pins == {path for path in tmp_path.rglob("*") if path.is_file() and path.stat().st_size >= len(raw)}
    finally:
        store.close()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_request_profile_rejects_invalid_context_limits(limit):
    with pytest.raises(ValueError, match="positive integer"):
        RequestProfile(parameters=set(), context_length=limit)


async def test_candidate_preview_requires_capabilities():
    agent = Agent(StubClient([]), ToolRegistry(), "current")
    with pytest.raises(ContextError, match="candidate's capabilities"):
        await agent._context_view([], 1, preview=True, candidate_model="candidate")


async def test_preview_does_not_update_live_editable_prompt(workspace, monkeypatch):
    from slipagent import agent as module
    registry = ToolRegistry(services={"workspace": workspace, "editable_prompts": True})
    agent = Agent(StubClient([]), registry, "current", system_prompt="original prompt")
    before = list(agent.messages)
    monkeypatch.setattr(module, "build_system_prompt", lambda *args: "updated prompt")
    await agent._context_view([], 1, preview=True, candidate_model="candidate",
        capabilities=RequestProfile(parameters={"tools"}, context_length=100_000))
    assert agent.messages == before
    assert agent.system_prompt == "original prompt"


async def test_corrections_are_sent_only_in_user_error_channel():
    registry = ToolRegistry()
    registry.context_notes["progress"] = "UNIQUE LOOP CORRECTION"
    agent = Agent(StubClient([]), registry, "test")
    view = await agent._context_view([], 1, repair="UNIQUE REPAIR CORRECTION")
    system = "\n".join(content_text(message.content) for message in view if message.role == "system")
    assert "UNIQUE LOOP CORRECTION" not in system
    assert "UNIQUE REPAIR CORRECTION" not in system
    wire = "\n".join(content_text(message.content) for message in view)
    assert wire.count("UNIQUE LOOP CORRECTION") == 1
    assert wire.count("UNIQUE REPAIR CORRECTION") == 1


async def test_new_compaction_work_uses_a_free_slot(monkeypatch):
    from collections import deque
    compactor = StepCompactor(lambda usage: None, lambda error: None)
    pending = deque([1])
    started = {number: asyncio.Event() for number in (1, 2)}
    release = asyncio.Event()

    async def work(number):
        started[number].set()
        await release.wait()

    def next_job(active, generation):
        if pending:
            number = pending.popleft()
            return ("step", number), work(number)
        return None

    monkeypatch.setattr(compactor, "slots", AsyncMock(return_value=3))
    monkeypatch.setattr(compactor, "_next_job", next_job)
    compactor.dispatch()
    try:
        await asyncio.wait_for(started[1].wait(), 1)
        pending.append(2)
        compactor.dispatch()
        await asyncio.wait_for(started[2].wait(), 1)
        assert not release.is_set()
    finally:
        release.set()
        await asyncio.gather(*tuple(compactor.jobs))


async def test_mcp_stray_stdout_does_not_lose_matching_response():
    client = MCPClient(ServerSpec("audit", "unused"))
    reader = asyncio.StreamReader()
    client._process = SimpleNamespace(stdout=reader, returncode=0)
    response = asyncio.get_running_loop().create_future()
    client._pending[1] = response
    reader.feed_data(b'noisy server startup\n{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n')
    reader.feed_eof()
    await client._read_stdout()
    assert (await response)["result"] == {"ok": True}
    assert any("non-JSON stdout" in line for line in client._stderr_tail)


async def test_mcp_reader_failure_disconnects_and_retains_cause():
    client = MCPClient(ServerSpec("audit", "unused"))
    process = await asyncio.create_subprocess_exec(sys.executable, "-c",
        "import sys; print('x' * 1024, flush=True); sys.stdin.read()",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, limit=128, start_new_session=sys.platform != "win32")
    client._process = process
    client._pump = asyncio.create_task(client._read_stdout())
    try:
        await asyncio.wait_for(client._pump, 2)
        assert not client.connected
        await client.disconnect()
        assert process.returncode is not None
        with pytest.raises(MCPError, match="line longer"):
            await client._send_request("tools/list", {}, None)
    finally:
        await client.disconnect()
