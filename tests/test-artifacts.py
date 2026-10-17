"""Inspection records preserve literal data, provenance, and session boundaries."""

import json
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from slipagent.artifacts import ArtifactStore, ReadArtifactTool
from slipagent.agent import Agent
from slipagent.context import CompletedStep
from slipagent.records import linked_tool_results, step_record
from slipagent.sessions import SessionJournal
from slipagent.tools import build_default_registry
from slipagent.tools.discovery import tool_catalog
from slipagent.tools.output import CommandArchive, ReadCommandOutputTool
from slipagent.types import Message, ToolCall
from test_agent import StubClient, completion


async def test_literal_json_records_retrieve_by_number_and_reject_paths(tmp_path):
    store = ArtifactStore(1_000_000)
    store.use_directory(tmp_path / "artifacts")
    record = store.save("printf '%s' \"a\\b\"\nprintf next", "{'x': '\\value'}\nα\n", kind="command_output",
                        error="stderr\n", step_id=7, call_id="call/one")
    path = store.directory / record["filename"]
    raw = json.loads(path.read_text())
    assert raw == {"command": record["command"], "output": "{'x': '\\value'}\nα\n", "error": "stderr\n",
                   "step_id": 7, "call_id": "call/one", "kind": "command_output"}
    assert path.stem.isdecimal()
    tool = ReadArtifactTool(store)
    assert (await tool.run(record["record_id"])).content["output"] == raw["output"]
    assert (await tool.run(1)).is_error
    assert (await tool.invoke({"record_id": "../../secret"})).is_error
    assert len((await tool.run()).content["records"]) == 1
    restored = ArtifactStore(1_000_000)
    restored.use_directory(store.directory)
    assert restored.read(record["record_id"])["error"] == "stderr\n"
    fork = ArtifactStore(1_000_000)
    fork.use_directory(tmp_path / "fork", source=store.directory)
    assert fork.read(record["record_id"])["call_id"] == "call/one"
    restored.clear()
    assert path.exists()
    with pytest.raises(ValueError):
        restored.read(record["record_id"])


def test_records_handle_collisions_concurrency_quota_and_missing_files(tmp_path, monkeypatch):
    store = ArtifactStore(100_000)
    store.use_directory(tmp_path)
    monkeypatch.setattr("slipagent.artifacts.time.time_ns", lambda: 100_000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda number: store.save("command", str(number), kind="command_output"), range(12)))
    assert len({record["record_id"] for record in records}) == 12
    assert len(json.loads((tmp_path / "index.json").read_text())) == 12
    target = records[0]
    (tmp_path / target["filename"]).unlink()
    with pytest.raises(OSError):
        store.read(target["record_id"])
    small = ArtifactStore(1)
    with pytest.raises(OSError, match="quota"):
        small.save("command", "output", kind="command_output")


async def test_command_records_keep_streams_separate_without_changing_log_offsets(tmp_path):
    store = ArtifactStore(1_000_000)
    store.use_directory(tmp_path / "records")
    archive = CommandArchive(1_000_000, directory=tmp_path)
    archive.artifacts = store
    log = archive.start("some command", step_id=3, call_id="call-3")
    log.write("stdout", "out\n")
    log.write("stderr", "err\n")
    log.finish(1, False)
    reference = log.metadata()["artifacts"][0]
    record = store.read(reference["record_id"])
    assert (record["output"], record["error"]) == ("out\n", "err\n")
    assert (await ReadCommandOutputTool(archive).run(log_id=log.id)).content["content"] == "out\n"
    log.finish(1, False)
    assert len(store.records) == 1
    await archive.aclose()


async def test_agent_edit_records_survive_compression_resume_and_fork(workspace, tmp_path):
    registry = build_default_registry(workspace)
    tool_catalog(registry).loaded.add("write_file")
    call = ToolCall("edit-1", "write_file", {"path": "new.txt", "content": "new contents\n"})
    agent = Agent(StubClient([completion(tool_calls=[call]), completion("Done")]), registry, "test")
    journal = SessionJournal(str(workspace.root), tmp_path / "sessions")
    registry.services["session_journal"] = journal
    journal.begin(agent)
    try:
        await agent.run("Write a file")
        await agent.wait_for_compaction()
        step = agent.history.steps[0]
        record = step_record(step.messages)
        reference = record["artifacts"][0]
        assert reference["call_id"] == "edit-1"
        assert "+new contents" in registry.services["artifact_store"].read(reference["record_id"])["output"]
        assert linked_tool_results(record)["tool_calls"][0]["artifacts"] == record["artifacts"]
        assert step.context_messages(compressed=True)[0].content["artifacts"] == record["artifacts"]
        saved = journal.load(journal.session_id)
        for fork in (False, True):
            restored_registry = build_default_registry(workspace)
            restored = Agent(StubClient([]), restored_registry, "test")
            other = SessionJournal(str(workspace.root), tmp_path / "sessions")
            restored_registry.services["session_journal"] = other
            try:
                other.restore(restored, saved, fork=fork)
                artifact = await restored_registry.invoke("read_artifact", {"record_id": reference["record_id"]})
                assert not artifact.is_error
                assert artifact.content["call_id"] == "edit-1"
                assert restored.history.steps[0].context_messages(compressed=True)[0].content["artifacts"]
            finally:
                await restored_registry.aclose()
    finally:
        await registry.aclose()


async def test_late_background_output_reference_is_attached_and_journalled(workspace, tmp_path):
    from slipagent.jobs import CommandJobsTool
    registry = build_default_registry(workspace)
    command = shlex.join([sys.executable, "-c", "import time; time.sleep(.1); print('background output')"])
    call = ToolCall("background-1", "run_command", {"command": command, "background": True})
    agent = Agent(StubClient([completion(tool_calls=[call]), completion("Started")]), registry, "test")
    journal = SessionJournal(str(workspace.root), tmp_path / "sessions")
    registry.services["session_journal"] = journal
    journal.begin(agent)
    try:
        await agent.run("Start background work")
        jobs = registry.services["command_jobs"]
        identifier = next(iter(jobs.jobs))
        await CommandJobsTool(jobs).run("wait", identifier)
        agent._notify_jobs()
        record = step_record(agent.history.steps[0].messages)
        reference = record["artifacts"][0]
        assert reference["call_id"] == "background-1"
        assert registry.services["artifact_store"].read(reference["record_id"])["output"] == "background output\n"
        saved = journal.load(journal.session_id)
        assert step_record(saved["history"].steps[0].messages)["artifacts"] == record["artifacts"]
    finally:
        await registry.aclose()


def test_version_two_pages_seek_and_detect_changed_payloads(tmp_path):
    import os

    output = tmp_path / "stdout"
    text = "".join(f"{number} é😀 line\n" for number in range(20_000))
    output.write_text(text, encoding="utf-8")
    store = ArtifactStore(10_000_000)
    store.use_directory(tmp_path / "records")
    record = store.save_command("command", {"stdout": output, "stderr": None}, kind="command_output",
                                step_id=1, call_id="call", error_suffix="[suffix]")
    pages, offset = [], 0
    while offset is not None:
        page = store.read_page(record["record_id"], "output", offset, 7_001)
        pages.append(page["output"])
        offset = page["next_offset"]
    assert "".join(pages) == text
    error = store.read_page(record["record_id"], "error", 2, 100)
    assert error["error"] == "uffix]" and error["eof"] and error["total_characters"] == 8
    payload = store.directory / "payloads" / record["payloads"]["output"]["sha256"]
    payload.write_bytes(payload.read_bytes().replace(b"19999", b"19998"))
    os.utime(payload, ns=(1, 1))
    with pytest.raises(ValueError, match="integrity"):
        store.read_page(record["record_id"], "output", 0, 10)
