"""Durable sessions round-trip originals and recover interruptions without replay."""

import json
import os
import shlex
import sys
from importlib import import_module

import pytest

from slipagent.agent import Agent
from slipagent.sessions import SessionError, SessionJournal
from slipagent.tools import build_default_registry
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message, ToolCall
from test_agent import StubClient, RecordingTool, completion
from test_cli_e2e import StubOpenRouter, run_cli, text_step


def attach(agent, workspace, tmp_path):
    journal = SessionJournal(str(workspace.root), tmp_path / "saved")
    agent.registry.services["session_journal"] = journal
    journal.begin(agent)
    return journal


async def test_originals_summaries_reasoning_usage_and_pending_survive_resume(workspace, tmp_path):
    reply = completion("original reply")
    reply.message.reasoning = "private reasoning for archive only"
    client = import_module("test-background-compaction").Client([reply])
    agent = Agent(client, ToolRegistry(), "test", system_prompt="old instructions")
    journal = attach(agent, workspace, tmp_path)
    await agent.run("exact prompt")
    await agent.wait_for_compaction()
    agent.enqueue("queued correction")
    parent = journal.session_id
    old_bytes = journal.path.read_bytes()
    loaded = journal.load(parent)
    agent.system_prompt = "current instructions"
    agent.messages[0] = Message.system("current instructions")
    journal.restore(agent, loaded)
    assert journal.session_id != parent
    assert (journal.directory / (parent + ".jsonl")).read_bytes() == old_bytes
    assert agent.messages[0].content == "current instructions"
    assert agent.pending == ["queued correction"]
    assert agent.history.posts[0].parts()["reasoning"] == "private reasoning for archive only"
    assert agent.history.posts[0].summary
    assert agent.usage.prompt_tokens > 0
    result = await agent.registry.invoke("recall_history", {"post_id": 1, "sections": ["prompt", "response"]})
    assert "exact prompt" in result.content and "original reply" in result.content
    view = await agent._context_view(agent.registry.specs(), 1)
    assert "private reasoning" not in str(view)
    assert agent.history.task.current_prompt_post == 1


@pytest.mark.parametrize("started", [False, True])
async def test_interrupted_tools_are_marked_and_never_replayed(workspace, tmp_path, started):
    tool = RecordingTool()
    agent = Agent(StubClient([]), ToolRegistry([tool]), "test")
    journal = attach(agent, workspace, tmp_path)
    agent.messages += [Message.user("do work"), Message.assistant("planned", [ToolCall("a", "record", {"value": "first"}), ToolCall("b", "record", {"value": "second"})])]
    agent._persist()
    if started:
        journal.tool_started(1, "a")
    journal.restore(agent, journal.load(journal.session_id))
    assert tool.seen == []
    observations = [message for message in agent.messages if message.role == "tool"]
    assert len(observations) == 2
    assert ("outcome unknown" if started else "not started") in observations[0].content
    assert "not started" in observations[1].content
    assert len(agent.history.posts) == 1


async def test_torn_final_record_recovers_but_complete_corruption_is_rejected(workspace, tmp_path):
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    journal = attach(agent, workspace, tmp_path)
    with journal.path.open("ab") as target:
        target.write(b'{"type":')
    assert journal.load(journal.session_id)["torn"]
    with journal.path.open("ab") as target:
        target.write(b'broken}\n')
    with pytest.raises(SessionError, match="not modified"):
        journal.load(journal.session_id)


async def test_failure_to_save_start_marker_prevents_tool_execution(workspace, tmp_path, monkeypatch):
    tool = RecordingTool()
    agent = Agent(StubClient([completion("run", [ToolCall("a", "record", {"value": "x"})])]), ToolRegistry([tool]), "test")
    journal = attach(agent, workspace, tmp_path)
    append = journal._append
    def fail(kind, **data):
        if kind == "tool_start":
            raise SessionError("simulated disk failure")
        append(kind, **data)
    monkeypatch.setattr(journal, "_append", fail)
    with pytest.raises(SessionError, match="disk failure"):
        await agent.run("change something")
    assert tool.seen == []


async def test_command_logs_survive_exit_reset_and_resume(workspace, tmp_path):
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([]), registry, "test")
    journal = attach(agent, workspace, tmp_path)
    command = shlex.join([sys.executable, "-c", "print('retained output')"])
    result = await registry.invoke("run_command", {"command": command})
    assert not result.is_error
    archive = registry.services["command_archive"]
    log_id = next(iter(archive.logs))
    saved = journal.session_id
    agent.reset()
    assert not archive.logs
    journal.restore(agent, journal.load(saved))
    result = await registry.invoke("read_command_output", {"log_id": log_id})
    assert "retained output" in result.content and not result.is_error
    path = archive.logs[log_id].streams["stdout"].path
    current = journal.session_id
    original_size = path.stat().st_size
    await registry.aclose()
    assert path.exists()
    with path.open("r+b") as stream:
        stream.truncate(3)
    # Restore detects storage loss even when the journal itself is complete.
    registry = build_default_registry(workspace)
    restored = Agent(StubClient([]), registry, "test")
    replacement = SessionJournal(str(workspace.root), tmp_path / "saved")
    registry.services["session_journal"] = replacement
    replacement.restore(restored, replacement.load(current))
    try:
        page = json.loads((await registry.invoke("read_command_output", {"log_id": log_id})).content)
        assert page["lost_bytes"] == original_size - 3
        assert "shorter than its journal" in page["retention_error"]
    finally:
        await registry.aclose()


def test_project_keys_preserve_symlink_spelling_and_private_paths(workspace, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(workspace.root, target_is_directory=True)
    first = SessionJournal(str(workspace.root), tmp_path / "state")
    second = SessionJournal(str(alias), tmp_path / "state")
    assert first.directory != second.directory
    assert len(first.directory.name) == 32 + 1 + 40
    assert first.directory.is_absolute() and "~" not in str(first.directory)
    if os.name == "posix":
        assert first.directory.stat().st_mode & 0o077 == 0


def test_project_key_does_not_collapse_parent_segments_across_symlinks(tmp_path):
    # link/.. can name a different directory from its lexical simplification.
    raw = str(tmp_path / "link" / ".." / "project")
    first = SessionJournal(raw, tmp_path / "state")
    second = SessionJournal(str(tmp_path / "project"), tmp_path / "state")
    assert first.project == raw
    assert first.directory != second.directory


def test_cli_resume_sends_previous_history_to_next_request(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    with StubOpenRouter([text_step("original answer"), text_step("continued answer")]) as stub:
        first = run_cli("-p", "original question", "--base-url", stub.base_url, cwd=project)
        assert first.returncode == 0, first.stderr
        second = run_cli("--resume", "latest", "-p", "continue", "--base-url", stub.base_url, cwd=project)
        assert second.returncode == 0, second.stderr
    assert "continued answer" in second.stdout
    request = json.dumps(stub.requests[-1])
    assert "original question" in request and "original answer" in request
