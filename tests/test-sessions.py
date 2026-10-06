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
from test_cli_e2e import StubOpenRouter, run_cli, run_repl_commands, text_step, metadata_server


def attach(agent, workspace, tmp_path):
    journal = SessionJournal(str(workspace.root), tmp_path / "saved")
    agent.registry.services["session_journal"] = journal
    journal.begin(agent)
    return journal


async def test_startup_commands_do_not_create_session_before_prompt(tmp_path, metadata_server):
    from slipagent import cli
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    journal = session.registry.services["session_journal"]
    try:
        assert journal.path is None
        assert journal.listing() == []
        for command in ("/help", "/sessions", "/overthinking off", "/rename First task"):
            await cli._handle_command(session, command)
        assert journal.path is None
        assert journal.listing() == []
        session.agent.client = StubClient([completion("Answer")])
        await session.agent.run("First prompt")
        await session.agent.wait_for_compaction()
        assert len(journal.listing()) == 1
        assert journal.title == "First task | SlipAgent"
        saved = journal.load(journal.session_id)
        assert any(message.role == "user" and message.content == "First prompt" for message in saved["messages"])
    finally:
        await cli._shutdown(session)


@pytest.mark.parametrize("startup_resume", [False, True])
async def test_startup_and_resume_create_no_empty_session(tmp_path, metadata_server, startup_resume):
    from slipagent import cli
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    journal = SessionJournal(str(tmp_path))
    agent.messages.append(Message.user("Saved prompt"))
    journal.begin(agent)
    parent = journal.session_id
    original = journal.path.read_bytes()
    args = ["--no-mcp", "-w", str(tmp_path)]
    if startup_resume:
        args += ["--resume", parent]
    session = await cli.build_session(cli.build_parser().parse_args(args))
    active = session.registry.services["session_journal"]
    try:
        if not startup_resume:
            assert len(active.listing()) == 1
            await cli._handle_command(session, f"/resume {parent}")
        assert len(active.listing()) == 2
        assert journal.path.read_bytes() == original
        assert all(any(message.role == "user" for message in active.load(entry["id"])["messages"])
                   for entry in active.listing())
    finally:
        await cli._shutdown(session)


async def test_fork_preserves_parent_and_continues_in_new_session(tmp_path, metadata_server):
    from slipagent import cli
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    journal = session.registry.services["session_journal"]
    session.agent.client = StubClient([completion("First answer"), completion("Fork answer")])
    try:
        await session.agent.run("Original prompt")
        await session.agent.wait_for_compaction()
        journal.rename("Original title")
        command = shlex.join([sys.executable, "-c", "print('forked output')"])
        result = await session.registry.invoke("run_command", {"command": command})
        assert not result.is_error
        archive = session.registry.services["command_archive"]
        log_id = next(iter(archive.logs))
        parent = journal.session_id
        parent_path = journal.path
        original = parent_path.read_bytes()
        await cli._handle_command(session, "/fork")
        assert journal.session_id != parent
        assert journal.title == "Original title | SlipAgent"
        assert len(journal.listing()) == 2
        assert journal.load(journal.session_id)["messages"][-1].content == "First answer"
        retained = await session.registry.invoke("read_command_output", {"log_id": log_id})
        assert "forked output" in retained.content and not retained.is_error
        await session.agent.run("New branch prompt")
        await session.agent.wait_for_compaction()
        assert parent_path.read_bytes() == original
        assert any(message.content == "New branch prompt" for message in journal.load(journal.session_id)["messages"])
        assert all(message.content != "New branch prompt" for message in journal.load(parent)["messages"])
    finally:
        await cli._shutdown(session)


@pytest.mark.parametrize("confirmation", [None, "cancel", "delete"])
async def test_delete_confirmation_removes_only_current_session(tmp_path, metadata_server, confirmation):
    import io
    from unittest.mock import AsyncMock, Mock
    from slipagent import cli
    from slipagent.terminal import TerminalUI
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    journal = session.registry.services["session_journal"]
    session.agent.client = StubClient([completion("Answer")])
    terminal = Mock(spec=TerminalUI)
    terminal.choose = AsyncMock(return_value=confirmation)
    output = io.StringIO()
    terminal.write.side_effect = lambda text, **kwargs: output.write(text)
    try:
        await session.agent.run("Original prompt")
        await session.agent.wait_for_compaction()
        parent = journal.session_id
        parent_path = journal.path
        original = parent_path.read_bytes()
        await cli._handle_command(session, "/fork")
        journal.rename("Delete this copy")
        current = journal.session_id
        current_path = journal.path
        current_original = current_path.read_bytes()
        # Disposable files stand in for all stored sidecar content.
        for suffix in ("-logs", "-requests"):
            directory = journal.directory / (current + suffix)
            directory.mkdir(exist_ok=True)
            (directory / "retained-data").write_text("Private session data")
        session.renderer.terminal = terminal
        await cli._handle_command(session, "/delete")
        terminal.choose.assert_awaited_once()
        assert "Delete this copy" in terminal.choose.call_args.args[0]
        assert parent_path.read_bytes() == original
        if confirmation == "delete":
            assert journal.path is None and journal.session_id == ""
            assert [entry["id"] for entry in journal.listing()] == [parent]
            assert not list(journal.directory.glob(current + "*"))
            assert all(message.role == "system" for message in session.agent.messages)
            assert not session.agent.pending and not session.agent.history.posts
            terminal.clear_transcript.assert_called_once()
            assert "SlipAgent — OpenRouter compatible coding agent" in output.getvalue()
            await cli._handle_command(session, "/sessions")
            assert journal.path is None
        else:
            assert current_path.read_bytes() == current_original
            assert journal.session_id == current
            assert len(journal.listing()) == 2
            terminal.clear_transcript.assert_not_called()
    finally:
        await cli._shutdown(session)


async def test_fork_and_delete_require_saved_session_and_delete_requires_terminal(tmp_path, metadata_server):
    from slipagent import cli
    session = await cli.build_session(cli.build_parser().parse_args(["--no-mcp", "-w", str(tmp_path)]))
    journal = session.registry.services["session_journal"]
    try:
        await cli._handle_command(session, "/fork")
        await cli._handle_command(session, "/delete")
        assert journal.path is None
        session.agent.messages.append(Message.user("Prompt"))
        session.agent._persist()
        original = journal.path.read_bytes()
        await cli._handle_command(session, "/delete")
        assert journal.path.read_bytes() == original
    finally:
        await cli._shutdown(session)


def test_session_titles_survive_reopen_resume_and_reset(workspace, tmp_path):
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    journal = attach(agent, workspace, tmp_path)
    default = f"{workspace.root} | SlipAgent"
    assert journal.title == default
    assert journal.listing()[0]["title"] == default
    original = journal.path.read_bytes()
    parent = journal.session_id
    journal.rename("Review café changes")
    title = "Review café changes | SlipAgent"
    assert journal.title == title
    assert journal.path.read_bytes() == original
    assert journal.listing()[0]["title"] == title
    if os.name == "posix":
        assert (journal.directory / (parent + ".title.json")).stat().st_mode & 0o077 == 0

    reopened = SessionJournal(str(workspace.root), tmp_path / "saved")
    restored = Agent(StubClient([]), ToolRegistry(), "test")
    restored.registry.services["session_journal"] = reopened
    reopened.restore(restored, reopened.load(parent))
    child = reopened.session_id
    assert child != parent and reopened.title == title
    assert reopened.load(child)["title"] == title
    reopened.rename("Follow-up")
    assert reopened.load(parent)["title"] == title
    assert reopened.load(child)["title"] == "Follow-up | SlipAgent"
    restored.reset()
    assert reopened.title == default
    assert reopened.load(child)["title"] == "Follow-up | SlipAgent"


def test_legacy_session_title_defaults_to_full_project_path(workspace, tmp_path):
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    journal = attach(agent, workspace, tmp_path)
    lines = journal.path.read_text().splitlines()
    header = json.loads(lines[0])
    del header["title"]
    lines[0] = json.dumps(header)
    journal.path.write_text("\n".join(lines) + "\n")
    assert journal.load(journal.session_id)["title"] == f"{workspace.root} | SlipAgent"
    assert journal.listing()[0]["title"] == f"{workspace.root} | SlipAgent"


def test_failed_rename_keeps_previous_saved_title(workspace, tmp_path, monkeypatch):
    from pathlib import Path
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    journal = attach(agent, workspace, tmp_path)
    journal.rename("Original")
    def fail(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(SessionError, match="Could not save"):
        journal.rename("Replacement")
    assert journal.title == "Original | SlipAgent"
    assert journal.load(journal.session_id)["title"] == journal.title
    assert not list(journal.directory.glob(".title-*"))


@pytest.mark.parametrize("name", ["", "   ", "bad\nname", "bad\x1b]2;title\x07", "bad\x9cname"])
def test_invalid_names_do_not_change_saved_title(workspace, tmp_path, name):
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    journal = attach(agent, workspace, tmp_path)
    before = journal.title
    with pytest.raises(SessionError, match="control characters"):
        journal.rename(name)
    assert journal.title == journal.load(journal.session_id)["title"] == before


@pytest.mark.parametrize("contents", ['{"title":', '{}', '{"title": 123}'])
def test_invalid_title_metadata_is_reported(workspace, tmp_path, contents):
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    journal = attach(agent, workspace, tmp_path)
    (journal.directory / (journal.session_id + ".title.json")).write_text(contents)
    with pytest.raises(SessionError):
        journal.load(journal.session_id)
    with pytest.raises(SessionError):
        journal.listing()


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
    assert "private reasoning for archive only" in str(view)
    agent.overthinking = False
    view = await agent._context_view(agent.registry.specs(), 1)
    assert "private reasoning for archive only" not in str(view)
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


def test_startup_resume_displays_saved_conversation_without_inference(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    with StubOpenRouter([text_step("Original saved answer")]) as stub:
        first = run_cli("-p", "Original saved question", "--base-url", stub.base_url, cwd=project)
        assert first.returncode == 0, first.stderr
        resumed = run_repl_commands(project, [], "--base-url", stub.base_url, "--resume", "latest")
        assert resumed.returncode == 0, resumed.stderr
        assert "> Original saved question" in resumed.stderr
        assert "Original saved answer" in resumed.stderr
        assert len(stub.requests) == 1
