"""Regression coverage for the remaining single-fix audit findings."""

import asyncio
import os
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slipagent.agent import Agent, STEP_LIMIT_NOTICE, STOP_NOTICE
from slipagent.context import CompletedStep
from slipagent.mcp import MCPError, MCPManager, MCPToolInfo, ServerSpec, ServerState, tool_alias
from slipagent.runtime import RuntimeFrame
from slipagent.sessions import SessionError, SessionJournal
from slipagent.tools.base import ToolRegistry
from slipagent.tools.git import GitOutput, GitRepository, GitStatusTool, GitTool
from slipagent.tools.shell import CommandOutput, RunCommandTool
from slipagent.types import Message, content_text
from slipagent.workspace import WorkspaceError
from test_agent import RecordingTool, StubClient, completion


async def test_decoded_loop_keeps_the_response_in_the_failed_turn(monkeypatch):
    """L-03: failure history retains the same reply that diagnostics receive."""
    from slipagent import requests
    from slipagent.progress import StreamLoopError

    class Guard:
        def feed(self, kind, text):
            if text == "decoded looping reply":
                raise StreamLoopError("repeated reply")

        def finish(self):
            pass

    monkeypatch.setattr(requests, "StreamLoopGuard", Guard)
    monkeypatch.setattr(requests, "parse_agent_response",
                        lambda *args, **kwargs: SimpleNamespace(text="decoded looping reply", calls=[]))
    registry = ToolRegistry()
    agent = Agent(StubClient([completion("Original reply")]), registry, "test")
    try:
        await agent.run("Inspect")
        failure = agent.history.steps[-1].request_failure
        assert failure["outcome"] == "loop_stopped"
        assert "Original reply" in failure["response_excerpt"]
    finally:
        await registry.aclose()


@pytest.mark.parametrize("stopped", [False, True])
async def test_stop_and_limit_notices_use_history_step_ids(stopped):
    """L-10: warning IDs follow history, independently of run request counts."""
    events = []
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, "test", max_steps=1, on_event=events.append)
    for number in range(1, 4):
        messages = [Message.user("Earlier"), Message.assistant("Done")]
        agent.messages.extend(messages)
        agent.history.steps.append(CompletedStep(number, "Earlier", messages))
    agent.history.cursor = len(agent.messages)
    agent.stop_requested = stopped
    agent._requests = 0 if stopped else 1
    try:
        answer = await agent._run("")
        expected = STOP_NOTICE if stopped else STEP_LIMIT_NOTICE
        assert answer == expected
        assert [(event.step, event.text) for event in events if event.kind == "warning"] == [(4, expected)]
    finally:
        await registry.aclose()


def test_unknown_process_exit_is_never_rendered_as_success():
    """L-14: shell, Git output, and Git validation preserve unknown status."""
    shell = CommandOutput()._render("command", None, "", "")
    git = GitOutput(["git", "status"], None, "", "")
    for result in (shell, git.result()):
        assert result.is_error
        assert "exit code: unknown" in content_text(result.content)
        assert "completed successfully" not in content_text(result.content)
    with pytest.raises(WorkspaceError, match="exit code: unknown"):
        GitTool._complete(git)


async def test_timed_out_process_callers_preserve_unknown_exit_status(workspace, monkeypatch):
    """L-14: callers must not convert a missing return code into zero."""
    from slipagent.tools import git, shell
    from slipagent.workspace import WorkspaceError

    process = SimpleNamespace(returncode=None)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", AsyncMock(return_value=process))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(shell, "capture_process", AsyncMock(return_value=("", "", True)))
    monkeypatch.setattr(git, "capture_process", AsyncMock(return_value=("", "", True)))
    result = await RunCommandTool(workspace).run(command="echo test", timeout=1)
    assert result.is_error and "exit code: unknown" in content_text(result.content)
    tool = GitStatusTool(workspace)
    with pytest.raises(WorkspaceError, match="exit code: unknown"):
        await tool._execute(GitRepository(workspace.root, workspace.root / ".git"), ["status"], 1)


@pytest.mark.skipif(os.name != "posix", reason="Directory fsync is a POSIX durability contract")
async def test_new_journal_and_private_directories_sync_their_entries(tmp_path, workspace, monkeypatch):
    """L-19/C-09: journal durability includes directory entries."""
    synced = []
    fsync = os.fsync

    def capture(descriptor):
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            synced.append(metadata.st_ino)
        fsync(descriptor)

    monkeypatch.setattr(os, "fsync", capture)
    registry = ToolRegistry()
    agent = Agent(StubClient([]), registry, "test")
    try:
        journal = SessionJournal(str(workspace.root), tmp_path / "state")
        assert (tmp_path / "state").stat().st_ino in synced
        synced.clear()
        journal.begin(agent)
        assert journal.directory.stat().st_ino in synced
        await journal.flush()
    finally:
        await registry.aclose()


async def test_journal_directory_sync_failure_stops_recording(tmp_path, workspace, monkeypatch):
    from slipagent import sessions

    journal = SessionJournal(str(workspace.root), tmp_path / "state")
    def fail_sync(path):
        raise OSError("sync failed")

    monkeypatch.setattr(sessions, "_sync_directory", fail_sync)
    registry = ToolRegistry()
    try:
        with pytest.raises(SessionError, match="sync failed"):
            journal.begin(Agent(StubClient([]), registry, "test"))
        assert journal.failed
    finally:
        await registry.aclose()


async def test_all_mcp_name_collisions_are_reported(tmp_path):
    """L-28: every skipped name remains visible in the server's status."""
    registry = ToolRegistry()
    manager = MCPManager(tmp_path, registry=registry)
    spec = ServerSpec(name="stub", command="unused")
    client = SimpleNamespace(spec=spec)
    state = ServerState(spec, client)
    try:
        for name in ("one", "two"):
            tool = RecordingTool()
            tool.name = tool_alias("stub", name)
            registry.register(tool)
        manager._register("stub", client, [MCPToolInfo(name, name, {"type": "object"})
                                            for name in ("one", "two", "three")], state)
        assert state.qualified == [tool_alias("stub", "three")]
        assert "skipped 2 colliding tools" in state.detail
        assert all(tool_alias("stub", name) in state.detail for name in ("one", "two"))
    finally:
        await registry.aclose()


async def test_mcp_shutdown_attempts_every_server_after_a_failure(tmp_path):
    """L-29: one server's shutdown failure cannot strand another server."""
    manager = MCPManager(tmp_path)
    broken = SimpleNamespace(disconnect=AsyncMock(side_effect=MCPError("stuck process")))
    healthy = SimpleNamespace(disconnect=AsyncMock())
    manager.servers = {name: ServerState(ServerSpec(name=name, command="unused"), client)
                       for name, client in (("broken", broken), ("healthy", healthy))}
    with pytest.raises(MCPError, match="broken: stuck process"):
        await manager.aclose()
    assert broken.disconnect.await_count == healthy.disconnect.await_count == 1


async def test_mcp_connection_error_retains_the_original_and_cleanup_failures(tmp_path, monkeypatch):
    """L-29: failing cleanup must not mask the reason connection failed."""
    from slipagent import mcp

    client = SimpleNamespace(connect=AsyncMock(side_effect=MCPError("handshake failed")),
                             disconnect=AsyncMock(side_effect=MCPError("stop failed")))
    monkeypatch.setattr(mcp, "MCPClient", lambda spec: client)
    manager = MCPManager(tmp_path)
    state = await manager.connect("stub", ServerSpec(name="stub", command="unused"))
    assert state.status == "error"
    assert "handshake failed" in state.detail and "cleanup failed: stop failed" in state.detail


def runtime_frame():
    frame = RuntimeFrame.__new__(RuntimeFrame)
    frame.session = SimpleNamespace(agent=SimpleNamespace(running=False))
    frame.busy = frame.reloading = False
    frame.force = True
    frame.generation = 0
    frame._applied = {}
    frame._attempted = frame._signature({})
    return frame


@pytest.mark.parametrize("committed", [False, True])
async def test_reload_diagnostics_match_the_committed_generation(committed, monkeypatch):
    """L-30: a post-commit reporting failure is not a rejected reload."""
    frame = runtime_frame()
    notices = []

    async def fail(sources, **kwargs):
        if committed:
            frame.generation += 1
        raise RuntimeError("report failed")

    monkeypatch.setattr(frame, "_reload", fail)
    monkeypatch.setattr(frame, "_notice", lambda text, **kwargs: notices.append(text))
    await frame.checkpoint(sources={})
    assert ("Reload applied (generation 1)" in notices[-1]) == committed
    assert ("Reload rejected" in notices[-1]) != committed


async def test_forced_reload_survives_deferral_with_unchanged_sources(monkeypatch):
    """L-31: staging while the session becomes busy preserves /reload."""
    frame = runtime_frame()
    monkeypatch.setattr(frame, "_stage", lambda *args: {})

    def overlay(*args):
        frame.busy = True
        return []

    monkeypatch.setattr(frame, "_overlay", overlay)
    await frame.checkpoint(sources={})
    assert frame.force and frame.generation == 0
    frame.busy = False
    reload = AsyncMock(return_value=True)
    monkeypatch.setattr(frame, "_reload", reload)
    await frame.checkpoint(sources={})
    assert reload.await_count == 1 and not frame.force


@pytest.mark.parametrize("committed", [False, True])
async def test_forced_reload_survives_cancellation_only_before_commit(committed, monkeypatch):
    """L-31: do not lose pending requests or repeat an applied generation."""
    frame = runtime_frame()

    async def cancel(sources, **kwargs):
        if committed:
            frame.generation += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(frame, "_reload", cancel)
    with pytest.raises(asyncio.CancelledError):
        await frame.checkpoint(sources={})
    assert frame.force != committed
    assert not frame.reloading
