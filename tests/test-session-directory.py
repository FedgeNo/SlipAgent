"""A removed, separate launch directory does not change the selected workspace."""

import os
import subprocess
import sys

import pytest

from slipagent.environment import session_environment
from test_cli_e2e import GUARD_DIRECTORY, cli_environment


def test_missing_process_directory_is_metadata_not_a_request_failure(monkeypatch):
    def missing():
        raise FileNotFoundError("launch directory removed")
    monkeypatch.setattr("slipagent.environment.os.getcwd", missing)
    info = session_environment()
    assert info["process_working_directory"] is None
    assert "launch directory removed" in info["process_working_directory_error"]
    assert info["machine"]


@pytest.mark.skipif(os.name != "posix", reason="Deleting the current directory is a POSIX scenario")
def test_live_session_commands_context_and_journal_work_after_launch_directory_is_removed(tmp_path):
    launch, project = tmp_path / "launch", tmp_path / "project"
    launch.mkdir()
    project.mkdir()
    script = '''
import asyncio, json, os, shlex, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import sitecustomize
sys.path.pop(0)
from slipagent.agent import Agent
from slipagent.sessions import SessionJournal
from slipagent.tools import build_default_registry
from slipagent.tools.shell import RunCommandTool
from slipagent.types import Completion, Message, content_text
from slipagent.workspace import Workspace

class Client:
    def __init__(self):
        self.requests = []
    async def chat(self, **kwargs):
        self.requests.append(kwargs)
        return Completion(Message.assistant("Still working."), "offline")

async def main():
    workspace = Workspace(sys.argv[1])
    registry = build_default_registry(workspace)
    journal = SessionJournal(str(workspace.root))
    registry.services["session_journal"] = journal
    client = Client()
    agent = Agent(client, registry, "offline")
    try:
        await agent.run("Before deletion")
        os.rmdir(os.getcwd())
        assert await agent.run("After deletion") == "Still working."
        context = "\\n".join(content_text(message.content) for message in client.requests[-1]["messages"])
        assert 'process_working_directory_error' in context
        assert str(workspace.root) in context
        command = shlex.join([sys.executable, "-I", "-c", "from pathlib import Path; Path('result.txt').write_text('workspace'); print(Path.cwd())"])
        result = await RunCommandTool(workspace).run(command)
        assert not result.is_error, result.content
        assert (workspace.root / 'result.txt').read_text() == 'workspace'
        saved = journal.load(journal.session_id)
        assert len(saved['history'].steps) == 2
    finally:
        await registry.aclose()
asyncio.run(main())
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(project), str(GUARD_DIRECTORY)], cwd=launch,
                            env=cli_environment(SLIPAGENT_STATE_DIR=str(tmp_path / "state")),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    assert not launch.exists()
    assert (project / "result.txt").read_text() == "workspace"
