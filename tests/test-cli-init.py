"""Initialization creates only confined guidance and applies it to the session."""

import io
from types import SimpleNamespace

import pytest

from slipagent.agent import Agent, build_system_prompt
from slipagent.cli import Renderer, Style, _execute_command, _init_command
from slipagent.instructions import load_project_instructions
from slipagent.tools import ToolRegistry
from slipagent.types import Message
from slipagent.workspace import Workspace, WorkspaceError


def session_for(workspace):
    prompt = build_system_prompt(str(workspace.root))
    agent = Agent(client=None, registry=ToolRegistry(), model="test", system_prompt=prompt)
    return SimpleNamespace(
        workspace=workspace, agent=agent,
        renderer=Renderer(Style(False), io.StringIO(), False),
        reloader=SimpleNamespace(_project_instructions=prompt),
    )


async def test_init_creates_only_project_scaffold_and_applies_it(workspace):
    session = session_for(workspace)
    session.agent.messages.append(Message.user("Keep my conversation."))
    await _init_command(session, Style(False), io.StringIO())
    assert sorted(path.name for path in workspace.root.iterdir()) == ["AGENTS.md"]
    content = (workspace.root / "AGENTS.md").read_text()
    assert "SlipAgent" not in content
    expected = build_system_prompt(str(workspace.root))
    assert session.agent.system_prompt == expected
    assert session.agent.messages[0].content == expected
    assert session.agent.messages[1].content == "Keep my conversation."
    assert session.reloader._project_instructions == expected
    context = await session.agent._context_view(session.agent.registry.specs(), 1)
    assert content in "\n".join(message.content or "" for message in context)
    session.agent.reset()
    assert session.agent.messages[0].content == expected


async def test_init_preserves_and_loads_existing_user_guidance(workspace):
    path = workspace.root / "AGENTS.md"
    path.write_text("Custom project instructions.\n")
    session = session_for(workspace)
    await _init_command(session, Style(False), io.StringIO())
    assert path.read_text() == "Custom project instructions.\n"
    context = await session.agent._context_view(session.agent.registry.specs(), 1)
    assert "Custom project instructions." in "\n".join(message.content or "" for message in context)


@pytest.mark.parametrize("existing", [False, True])
async def test_init_rejects_outside_guidance_symlink(tmp_path, existing):
    root = tmp_path / "project"
    root.mkdir()
    target = tmp_path / "outside.md"
    if existing:
        target.write_text("Outside instructions.")
    (root / "AGENTS.md").symlink_to(target)
    session = session_for(Workspace(root))
    original = session.agent.system_prompt
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        await _init_command(session, Style(False), io.StringIO())
    assert target.exists() == existing
    if existing:
        assert target.read_text() == "Outside instructions."
    assert session.agent.system_prompt == original


async def test_init_waits_for_active_turn_to_finish(workspace):
    session = session_for(workspace)
    session.agent.running = True
    assert await _execute_command(session, "/init") is False
    assert not (workspace.root / "AGENTS.md").exists()
    assert "use it after this turn finishes" in session.renderer.stream.getvalue()
