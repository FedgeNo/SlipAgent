"""Manual boundaries change history projection without changing originals."""

import io
import json
from importlib import import_module
from types import SimpleNamespace

import pytest

from data_text_reader import request_input
from slipagent.agent import Agent
from slipagent.cli import CommandRunner, Renderer, Style, command_menu_options
from slipagent.commands import BY_NAME, CommandContext, command_help
from slipagent.context import ConversationHistory, RecallHistoryTool
from slipagent.sessions import SessionError, SessionJournal
from slipagent.tools.base import ToolRegistry, ToolResult
from slipagent.types import Message, ToolCall, content_text
from slipagent.work import SessionWork
from test_agent import RecordingTool, StubClient, call, completion


async def project(history, messages, **options):
    return await history.view(messages, [], keep_steps=50, context_length=1_000_000,
                              max_output=1000, overthinking=False, **options)


def conversation():
    messages = []
    for number in range(1, 4):
        messages.extend([Message.user(f"request {number}"), Message.assistant(f"ORIGINAL {number} " * 1000)])
    history = ConversationHistory()
    history.sync(messages)
    for step in history.steps:
        step.summary = f"summary {step.id}"
    return history, messages


async def test_boundary_compresses_immediately_when_full_history_fits():
    history, messages = conversation()
    before = request_input(await project(history, messages))["history"]
    assert all(record["representation"] == "full" for record in before)
    history.compact_through = 2
    view = await project(history, messages)
    records = request_input(view)["history"]
    assert [record["representation"] for record in records] == ["compressed", "compressed", "full"]
    assert records[0]["compressed_summary"] == "summary 1"
    assert "ORIGINAL 3" in records[2]["agent_response"]
    system = "\n".join(content_text(message.content) for message in view if message.role == "system")
    assert "compact_through" in system and "boundary through step 2" in system
    recalled = await RecallHistoryTool(history).run(step_id=1, section="response")
    assert "ORIGINAL 1" in content_text(recalled.content)
    assert "ORIGINAL 1" in history.steps[0].messages[1].content


async def test_missing_and_larger_summaries_keep_originals_and_late_summaries_apply():
    history, messages = conversation()
    history.compact_through = 3
    history.steps[0].summary = None
    history.steps[1].summary = "larger summary " * 10_000
    records = request_input(await project(history, messages))["history"]
    assert [record["representation"] for record in records] == ["full", "full", "compressed"]
    history.steps[0].summary = "late summary"
    records = request_input(await project(history, messages))["history"]
    assert records[0]["representation"] == "compressed"
    assert records[0]["compressed_summary"] == "late summary"


async def test_unseen_tool_result_is_delivered_before_boundary_compresses_it():
    calls = [ToolCall("read", "read_file", {"path": "file.txt"})]
    messages = [Message.user("Inspect"), Message.assistant("Reading", calls),
                Message.tool_result("read", {"tool": "read_file", "call_id": "read", "status": "success",
                                             "content": "EXACT OUTPUT " * 1000})]
    history = ConversationHistory()
    history.sync(messages)
    history.steps[0].summary = "Read the file"
    history.compact_through = 1
    records = request_input(await project(history, messages))["history"]
    assert records[0]["representation"] == "full"
    assert "EXACT OUTPUT" in content_text(records[0])
    # Merely building the request does not establish delivery.
    assert not history.steps[0].observed
    history.steps[0].observed = True
    records = request_input(await project(history, messages))["history"]
    assert records[0]["representation"] == "compressed"


async def test_manual_compression_does_not_reinsert_original_reasoning():
    history, messages = conversation()
    messages[1].reasoning = "RAW THOUGHTS " * 1000
    history.compact_through = 1
    view = await history.view(messages, [], keep_steps=50, context_length=1_000_000,
                              max_output=1000, overthinking=True)
    record = request_input(view)["history"][0]
    assert record["representation"] == "compressed"
    assert "RAW THOUGHTS" not in content_text(record)
    assert history.steps[0].messages[1].reasoning == "RAW THOUGHTS " * 1000


async def test_narratives_respect_the_boundary():
    fixture = import_module("test-history-merge")
    history, messages = fixture.history_with_posts(20, size=1000)
    fixture.attach_layers(history)
    history.compact_through = 19
    records = request_input(await project(history, messages))["history"]
    assert not any(entry["representation"] == "historical_summary" for entry in records)
    history.compact_through = 20
    records = request_input(await project(history, messages))["history"]
    assert len(records) == 1 and records[0]["step_ids"] == list(range(1, 21))


async def test_manual_boundary_uses_smaller_eras():
    fixture = import_module("test-history-merge")
    history, messages = fixture.history_with_posts(100, size=1000)
    fixture.attach_layers(history)
    history.compact_through = 100
    records = request_input(await project(history, messages))["history"]
    assert len(records) == 1
    assert records[0]["representation"] == "era_summary"
    assert records[0]["step_ids"] == list(range(1, 101))


async def compact_command(agent, argument=""):
    out = io.StringIO()
    context = CommandContext(SimpleNamespace(agent=agent), Style(False), out)
    await BY_NAME["compact"].run(context, argument)
    return out.getvalue()


async def test_boundary_stays_fixed_until_explicit_command_and_reset_clears_it():
    client = StubClient([])
    agent = Agent(client, ToolRegistry(), "test")
    _, messages = conversation()
    agent.messages.extend(messages[:4])
    try:
        output = await compact_command(agent)
        assert agent.history.compact_through == 2
        assert "step 2" in output and "lack individual summaries" in output
        assert not client.calls and not client.summary_calls
        agent.messages.extend(messages[4:])
        agent.history.sync(agent.messages)
        assert agent.history.compact_through == 2
        await compact_command(agent)
        assert agent.history.compact_through == 3
        agent.reset(new_session=False)
        assert agent.history.compact_through == 0
    finally:
        await agent.registry.aclose()


@pytest.mark.parametrize("argument", ["", "unexpected"])
async def test_empty_history_and_invalid_arguments_do_not_set_boundary(argument):
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    output = await compact_command(agent, argument)
    assert agent.history.compact_through == 0
    assert "usage: /compact" in output if argument else "No completed history" in output


def test_command_is_discoverable_and_queues_at_a_batch_boundary():
    runner = CommandRunner(None)
    assert runner.changes_session("compact", "")
    assert not runner.runs_in_background("/compact")
    assert runner.at_boundary("/compact")
    assert "/compact" in command_help("compact")
    assert "/compact" in dict(command_menu_options())


async def test_busy_command_applies_before_the_tool_continuation(monkeypatch):
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    client = StubClient([completion(tool_calls=[call(value="inspect")]), completion("Done")])
    agent = Agent(client, registry, "test")
    history, messages = conversation()
    agent.messages.extend(messages[:2])
    agent.history.sync(agent.messages)
    agent.history.steps[0].summary = "Earlier inspection"
    extensions = {}
    session = SimpleNamespace(agent=agent, registry=registry, extensions=extensions,
        renderer=Renderer(Style(False), io.StringIO(), False), work=SessionWork(extensions))
    runner = CommandRunner(session)

    async def run(**arguments):
        await runner.execute("/compact")
        assert session.work.deferred_commands == ["/compact"]
        assert not agent.stop_requested
        return ToolResult.ok("UNSEEN OBSERVATION " * 1000)

    async def boundary():
        commands = session.work.deferred_commands or []
        while commands and runner.at_boundary(commands[0]):
            await runner.execute(commands.pop(0), at_boundary=True)

    monkeypatch.setattr(tool, "run", run)
    agent.on_boundary = boundary
    try:
        assert await agent.run("Continue inspection") == "Done"
        await agent.wait_for_compaction()
        assert len(client.calls) == 2
        records = request_input(client.calls[1]["messages"])["history"]
        assert records[0]["representation"] == "compressed"
        assert records[1]["representation"] == "full"
        assert "UNSEEN OBSERVATION" in content_text(records[1])
        assert agent.history.compact_through == 2
        view = request_input(await project(agent.history, agent.messages))["history"]
        assert view[1]["representation"] == "compressed"
        assert view[2]["representation"] == "full"
    finally:
        await registry.aclose()


async def test_boundary_survives_resume_and_fork(tmp_path):
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
    agent.registry.services["session_journal"] = journal
    _, messages = conversation()
    agent.messages.extend(messages)
    try:
        await compact_command(agent)
        parent = journal.session_id
        data = journal.load(parent)
        assert data["history"].compact_through == 3
        journal.restore(agent, data)
        assert agent.history.compact_through == 3
        journal.restore(agent, journal.load(parent), fork=True)
        assert journal.session_id != parent
        assert journal.load(journal.session_id)["history"].compact_through == 3
        assert journal.load(parent)["history"].compact_through == 3
    finally:
        await agent.registry.aclose()


@pytest.mark.parametrize("boundary", [-1, 4, True, "1"])
def test_invalid_saved_boundary_is_rejected(tmp_path, boundary):
    agent = Agent(StubClient([]), ToolRegistry(), "test")
    history, messages = conversation()
    agent.messages = messages
    agent.history = history
    journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
    journal.begin(agent)
    with journal.path.open("a") as target:
        target.write(json.dumps({"type": "state", **journal.state, "compact_through": boundary}) + "\n")
    with pytest.raises(SessionError, match="manual compaction boundary"):
        journal.load(journal.session_id)
