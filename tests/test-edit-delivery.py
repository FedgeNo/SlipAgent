"""Saved edits fit bounded requests and acknowledge only accepted revisions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from data_text_reader import request_input
from slipagent.agent import Agent
from slipagent.artifacts import ArtifactStore, ReadArtifactTool
from slipagent.context import message_tokens
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry
from slipagent.useredits import UserEdits
from test_agent import StubClient, completion


def setup_edits(tmp_path, responses, **options):
    store = ArtifactStore(10_000_000)
    store.use_directory(tmp_path / "artifacts")
    registry = ToolRegistry([ReadArtifactTool(store)])
    registry.resources.artifacts = store
    client = StubClient(responses)
    agent = Agent(client, registry, "test", **options)
    terminal = SimpleNamespace(_touched_items={}, write_artifact_reference=Mock(), notify=Mock(), write=Mock())
    edits = UserEdits(SimpleNamespace(agent=agent, registry=registry), terminal)
    return agent, client, edits, store


async def test_large_saved_diff_fits_and_retires_into_retrievable_history(tmp_path):
    agent, client, edits, store = setup_edits(tmp_path, [completion("Done"), completion("Next")],
                                             context_tokens=16_000, max_tokens=512)
    diff = "@@ -1 +1 @@\n" + "-before\n+after α 👩‍💻\n" * 20_000
    edits.record(tmp_path / "file", diff, "edit")
    assert await agent.run("Continue") == "Done"
    payload = request_input(client.calls[0]["messages"])
    notice = payload["user_edits"][0]
    assert len(notice["diff"]) < len(diff) and notice["diff_truncated"]
    assert message_tokens(client.calls[0]["messages"]) < 16_000
    assert store.read(notice["artifact"]["record_id"])["output"] == diff
    assert await agent.run("Next") == "Next"
    assert "user_edits" not in request_input(client.calls[-1]["messages"])
    assert agent.history.steps[0].parts()["user_edits"][0]["revision"] == notice["revision"]
    assert agent.history.steps[0].context_messages(compressed=True)[0].content["artifacts"]
    await agent.registry.aclose()


async def test_large_edit_projection_alone_fits_an_8192_token_context():
    from slipagent.context import ConversationHistory
    from slipagent.types import Message
    history = ConversationHistory()
    view = await history.view([Message.user("Continue")], [], keep_steps=1,
                              context_length=8192, max_output=512,
                              user_edits=[{"path": "example.txt", "diff": "+changed\n" * 20_000}])
    payload = request_input(view)
    assert payload["user_edits"][0]["diff_truncated"]
    assert message_tokens(view) <= int(8192 * .85) - 512


async def test_preview_and_rejected_response_do_not_acknowledge_edit(tmp_path):
    invalid = completion('{"tool_calls":[{"name":"read_artifact","arguments":"["}]}')
    agent, client, edits, _ = setup_edits(tmp_path, [invalid, completion("Accepted")], max_steps=1)
    edits.record(tmp_path / "file", "+pending", "edit")
    await agent._context_view(agent.registry.specs(), 1, preview=True)
    assert not agent.registry.context_notes["user_edits"]["edit"].get("delivered")
    await agent.run("Work")
    assert not agent.registry.context_notes["user_edits"]["edit"].get("delivered")
    assert await agent.run("Continue") == "Accepted"
    assert request_input(client.calls[-1]["messages"])["user_edits"][0]["diff"] == "+pending"
    assert agent.registry.context_notes["user_edits"]["edit"]["delivered"]
    await agent.registry.aclose()


async def test_new_save_during_request_survives_earlier_acceptance_and_revert_is_notified(tmp_path):
    agent, client, edits, store = setup_edits(tmp_path, [completion("First"), completion("Second"), completion("Reverted")])
    path = tmp_path / "file"
    edits.record(path, "+first", "edit")
    original = client.chat
    async def chat(**kwargs):
        if not client.calls:
            edits.record(path, "+second", "edit")
        return await original(**kwargs)
    client.chat = chat
    assert await agent.run("Work") == "First"
    assert not agent.registry.context_notes["user_edits"]["edit"].get("delivered")
    assert await agent.run("Continue") == "Second"
    notices = [request_input(call["messages"])["user_edits"][0] for call in client.calls]
    assert notices[0]["revision"] != notices[1]["revision"]
    count = len(store.records)
    edits.record(path, "+second", "edit")
    assert len(store.records) == count
    assert agent.registry.context_notes["user_edits"]["edit"]["delivered"]
    edits.record(path, "", "edit")
    assert await agent.run("Continue") == "Reverted"
    assert request_input(client.calls[-1]["messages"])["user_edits"][0]["reverted_to_editor_baseline"]
    await agent.registry.aclose()


async def test_delivered_edit_history_survives_resume_and_fork(tmp_path):
    agent, client, edits, _ = setup_edits(tmp_path, [completion("Done")])
    journal = SessionJournal(str(tmp_path), tmp_path / "sessions")
    agent.registry.resources.journal = journal
    edits.record(tmp_path / "file", "+saved", "edit")
    await agent.run("Continue")
    await agent.wait_for_compaction()
    data = journal.load(journal.session_id)
    restored = Agent(client, ToolRegistry(), "test")
    journal.restore(restored, data, fork=True)
    assert restored.history.steps[0].parts()["user_edits"][0]["diff"] == "+saved"
    assert restored.registry.context_notes["user_edits"]["edit"]["delivered"]
    assert journal.load(journal.session_id)["history"].steps[0].parts()["user_edits"]
    await restored.registry.aclose()
    await agent.registry.aclose()


async def test_artifact_pages_reconstruct_unicode_and_list_in_bounded_batches(tmp_path, monkeypatch):
    from pathlib import Path
    store = ArtifactStore(1_000_000)
    store.use_directory(tmp_path / "records")
    original = "α👩‍💻\n\\quoted\" " * 1000
    records = [store.save("command", original, kind="diff", error="error\n") for _ in range(3)]
    def no_whole_reads(*args, **kwargs):
        pytest.fail("Paged retrieval loaded an entire record")
    monkeypatch.setattr(Path, "read_text", no_whole_reads)
    tool = ReadArtifactTool(store)
    offset, pieces = 0, []
    while True:
        result = await tool.run(records[0]["record_id"], offset=offset, limit=317)
        assert not result.is_error
        page = result.content
        assert len(page["output"]) <= 317
        pieces.append(page["output"])
        if page["eof"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert "".join(pieces) == original
    assert (await tool.run(records[0]["record_id"], field="error" )).content["error"] == "error\n"
    first = (await tool.run(limit=2)).content
    second = (await tool.run(offset=first["next_offset"], limit=2)).content
    assert [r["record_id"] for r in first["records"] + second["records"]] == [r["record_id"] for r in records]
    assert second["eof"]
    await store.aclose()


async def test_transport_failure_keeps_saved_edit_pending(tmp_path):
    from slipagent.openrouter import OpenRouterError
    agent, client, edits, _ = setup_edits(tmp_path, [completion("Accepted")], max_steps=1)
    edits.record(tmp_path / "file", "+pending", "edit")
    original = client.chat
    async def failed(**kwargs):
        raise OpenRouterError("connection dropped")
    client.chat = failed
    with pytest.raises(OpenRouterError, match="connection dropped"):
        await agent.run("Work")
    assert not agent.registry.context_notes["user_edits"]["edit"]["delivered"]
    client.chat = original
    await agent.run("Continue")
    assert agent.registry.context_notes["user_edits"]["edit"]["delivered"]
    assert agent.history.steps[-1].parts()["user_edits"][0]["diff"] == "+pending"
    await agent.registry.aclose()


async def test_many_saved_files_share_budget_and_only_included_revisions_retire(tmp_path):
    agent, client, edits, _ = setup_edits(tmp_path, [completion("Done"), completion("Next")],
                                         context_tokens=16_000, max_tokens=512)
    for number in range(60):
        edits.record(tmp_path / f"file-{number}", "+changed\n" * 1000, f"edit-{number}")
    await agent.run("Work")
    payload = request_input(client.calls[0]["messages"])
    included = {notice["edit_id"] for notice in payload["user_edits"]}
    assert 0 < len(included) < 60
    assert payload["user_edits_omitted"] == 60 - len(included)
    assert message_tokens(client.calls[0]["messages"]) < 16_000
    pending = agent.registry.context_notes["user_edits"]
    assert {key for key, entry in pending.items() if entry["delivered"]} == included
    await agent.run("Continue")
    following = request_input(client.calls[-1]["messages"])["user_edits"]
    assert not included.intersection(notice["edit_id"] for notice in following)
    await agent.registry.aclose()


async def test_delivery_receipt_recovers_before_pending_state_update_and_tools(tmp_path):
    from slipagent.editdelivery import notifications
    from slipagent.types import Message, ToolCall
    agent, client, edits, _ = setup_edits(tmp_path, [])
    journal = SessionJournal(str(tmp_path), tmp_path / "sessions")
    agent.registry.resources.journal = journal
    edits.record(tmp_path / "file", "+saved", "edit")
    agent.messages.append(Message.user("Continue"))
    view = await agent._context_view(agent.registry.specs(), 1, preview=True)
    delivered = notifications(view)
    agent.messages.append(Message.assistant("Read next", [ToolCall("call", "read_artifact", {"record_id": delivered[0]["artifact"]["record_id"]})]))
    agent._persist()
    journal.edits_delivered(1, delivered)
    # Simulate process death after the durable receipt, before acknowledgement
    # and before any accepted tool can start.
    assert not agent.registry.context_notes["user_edits"]["edit"]["delivered"]
    data = journal.load(journal.session_id)
    restored = Agent(client, ToolRegistry(), "test")
    journal.restore(restored, data)
    assert restored.registry.context_notes["user_edits"]["edit"]["delivered"]
    step = restored.history.steps[0]
    assert step.parts()["user_edits"] == delivered
    assert step.tool_results and not client.calls
    assert "not" in str(step.tool_results[0].content).lower()
    await restored.registry.aclose()
    await agent.registry.aclose()


async def test_unarchived_large_legacy_diff_remains_pending_after_acceptance(tmp_path):
    agent, _, edits, _ = setup_edits(tmp_path, [completion("Done")])
    original = "+legacy\n" * 20_000
    agent.registry.context_notes["user_edits"] = {"legacy": {"path": str(tmp_path / "file"), "diff": original}}
    await agent.run("Continue")
    entry = agent.registry.context_notes["user_edits"]["legacy"]
    assert entry["diff"] == original and not entry.get("delivered")
    notice = agent.history.steps[0].parts()["user_edits"][0]
    assert notice["archive_unavailable"] and notice["diff_truncated"]
    assert notice["revision"].startswith("legacy-")
    await agent.registry.aclose()


async def test_saved_edit_compaction_recall_and_excerpt_preserve_origin_and_reference(tmp_path):
    from slipagent.context import RecallHistoryTool
    from slipagent.types import decode_json_content
    agent, client, edits, _ = setup_edits(tmp_path, [completion("Done")])
    edits.record(tmp_path / "file", "+saved", "edit")
    await agent.run("Continue")
    await agent.wait_for_compaction()
    step = agent.history.steps[0]
    source = decode_json_content(client.summary_calls[0]["messages"][1].content)
    assert source["user_edits"][0]["source"] == "user"
    recalled = await RecallHistoryTool(agent.history).run(step_id=1, section="user_edits")
    notices = recalled.content["content"]
    assert notices == step.parts()["user_edits"]
    excerpt = step.excerpt_context_messages(1000)
    assert excerpt and message_tokens(excerpt) <= 1000
    assert excerpt[0].content["user_edits"][0]["revision"] == notices[0]["revision"]
    assert excerpt[0].content["artifacts"][0]["record_id"] == notices[0]["artifact"]["record_id"]
    await agent.registry.aclose()


async def test_artifact_page_limits_bound_commands_lists_and_reject_corruption(tmp_path):
    import json
    store = ArtifactStore(1_000_000)
    store.use_directory(tmp_path / "records")
    command = "c" * 20_000
    record = store.save(command, "x" * 30_000, kind="diff")
    tool = ReadArtifactTool(store)
    page = (await tool.run(record["record_id"], limit=100_000)).content
    assert len(page["output"]) == 12_000
    assert len(page["command"]) == 500 and page["command_truncated"]
    assert (await tool.run(record["record_id"], field="command", offset=500, limit=200)).content["command"] == command[500:700]
    end = (await tool.run(record["record_id"], offset=30_000)).content
    assert end["eof"] and end["next_offset"] is None and end["output"] == ""
    for arguments in ({"offset": -1}, {"offset": True}, {"limit": 0}, {"field": "unknown"}):
        assert (await tool.run(record["record_id"], **arguments)).is_error
    for _ in range(100):
        store.save("small", "", kind="diff")
    listing = (await tool.run(limit=1000)).content
    assert len(listing["records"]) == 100 and listing["next_offset"] == 100
    path = store.record_path(record["record_id"])
    data = json.loads(path.read_text())
    data["kind"] = "unexpected"
    path.write_text(json.dumps(data))
    assert (await tool.run(record["record_id"], offset=10)).is_error
    await store.aclose()
