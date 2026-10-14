"""Percentage selection and occasional, source-linked historical narratives."""

import asyncio
import copy
import json

import pytest

from data_text_reader import request_input
from slipagent.compaction import StepCompactor
from slipagent.capabilities import RequestProfile
from slipagent.context import ConversationHistory, RecallHistoryTool, message_tokens
from slipagent.historymerge import historical_records, schedule_history_merge
from slipagent.sessions import SessionJournal
from slipagent.types import Completion, Message, ToolCall, Usage, content_text
from slipagent.agent import Agent
from slipagent.tools.base import ToolRegistry
from test_agent import StubClient


def make_agent(responses):
    client = StubClient(responses)
    return Agent(client, ToolRegistry(), "test"), client


def history_with_posts(total, size=3000):
    history = ConversationHistory()
    messages = []
    for index in range(1, total + 1):
        call_id = f"read-{index}"
        messages.extend([Message.user(f"Inspect file-{index}.py"),
                         Message.assistant("Reading", [ToolCall(call_id, "read_file", {"path": f"file-{index}.py"})]),
                         Message.tool_result(call_id, {"tool": "read_file", "call_id": call_id, "status": "success",
                             "content": "ORIGINAL " * 1000,
                             "artifacts": [{"record_id": index, "step_id": index, "call_id": call_id}]})])
    history.sync(messages)
    for step in history.steps:
        step.summary = f"Turn {step.id}: observations " + "detail " * (size // 7)
        step.reasoning_summary = ""
        step.results_summarized = True
        step.observed = True
    return history, messages


class MergeClient:
    def __init__(self, *, malformed=False, pause=False):
        self.calls = []
        self.malformed = malformed
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not pause:
            self.release.set()

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        text = "bad json" if self.malformed else json.dumps({"summary":
            "The user requested inspection. Tools returned observations (step_id 1, call_id read-1); further inspections followed."})
        return Completion(Message.assistant(text), "test", usage=Usage(prompt_tokens=100, completion_tokens=30))


def schedule(owner, history, client):
    schedule_history_merge(owner, history.steps, client, "test", None, None)


def append_posts(history, messages, count, size=3000):
    added, _ = history_with_posts(count, size)
    for step in added.steps:
        index = len(history.steps) + 1
        messages.extend([Message.user(f"Next {index}"), Message.assistant("ORIGINAL " * 1000)])
        history.sync(messages)
        history.steps[-1].summary = step.summary
        history.steps[-1].reasoning_summary = ""


async def test_large_summary_batches_merge_occasionally_and_preserve_source_metadata():
    history, messages = history_with_posts(29)
    persisted, usage, errors = [], [], []
    owner = StepCompactor(usage.append, errors.append)
    owner.on_step = lambda step: persisted.append(step.id)
    client = MergeClient()
    schedule(owner, history, client)
    assert not owner.jobs
    append_posts(history, messages, 1)
    originals = [step.full_text() for step in history.steps]
    first_pass = [step.summary for step in history.steps]
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 1 and not errors
    record = historical_records(history.steps)[0]
    assert record["step_ids"] == list(range(1, 21))
    assert record["created_at_step"] == 30
    assert record["sources"][0] == {"step_id": 1, "tool_calls": [{"call_id": "read-1", "tool_name": "read_file"}],
                                      "artifacts": [{"record_id": 1, "step_id": 1, "call_id": "read-1"}]}
    prompt = content_text(client.calls[0]["messages"][0].content)
    assert "one compact narrative" in prompt and "original `step_id` numbers" in prompt and "call_id" in prompt
    assert [step.full_text() for step in history.steps] == originals
    assert [step.summary for step in history.steps] == first_pass
    assert persisted == [30, 20] and len(usage) == 1
    for _ in range(19):
        append_posts(history, messages, 1)
        schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 1
    append_posts(history, messages, 11)
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 2
    second = historical_records(history.steps)[1]
    assert second["step_ids"] == list(range(21, 41))
    assert second["created_at_step"] == 60
    source = client.calls[1]["messages"][1].content
    assert [step["step_id"] for step in source["steps"]] == list(range(21, 41))
    assert all("compressed_summary" in step for step in source["steps"])


async def test_small_normal_turns_do_not_merge_just_because_twenty_turns_passed():
    history, _ = history_with_posts(90, size=70)
    client = MergeClient()
    owner = StepCompactor(lambda _: None, lambda _: None)
    schedule(owner, history, client)
    await owner.wait()
    assert not client.calls
    assert not any(getattr(step, "history_merge_attempt", 0) for step in history.steps)


async def test_failed_merge_waits_twenty_turns_before_another_model_call():
    history, messages = history_with_posts(30)
    client = MergeClient(malformed=True)
    errors = []
    owner = StepCompactor(lambda _: None, errors.append)
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 1 and len(errors) == 1 and not historical_records(history.steps)
    for _ in range(19):
        append_posts(history, messages, 1)
        schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 1
    append_posts(history, messages, 1)
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 2


async def test_inflight_merge_cannot_duplicate_or_publish_after_reset():
    history, messages = history_with_posts(30)
    client = MergeClient(pause=True)
    usage = []
    owner = StepCompactor(usage.append, lambda _: None)
    schedule(owner, history, client)
    await client.started.wait()
    append_posts(history, messages, 40)
    schedule(owner, history, client)
    assert len(client.calls) == 1
    owner.reset()
    client.release.set()
    await owner.wait()
    assert not historical_records(history.steps) and not usage


async def test_merge_fits_frozen_provider_budget_without_truncating_source_summaries():
    history, _ = history_with_posts(45)
    owner = StepCompactor(lambda _: None, lambda _: None)
    client = MergeClient()
    small = RequestProfile(context_length=20000, max_prompt_tokens=20000, parameters=set())
    schedule_history_merge(owner, history.steps, client, "test", small, None)
    await owner.wait()
    assert not client.calls  # Twenty complete source summaries cannot fit.
    larger = RequestProfile(context_length=60000, parameters={"max_tokens"})
    schedule_history_merge(owner, history.steps, client, "test", larger, None)
    await owner.wait()
    assert len(client.calls) == 1
    call = client.calls[0]
    assert message_tokens(call["messages"]) + call["max_tokens"] <= int(60000 * .85)
    for source in call["messages"][1].content["steps"]:
        assert source["compressed_summary"] == history.steps[source["step_id"] - 1].summary


async def test_missing_summaries_are_gaps_rather_than_claimed_source_turns():
    history, _ = history_with_posts(33)
    history.steps[2].summary = None
    history.steps[9].summary = None
    owner = StepCompactor(lambda _: None, lambda _: None)
    schedule(owner, history, MergeClient())
    await owner.wait()
    record = historical_records(history.steps)[0]
    assert record["step_ids"] == [i for i in range(1, 23) if i not in (3, 10)]
    assert [source["step_id"] for source in record["sources"]] == record["step_ids"]


async def test_second_compactor_is_wired_into_completed_agent_steps():
    history, messages = history_with_posts(29)

    class Client(MergeClient):
        async def chat(self, **kwargs):
            if content_text(kwargs["messages"][0].content).startswith("\n# Summarizing the Previous Turn"):
                return Completion(Message.assistant(json.dumps({"summary": "Completed the newest turn.", "reasoning_summary": ""})), "test")
            return await super().chat(**kwargs)

    agent = Agent(Client(), ToolRegistry(), "test")
    agent.messages = messages
    agent.history = history
    agent.messages.extend([Message.user("Next"), Message.assistant("Newest answer")])
    await agent._archive_step(None)
    await agent.wait_for_compaction()
    assert historical_records(agent.history.steps)[0]["step_ids"] == list(range(1, 21))
    assert agent.history.steps[-1].summary == "Completed the newest turn."
    await agent.registry.aclose()


async def test_merged_narratives_enter_context_without_duplicate_sources_and_recall_remains_exact():
    history, messages = history_with_posts(30)
    owner = StepCompactor(lambda _: None, lambda _: None)
    schedule(owner, history, MergeClient())
    await owner.wait()
    messages.append(Message.user("Continue"))
    view = await history.view(messages, [], keep_steps=50, context_length=100000, max_output=1000)
    records = request_input(view)["history"]
    narrative = next(record for record in records if record["representation"] == "historical_summary")
    assert narrative["step_ids"] == list(range(1, 21))
    assert not any(record["step_id"] < 20 for record in records)
    assert [record["step_id"] for record in records if record["representation"] == "full"] == list(range(21, 31))
    recalled = await RecallHistoryTool(history).invoke({"step_id": 1, "call_id": "read-1"})
    assert "ORIGINAL " * 1000 in content_text(recalled.content)


async def test_percentage_selection_preserves_broad_history_under_large_read_pressure():
    history, messages = history_with_posts(60, size=70)
    messages.append(Message.user("Continue"))
    originals = [step.full_text() for step in history.steps]
    view = await history.view(messages, [], keep_steps=50, context_length=45000, max_output=1000)
    records = request_input(view)["history"]
    full = [r for r in records if r["representation"] == "full"]
    compressed = [r for r in records if r["representation"] == "compressed"]
    assert len(full) == (len(records) + 2) // 3
    assert len(compressed) >= 2 * len(full) - 2
    assert len(records) >= 20
    assert message_tokens(view) <= int(45000 * .85) - 1000
    assert [step.full_text() for step in history.steps] == originals


async def test_missing_summary_preserves_original_evidence_when_budget_allows():
    history, messages = history_with_posts(9, size=70)
    history.steps[0].summary = None
    messages.append(Message.user("Continue"))
    records = request_input(await history.view(messages, [], keep_steps=50,
                            context_length=100000, max_output=1000))["history"]
    assert records[0]["representation"] == "full"
    assert "ORIGINAL" in records[0]["tool_calls"][0]["result"]


async def test_narrative_and_attempt_spacing_survive_session_resume_and_fork(tmp_path):
    history, messages = history_with_posts(30)
    agent, _ = make_agent([])
    agent.messages = messages
    agent.history = history
    journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
    journal.begin(agent)
    owner = StepCompactor(lambda _: None, lambda _: None)
    owner.on_step = journal.step
    schedule(owner, history, MergeClient())
    await owner.wait()
    expected = copy.deepcopy(historical_records(history.steps))
    data = journal.load(journal.session_id)
    assert historical_records(data["history"].steps) == expected
    assert data["history"].steps[-1].history_merge_attempt == 30
    restored, _ = make_agent([])
    journal.restore(restored, data, fork=True)
    assert historical_records(restored.history.steps) == expected
    child = journal.load(journal.session_id)
    assert historical_records(child["history"].steps) == expected
    assert child["history"].steps[-1].history_merge_attempt == 30
    await agent.registry.aclose()
    await restored.registry.aclose()
