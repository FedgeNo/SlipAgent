"""Layered 20-turn narratives and 100-turn eras, served oldest first under budget pressure."""

import asyncio
import copy
import json

import pytest

from data_text_reader import request_input
from slipagent.compaction import StepCompactor
from slipagent.capabilities import RequestProfile
from slipagent.context import ConversationHistory, RecallHistoryTool, message_tokens
from slipagent.historymerge import era_records, historical_records
from slipagent.sessions import SessionJournal
from slipagent.types import Completion, Message, ToolCall, Usage, content_text
from slipagent.agent import Agent
from slipagent.tools.base import ToolRegistry
from test_agent import StubClient


def make_agent(responses):
    client = StubClient(responses)
    return Agent(client, ToolRegistry(), "test"), client


def history_with_posts(total, size=3000, run_length=20):
    """Steps that each read a file; a user request opens every run of run_length steps."""
    history = ConversationHistory()
    messages = []
    for index in range(1, total + 1):
        call_id = f"read-{index}"
        if (index - 1) % run_length == 0:
            messages.append(Message.user(f"Inspect files from {index}"))
        messages.extend([Message.assistant("Reading", [ToolCall(call_id, "read_file", {"path": f"file-{index}.py"})]),
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


def schedule(owner, history, client, capabilities=None):
    owner.track(history.steps, client, "test", capabilities, None)


def append_posts(history, messages, count, size=3000):
    """Continue the current run with more file reads."""
    added, _ = history_with_posts(count, size)
    for step in added.steps:
        index = len(history.steps) + 1
        call_id = f"next-{index}"
        messages.extend([Message.assistant("Reading", [ToolCall(call_id, "read_file", {"path": f"next-{index}.py"})]),
                         Message.tool_result(call_id, "ORIGINAL " * 1000)])
        history.sync(messages)
        history.steps[-1].summary = step.summary
        history.steps[-1].reasoning_summary = ""


async def test_grouped_history_reads_individual_summary_files(tmp_path):
    from slipagent.sources import FileText
    from slipagent.turnstore import TurnStore

    history, messages = history_with_posts(32)
    store = TurnStore(tmp_path)
    for step in history.steps:
        store.archive(step, messages)
    client = MergeClient()
    errors = []
    owner = StepCompactor(lambda usage: None, errors.append)
    owner.on_step = store.summarized
    schedule(owner, history, client)
    await owner.wait()
    assert not errors
    source = client.calls[0]["messages"][1].content
    for item in source["steps"]:
        assert isinstance(item["compressed_summary"], FileText)
        assert item["compressed_summary"].read() == history.steps[item["step_id"] - 1].summary
    saved = list(tmp_path.glob("*.grouped.json"))
    assert len(saved) == 1
    record = json.loads(saved[0].read_text())
    assert record["step_ids"] == [item["step_id"] for item in source["steps"]] == list(range(1, 21))
    assert record["sources"][0]["tool_calls"][0]["call_id"] == "read-1"
    store.close()


async def test_each_fixed_block_of_twenty_gets_a_narrative_as_soon_as_its_summaries_exist():
    history, messages = history_with_posts(19, size=70)
    persisted, usage, errors = [], [], []
    owner = StepCompactor(usage.append, errors.append)
    owner.on_step = lambda step: persisted.append(step.id)
    client = MergeClient()
    schedule(owner, history, client)
    await owner.wait()
    assert not client.calls
    append_posts(history, messages, 1, size=70)
    originals = [step.full_text() for step in history.steps]
    first_pass = [step.summary for step in history.steps]
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 1 and not errors
    record = historical_records(history.steps)[0]
    assert record["step_ids"] == list(range(1, 21)) and record["step_id"] == 20
    assert record["created_at_step"] == 20
    assert record["sources"][0] == {"step_id": 1, "run": 1, "tool_calls": [{"call_id": "read-1", "tool_name": "read_file"}],
                                      "artifacts": [{"record_id": 1, "step_id": 1, "call_id": "read-1"}]}
    prompt = content_text(client.calls[0]["messages"][0].content)
    assert "one compact narrative" in prompt and "original `step_id` numbers" in prompt and "call_id" in prompt
    assert [step.full_text() for step in history.steps] == originals
    assert [step.summary for step in history.steps] == first_pass
    assert persisted == [20, 20] and len(usage) == 1
    append_posts(history, messages, 19, size=70)
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 1
    append_posts(history, messages, 1, size=70)
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 2
    second = historical_records(history.steps)[1]
    assert second["step_ids"] == list(range(21, 41)) and second["created_at_step"] == 40
    source = client.calls[1]["messages"][1].content
    assert [step["step_id"] for step in source["steps"]] == list(range(21, 41))
    assert all("compressed_summary" in step for step in source["steps"])


async def test_five_narratives_become_an_era_and_every_level_is_kept():
    history, _ = history_with_posts(100, size=70)
    client = MergeClient()
    owner = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 6
    assert [record["step_ids"] for record in historical_records(history.steps)] == [
        list(range(start, start + 20)) for start in range(1, 101, 20)]
    era = era_records(history.steps)[0]
    assert era["representation"] == "era_summary" and era["step_id"] == 100
    assert era["step_ids"] == list(range(1, 101))
    era_prompt = content_text(client.calls[-1]["messages"][0].content)
    assert "five supplied narratives" in era_prompt
    era_source = client.calls[-1]["messages"][1].content
    assert [item["step_ids"][0] for item in era_source["narratives"]] == [1, 21, 41, 61, 81]
    assert all(step.summary for step in history.steps)


async def test_unsettled_first_passes_hold_a_block_and_failed_ones_are_gaps():
    history, _ = history_with_posts(33)
    history.steps[2].summary = None
    history.steps[9].summary = None
    owner = StepCompactor(lambda _: None, lambda _: None)
    client = MergeClient()
    schedule(owner, history, client)
    await owner.wait()
    assert not client.calls
    history.steps[2].compaction_status = "failed"
    history.steps[9].compaction_status = "failed"
    schedule(owner, history, client)
    await owner.wait()
    record = historical_records(history.steps)[0]
    assert record["step_ids"] == [i for i in range(1, 21) if i not in (3, 10)]
    assert record["step_id"] == 20
    assert [source["step_id"] for source in record["sources"]] == record["step_ids"]


@pytest.mark.parametrize("outcome", ["recover", "reset", "permanent"])
@pytest.mark.parametrize("status", [429, 503, "timeout", "transport"])
async def test_history_merge_retries_until_recovery_or_reset(monkeypatch, outcome, status):
    import httpx
    from slipagent.openrouter import OpenRouterClient

    history, _ = history_with_posts(30)
    requests, waits, errors = [], [], []
    owner = StepCompactor(lambda usage: None, errors.append)
    originals = [step.summary for step in history.steps]
    def handle(request):
        requests.append(json.loads(request.content))
        if outcome == "permanent" or len(requests) < 10 or outcome == "reset":
            if outcome != "permanent" and status == "timeout":
                raise TimeoutError("request deadline")
            if outcome != "permanent" and status == "transport":
                raise httpx.ReadError("connection lost")
            return httpx.Response(401 if outcome == "permanent" else status, json={"error": {"message": "unavailable"}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps({"summary": "Completed the requested inspections."})}, "finish_reason": "stop"}]})
    real_sleep = asyncio.sleep
    async def sleep(delay):
        waits.append(delay)
        assert not errors
        if outcome == "reset" and len(requests) == 10:
            owner.reset()
            await real_sleep(0)
    monkeypatch.setattr("slipagent.api.asyncio.sleep", sleep)
    monkeypatch.setattr("slipagent.api.random.uniform", lambda low, high: high)
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        schedule(owner, history, client)
        await owner.wait()
    assert len(requests) == (1 if outcome == "permanent" else 10)
    assert all(request == requests[0] for request in requests)
    assert [step.summary for step in history.steps] == originals
    assert bool(historical_records(history.steps)) == (outcome == "recover")
    assert bool(errors) == (outcome == "permanent")
    assert waits == ([] if outcome == "permanent" else [1, 2, 4, 8, 16, 32] + [60] * (4 if outcome == "reset" else 3))


async def test_failed_block_exhausts_retries_then_waits_twenty_turns(monkeypatch):
    from unittest.mock import AsyncMock
    sleep = AsyncMock()
    monkeypatch.setattr("slipagent.compaction.asyncio.sleep", sleep)
    history, messages = history_with_posts(20)
    client = MergeClient(malformed=True)
    errors = []
    owner = StepCompactor(lambda _: None, errors.append)
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 6 and len(errors) == 1 and not historical_records(history.steps)
    assert "turns 1–20" in errors[0]
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2, 4, 8, 16]
    for _ in range(19):
        append_posts(history, messages, 1)
        schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 6
    append_posts(history, messages, 1)
    schedule(owner, history, client)
    await owner.wait()
    # The first block is due again and the second block is now complete.
    assert len(client.calls) == 18


@pytest.mark.parametrize("invalid", ["bad json", '{}', '{"summary":""}',
                                     json.dumps({"summary": "x" * 12001})],
                         ids=["json", "fields", "empty", "oversized"])
async def test_grouped_summary_retries_invalid_replies_before_publishing(monkeypatch, invalid):
    from unittest.mock import AsyncMock
    monkeypatch.setattr("slipagent.compaction.asyncio.sleep", AsyncMock())
    history, _ = history_with_posts(30)
    originals = [step.full_text() for step in history.steps]
    client = MergeClient()
    chat = client.chat

    async def respond(**kwargs):
        completion = await chat(**kwargs)
        if len(client.calls) == 1:
            completion.message.content = invalid
        return completion

    client.chat = respond
    usage, errors = [], []
    owner = StepCompactor(usage.append, errors.append)
    schedule(owner, history, client)
    await owner.wait()
    assert len(client.calls) == 2 and len(usage) == 2 and not errors
    assert client.calls[0] == client.calls[1]
    assert len(historical_records(history.steps)) == 1
    assert [step.full_text() for step in history.steps] == originals


async def test_inflight_merge_cannot_duplicate_or_publish_after_reset():
    history, messages = history_with_posts(30)
    client = MergeClient(pause=True)
    usage = []
    owner = StepCompactor(usage.append, lambda _: None)
    schedule(owner, history, client)
    await client.started.wait()
    schedule(owner, history, client)
    assert len(client.calls) == 1
    owner.reset()
    client.release.set()
    await owner.wait()
    assert not historical_records(history.steps) and not usage


async def test_grouped_retry_wait_is_cancelled_at_session_grace_deadline(monkeypatch):
    import httpx
    from slipagent.openrouter import OpenRouterClient

    monkeypatch.setattr("slipagent.compaction.COMPACTION_GRACE_PERIOD", .02)
    history, _ = history_with_posts(30)
    originals = [step.full_text() for step in history.steps]
    requests, errors = [], []

    def handle(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "60"}, json={"error": "unavailable"})

    owner = StepCompactor(lambda _: None, errors.append)
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        schedule(owner, history, client)
        await asyncio.wait_for(owner.settle(), 1)
    assert len(requests) == 1 and not owner.jobs and not errors
    assert not historical_records(history.steps)
    assert [step.full_text() for step in history.steps] == originals


async def test_merge_fits_frozen_provider_budget_without_truncating_source_summaries():
    history, _ = history_with_posts(45)
    owner = StepCompactor(lambda _: None, lambda _: None)
    client = MergeClient()
    small = RequestProfile(context_length=20000, max_prompt_tokens=20000, parameters=set())
    schedule(owner, history, client, small)
    await owner.wait()
    assert not client.calls  # Twenty complete source summaries cannot fit.
    larger = RequestProfile(context_length=60000, parameters={"max_tokens"})
    schedule(owner, history, client, larger)
    await owner.wait()
    assert len(client.calls) == 2
    call = client.calls[0]
    assert message_tokens(call["messages"]) + call["max_tokens"] <= int(60000 * .85)
    for source in call["messages"][1].content["steps"]:
        assert source["compressed_summary"] == history.steps[source["step_id"] - 1].summary


async def test_second_compactor_is_wired_into_completed_agent_steps():
    history, messages = history_with_posts(19)

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
    assert agent.history.steps[-1].summary == "Completed the newest turn."
    # The new request ends the 19-turn run, which closes its block.
    assert historical_records(agent.history.steps)[0]["step_ids"] == list(range(1, 20))
    await agent.registry.aclose()


async def test_narratives_replace_their_block_only_under_budget_pressure():
    history, messages = history_with_posts(30)
    owner = StepCompactor(lambda _: None, lambda _: None)
    schedule(owner, history, MergeClient())
    await owner.wait()
    messages.append(Message.user("Continue"))
    roomy = request_input(await history.view(messages, [], keep_steps=50, context_length=1_000_000, max_output=1000))
    assert all(record["representation"] == "full" for record in roomy["history"])
    view = await history.view(messages, [], keep_steps=50, context_length=66_000, max_output=1000)
    records = request_input(view)["history"]
    narrative = next(record for record in records if record["representation"] == "historical_summary")
    assert narrative["step_ids"] == list(range(1, 21))
    assert not any(record["step_id"] < 20 for record in records)
    assert [record["step_id"] for record in records if record["representation"] == "full"] == list(range(21, 31))
    assert message_tokens(view) <= int(66_000 * .85) - 1000
    recalled = await RecallHistoryTool(history).invoke({"step_id": 1, "call_id": "read-1"})
    assert "ORIGINAL " * 1000 in content_text(recalled.content)


def attach_layers(history):
    """Narratives for every complete block and eras for every complete era, as the compactor stores them."""
    for block in range(len(history.steps) // 20):
        anchor = history.steps[block * 20 + 19]
        ids = list(range(block * 20 + 1, block * 20 + 21))
        anchor.historical_summary = {
            "record_type": "history_step", "representation": "historical_summary", "step_id": anchor.id,
            "step_ids": ids, "sources": [{"step_id": i, "tool_calls": [], "artifacts": []} for i in ids],
            "compressed_summary": f"NARRATIVE {ids[0]}-{ids[-1]} " + "story " * 300, "created_at_step": anchor.id}
    for era in range(len(history.steps) // 100):
        anchor = history.steps[era * 100 + 99]
        anchor.era_summary = {
            "record_type": "history_step", "representation": "era_summary", "step_id": anchor.id,
            "step_ids": list(range(era * 100 + 1, era * 100 + 101)),
            "compressed_summary": f"ERA {era * 100 + 1}-{era * 100 + 100} " + "saga " * 300, "created_at_step": anchor.id}


async def test_compression_levels_advance_from_the_oldest_end_as_the_budget_shrinks():
    history, messages = history_with_posts(240, size=700)
    attach_layers(history)
    messages.append(Message.user("Continue"))
    seen = []
    for length in (4_000_000, 600_000, 300_000, 150_000, 60_000):
        view = await history.view(messages, [], keep_steps=50, context_length=length, max_output=1000)
        assert message_tokens(view) <= int(length * .85) - 1000
        records = request_input(view)["history"]
        order = {"era_summary": 4, "historical_summary": 3, "compressed": 2, "full": 1}
        levels = [order[record["representation"]] for record in records]
        assert levels == sorted(levels, reverse=True)  # Never more compressed than older history.
        assert records[-1]["step_id"] == 240
        seen.append(records)
    assert all(record["representation"] == "full" for record in seen[0])
    assert {record["representation"] for record in seen[1]} == {"compressed", "full"}
    assert any(record["representation"] == "historical_summary" for record in seen[2])
    assert any(record["representation"] == "era_summary" for record in seen[3])
    smallest = seen[4]
    # Dropping starts at the very beginning; the newest turns stay.
    assert smallest[0]["step_id"] > 1 and smallest[-1]["step_id"] == 240


async def test_missing_summary_preserves_original_evidence_when_budget_allows():
    history, messages = history_with_posts(9, size=70)
    history.steps[0].summary = None
    messages.append(Message.user("Continue"))
    records = request_input(await history.view(messages, [], keep_steps=50,
                            context_length=100000, max_output=1000))["history"]
    assert records[0]["representation"] == "full"
    assert "ORIGINAL" in records[0]["tool_calls"][0]["result"]


async def test_narratives_eras_and_attempts_survive_session_resume_and_fork(tmp_path):
    history, messages = history_with_posts(100, size=70)
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
    eras = copy.deepcopy(era_records(history.steps))
    assert len(expected) == 5 and len(eras) == 1
    data = journal.load(journal.session_id)
    assert historical_records(data["history"].steps) == expected
    assert era_records(data["history"].steps) == eras
    assert data["history"].steps[19].history_merge_attempt == 100
    assert data["history"].steps[99].era_attempt == 100
    restored, _ = make_agent([])
    journal.restore(restored, data, fork=True)
    assert historical_records(restored.history.steps) == expected
    child = journal.load(journal.session_id)
    assert historical_records(child["history"].steps) == expected
    assert era_records(child["history"].steps) == eras
    await agent.registry.aclose()
    await restored.registry.aclose()


async def test_resume_ignores_narratives_from_before_fixed_blocks(tmp_path):
    history, messages = history_with_posts(30, size=70)
    agent, _ = make_agent([])
    agent.messages = messages
    agent.history = history
    journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
    journal.begin(agent)
    legacy = history.steps[24]
    legacy.historical_summary = {
        "record_type": "history_step", "representation": "historical_summary", "step_id": 25,
        "step_ids": list(range(1, 26)), "sources": [{"step_id": i, "tool_calls": [], "artifacts": []} for i in range(1, 26)],
        "compressed_summary": "Older narrative.", "created_at_step": 30}
    legacy.history_merge_attempt = 30
    journal.step(legacy)
    data = journal.load(journal.session_id)
    assert not historical_records(data["history"].steps)
    await agent.registry.aclose()


def history_with_runs(lengths, *, last_ended=True):
    """Runs of the given lengths; each opens with a request and ends with a reply that calls no tools."""
    history = ConversationHistory()
    messages = []
    for run, length in enumerate(lengths, 1):
        messages.append(Message.user(f"Request {run}"))
        for turn in range(length):
            index = sum(1 for message in messages if message.role == "assistant") + 1
            if turn == length - 1 and (run < len(lengths) or last_ended):
                messages.append(Message.assistant(f"Run {run} done"))
                continue
            call_id = f"read-{index}"
            messages.extend([Message.assistant("Reading", [ToolCall(call_id, "read_file", {"path": f"f{index}.py"})]),
                             Message.tool_result(call_id, "ORIGINAL " * 100)])
    history.sync(messages)
    for step in history.steps:
        step.summary = f"Turn {step.id}: observations " + "detail " * 10
        step.reasoning_summary = ""
        step.results_summarized = True
        step.observed = True
    return history, messages


def spans(history):
    from slipagent.historymerge import blocks, run_numbers
    return [(history.steps[first].id, history.steps[last].id)
            for first, last in blocks(history.steps, run_numbers(history.steps))]


def test_a_request_or_a_reply_without_tools_starts_the_next_run():
    from slipagent.historymerge import run_numbers
    history = ConversationHistory()
    history.sync([Message.user("First"), Message.assistant("Look", [ToolCall("a", "list_dir", {})]),
                  Message.tool_result("a", "x"),
                  # A prompt sent mid-run starts a new run.
                  Message.user("Also check y"), Message.assistant("Look", [ToolCall("b", "list_dir", {})]),
                  Message.tool_result("b", "y"), Message.assistant("Done"),
                  # A turn after a reply without tools starts a run even without a prompt.
                  Message.assistant("Following up", [ToolCall("c", "list_dir", {})]), Message.tool_result("c", "z")])
    assert run_numbers(history.steps) == [1, 2, 2, 3]


def test_blocks_close_at_run_ends_after_ten_turns_and_split_long_runs_at_twenty():
    history, messages = history_with_runs([3, 4, 5, 25, 2])
    # Short runs share a block until it reaches ten turns; a long run is cut at 20.
    assert spans(history) == [(1, 12), (13, 32)]
    extended, _ = history_with_runs([3, 4, 5, 25, 2, 5])
    # Turns 33–39 waited for more; earlier blocks do not move.
    assert spans(extended) == [(1, 12), (13, 32), (33, 44)]
    running, _ = history_with_runs([3, 4, 5], last_ended=False)
    assert spans(running) == []  # The run is still going and the block is under 20 turns.


async def test_run_blocks_get_narratives_with_their_runs_and_five_make_an_era():
    history, _ = history_with_runs([10, 12, 10, 15, 11])
    client = MergeClient()
    owner = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    schedule(owner, history, client)
    await owner.wait()
    narratives = historical_records(history.steps)
    assert [(record["step_ids"][0], record["step_ids"][-1]) for record in narratives] == [
        (1, 10), (11, 22), (23, 32), (33, 47), (48, 58)]
    assert [record["runs"] for record in narratives] == [[1], [2], [3], [4], [5]]
    assert {source["run"] for source in narratives[1]["sources"]} == {2}
    source = client.calls[0]["messages"][1].content
    assert {item["run"] for item in source["steps"]} == {1}
    prompt = content_text(client.calls[0]["messages"][0].content)
    assert "Organize the narrative by run" in prompt and "recall_history` search" not in prompt
    era = era_records(history.steps)[0]
    assert era["step_ids"] == list(range(1, 59)) and era["runs"] == [1, 2, 3, 4, 5] and era["step_id"] == 58
    era_source = client.calls[-1]["messages"][1].content
    assert [item["runs"] for item in era_source["narratives"]] == [[1], [2], [3], [4], [5]]


def test_sessions_with_fixed_block_narratives_keep_them_and_continue_with_run_blocks():
    history, messages = history_with_posts(55, size=70)
    attach_layers(history)
    # The fixed blocks through the last stored narrative stay; later turns follow runs,
    # and the run that began at turn 41 is still going.
    assert spans(history) == [(1, 20), (21, 40)]
    append_posts(history, messages, 5, size=70)
    assert spans(history) == [(1, 20), (21, 40), (41, 60)]


async def test_requests_show_each_turns_run_and_the_current_run():
    history, messages = history_with_runs([2, 3])
    messages.append(Message.user("Next request"))
    view = await history.view(messages, [], keep_steps=50, context_length=200_000, max_output=1000)
    records = request_input(view)["history"]
    assert [record["run"] for record in records] == [1, 1, 2, 2, 2]
    from test_agent import context_records
    assert context_records(view)[-1]["run"] == 3


async def test_run_block_narratives_with_gaps_survive_resume(tmp_path):
    history, messages = history_with_runs([12, 3])
    history.steps[4].summary = None
    history.steps[4].compaction_status = "failed"
    agent, _ = make_agent([])
    agent.messages = messages
    agent.history = history
    journal = SessionJournal(str(tmp_path), directory=tmp_path / "state")
    journal.begin(agent)
    owner = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    owner.on_step = journal.step
    schedule(owner, history, MergeClient())
    await owner.wait()
    expected = copy.deepcopy(historical_records(history.steps))
    assert [record["step_ids"] for record in expected] == [[i for i in range(1, 13) if i != 5]]
    data = journal.load(journal.session_id)
    assert historical_records(data["history"].steps) == expected
    await agent.registry.aclose()
