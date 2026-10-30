"""Oversized turns retain exact source coverage and publish only a complete merge."""

import asyncio
import json

import httpx
import pytest

from data_text_reader import read_data
from slipagent.capabilities import RequestProfile
from slipagent.compaction import StepCompactor
from slipagent.compactionchunks import OVERLAP_CHARACTERS, chunk_messages, plan_chunks
from slipagent.context import CompletedStep, message_tokens
from slipagent.openrouter import OpenRouterClient
from slipagent.sources import materialize, spool_directory, write_indexed_json
from slipagent.types import Completion, Message, ToolCall


def large_step():
    text = ("α² + β² = γ²; preserve equation, condition, and units.\n" * 2500) + "EXACT=1.23456789012345"
    call = ToolCall("call_local", "read_file", {"path": "equations.txt"})
    step = CompletedStep(41, "Keep the exact equation and final value.", [
        Message.user("Keep the exact equation and final value."), Message.assistant("Reading.", [call]),
        Message.tool_result(call.id, {"empty": [], "nested": {"equation": text}})])
    source = step.compaction_input()
    source.update(history=[], history_omitted=0)
    return step, source, text


def profile():
    return RequestProfile(parameters={"max_tokens"}, context_length=32768, max_completion_tokens=8192)


@pytest.mark.parametrize("file_backed", [False, True])
def test_chunks_cover_the_original_exactly_with_original_overlap_and_stable_identity(tmp_path, file_backed):
    step, source, text = large_step()
    if file_backed:
        source = write_indexed_json(tmp_path / "source.json", source)
    budget = int(profile().context_length * .85) - 8192
    plan = plan_chunks(step.id, source, budget)
    directory = plan.directory.name
    try:
        assert len(plan.chunks) > 1
        records = []
        for index, chunk in enumerate(plan.chunks, 1):
            assert message_tokens(chunk_messages(plan.shared, chunk["records"], index, len(plan.chunks))) <= budget
            records.extend(materialize(chunk["records"]))
        fragments = [record for record in records if "text" in record]
        assert fragments
        offset = 0
        for record in fragments:
            assert record["path"] == ["tool_results", 0, "content", "nested", "equation"]
            assert record["start"] == offset
            assert record["text"] == text[record["start"]:record["end"]]
            assert record["overlap_before"] == text[max(0, offset - OVERLAP_CHARACTERS):offset]
            assert record["overlap_after"] == text[record["end"]:record["end"] + OVERLAP_CHARACTERS]
            assert record["tool_identity"] == {"call_id": "call_local", "name": "read_file"}
            offset = record["end"]
        assert offset == len(text)
        assert "".join(record["text"] for record in fragments) == text
        assert any(record.get("value") == [] for record in records)
    finally:
        plan.close()
    from pathlib import Path
    assert not Path(directory).exists()


def test_small_tool_call_and_result_stay_in_one_structural_group():
    step, source, _ = large_step()
    source["tool_results"][0]["content"] = "read successfully"
    plan = plan_chunks(step.id, source, 20000)
    try:
        records = [record for chunk in plan.chunks for record in materialize(chunk["records"])]
        group = next(record["call_and_results"] for record in records if "call_and_results" in record)
        assert [record["path"] for record in group] == [["tool_calls", 0], ["tool_results", 0]]
    finally:
        plan.close()


async def test_chunk_review_and_merge_use_only_the_free_route_and_never_execute_tools(free_compaction):
    step, source, _ = large_step()
    original = step.full_text()
    phases, hosts, published = [], [], []
    budget = int(profile().context_length * .85) - 8192

    def handle(request):
        body = json.loads(request.content)
        record = read_data(body["messages"][1]["content"])
        phases.append(record["phase"])
        hosts.append(request.url.host)
        assert body.get("tools") is None
        assert step.summary is None
        assert message_tokens([Message.from_api(message) for message in body["messages"]]) <= budget
        assert not published
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps({
            "summary": "Boundary facts preserved." if record["phase"] == "boundary_review" else "The exact result was retained.",
            "reasoning_summary": ""})}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    compactor = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    compactor.on_step = lambda completed: published.append(completed.summary)
    async with OpenRouterClient("paid-primary", transport=httpx.MockTransport(handle)) as working:
        free_compaction(compactor, working, profile=profile())
        await compactor._compact(step, source, working, "paid", profile(), 8192, 0)
    assert phases.count("chunk") > 1
    assert phases.count("boundary_review") == phases.count("chunk") - 1
    assert phases[-1] == "merge"
    assert all(host == "integrate.api.nvidia.com" for host in hosts)
    assert published == ["The exact result was retained."]
    assert step.compaction_status == "complete"
    assert step.full_text() == original
    assert not list(spool_directory().glob("compaction-chunks-*"))


@pytest.mark.parametrize("failure", ["cancel", "bad_boundary", "bad_merge"])
async def test_incomplete_chunk_pipeline_never_publishes_partial_memory(failure):
    step, source, _ = large_step()
    original = step.full_text()
    ready = asyncio.Event()
    published = []

    class Client:
        async def chat(self, **kwargs):
            record = kwargs["messages"][1].content
            if record["phase"] == "merge" and failure == "cancel":
                ready.set()
                await asyncio.Event().wait()
            bad = record["phase"] == {"bad_boundary": "boundary_review", "bad_merge": "merge"}.get(failure)
            return Completion(Message.assistant("bad JSON" if bad else json.dumps({"summary": "Intermediate only.", "reasoning_summary": ""})), "test")

    compactor = StepCompactor(lambda _: None, lambda _: None)
    compactor.on_step = lambda completed: published.append(completed.summary)
    task = asyncio.create_task(compactor._compact_once(step, source, Client(), "test", profile(), 8192, 0))
    if failure == "cancel":
        await asyncio.wait_for(ready.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(ValueError, match="invalid JSON"):
            await task
    assert step.summary is None and not published
    assert step.full_text() == original
    assert not list(spool_directory().glob("compaction-chunks-*"))


async def test_merge_reduces_long_intermediate_summaries_until_every_request_fits():
    step, source, _ = large_step()
    phases = []
    budget = int(profile().context_length * .85) - 8192
    class Client:
        async def chat(self, **kwargs):
            assert message_tokens(kwargs["messages"]) <= budget
            record = kwargs["messages"][1].content
            phases.append(record["phase"])
            return Completion(Message.assistant(json.dumps({"summary": "Evidence. " * 590,
                "reasoning_summary": "Uncertainty. " * 450})), "test")
    compactor = StepCompactor(lambda _: None, lambda _: None)
    await compactor._compact_once(step, source, Client(), "test", profile(), 8192, 0)
    assert phases.count("merge") > 1
    assert step.compaction_status == "complete"
    assert not list(spool_directory().glob("compaction-chunks-*"))


async def test_legacy_oversized_failure_is_retried_and_old_error_is_cleared():
    from slipagent.compaction import LEGACY_OVERSIZED_ERROR
    step, _, _ = large_step()
    step.compaction_status, step.compaction_error = "failed", LEGACY_OVERSIZED_ERROR
    following = CompletedStep(42, "Continue.", [Message.user("Continue."), Message.assistant("Done.")])
    class Client:
        async def chat(self, **kwargs):
            return Completion(Message.assistant(json.dumps({"summary": "Recovered original step.", "reasoning_summary": ""})), "test")
    compactor = StepCompactor(lambda _: None, lambda error: pytest.fail(error))
    compactor.submit(following, Client(), "test", profile(), 8192, conversation=[step, following])
    await compactor.wait()
    assert step.compaction_status == "complete"
    assert step.compaction_error is None
