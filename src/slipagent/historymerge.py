"""Occasional narratives over immutable first-pass turn summaries."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import TYPE_CHECKING, Any

from .capabilities import RequestProfile, model_context_limit
from .prompts import load_prompt
from .records import step_record
from .types import Message

if TYPE_CHECKING:
    from .api import APIClient
    from .compaction import StepCompactor
    from .context import HistoryStep

MERGE_INTERVAL = 20
MERGE_MIN_TOKENS = 12_000
MERGE_MAX_CHARS = 12_000


def historical_records(steps: list[HistoryStep]) -> list[dict[str, Any]]:
    return [record for step in steps
            if (record := getattr(step, "historical_summary", None)) is not None]


def validate_historical_record(record: Any, anchor: int, total: int) -> None:
    """Validate persisted source identities before they can hide originals."""
    if not isinstance(record, dict) or set(record) != {
        "record_type", "representation", "step_id", "step_ids", "sources",
        "compressed_summary", "created_at_step",
    }:
        raise ValueError("invalid historical summary record")
    ids = record["step_ids"]
    if (not isinstance(ids, list) or not ids
            or any(type(i) is not int or not 1 <= i <= total for i in ids)
            or ids != sorted(set(ids)) or ids[-1] != anchor
            or record["step_id"] != anchor
            or record["record_type"] != "history_step"
            or record["representation"] != "historical_summary"
            or type(record["created_at_step"]) is not int
            or not anchor <= record["created_at_step"] <= total):
        raise ValueError("invalid historical summary source steps")
    summary = record["compressed_summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > MERGE_MAX_CHARS:
        raise ValueError("invalid historical narrative")
    sources = record["sources"]
    if not isinstance(sources, list) or len(sources) != len(ids):
        raise ValueError("invalid historical retrieval metadata")
    for source, step_id in zip(sources, ids):
        if (not isinstance(source, dict) or set(source) != {"step_id", "tool_calls", "artifacts"}
                or source["step_id"] != step_id or not isinstance(source["tool_calls"], list)
                or not isinstance(source["artifacts"], list)):
            raise ValueError("invalid historical retrieval metadata")
        for call in source["tool_calls"]:
            if (not isinstance(call, dict) or set(call) != {"call_id", "tool_name"}
                    or any(not isinstance(call[key], str) for key in call)):
                raise ValueError("invalid historical tool identity")


def schedule_history_merge(owner: StepCompactor, steps: list[HistoryStep], client: APIClient,
                           model: str, capabilities: RequestProfile | None,
                           max_tokens: int | None) -> None:
    from .context import estimate_tokens, message_tokens

    total = len(steps)
    last_attempt = max((getattr(step, "history_merge_attempt", 0) for step in steps), default=0)
    name = f"slipagent-history-merge-{owner.generation}"
    if total - last_attempt < MERGE_INTERVAL or any(job.get_name() == name for job in owner.jobs):
        return
    records = historical_records(steps)
    covered = {step_id for record in records for step_id in record["step_ids"]}
    boundary = total - (total + 2) // 3
    limit = 128_000
    if capabilities is not None:
        limit = model_context_limit(model, capabilities,
                    min(capabilities.context_length, capabilities.max_prompt_tokens or capabilities.context_length))
    reserve = min(max_tokens or 8192, capabilities.max_completion_tokens or 8192) if capabilities else (max_tokens or 8192)
    system = Message.system(load_prompt("historical-summary-prompt.md", max_characters=MERGE_MAX_CHARS))
    source: dict[str, Any] = {"steps": []}
    batch = []
    for step in steps[:boundary]:
        if step.id in covered or step.summary is None:
            continue
        original = step_record(step.messages)
        metadata = {"step_id": step.id,
                    "tool_calls": [{"call_id": call["call_id"], "tool_name": call["tool_name"]}
                                   for call in original["tool_calls"]],
                    "artifacts": step.context_messages(compressed=True)[0].content.get("artifacts", [])}
        source["steps"].append({**metadata, "compressed_summary": step.summary})
        if message_tokens([system, Message.user(source)]) + reserve > int(limit * .85):
            source["steps"].pop()
            break
        batch.append(step)
    if len(batch) < MERGE_INTERVAL or estimate_tokens(json.dumps(source, ensure_ascii=False)) < MERGE_MIN_TOKENS:
        return
    # Record attempts, including failures, so retries cannot turn this into an
    # every-turn request. The marker and source summaries survive session resume.
    setattr(steps[-1], "history_merge_attempt", total)
    if owner.on_step is not None:
        owner.on_step(steps[-1])  # type: ignore[arg-type]
    job = asyncio.create_task(_merge(owner, batch, copy.deepcopy(source), system, client,
                                    model, capabilities, reserve, total, owner.generation), name=name)
    owner.jobs.add(job)
    job.add_done_callback(owner.jobs.discard)


async def _merge(owner: StepCompactor, batch: list[HistoryStep], source: dict[str, Any],
                 system: Message, client: APIClient, model: str,
                 capabilities: RequestProfile | None, reserve: int,
                 created_at: int, generation: int) -> None:
    from .api import APIClient
    from .context import message_tokens

    try:
        options: dict[str, Any] = {}
        if isinstance(client, APIClient):
            options.update(request_profile=capabilities, disable_timeout=True, single_attempt=True)
        async with asyncio.timeout(20 * 60):
            completion = await client.chat(
                model=model, messages=[system, Message.user(source)], tools=None,
                temperature=.2,
                max_tokens=reserve if capabilities is not None and "max_tokens" in capabilities.parameters else None,
                extra_body={}, **options,
            )
        if generation != owner.generation:
            return
        owner.on_usage(completion.usage)
        if completion.response_error or completion.tool_calls or completion.finish_reason == "length":
            raise ValueError(completion.response_error or "the historical summary was incomplete or requested tools")
        value = json.loads(completion.text)
        if not isinstance(value, dict) or set(value) != {"summary"}:
            raise ValueError("the historical compactor must return only summary")
        record = {"record_type": "history_step", "representation": "historical_summary",
                  "step_id": batch[-1].id, "step_ids": [step.id for step in batch],
                  "sources": [{key: item[key] for key in ("step_id", "tool_calls", "artifacts")}
                              for item in source["steps"]],
                  "compressed_summary": value["summary"], "created_at_step": created_at}
        validate_historical_record(record, batch[-1].id, created_at)
        originals = [Message.user(step.context_messages(compressed=True)[0].content) for step in batch]
        if message_tokens([Message.user(record)]) >= message_tokens(originals):
            raise ValueError("the historical narrative did not reduce the supplied summaries")
        setattr(batch[-1], "historical_summary", record)
        if owner.on_step is not None:
            owner.on_step(batch[-1])  # type: ignore[arg-type]
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if generation == owner.generation:
            owner.on_error(f"Could not merge older history: {exc}. Individual summaries and originals remain available; another attempt waits at least 20 turns.")
