"""Layered summaries over immutable first-pass turn summaries.

Each fixed block of 20 turns (1–20, 21–40, ...) gets one narrative as soon as
all its turns have finished first-pass compaction, built from their summaries.
Each fixed block of five narratives (turns 1–100, 101–200, ...) gets one era as
soon as those narratives exist. Every level is kept; the context view chooses
which one to serve. The step compactor schedules these jobs with turn summaries.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeGuard

from .capabilities import RequestProfile, model_context_limit
from .prompts import load_prompt
from .records import step_record
from .types import Message

if TYPE_CHECKING:
    from .api import APIClient
    from .compaction import StepCompactor
    from .context import HistoryStep

MERGE_INTERVAL = 20
ERA_NARRATIVES = 5
ERA_INTERVAL = MERGE_INTERVAL * ERA_NARRATIVES
MERGE_MAX_CHARS = 12_000

Route = tuple["APIClient", str, RequestProfile | None, int | None]


def aligned(record: Any, interval: int) -> TypeGuard[dict[str, Any]]:
    """Whether a record covers exactly one fixed block ending at its anchor."""
    if not isinstance(record, dict):
        return False
    anchor, ids = record.get("step_id"), record.get("step_ids")
    return (type(anchor) is int and anchor > 0 and anchor % interval == 0 and isinstance(ids, list) and bool(ids)
            and all(type(i) is int and anchor - interval < i <= anchor for i in ids))


def historical_records(steps: list[HistoryStep]) -> list[dict[str, Any]]:
    """Narratives for fixed 20-turn blocks, oldest first."""
    return [getattr(step, "_disk_group", record) for step in steps
            if aligned(record := getattr(step, "historical_summary", None), MERGE_INTERVAL)]


def era_records(steps: list[HistoryStep]) -> list[dict[str, Any]]:
    """Eras for fixed 100-turn blocks, oldest first."""
    return [getattr(step, "_disk_era", record) for step in steps
            if aligned(record := getattr(step, "era_summary", None), ERA_INTERVAL)]


def _validate_ids(record: dict[str, Any], anchor: int, total: int, interval: int, representation: str) -> None:
    ids = record["step_ids"]
    if (not aligned(record, interval) or record["step_id"] != anchor
            or not 1 <= anchor <= total or ids != sorted(set(ids))
            or record["record_type"] != "history_step"
            or record["representation"] != representation
            or type(record["created_at_step"]) is not int
            or not anchor <= record["created_at_step"] <= total):
        raise ValueError("invalid historical summary source steps")
    summary = record["compressed_summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > MERGE_MAX_CHARS:
        raise ValueError("invalid historical narrative")


def validate_historical_record(record: Any, anchor: int, total: int) -> None:
    """Validate persisted source identities before they can hide originals."""
    if not isinstance(record, dict) or set(record) != {
        "record_type", "representation", "step_id", "step_ids", "sources",
        "compressed_summary", "created_at_step",
    }:
        raise ValueError("invalid historical summary record")
    _validate_ids(record, anchor, total, MERGE_INTERVAL, "historical_summary")
    sources = record["sources"]
    ids = record["step_ids"]
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


def validate_era_record(record: Any, anchor: int, total: int) -> None:
    if not isinstance(record, dict) or set(record) != {
        "record_type", "representation", "step_id", "step_ids", "compressed_summary", "created_at_step",
    }:
        raise ValueError("invalid era summary record")
    _validate_ids(record, anchor, total, ERA_INTERVAL, "era_summary")


def _limits(owner: StepCompactor, model: str, capabilities: RequestProfile | None,
            max_tokens: int | None) -> tuple[int, int]:
    """The provisional source budget before a route is chosen, and the reply reserve."""
    routed = getattr(owner, "routing", None) is not None
    limit = 128_000
    if routed:
        limit = 1_000_000
    elif capabilities is not None:
        limit = model_context_limit(model, capabilities,
                    min(capabilities.context_length, capabilities.max_prompt_tokens or capabilities.context_length))
    reserve = (8192 if routed else min(max_tokens or 8192, capabilities.max_completion_tokens or 8192)
               if capabilities else (max_tokens or 8192))
    return int(limit * .85) - reserve, reserve


def _due(anchor: HistoryStep, attribute: str, total: int) -> bool:
    # Record attempts, including failures, so a failing block waits twenty
    # turns rather than retrying every turn. The marker survives resume.
    attempt = getattr(anchor, attribute, 0)
    return not attempt or total - attempt >= MERGE_INTERVAL


def _narrative_source(steps: list[HistoryStep], block: int) -> tuple[list[HistoryStep], dict[str, Any]] | None:
    """The block's summarized turns, or None while any turn's first pass is unsettled."""
    members = steps[block * MERGE_INTERVAL:(block + 1) * MERGE_INTERVAL]
    if len(members) < MERGE_INTERVAL:
        return None
    batch = []
    source: dict[str, Any] = {"steps": []}
    for step in members:
        if step.summary is None:
            # A failed first pass is a permanent gap; anything else may still finish.
            if getattr(step, "compaction_status", None) != "failed":
                return None
            continue
        original = step._disk["model"] if hasattr(step, "_disk") else step_record(step.messages)
        summary = step._disk_summary["summary"] if hasattr(step, "_disk_summary") else step.summary
        source["steps"].append({
            "step_id": step.id,
            "tool_calls": [{"call_id": call["call_id"], "tool_name": call["tool_name"]}
                           for call in original["tool_calls"]],
            "artifacts": step.context_messages(compressed=True)[0].content.get("artifacts", []),
            "compressed_summary": summary,
        })
        batch.append(step)
    return (batch, source) if batch else None


def schedule_history_merge(owner: StepCompactor, steps: list[HistoryStep], client: APIClient,
                           model: str, capabilities: RequestProfile | None, max_tokens: int | None) -> None:
    """Entry point for a compactor worker started before a reload; the dispatcher owns this work."""
    owner.track(steps, client, model, capabilities, max_tokens)


def ready_job(owner: StepCompactor, steps: list[HistoryStep], active: set[Any],
              route: Route) -> tuple[Any, Coroutine[Any, Any, None]] | None:
    """The oldest narrative, then the oldest era, whose inputs are all available."""
    from .context import message_tokens

    client, model, capabilities, max_tokens = route
    total = len(steps)
    budget, reserve = _limits(owner, model, capabilities, max_tokens)
    narrative_prompt = Message.system(load_prompt("historical-summary-prompt.md", max_characters=MERGE_MAX_CHARS))
    for block in range(total // MERGE_INTERVAL):
        anchor = steps[(block + 1) * MERGE_INTERVAL - 1]
        key = ("narrative", block)
        if (key in active or aligned(getattr(anchor, "historical_summary", None), MERGE_INTERVAL)
                or not _due(anchor, "history_merge_attempt", total)):
            continue
        prepared = _narrative_source(steps, block)
        if prepared is None:
            continue
        batch, source = prepared
        if message_tokens([narrative_prompt, Message.user(source)]) > budget:
            continue  # Complete source summaries cannot fit this route; never truncate them.
        setattr(anchor, "history_merge_attempt", total)
        if owner.on_step is not None:
            owner.on_step(anchor)  # type: ignore[arg-type]
        return key, _merge(owner, anchor, batch, copy.deepcopy(source), narrative_prompt, client,
                           model, capabilities, reserve, total, owner.generation)
    era_prompt = Message.system(load_prompt("era-summary-prompt.md", max_characters=MERGE_MAX_CHARS))
    for era in range(total // ERA_INTERVAL):
        anchor = steps[(era + 1) * ERA_INTERVAL - 1]
        key = ("era", era)
        if (key in active or aligned(getattr(anchor, "era_summary", None), ERA_INTERVAL)
                or not _due(anchor, "era_attempt", total)):
            continue
        narratives = []
        for block in range(era * ERA_NARRATIVES, (era + 1) * ERA_NARRATIVES):
            record = getattr(steps[(block + 1) * MERGE_INTERVAL - 1], "historical_summary", None)
            if not aligned(record, MERGE_INTERVAL):
                break
            narratives.append(record)
        if len(narratives) < ERA_NARRATIVES:
            continue
        era_source = {"narratives": [{"step_ids": record["step_ids"], "compressed_summary": record["compressed_summary"]}
                                     for record in narratives]}
        if message_tokens([era_prompt, Message.user(era_source)]) > budget:
            continue
        setattr(anchor, "era_attempt", total)
        if owner.on_step is not None:
            owner.on_step(anchor)  # type: ignore[arg-type]
        return key, _era(owner, anchor, narratives, copy.deepcopy(era_source), era_prompt, client,
                         model, capabilities, reserve, total, owner.generation)
    return None


async def _summarize(owner: StepCompactor, system: Message, source: dict[str, Any], replaced: list[Message],
                     destination: APIClient, selected: str, profile: RequestProfile | None,
                     allowance: int | None, generation: int, build: Any) -> dict[str, Any]:
    from .api import APIClient
    from .compaction import COMPACTION_TIMEOUT, CompactionBudgetError, decode_summary
    from .context import message_tokens

    options: dict[str, Any] = {}
    if isinstance(destination, APIClient):
        options.update(request_profile=profile, disable_timeout=True, single_attempt=True)
    limit = min(profile.context_length, profile.max_prompt_tokens or profile.context_length) if profile else 128_000
    limit = model_context_limit(selected, profile, limit)
    if message_tokens([system, Message.user(source)]) + (allowance or 8192) > int(limit * .85):
        raise CompactionBudgetError("the grouped summaries exceed the compression provider's context budget")
    async with asyncio.timeout(COMPACTION_TIMEOUT):
        completion = await destination.chat(
            model=selected, messages=[system, Message.user(source)], tools=None,
            max_tokens=allowance if profile is not None and "max_tokens" in profile.parameters else None,
            extra_body={}, **options,
        )
    if generation != owner.generation:
        return {}
    owner.on_usage(completion.usage)
    value = decode_summary(completion)
    if not isinstance(value, dict) or set(value) != {"summary"}:
        raise ValueError("the historical compactor must return only summary")
    record: dict[str, Any] = build(value["summary"])
    if message_tokens([Message.user(record)]) >= message_tokens(replaced):
        raise ValueError("the historical narrative did not reduce the supplied summaries")
    return record


async def _publish(owner: StepCompactor, anchor: HistoryStep, attribute: str, operation: Any,
                   route: Route, generation: int, label: str) -> None:
    client, model, capabilities, reserve = route
    try:
        record = await owner.request(operation, client, model, capabilities, reserve, generation)
        if generation != owner.generation or not record:
            return
        setattr(anchor, attribute, record)
        if owner.on_step is not None:
            owner.on_step(anchor)  # type: ignore[arg-type]
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if generation == owner.generation:
            owner.on_error(f"Could not summarize {label}: {exc}. Individual summaries and originals remain "
                           "available; another attempt waits at least 20 turns.")


async def _merge(owner: StepCompactor, anchor: HistoryStep, batch: list[HistoryStep], source: dict[str, Any],
                 system: Message, client: APIClient, model: str, capabilities: RequestProfile | None,
                 reserve: int, created_at: int, generation: int) -> None:
    def build(summary: str) -> dict[str, Any]:
        record = {"record_type": "history_step", "representation": "historical_summary",
                  "step_id": anchor.id, "step_ids": [step.id for step in batch],
                  "sources": [{key: item[key] for key in ("step_id", "tool_calls", "artifacts")}
                              for item in source["steps"]],
                  "compressed_summary": summary, "created_at_step": created_at}
        validate_historical_record(record, anchor.id, created_at)
        return record

    replaced = [Message.user(step.context_messages(compressed=True)[0].content) for step in batch]

    async def operation(destination: APIClient, selected: str, profile: RequestProfile | None,
                        allowance: int | None) -> dict[str, Any]:
        return await _summarize(owner, system, source, replaced, destination, selected, profile,
                                allowance, generation, build)

    first, last = anchor.id - MERGE_INTERVAL + 1, anchor.id
    await _publish(owner, anchor, "historical_summary", operation, (client, model, capabilities, reserve),
                   generation, f"turns {first}–{last}")


async def _era(owner: StepCompactor, anchor: HistoryStep, narratives: list[dict[str, Any]], source: dict[str, Any],
               system: Message, client: APIClient, model: str, capabilities: RequestProfile | None,
               reserve: int, created_at: int, generation: int) -> None:
    def build(summary: str) -> dict[str, Any]:
        record = {"record_type": "history_step", "representation": "era_summary", "step_id": anchor.id,
                  "step_ids": [step_id for narrative in narratives for step_id in narrative["step_ids"]],
                  "compressed_summary": summary, "created_at_step": created_at}
        validate_era_record(record, anchor.id, created_at)
        return record

    replaced = [Message.user(narrative) for narrative in narratives]

    async def operation(destination: APIClient, selected: str, profile: RequestProfile | None,
                        allowance: int | None) -> dict[str, Any]:
        return await _summarize(owner, system, source, replaced, destination, selected, profile,
                                allowance, generation, build)

    first, last = anchor.id - ERA_INTERVAL + 1, anchor.id
    await _publish(owner, anchor, "era_summary", operation, (client, model, capabilities, reserve),
                   generation, f"the era of turns {first}–{last}")
