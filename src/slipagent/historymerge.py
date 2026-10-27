"""Layered summaries over immutable first-pass turn summaries.

A run starts with a user request and ends when the model replies without tool
calls or the user prompts again. History is cut into blocks that close at a run
boundary once they hold ten turns, or at 20 turns within a long run. Each block
gets one narrative as soon as all its turns have finished first-pass compaction,
built from their summaries; each five consecutive blocks get one era as soon as
those narratives exist. Sessions that already have fixed 20-turn narratives keep
fixed blocks through the last of them. Every level is kept; the context view
chooses which one to serve. The step compactor schedules these jobs with turn
summaries.
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
# A shorter block's narrative rarely shrinks its turn summaries enough to be accepted.
NARRATIVE_MIN_STEPS = 10
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


def servable(record: Any, interval: int) -> TypeGuard[dict[str, Any]]:
    """A fixed-block record from before runs, or a run-block record within one span ending at its anchor.

    Turns whose first pass failed are gaps, so the listed turns need not be consecutive.
    """
    if not isinstance(record, dict):
        return False
    if "runs" not in record:
        return aligned(record, interval)
    anchor, ids = record.get("step_id"), record.get("step_ids")
    return (type(anchor) is int and anchor > 0 and isinstance(ids, list) and bool(ids)
            and all(type(i) is int and anchor - interval < i <= anchor for i in ids))


def historical_records(steps: list[HistoryStep]) -> list[dict[str, Any]]:
    """Narratives, oldest first."""
    return [getattr(step, "_disk_group", record) for step in steps
            if servable(record := getattr(step, "historical_summary", None), MERGE_INTERVAL)]


def era_records(steps: list[HistoryStep]) -> list[dict[str, Any]]:
    """Eras, oldest first."""
    return [getattr(step, "_disk_era", record) for step in steps
            if servable(record := getattr(step, "era_summary", None), ERA_INTERVAL)]


def _model_record(step: HistoryStep) -> dict[str, Any] | None:
    disk = getattr(step, "_disk", None)
    return disk["model"] if disk is not None else None


def _prompted(step: HistoryStep) -> bool:
    record = _model_record(step)
    if record is not None:
        return bool(record.get("user_prompts"))
    return any(message.role == "user" for message in step.messages)


def _calls_tools(step: HistoryStep) -> bool:
    record = _model_record(step)
    if record is not None:
        return bool(record.get("tool_calls"))
    return any(message.tool_calls for message in step.messages if message.role == "assistant")


def run_numbers(steps: list[HistoryStep]) -> list[int]:
    """Each turn's run, numbered from one: user input starts a run, as does a turn after a reply without tools."""
    runs: list[int] = []
    run, ended = 0, True
    for step in steps:
        if ended or _prompted(step):
            run += 1
        runs.append(run)
        ended = not _calls_tools(step)
    return runs


def next_run(steps: list[HistoryStep], runs: list[int], prompted: bool) -> int:
    """The run of the turn being requested."""
    if not runs:
        return 1
    return runs[-1] + 1 if prompted or not _calls_tools(steps[-1]) else runs[-1]


def blocks(steps: list[HistoryStep], runs: list[int]) -> list[tuple[int, int]]:
    """Closed narrative blocks as inclusive turn-index ranges, oldest first.

    A block never reopens: it closes on information from its own turns and the
    turn after it, so later turns cannot move its boundaries.
    """
    fixed_end = max((step.id for step in steps if "runs" not in (record := getattr(step, "historical_summary", None) or {})
                     and aligned(record, MERGE_INTERVAL)), default=0)
    result = [(start, start + MERGE_INTERVAL - 1) for start in range(0, fixed_end, MERGE_INTERVAL)]
    first = fixed_end
    for index in range(fixed_end, len(steps)):
        size = index - first + 1
        run_ended = runs[index + 1] != runs[index] if index + 1 < len(steps) else not _calls_tools(steps[index])
        if size == MERGE_INTERVAL or size >= NARRATIVE_MIN_STEPS and run_ended:
            result.append((first, index))
            first = index + 1
    return result


def _covers(record: Any, steps: list[HistoryStep], first: int, last: int) -> bool:
    """Whether a stored record summarizes this block, gaps aside."""
    return (isinstance(record, dict) and record.get("step_id") == steps[last].id
            and bool(record.get("step_ids")) and record["step_ids"][0] >= steps[first].id)


def _validate_ids(record: dict[str, Any], anchor: int, total: int, interval: int, representation: str) -> None:
    ids = record["step_ids"]
    if (not servable(record, interval) or record["step_id"] != anchor
            or not 1 <= anchor <= total or ids != sorted(set(ids))
            or record["record_type"] != "history_step"
            or record["representation"] != representation
            or type(record["created_at_step"]) is not int
            or not anchor <= record["created_at_step"] <= total):
        raise ValueError("invalid historical summary source steps")
    summary = record["compressed_summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > MERGE_MAX_CHARS:
        raise ValueError("invalid historical narrative")
    runs = record.get("runs")
    if runs is not None and (not isinstance(runs, list) or not runs or runs != sorted(set(runs))
                             or any(type(run) is not int or run < 1 for run in runs)):
        raise ValueError("invalid historical summary runs")


def validate_historical_record(record: Any, anchor: int, total: int) -> None:
    """Validate persisted source identities before they can hide originals."""
    keys = {"record_type", "representation", "step_id", "step_ids", "sources", "compressed_summary", "created_at_step"}
    if not isinstance(record, dict) or set(record) - {"runs"} != keys:
        raise ValueError("invalid historical summary record")
    _validate_ids(record, anchor, total, MERGE_INTERVAL, "historical_summary")
    sources = record["sources"]
    ids = record["step_ids"]
    if not isinstance(sources, list) or len(sources) != len(ids):
        raise ValueError("invalid historical retrieval metadata")
    source_keys = {"step_id", "tool_calls", "artifacts"} | ({"run"} if "runs" in record else set())
    for source, step_id in zip(sources, ids):
        if (not isinstance(source, dict) or set(source) != source_keys
                or source["step_id"] != step_id or "run" in source and source["run"] not in record["runs"]
                or not isinstance(source["tool_calls"], list)
                or not isinstance(source["artifacts"], list)):
            raise ValueError("invalid historical retrieval metadata")
        for call in source["tool_calls"]:
            if (not isinstance(call, dict) or set(call) != {"call_id", "tool_name"}
                    or any(not isinstance(call[key], str) for key in call)):
                raise ValueError("invalid historical tool identity")


def validate_era_record(record: Any, anchor: int, total: int) -> None:
    keys = {"record_type", "representation", "step_id", "step_ids", "compressed_summary", "created_at_step"}
    if not isinstance(record, dict) or set(record) - {"runs"} != keys:
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


def _narrative_source(steps: list[HistoryStep], runs: list[int], first: int,
                      last: int) -> tuple[list[HistoryStep], dict[str, Any]] | None:
    """The block's summarized turns, or None while any turn's first pass is unsettled."""
    batch = []
    source: dict[str, Any] = {"steps": []}
    for index in range(first, last + 1):
        step = steps[index]
        if step.summary is None:
            # A failed first pass is a permanent gap; anything else may still finish.
            if getattr(step, "compaction_status", None) != "failed":
                return None
            continue
        original = step._disk["model"] if hasattr(step, "_disk") else step_record(step.messages)
        summary = step._disk_summary["summary"] if hasattr(step, "_disk_summary") else step.summary
        source["steps"].append({
            "step_id": step.id,
            "run": runs[index],
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
    runs = run_numbers(steps)
    closed = blocks(steps, runs)
    narrative_prompt = Message.system(load_prompt("historical-summary-prompt.md", max_characters=MERGE_MAX_CHARS))
    for first, last in closed:
        anchor = steps[last]
        key = ("narrative", anchor.id)
        if (key in active or _covers(getattr(anchor, "historical_summary", None), steps, first, last)
                or not _due(anchor, "history_merge_attempt", total)):
            continue
        prepared = _narrative_source(steps, runs, first, last)
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
    for start in range(0, len(closed) - ERA_NARRATIVES + 1, ERA_NARRATIVES):
        group = closed[start:start + ERA_NARRATIVES]
        anchor = steps[group[-1][1]]
        key = ("era", anchor.id)
        if (key in active or _covers(getattr(anchor, "era_summary", None), steps, group[0][0], group[-1][1])
                or not _due(anchor, "era_attempt", total)):
            continue
        narratives = []
        for first, last in group:
            record = getattr(steps[last], "historical_summary", None)
            if not _covers(record, steps, first, last):
                break
            narratives.append(record)
        if len(narratives) < ERA_NARRATIVES:
            continue
        era_source = {"narratives": [{"step_ids": record["step_ids"],
                                      "runs": sorted({runs[step_id - 1] for step_id in record["step_ids"]}),
                                      "compressed_summary": record["compressed_summary"]}
                                     for record in narratives]}
        if message_tokens([era_prompt, Message.user(era_source)]) > budget:
            continue
        setattr(anchor, "era_attempt", total)
        if owner.on_step is not None:
            owner.on_step(anchor)  # type: ignore[arg-type]
        return key, _era(owner, anchor, narratives, copy.deepcopy(era_source), runs, era_prompt, client,
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
                  "runs": sorted({item["run"] for item in source["steps"]}),
                  "sources": [{key: item[key] for key in ("step_id", "run", "tool_calls", "artifacts")}
                              for item in source["steps"]],
                  "compressed_summary": summary, "created_at_step": created_at}
        validate_historical_record(record, anchor.id, created_at)
        return record

    replaced = [Message.user(step.context_messages(compressed=True)[0].content) for step in batch]

    async def operation(destination: APIClient, selected: str, profile: RequestProfile | None,
                        allowance: int | None) -> dict[str, Any]:
        return await _summarize(owner, system, source, replaced, destination, selected, profile,
                                allowance, generation, build)

    await _publish(owner, anchor, "historical_summary", operation, (client, model, capabilities, reserve),
                   generation, f"turns {batch[0].id}–{batch[-1].id}")


async def _era(owner: StepCompactor, anchor: HistoryStep, narratives: list[dict[str, Any]], source: dict[str, Any],
               runs: list[int], system: Message, client: APIClient, model: str, capabilities: RequestProfile | None,
               reserve: int, created_at: int, generation: int) -> None:
    def build(summary: str) -> dict[str, Any]:
        ids = [step_id for narrative in narratives for step_id in narrative["step_ids"]]
        record = {"record_type": "history_step", "representation": "era_summary", "step_id": anchor.id,
                  "step_ids": ids, "runs": sorted({runs[step_id - 1] for step_id in ids}),
                  "compressed_summary": summary, "created_at_step": created_at}
        validate_era_record(record, anchor.id, created_at)
        return record

    replaced = [Message.user(narrative) for narrative in narratives]

    async def operation(destination: APIClient, selected: str, profile: RequestProfile | None,
                        allowance: int | None) -> dict[str, Any]:
        return await _summarize(owner, system, source, replaced, destination, selected, profile,
                                allowance, generation, build)

    ids = [step_id for narrative in narratives for step_id in narrative["step_ids"]]
    await _publish(owner, anchor, "era_summary", operation, (client, model, capabilities, reserve),
                   generation, f"the era of turns {ids[0]}–{ids[-1]}")
