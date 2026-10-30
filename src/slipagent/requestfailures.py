"""Durable failed-request turns and model-visible retry counts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import HistoryStep

FAILURE_OUTCOMES = {"request_error", "context_overflow", "rejected", "loop_stopped", "cancelled"}
# The stream stopped before the reply finished, and its partial reply and
# reasoning are kept. Rejected and looping replies did finish.
INCOMPLETE_OUTCOMES = {"request_error", "cancelled"}


def retry_state(steps: list[HistoryStep]) -> dict[str, Any] | None:
    failed = [step for step in steps if getattr(step, "request_failure", None) is not None]
    if not failed or failed[-1] is not steps[-1]:
        return None
    consecutive = []
    for step in reversed(steps):
        if getattr(step, "request_failure", None) is None:
            break
        consecutive.append(step.id)
    return {"consecutive_failures": len(consecutive), "total_failed_attempts": len(failed),
            "failed_step_ids": list(reversed(consecutive)), "last_failed_step": failed[-1].id}


def validate_failure(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "outcome", "error", "response_excerpt", "diagnostic_attempt", "request_number",
        "consecutive_failures", "total_failed_attempts", "tools_executed",
    }:
        raise ValueError("invalid failed request turn")
    if value["outcome"] not in FAILURE_OUTCOMES or value["tools_executed"] is not False:
        raise ValueError("invalid failed request outcome")
    if any(not isinstance(value[key], str) for key in ("error", "response_excerpt")):
        raise ValueError("invalid failed request details")
    for key in ("request_number", "consecutive_failures", "total_failed_attempts"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError("invalid failed request counter")
    attempt = value["diagnostic_attempt"]
    if attempt is not None and (type(attempt) is not int or attempt < 1):
        raise ValueError("invalid failed request diagnostic reference")


def is_incomplete(failure: dict[str, Any]) -> bool:
    return failure["outcome"] in INCOMPLETE_OUTCOMES


def failure_summary(failure: dict[str, Any]) -> str:
    error: str = failure["error"]
    return (f"Harness-recorded failed model request: {failure['outcome']}. "
            + error[:2000] + ("…" if len(error) > 2000 else "")
            + f" Consecutive failures: {failure['consecutive_failures']}; "
            + f"session failed attempts: {failure['total_failed_attempts']}. "
            + "No tools from this attempt executed. "
            + ("The partial response is kept as an incomplete turn and was not accepted." if is_incomplete(failure)
               else "The partial response was not accepted."))


def failed_steps(steps: list[HistoryStep]) -> dict[int, dict[str, Any]]:
    """Each failed turn's record by step ID."""
    return {step.id: failure for step in steps if (failure := getattr(step, "request_failure", None)) is not None}
