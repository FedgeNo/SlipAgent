"""Whole-step summaries and filtered thoughts, queued after complete tool batches.

Each background job freezes its step, model, and
request profile and up to 25 preceding steps as context. It never executes tools.
Reset cancels jobs and invalidates late results. The registry owns this service
so it survives behavior reloads and closes before the shared API client.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from typing import Any

from .capabilities import RequestProfile
from .context import SUMMARY_MAX_CHARS, CompletedStep, HistoryStep, message_tokens
from .api import APIClient, APIResponseError, APITransportError, RETRYABLE_STATUS
from .types import Message, Usage
from .prompts import load_prompt

COMPACTION_TIMEOUT = 20 * 60
COMPACTION_RETRIES = 5

class StepCompactor:
    def __init__(self, on_usage: Callable[[Usage], None], on_error: Callable[[str], None]) -> None:
        self.on_usage = on_usage
        self.on_error = on_error
        self.jobs: set[asyncio.Task[None]] = set()
        self.generation = 0
        self.on_step: Callable[[CompletedStep], None] | None = None

    def submit(self, step: CompletedStep, client: APIClient, model: str,
               capabilities: RequestProfile | None, max_tokens: int | None, *,
               history: list[HistoryStep] | None = None) -> None:
        source = copy.deepcopy(step.compaction_input())
        source['history'] = copy.deepcopy([prior.context_messages(compressed=prior.summary is not None)[0].content
                                          for prior in (history or [])[-25:] if prior.id < step.id])
        source['history_omitted'] = 0
        generation = self.generation
        job = asyncio.create_task(self._compact(step, source, client, model, capabilities, max_tokens, generation))
        self.jobs.add(job)
        job.add_done_callback(self.jobs.discard)

    async def _compact(self, step: CompletedStep, source: dict[str, Any], client: APIClient, model: str,
                       capabilities: RequestProfile | None, max_tokens: int | None, generation: int) -> None:
        try:
            attempt = 0
            while True:
                try:
                    await self._compact_once(step, source, client, model, capabilities, max_tokens, generation)
                    return
                except Exception as exc:
                    if generation != self.generation:
                        return
                    if isinstance(client, APIClient) and isinstance(exc, APIResponseError):
                        transient = isinstance(exc, APITransportError) or exc.status_code in RETRYABLE_STATUS
                        if transient and await client.wait_retry(attempt, exc):
                            attempt += 1
                            continue
                    elif attempt < COMPACTION_RETRIES:
                        delay = (client._backoff(attempt, getattr(exc, "retry_after", None))
                                 if isinstance(client, APIClient) else min(2 ** attempt, 30))
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    step.compaction_status = "failed"
                    step.compaction_error = f"{type(exc).__name__}: {exc}"
                    self.on_error(f"Could not summarize step {step.id}: {exc}. Its full original remains available through recall_history.")
                    callback = getattr(self, "on_step", None)
                    if callback is not None:
                        callback(step)
                    return
        except asyncio.CancelledError:
            step.compaction_status = "cancelled"
            raise

    async def _compact_once(self, step: CompletedStep, source: dict[str, Any], client: APIClient, model: str,
                            capabilities: RequestProfile | None, max_tokens: int | None, generation: int) -> None:
        try:
            if generation != self.generation:
                return
            messages = [Message.system(load_prompt("background-summary-prompt.md")),
                        Message.user(source)]
            reserve = max_tokens or 8192
            if capabilities is not None:
                context_limit = min(capabilities.context_length, capabilities.max_prompt_tokens or capabilities.context_length)
                if max_tokens is None:
                    reserve = min(reserve, max(256, context_limit // 8))
                reserve = min(reserve, capabilities.max_completion_tokens or reserve)
                while source['history'] and message_tokens(messages) + reserve > int(context_limit * .85):
                    source['history'].pop(0)
                    source['history_omitted'] += 1
                if message_tokens(messages) + reserve > int(context_limit * .85):
                    raise ValueError("the complete step exceeds the selected model's compaction context budget")
            options: dict[str, Any] = {}
            if isinstance(client, APIClient):
                options["request_profile"] = capabilities
                options["disable_timeout"] = True
                options["single_attempt"] = True
            async with asyncio.timeout(COMPACTION_TIMEOUT):
                completion = await client.chat(
                    model=model, messages=messages, tools=None, temperature=.2,
                    max_tokens=reserve if capabilities is not None and "max_tokens" in capabilities.parameters else None,
                    extra_body={},
                    **options,
                )
            if generation != self.generation:
                return
            self.on_usage(completion.usage)
            if completion.response_error or completion.tool_calls or completion.finish_reason == "length":
                raise ValueError(completion.response_error or "the compactor returned tools or an incomplete summary")
            value = json.loads(completion.text)
            if not isinstance(value, dict) or set(value) != {'summary', 'reasoning_summary'}:
                raise ValueError('the compactor must return summary and reasoning_summary as separate JSON fields')
            summary = value['summary']
            thoughts = value['reasoning_summary']
            if not isinstance(summary, str) or not summary.strip() or len(summary) > SUMMARY_MAX_CHARS:
                raise ValueError("the compactor returned an empty or oversized summary")
            if not isinstance(thoughts, str) or len(thoughts) > SUMMARY_MAX_CHARS:
                raise ValueError('the compactor returned invalid or oversized filtered thoughts')
            summary.encode("utf-8")
            thoughts.encode('utf-8')
            # Publish both fields together after validation. Empty means filtering
            # found nothing useful; None means filtering has not completed.
            step.summary = summary.strip()
            step.reasoning_summary = thoughts.strip() if source['reasoning'] else ''
            step.results_summarized = True
            step.compaction_status = "complete"
            callback = getattr(self, "on_step", None)
            if callback is not None:
                callback(step)
        except asyncio.CancelledError:
            step.compaction_status = "cancelled"
            raise

    def reset(self) -> None:
        self.generation += 1
        for job in tuple(self.jobs):
            job.cancel()

    async def wait(self) -> None:
        while self.jobs:
            jobs = tuple(self.jobs)
            await asyncio.gather(*jobs, return_exceptions=True)
            # Completed-task callbacks can still be queued when gather returns.
            # Remove this settled batch directly rather than spin before they run.
            self.jobs.difference_update(jobs)

    async def aclose(self) -> None:
        self.reset()
        await self.wait()
