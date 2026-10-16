"""Whole-step summaries and filtered thoughts, queued after complete tool batches.

One worker summarizes immutable completed steps in order. A bounded ready queue
holds references; deferred steps retain request metadata on their originals.
The worker materializes one request at a time, rotating configured background
providers and using up to 25 preceding steps as context. It never executes tools.
Reset cancels jobs and invalidates late results. The registry owns this service
so it survives behavior reloads and closes before the shared API client.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections import deque
from itertools import chain
from collections.abc import Callable
from typing import Any

from .capabilities import RequestProfile, model_context_limit
from .context import SUMMARY_MAX_CHARS, CompletedStep, HistoryStep, message_tokens
from .api import APIClient, APIResponseError, APITransportError, RETRYABLE_STATUS
from .types import Completion, Message, Usage
from .compressionproviders import CompressionProviders, CompressionUnavailableError
from .lifecycle import finish_cleanup
from .prompts import load_prompt

COMPACTION_TIMEOUT = 20 * 60
COMPACTION_RETRIES = 5
COMPACTION_QUEUE_SIZE = 32
COMPACTION_GRACE_PERIOD = 120


class CompactionBudgetError(ValueError):
    """The step cannot fit the compaction request; another attempt gives the same answer."""


def decode_summary(completion: Completion) -> Any:
    """Accept a complete JSON answer, including a single Markdown code fence."""
    if completion.response_error or completion.tool_calls or completion.finish_reason == "length":
        raise ValueError(completion.response_error or "the compactor returned tools or an incomplete summary")
    text = completion.text.strip().removeprefix("\ufeff").strip()
    if not text:
        reasoning = len(completion.message.reasoning or "")
        raise ValueError(f"the compactor returned no answer text (finish: {completion.finish_reason or 'unknown'}; "
                         f"reasoning: {reasoning} characters)")
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence[1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        preview = json.dumps(text[:80], ensure_ascii=True)
        raise ValueError(f"the compactor returned invalid JSON at line {exc.lineno}, column {exc.colno}; "
                         f"answer starts with {preview}") from exc


def _frozen_step(step: HistoryStep) -> HistoryStep:
    """Share immutable text while freezing message and metadata bindings."""
    messages = getattr(step, "_compaction_messages", None)
    if messages is None:
        messages = [copy.copy(message) for message in step.messages]
        if isinstance(step, CompletedStep):
            step.__dict__["_compaction_messages"] = messages
    frozen = copy.copy(step)
    frozen.messages = messages
    if isinstance(frozen, CompletedStep):
        frozen.__dict__.pop("_compaction_request", None)
        frozen.tool_results = [message for message in messages if message.role == "tool"]
    return frozen

class StepCompactor:
    def __init__(self, on_usage: Callable[[Usage], None], on_error: Callable[[str], None]) -> None:
        self.on_usage = on_usage
        self.on_error = on_error
        self.jobs: set[asyncio.Task[None]] = set()
        self.generation = 0
        self.on_step: Callable[[CompletedStep], None] | None = None
        self.routing: CompressionProviders | None = None

    def submit(self, step: CompletedStep, client: APIClient, model: str,
               capabilities: RequestProfile | None, max_tokens: int | None, *,
               history: list[HistoryStep] | None = None,
               conversation: list[HistoryStep] | None = None) -> None:
        # Deferred work stays on the already-retained original step. Only the
        # active request materializes text; the ready queue holds bounded refs.
        step.__dict__["_compaction_request"] = (client, model, capabilities, max_tokens,
                                               tuple(_frozen_step(prior) for prior in (history or [])[-25:]),
                                               conversation, _frozen_step(step))
        queue = self.__dict__.setdefault("_ready", deque())
        if conversation is not None:
            self.__dict__["_conversation"] = conversation
        else:
            self.__dict__.setdefault("_standalone_steps", []).append(step)
        if len(queue) < COMPACTION_QUEUE_SIZE:
            queue.append(step)
        else:
            step.compaction_status = "deferred"
        generation = self.generation
        name = f"slipagent-step-compactor-{generation}"
        if not any(job.get_name() == name and not job.done() for job in self.jobs):
            previous = tuple(job for job in self.jobs if not job.done())
            job = asyncio.create_task(self._drain(generation, previous), name=name)
            self.jobs.add(job)
            job.add_done_callback(self.jobs.discard)
        if conversation is not None:
            from .historymerge import schedule_history_merge
            schedule_history_merge(self, conversation, client, model, capabilities, max_tokens)

    async def _drain(self, generation: int, previous: tuple[asyncio.Task[None], ...]) -> None:
        if previous:
            await asyncio.gather(*previous, return_exceptions=True)
        queue = self.__dict__["_ready"]
        while generation == self.generation:
            if not queue:
                for step in chain(self.__dict__.get("_conversation", []), self.__dict__.get("_standalone_steps", [])):
                    if getattr(step, "_compaction_request", None) is not None:
                        queue.append(step)
                        if len(queue) == COMPACTION_QUEUE_SIZE:
                            break
            if not queue:
                self.__dict__.pop("_standalone_steps", None)
                return
            step = queue.popleft()
            request = step.__dict__.pop("_compaction_request", None)
            if request is None:
                continue
            client, model, capabilities, max_tokens, history, conversation, frozen = request
            source = copy.deepcopy(frozen.compaction_input())
            if not isinstance(client, APIClient):
                from .sources import materialize
                source = materialize(source)
            source['history'] = copy.deepcopy([prior.context_messages(compressed=prior.summary is not None)[0].content
                                              for prior in history if prior.id < step.id])
            source['history_omitted'] = 0
            step.compaction_status = "pending"
            await self._compact(step, source, client, model, capabilities, max_tokens, generation)
            if conversation is not None and generation == self.generation:
                from .historymerge import schedule_history_merge
                schedule_history_merge(self, conversation, client, model, capabilities, max_tokens)

    async def _compact(self, step: CompletedStep, source: dict[str, Any], client: APIClient, model: str,
                       capabilities: RequestProfile | None, max_tokens: int | None, generation: int) -> None:
        try:
            attempt = 0
            while True:
                try:
                    routing = getattr(self, "routing", None)
                    if routing is not None:
                        client, model, capabilities, max_tokens = await routing.select()
                        self.__dict__.pop("_routing_error", None)
                    await self._compact_once(step, source, client, model, capabilities, max_tokens, generation)
                    return
                except Exception as exc:
                    if generation != self.generation:
                        return
                    if await self.wait_request_retry(client, attempt, exc):
                        attempt += 1
                        continue
                    step.compaction_status = "failed"
                    step.compaction_error = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, CompressionUnavailableError):
                        if self.__dict__.get("_routing_error") != str(exc):
                            self.__dict__["_routing_error"] = str(exc)
                            self.on_error(f"Compression unavailable: {exc}. Original turns remain available through recall_history.")
                    else:
                        self.on_error(f"Could not summarize step {step.id}: {exc}. Its full original remains available through recall_history.")
                    callback = getattr(self, "on_step", None)
                    if callback is not None:
                        callback(step)
                    return
        except asyncio.CancelledError:
            step.compaction_status = "cancelled"
            raise
        finally:
            if generation != self.generation and step.compaction_status == "pending":
                step.compaction_status = "cancelled"

    async def wait_request_retry(self, client: APIClient, attempt: int, error: Exception) -> bool:
        """Share transport and malformed-reply recovery across both compactors."""
        if isinstance(client, APIClient) and isinstance(error, (APIResponseError, TimeoutError)):
            if isinstance(error, TimeoutError):
                error = APITransportError("Background compaction request timed out")
            transient = isinstance(error, APITransportError) or error.status_code in RETRYABLE_STATUS
            return transient and await client.wait_retry(attempt, error, persistent=True)
        if attempt >= COMPACTION_RETRIES or isinstance(error, (CompactionBudgetError, CompressionUnavailableError)):
            return False
        delay = (client._backoff(attempt, getattr(error, "retry_after", None))
                 if isinstance(client, APIClient) else min(2 ** attempt, 30))
        await asyncio.sleep(delay)
        return True

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
                context_limit = model_context_limit(model, capabilities, context_limit)
                if max_tokens is None:
                    reserve = min(reserve, max(256, context_limit // 8))
                reserve = min(reserve, capabilities.max_completion_tokens or reserve)
                while source['history'] and message_tokens(messages) + reserve > int(context_limit * .85):
                    source['history'].pop(0)
                    source['history_omitted'] += 1
                if message_tokens(messages) + reserve > int(context_limit * .85):
                    raise CompactionBudgetError("the complete step exceeds the selected model's compaction context budget")
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
            value = decode_summary(completion)
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
        for step in chain(self.__dict__.get("_ready", []), self.__dict__.get("_conversation", []),
                          self.__dict__.get("_standalone_steps", [])):
            if getattr(step, "_compaction_request", None) is not None:
                delattr(step, "_compaction_request")
                step.compaction_status = "cancelled"
        self.__dict__.pop("_ready", None)
        self.__dict__.pop("_conversation", None)
        self.__dict__.pop("_standalone_steps", None)
        for job in tuple(self.jobs):
            job.cancel()

    async def wait(self) -> None:
        while self.jobs:
            jobs = tuple(self.jobs)
            await asyncio.gather(*jobs, return_exceptions=True)
            # Completed-task callbacks can still be queued when gather returns.
            # Remove this settled batch directly rather than spin before they run.
            self.jobs.difference_update(jobs)

    async def settle(self) -> None:
        """Allow one grace period before cancelling work for a session transition."""
        waiter = asyncio.create_task(self.wait(), name="settle compaction")
        try:
            async with asyncio.timeout(COMPACTION_GRACE_PERIOD):
                await asyncio.shield(waiter)
        except TimeoutError:
            pass
        finally:
            if not waiter.done():
                self.reset()
                await finish_cleanup(waiter)

    async def aclose(self) -> None:
        self.reset()
        await self.wait()
        routing = getattr(self, "routing", None)
        if routing is not None:
            await routing.aclose()
