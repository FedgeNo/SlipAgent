"""Isolated whole-step summaries, queued immediately after complete tool batches.

Each background job freezes its step, model, and
request profile; it never receives the conversation context or executes tools.
Reset cancels jobs and invalidates late results. The registry owns this service
so it survives behavior reloads and closes before the shared API client.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from .capabilities import ModelCapabilities
from .context import SUMMARY_MAX_CHARS, CompletedStep, message_tokens
from .openrouter import OpenRouterClient
from .types import Message, Usage
from .prompts import COMPACTION_PROMPT

COMPACTION_TIMEOUT = 20 * 60

class StepCompactor:
    def __init__(self, on_usage: Callable[[Usage], None], on_error: Callable[[str], None]) -> None:
        self.on_usage = on_usage
        self.on_error = on_error
        self.jobs: set[asyncio.Task[None]] = set()
        self.generation = 0
        self.on_step: Callable[[CompletedStep], None] | None = None

    def submit(self, step: CompletedStep, client: OpenRouterClient, model: str,
               capabilities: ModelCapabilities | None, max_tokens: int | None) -> None:
        source = step.compaction_input()
        generation = self.generation
        job = asyncio.create_task(self._compact(step, source, client, model, capabilities, max_tokens, generation))
        self.jobs.add(job)
        job.add_done_callback(self.jobs.discard)

    async def _compact(self, step: CompletedStep, source: str, client: OpenRouterClient, model: str,
                       capabilities: ModelCapabilities | None, max_tokens: int | None, generation: int) -> None:
        try:
            if generation != self.generation:
                return
            messages = [Message.system(COMPACTION_PROMPT.rstrip("\n")),
                        Message.user(source)]
            reserve = max_tokens or 8192
            if capabilities is not None:
                context_limit = min(capabilities.context_length, capabilities.max_prompt_tokens or capabilities.context_length)
                if max_tokens is None:
                    reserve = min(reserve, max(256, context_limit // 8))
                reserve = min(reserve, capabilities.max_completion_tokens or reserve)
                if message_tokens(messages) + reserve > int(context_limit * .85):
                    raise ValueError("the complete step exceeds the selected model's compaction context budget")
            options: dict[str, Any] = {}
            if isinstance(client, OpenRouterClient):
                options["request_profile"] = capabilities
                options["disable_timeout"] = True
            async with asyncio.timeout(COMPACTION_TIMEOUT):
                completion = await client.chat(
                    model=model, messages=messages, tools=None, temperature=.2,
                    max_tokens=reserve if capabilities is not None and "max_tokens" in capabilities.parameters else None,
                    extra_body={"provider": capabilities.provider_preferences() if capabilities is not None else {"require_parameters": True}},
                    **options,
                )
            if generation != self.generation:
                return
            self.on_usage(completion.usage)
            if completion.response_error or completion.tool_calls or completion.finish_reason == "length":
                raise ValueError(completion.response_error or "the compactor returned tools or an incomplete summary")
            summary = completion.text.strip()
            # Some schema-oriented models wrap even an ordinary summary.
            # Unwrap only a single recognized text field, without repairs.
            try:
                value = json.loads(summary)
            except ValueError:
                value = None
            if isinstance(value, dict) and set(value) in ({"summary"}, {"response"}):
                summary = next(iter(value.values()))
            if not isinstance(summary, str) or not summary.strip() or len(summary) > SUMMARY_MAX_CHARS:
                raise ValueError("the compactor returned an empty or oversized summary")
            summary.encode("utf-8")
            step.summary = summary.strip()
            step.results_summarized = True
            step.compaction_status = "complete"
            callback = getattr(self, "on_step", None)
            if callback is not None:
                callback(step)
        except asyncio.CancelledError:
            step.compaction_status = "cancelled"
            raise
        except Exception as exc:
            if generation == self.generation:
                step.compaction_status = "failed"
                step.compaction_error = f"{type(exc).__name__}: {exc}"
                self.on_error(f"Could not summarize step {step.id}: {exc}. Its full original remains available through recall_history.")
                callback = getattr(self, "on_step", None)
                if callback is not None:
                    callback(step)

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
