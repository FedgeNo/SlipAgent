"""Own request attempts, recovery, diagnostics, and response acceptance for one step."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

from .api import APIClient, APIContextError, APIResponseError, APITransportError, RETRYABLE_STATUS
from .budget import ContextBudget
from .capabilities import RequestProfile
from .context import ContextError
from .contextlegacy import _visible_response
from .progress import StreamLoopError, StreamLoopGuard
from .prompts import load_prompt, model_prompt_profile
from .protocol import AgentResponse, ResponseFormatError, model_response_format, parse_agent_response
from .streaming import ResponseStream
from .tools.base import ToolRegistry
from .types import Completion, Message, ToolSpec, Usage

from .events import AgentEvent

MAX_MEMORY_ATTEMPTS = 3


class ContextStepLimit(Exception):
    """The current run has exhausted its model-request budget."""


@dataclass(frozen=True)
class RequestSettings:
    model: str
    temperature: float | None
    max_tokens: int | None
    session_id: str


@dataclass(frozen=True)
class RequestHooks:
    context: Callable[[str, float | None], Awaitable[list[Message]]]
    reserve: Callable[[], None]
    exhausted: Callable[[], bool]
    has_pending: Callable[[], bool]
    delivered: Callable[[], None]
    account: Callable[[Usage], None]
    persist: Callable[[], None]
    emit: Callable[[AgentEvent], None]
    stop_event: Callable[[], asyncio.Event]
    stop_requested: Callable[[], bool]
    stop_notice: Callable[[int], str]


@dataclass(frozen=True)
class RequestAccepted:
    completion: Completion
    record: AgentResponse
    usage: Usage


@dataclass(frozen=True)
class RequestStopped:
    text: str
    loop_stopped: bool = False


@dataclass
class RequestDriver:
    client: APIClient
    registry: ToolRegistry
    budget: ContextBudget
    settings: RequestSettings
    hooks: RequestHooks

    async def run(
        self, context: list[Message], specs: list[ToolSpec], step: int, run_step: int,
        capabilities: RequestProfile | None, native_tools: bool,
    ) -> RequestAccepted | RequestStopped:
        self.registry.services.pop("accepted_user_edits", None)
        extra_body: dict[str, Any] = {}
        response_format = model_response_format(self.settings.model, capabilities, native_tools=native_tools)
        if response_format is not None:
            extra_body["response_format"] = response_format
        step_usage = Usage()
        attempts = 0
        overflows = 0
        transport_retries = 0
        budget_fraction = self.budget.fraction
        repair = ""
        start_next_step = False
        while True:
            if start_next_step:
                self.hooks.emit(AgentEvent(kind="step_start", step=step))
                start_next_step = False
            if self.hooks.has_pending():
                context = await self.hooks.context(repair, budget_fraction)
            self.hooks.reserve()
            request_estimate = 0
            diagnostic_id: int | None = None
            diagnostics = self.registry.resources.diagnostics
            xml_tool_calls = model_prompt_profile(self.settings.model).get("xml_tool_calls", False)
            preview = ResponseStream(xml_tool_calls=xml_tool_calls)
            received_content = False
            def end_preview(*, failed: bool = False) -> None:
                self.hooks.emit(AgentEvent(kind="stream_end", step=step))
                if failed and preview.text:
                    self.hooks.emit(AgentEvent(kind="warning", step=step, text=(
                        "Response preview was not accepted. No tools from this attempt were executed."
                    )))
            def finish_attempt(outcome: str, *, detail: str = "", response: str = "", usage: Usage | None = None) -> None:
                if diagnostics is not None:
                    diagnostics.finish(diagnostic_id, outcome, detail=detail, response=response,
                                       usage=asdict(usage) if usage is not None else None)
                    if diagnostics.error and self.registry.context_notes.get("request_diagnostics") != diagnostics.error:
                        self.registry.context_notes["request_diagnostics"] = diagnostics.error
                        self.hooks.emit(AgentEvent(kind="warning", text=f"Request diagnostics: {diagnostics.error}"))
            def record_failure(outcome: str, detail: str, response: str, usage: Usage) -> None:
                nonlocal step, run_step, step_usage, start_next_step
                recorder = self.registry.services["request_failure_recorder"]
                step = recorder({"outcome": outcome, "error": detail,
                                 "response_excerpt": response[:16000],
                                 "diagnostic_attempt": diagnostic_id}, usage)
                run_step += 1
                step_usage = Usage()
                start_next_step = True
            def request_sent(request: dict[str, Any]) -> None:
                nonlocal request_estimate, diagnostic_id
                request_estimate = self.budget.estimate(request)
                self.hooks.delivered()
                if diagnostics is not None:
                    diagnostic_id = diagnostics.begin(request, step=run_step, step_id=step)
                instructions = self.registry.resources.instructions
                if instructions is not None:
                    instructions.presented(self.registry.services.get("instruction_snapshot", {}))
                self.hooks.emit(AgentEvent(kind="context", step=step, text=json.dumps(request, ensure_ascii=False)))
            def delta(kind: str, chunk: str) -> None:
                nonlocal received_content
                if kind == "content":
                    received_content = True
                stream_guard.feed(kind, chunk)
                if kind == "reasoning" and chunk:
                    self.hooks.emit(AgentEvent(kind="reasoning_delta", step=step, text=chunk))
                elif kind == "content" and chunk:
                    visible = preview.push(chunk)
                    if visible:
                        self.hooks.emit(AgentEvent(kind="assistant_delta", step=step, text=visible))
            def response_received(body: dict[str, Any]) -> None:
                if diagnostics is not None:
                    diagnostics.response(diagnostic_id, body)
            stream_guard = StreamLoopGuard()
            options: dict[str, Any] = {
                "on_delta": delta,
                "on_request": request_sent,
                "on_response": response_received,
                "single_attempt": True,
            } if isinstance(self.client, APIClient) else {}
            try:
                if not options:
                    self.hooks.delivered()
                    instructions = self.registry.resources.instructions
                    if instructions is not None:
                        instructions.presented(self.registry.services.get("instruction_snapshot", {}))
                completion = await self.client.chat(
                    model=self.settings.model,
                    messages=context,
                    tools=(specs or None) if native_tools else None,
                    extra_body=extra_body,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    session_id=self.settings.session_id,
                    **options,
                )
                try:
                    if not options:
                        stream_guard.feed("reasoning", completion.message.reasoning or "")
                    if not received_content:
                        stream_guard.feed("content", completion.text)
                    stream_guard.finish()
                except StreamLoopError as exc:
                    exc.usage = completion.usage
                    exc.partial_response = json.dumps(completion.message.to_api(), ensure_ascii=False)
                    raise
            except StreamLoopError as exc:
                self.hooks.account(exc.usage)
                finish_attempt("loop_stopped", detail=str(exc), response=exc.partial_response, usage=exc.usage)
                end_preview(failed=True)
                self.hooks.emit(AgentEvent(kind="warning", step=step, text=str(exc)))
                record_failure("loop_stopped", str(exc), exc.partial_response, exc.usage)
                return RequestStopped(str(exc), loop_stopped=True)
            except APIContextError as exc:
                self.hooks.account(exc.usage)
                finish_attempt("context_overflow", detail=str(exc))
                end_preview(failed=True)
                try:
                    if self.hooks.stop_requested():
                        return RequestStopped(self.hooks.stop_notice(step))
                    if overflows >= 2:
                        raise
                    overflows += 1
                    budget_fraction *= .65
                    # Compare before adding failure bookkeeping: a new turn ID
                    # and error record do not establish that history shrank.
                    smaller = await self.hooks.context(repair, budget_fraction)
                finally:
                    record_failure("context_overflow", str(exc), exc.partial_response or exc.body or "", exc.usage)
                if [m.to_api() for m in smaller] == [m.to_api() for m in context]:
                    raise ContextError("The provider rejected the current input even without removable history. Originals are preserved; shorten the input or select a larger-context model.")
                context = await self.hooks.context(repair, budget_fraction)
                self.budget.fraction = budget_fraction
                self.hooks.emit(AgentEvent(kind="retry", step=step, text=f"Provider context limit reached; retrying with less history ({overflows}/2). Originals are preserved."))
                continue
            except APIResponseError as exc:
                self.hooks.account(exc.usage)
                step_usage = step_usage + exc.usage
                finish_attempt("request_error", detail=str(exc), response=exc.partial_response or exc.body or "", usage=exc.usage)
                end_preview(failed=True)
                record_failure("request_error", str(exc), exc.partial_response or exc.body or "", exc.usage)
                if self.hooks.stop_requested():
                    return RequestStopped(self.hooks.stop_notice(step))
                transient = isinstance(exc, APITransportError) or exc.status_code in RETRYABLE_STATUS
                if not transient:
                    raise
                if self.hooks.exhausted():
                    raise ContextStepLimit from exc
                def countdown(error: str, seconds: int) -> None:
                    self.hooks.emit(AgentEvent(kind="retry_wait", step=seconds, text=error))
                retry = await self.client.wait_retry(transport_retries, exc, stop=self.hooks.stop_event(), on_retry=countdown)
                if self.hooks.stop_event().is_set():
                    return RequestStopped(self.hooks.stop_notice(step))
                if not retry:
                    raise
                transport_retries += 1
                context = await self.hooks.context(repair, budget_fraction)
                continue
            except BaseException as exc:
                finish_attempt("cancelled" if isinstance(exc, asyncio.CancelledError) else "request_error", detail=str(exc))
                end_preview(failed=True)
                if isinstance(exc, asyncio.CancelledError):
                    record_failure("cancelled", "Request cancelled before acceptance.", preview.text, Usage())
                raise
            attempts += 1
            self.budget.observe_estimate(request_estimate, completion.usage.prompt_tokens)
            self.hooks.account(completion.usage)
            self.hooks.persist()
            step_usage = step_usage + completion.usage
            try:
                if completion.response_error:
                    raise ResponseFormatError(completion.response_error, excerpt=completion.response_excerpt)
                record = parse_agent_response(completion.text, completion.tool_calls, tools=specs,
                                              xml_tool_calls=xml_tool_calls)
                record.text = _visible_response(record.text)
                if not record.calls and not record.text.strip():
                    raise ResponseFormatError(load_prompt("response-bookkeeping-error.md"))
                if record.calls and completion.finish_reason == "length":
                    raise ResponseFormatError(load_prompt("response-truncated-error.md"))
                # Check decoded reply text too: a JSON response envelope is
                # structured output, but the prose inside it can still loop.
                response_guard = StreamLoopGuard()
                response_guard.feed("content", record.text)
                response_guard.finish()
            except StreamLoopError as exc:
                finish_attempt("loop_stopped", detail=str(exc),
                               response=json.dumps(completion.message.to_api(), ensure_ascii=False), usage=completion.usage)
                end_preview(failed=True)
                self.hooks.emit(AgentEvent(kind="warning", step=step, text=str(exc)))
                record_failure("loop_stopped", str(exc), exc.partial_response, completion.usage)
                return RequestStopped(str(exc), loop_stopped=True)
            except ResponseFormatError as exc:
                finish_attempt("rejected", detail=str(exc), response=completion.response_excerpt or json.dumps(completion.message.to_api(), ensure_ascii=False), usage=completion.usage)
                rejection = str(exc)
                rejected_excerpt = exc.excerpt
            else:
                finish_attempt("accepted", response=json.dumps(completion.message.to_api(), ensure_ascii=False), usage=completion.usage)
                from .editdelivery import notifications
                self.registry.services["accepted_user_edits"] = notifications(context)
                accepted_text = record.text.strip()
                if preview.text and not accepted_text.startswith(preview.text):
                    end_preview()
                    self.hooks.emit(AgentEvent(kind="warning", step=step, text=(
                        "Response preview differs from the accepted reply; showing the validated reply below."
                    )))
                else:
                    remaining = preview.finish(accepted_text)
                    if remaining:
                        self.hooks.emit(AgentEvent(kind="assistant_delta", step=step, text=remaining))
                break
            end_preview(failed=True)
            record_failure("rejected", rejection,
                           completion.response_excerpt or json.dumps(completion.message.to_api(), ensure_ascii=False),
                           completion.usage)
            if self.hooks.stop_requested():
                return RequestStopped(self.hooks.stop_notice(step))
            if self.hooks.exhausted():
                raise ContextStepLimit
            if attempts >= MAX_MEMORY_ATTEMPTS:
                raise ContextError(f"The model returned invalid structured output in 3 consecutive responses: {rejection} No tools from these responses were executed; accepted history is preserved.")
            self.hooks.emit(AgentEvent(kind="retry", step=step, text=(
                f"Response rejected: {rejection} "
                f"Retrying from the start (retry {attempts}/{MAX_MEMORY_ATTEMPTS - 1})."
            )))
            # Retry the complete round without adding unexecuted tool calls to history.
            repair = load_prompt("response-repair.md", rejection=rejection,
                                 tool_channel=load_prompt("repair-native-tools.md" if native_tools else "repair-json-tools.md"))
            if rejected_excerpt:
                repair += (
                    load_prompt('rejected-output.md') + '\n\n'
                    + rejected_excerpt
                )
            # Include diagnostics in the budget calculation, rather than append
            # them to a request that might already fill the available context.
            context = await self.hooks.context(repair, budget_fraction)

        return RequestAccepted(completion, record, step_usage)
