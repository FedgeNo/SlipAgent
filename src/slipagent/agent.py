"""The agent loop.

Owns the cycle of asking, validating actions, executing a complete tool batch,
and supplying observations. Replies may be ordinary text or JSON envelopes.
Completed steps are summarized separately without blocking the next action.
Events leave presentation and lifecycle controls with the CLI.
"""

from __future__ import annotations

import uuid
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .events import AgentEvent as AgentEvent, EventKind as EventKind, EventHandler as EventHandler
from .requests import ContextStepLimit as ContextStepLimit, MAX_MEMORY_ATTEMPTS as MAX_MEMORY_ATTEMPTS
from .requests import RequestAccepted, RequestStopped
from .api import APIClient, APIError
from .config import DEFAULT_MAX_STEPS, DEFAULT_CONTEXT_STEPS
from .context import (
    DEFAULT_CONTEXT_LENGTH,
    ContextError, ContextStopped, ConversationHistory,
    RecallHistoryTool, CompletedStep,
)
from .compaction import StepCompactor
from .progress import LoopGuard
from .budget import ContextBudget
from .checkpoints import FileCheckpoints
from .prompts import PromptError, load_prompt, model_prompt_profile
from .capabilities import RequestProfile
from .tools.base import ToolRegistry, ToolResult
from .types import Message, ToolCall, ToolSpec, Usage
from .toolbatch import tool_message as _tool_message
from .inputqueue import input_queue

def build_system_prompt(workspace: str, *, project_instructions: str = "") -> str:
    """Render the default system prompt for a given workspace root."""
    try:
        prompt = load_prompt("system-prompt.md").format(
            workspace=workspace, interpreter=sys.executable
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise PromptError(f"Invalid prompt template system-prompt.md: {exc}") from None
    prompt += load_prompt("response-common.md")
    if project_instructions:
        prompt += (
            load_prompt('project-instructions-initial.md', project_instructions=project_instructions)
        )
    return prompt


STEP_LIMIT_NOTICE = (
    "Stopped: reached the step limit without a final answer. "
    "The conversation so far is intact — ask me to continue to pick up where "
    "we left off."
)

STOP_NOTICE = "Stopped after the current step. Results are saved in this conversation; ask me to continue when ready."
@dataclass(slots=True)
class Agent:
    """Stateful conversation driver."""

    client: APIClient
    registry: ToolRegistry
    model: str
    max_steps: int = DEFAULT_MAX_STEPS
    temperature: float | None = None
    max_tokens: int | None = None
    context_steps: int = DEFAULT_CONTEXT_STEPS
    system_prompt: str | None = None
    on_event: EventHandler | None = None
    on_boundary: Callable[[], Awaitable[None]] | None = None
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    overthinking: bool = True
    planning: bool = True
    context_tokens: int | None = None
    reasoning_history_steps: int = 25
    filtered_thoughts: bool = True
    exposed_tools: tuple[str, ...] | None = None
    # Input received while busy joins the conversation at a request or batch
    # boundary, without separating tool calls from their results.
    pending: list[str] = field(default_factory=list)
    running: bool = field(default=False, init=False)
    stop_requested: bool = field(default=False, init=False)
    stopped: bool = field(default=False, init=False)
    history: ConversationHistory = field(default_factory=ConversationHistory, init=False)
    _context_lengths: dict[str, int] = field(default_factory=dict, init=False)
    _requests: int = field(default=0, init=False)
    _summary_progress: dict[str, tuple[int, str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.context_steps < 1:
            raise ValueError("context_steps must be at least 1")
        if self.context_tokens is not None and self.context_tokens < 4096:
            raise ValueError("context_tokens must be at least 4096")
        if not 0 <= self.reasoning_history_steps <= 25:
            raise ValueError("reasoning_history_steps must be between 0 and 25")
        if self.system_prompt is not None:
            self.messages.append(Message.system(self.system_prompt))
        self.registry.register(RecallHistoryTool(self.history))
        self.set_planning(self.planning)
        self._compactor()
        jobs = self.registry.resources.jobs
        if jobs is not None:
            # Idle is a safe display boundary, but must never launch model work.
            # Resolve the current method at delivery so behavior reloads apply.
            jobs.on_completion = lambda: self._notify_jobs() if not self.running else None

    def _compactor(self) -> StepCompactor:
        # A behavior reload can introduce this service into an existing session
        # without running Agent.__post_init__ again. The shared registry owns it.
        result = self.registry.resources.compactor
        if result is None:
            result = StepCompactor(
                self._compaction_usage,
                lambda text: self._emit(AgentEvent(kind="warning", text=text)),
            )
            self.registry.resources.compactor = result
        result.on_step = self._persist_step
        return result

    def _persist(self) -> None:
        journal = self.registry.resources.journal
        if journal is not None:
            journal.record(self)
        self._archive_history()

    def _archive_history(self) -> None:
        from .turnstore import TurnStore
        if not self.history.steps:
            return
        journal = self.registry.resources.journal
        if self.history.store is None:
            directory = journal.directory / (journal.session_id + "-turns") if journal is not None and journal.path is not None else None
            self.history.store = TurnStore(directory)
            self.registry.resources.turns = self.history.store
        for step in self.history.steps:
            if isinstance(step, CompletedStep):
                self.history.store.archive(step, self.messages)
        if journal is not None and self.messages:
            journal.last_message = journal._message(self.messages[-1])

    def _persist_step(self, step: CompletedStep) -> None:
        journal = self.registry.resources.journal
        if journal is not None:
            journal.step(step)
        if self.history.store is not None:
            self.history.store.summarized(step)

    def _compaction_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        self._persist()

    def _loop_guard(self) -> LoopGuard:
        guard = self.registry.resources.loop_guard
        if guard is None:
            guard = LoopGuard()
            self.registry.resources.loop_guard = guard
        return guard

    def _checkpoints(self) -> FileCheckpoints | None:
        workspace = self.registry.resources.workspace
        if workspace is None:
            return None
        checkpoints = self.registry.resources.checkpoints
        if checkpoints is None:
            checkpoints = FileCheckpoints(workspace)
            self.registry.resources.checkpoints = checkpoints
        return checkpoints

    def _budget(self) -> ContextBudget:
        budget = self.registry.resources.budget
        if budget is None:
            budget = ContextBudget()
            self.registry.resources.budget = budget
        return budget

    async def wait_for_compaction(self) -> None:
        """Settle summaries within the session-transition grace period."""
        await self._compactor().settle()

    async def _archive_step(self, capabilities: RequestProfile | None) -> None:
        self.history.sync(self.messages)
        step = self.history.steps[-1]
        if isinstance(step, CompletedStep):
            edits = self.registry.services.get("step_user_edits", {}).pop(step.id, None)
            if edits:
                step.user_edits = edits
            self._persist()
            if self.registry.services.get("interrupt_requested"):
                return  # Preserve interrupted results without starting new work.
            self._compactor().submit(
                step, self.client, self.model, capabilities, self.max_tokens,
                history=self.history.steps[:-1][-25:],
                conversation=self.history.steps,
            )
            # Start the isolated request now without waiting for its response.
            await asyncio.sleep(0)

    def reset(self, *, new_session: bool = True) -> None:
        """Clear the conversation, keeping the system prompt."""
        jobs = self.registry.resources.jobs
        if jobs is not None:
            jobs.clear()
        diagnostics = self.registry.resources.diagnostics
        if diagnostics is not None:
            diagnostics.clear()
        self._compactor().reset()
        reset_routing = self.registry.services.get("reset_model_routing")
        if reset_routing is not None:
            reset_routing()
        catalog = self.registry.services.get("tool_catalog")
        if catalog is not None:
            catalog.loaded.clear()
        prefix = self.messages[:1] if self.messages and self.messages[0].role == "system" else []
        self.messages = list(prefix)
        self.usage = Usage()
        self.pending.clear()
        self.queued_messages.clear()
        self.registry.services.pop("input_queue", None)
        self._emit(AgentEvent(kind="user_queue_reset"))
        self.stop_requested = False
        self.stopped = False
        self.history.clear()
        archive = self.registry.resources.archive
        if archive is not None:
            archive.clear()
        artifacts = self.registry.resources.artifacts
        if artifacts is not None:
            artifacts.clear()
        self._summary_progress.clear()
        self._loop_guard().reset()
        checkpoints = self.registry.resources.checkpoints
        if checkpoints is not None:
            checkpoints.clear()
        self.registry.context_notes.pop("file_rewind", None)
        self.registry.context_notes.pop("progress", None)
        self.registry.context_notes.pop("resume", None)
        self.registry.context_notes.pop("background_commands", None)
        self.registry.context_notes.pop("request_diagnostics", None)
        self.registry.context_notes.pop("user_edits", None)
        self.registry.services.pop("step_user_edits", None)
        self.registry.services.pop("accepted_user_edits", None)
        journal = self.registry.resources.journal
        if journal is not None and new_session:
            journal.begin(self)

    def request_stop(self) -> bool:
        """Finish the active response and tool batch without requesting another."""
        if not self.running:
            return False
        self.stop_requested = True
        self._stop_event().set()
        return True

    def _stop_event(self) -> asyncio.Event:
        event: asyncio.Event = self.registry.services.setdefault("request_stop_event", asyncio.Event())
        return event

    def _stop_notice(self, step: int) -> str:
        self.stopped = True
        self._emit(AgentEvent(kind="warning", step=step, text=STOP_NOTICE))
        return STOP_NOTICE

    def _notify_jobs(self) -> None:
        jobs = self.registry.resources.jobs
        if jobs is None:
            return
        for monitor in jobs.monitors.notices():
            outcome = "result ready" if monitor["state"] == "ready" else "matched output"
            self._emit(AgentEvent(kind="notice", text=f"Monitor {monitor['monitor_id']} initialized in turn {monitor['initialized_turn']}: {outcome}; {monitor['suppressed_notifications']} notifications suppressed."))
            if monitor.get("artifact"):
                self._attach_artifact(monitor["artifact"])
                self._emit(AgentEvent(kind="artifact", result=ToolResult.ok(monitor["artifact"])))
        completed = jobs.completions()
        if completed:
            self.registry.context_notes["background_commands"] = {
                "completed": completed, "guidance": load_prompt('background-command-results.md'),
            }
            for job in completed:
                self._emit(AgentEvent(kind="notice", text=f"Background command {job['job_id']} {job['state']} (exit {job['returncode']})."))
                for reference in job.get("artifacts", []):
                    self._attach_artifact(reference)
                    self._emit(AgentEvent(kind="artifact", result=ToolResult.ok(reference)))

    def _attach_artifact(self, reference: dict[str, Any]) -> None:
        step_id = reference.get("step_id")
        if type(step_id) is not int or not 1 <= step_id <= len(self.history.steps):
            return
        for message in self.history.steps[step_id - 1].messages:
            if message.role != "tool" or message.tool_call_id != reference["call_id"] or not isinstance(message.content, dict):
                continue
            content = message._stored["content"] if hasattr(message, "_stored") else message.content
            references = content.setdefault("artifacts", [])
            if any(saved["record_id"] == reference["record_id"] for saved in references):
                return
            references.append(reference)
            if self.history.store is not None:
                self.history.store.attach_artifact(self.history.steps[step_id - 1], reference["call_id"], reference)
            journal = self.registry.resources.journal
            if journal is not None:
                for index, original in enumerate(self.messages):
                    if original is message:
                        journal.replace_message(index, message)
                        break

    def extend(self, messages: Sequence[Message]) -> None:
        """Append pre-built messages (used for resuming saved conversations)."""
        self.messages.extend(messages)
        self.history.sync(self.messages)

    async def _context_length(self) -> int:
        if isinstance(self.client, APIClient):
            capabilities = await self.client.model_capabilities(self.model)
            if capabilities is not None:
                length = capabilities.context_length
                if capabilities.max_prompt_tokens is not None:
                    length = min(length, capabilities.max_prompt_tokens)
                return length
        if self.model not in self._context_lengths:
            length = DEFAULT_CONTEXT_LENGTH
            if isinstance(self.client, APIClient):
                cached_length = self.client.catalog_context_length(self.model)
                if cached_length is not None:
                    length = cached_length
            self._context_lengths[self.model] = length
        return self._context_lengths[self.model]

    def _account_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    def _record_request_failure(self, failure: dict[str, Any], usage: Usage,
                                partial: tuple[str, str] | None = None) -> int:
        from .requestfailures import failure_summary, retry_state, validate_failure

        self.history.sync(self.messages)
        previous = retry_state(self.history.steps)
        total = sum(getattr(step, "request_failure", None) is not None for step in self.history.steps)
        failure = {**failure, "request_number": self._requests,
                   "consecutive_failures": previous["consecutive_failures"] + 1 if previous else 1,
                   "total_failed_attempts": total + 1, "tools_executed": False}
        validate_failure(failure)
        summary = failure_summary(failure)
        # The partial reply and reasoning are stored like a completed step's, marked
        # incomplete by the failure record. Tool calls are never kept: they did not run.
        content, reasoning = partial or ("", "")
        self.messages.append(Message(role="assistant", content=content or None, reasoning=reasoning or None,
                                     reasoning_model=self.model if reasoning else None))
        self.history.sync(self.messages)
        failed = self.history.steps[-1]
        setattr(failed, "request_failure", failure)
        failed.summary = summary
        failed.results_summarized = True
        if isinstance(failed, CompletedStep):
            # No compactor runs for a failed step, so no filtered thoughts exist;
            # the original reasoning is shown.
            failed.reasoning_summary = None
            failed.compaction_status = "complete"
            failed.observed = False
        self._persist()
        self._emit(AgentEvent(kind="step_end", step=failed.id, usage=usage))
        return failed.id + 1

    def _reserve_request(self) -> None:
        if self._requests >= self.max_steps:
            raise ContextStepLimit
        self._requests += 1

    async def _name_session(self, step: int, title: str) -> None:
        """Adopt the model's title only while the session keeps its default one."""
        journal = self.registry.resources.journal
        if not title or journal is None or journal.named:
            return
        from .sessions import SessionError
        from .tools.blocking import run_blocking
        try:
            await run_blocking(journal.rename, title)
        except SessionError as exc:
            self._emit(AgentEvent(kind="warning", step=step, text=str(exc)))
            return
        self._emit(AgentEvent(kind="session_title", step=step, text=journal.title))

    def _record_edit_delivery(self, step: int, edits: list[dict[str, Any]]) -> None:
        from .editdelivery import acknowledge

        if not edits:
            return
        # Save the accepted reply before the receipt, and the receipt before
        # retiring pending revisions or dispatching any accepted tools.
        self._persist()
        journal = self.registry.resources.journal
        if journal is not None:
            journal.edits_delivered(step, edits)
        self.registry.services.setdefault("step_user_edits", {})[step] = edits
        acknowledge(self.registry.context_notes.get("user_edits", {}), edits)
        self._persist()

    async def _context_view(self, specs: list[ToolSpec], step: int, *,
                            capabilities: RequestProfile | None = None, preview: bool = False,
                            repair: str = "", budget_fraction: float | None = None,
                            candidate_model: str | None = None) -> list[Message]:
        """Assemble the exact model-visible state within its endpoint limits.

        Selection previews must not register input or archive new steps on
        the active history. Runtime diagnostics and service metadata are explicit
        input because the model cannot see terminal notices or local state.
        """
        model = self.model if candidate_model is None else candidate_model
        if candidate_model is not None and capabilities is None:
            raise ContextError("Candidate-model previews require the candidate's capabilities.")
        specs = self._visible_specs(specs)
        if capabilities is None and isinstance(self.client, APIClient):
            capabilities = await self.client.model_capabilities(self.model)
        length = await self._context_length() if capabilities is None else capabilities.context_length
        if capabilities is not None:
            if capabilities.max_prompt_tokens is not None:
                length = min(length, capabilities.max_prompt_tokens)
            try:
                capabilities.validate_output_limit(self.max_tokens)
            except ValueError as exc:
                raise ContextError(str(exc)) from exc
        if self.context_tokens is not None:
            length = min(length, self.context_tokens)
        workspace = self.registry.resources.workspace
        messages = self.messages
        if workspace is not None and self.registry.services.get("editable_prompts"):
            prompt = build_system_prompt(str(workspace.root))
            if preview:
                messages = list(self.messages)
            else:
                self.system_prompt = prompt
            if messages and messages[0].role == "system":
                messages[0] = Message.system(prompt)
        from .contextbuilder import ContextBuilder, ContextSettings

        builder = ContextBuilder(
            self.registry, self.history, messages, self._budget(),
            ContextSettings(model, self.planning, self.context_steps, self.overthinking,
                            self.reasoning_history_steps, self.filtered_thoughts, self.max_tokens),
        )
        return await builder.build(
            specs, length, self._uses_native_tools(capabilities, model=model), capabilities=capabilities,
            preview=preview, repair=repair, budget_fraction=budget_fraction,
        )

    def _uses_native_tools(self, capabilities: RequestProfile | None, *, model: str | None = None) -> bool:
        channel = model_prompt_profile(self.model if model is None else model).get("tool_channel")
        if channel in ("embedded", "glm_xml"):
            return False
        return capabilities is None or capabilities.native_tools

    @property
    def total_cost(self) -> float | None:
        return self.usage.cost

    def enqueue(self, text: str) -> None:
        """Queue a user message typed while the agent was mid-step.

        The REPL stays interactive during a step, so input can arrive before
        the current step finishes. It joins at a request or batch boundary,
        keeping assistant tool calls together with their matching results.
        """
        message = text.strip()
        if message:
            queue = input_queue(self)
            queue.snapshot()
            self.pending.append(message)
            queue.appended(message)
            self._persist()
            self._emit(AgentEvent(kind="user_message", text=message))

    def _drain_pending(self) -> None:
        """Move queued user input into the conversation as user messages."""
        if not self.pending:
            return
        queued, self.pending = self.pending, []
        self._loop_guard().reset()
        self.registry.context_notes.pop("progress", None)
        for message in queued:
            self.queued_messages[len(self.messages)] = message
            self.messages.append(Message.user(message))

    @property
    def has_queued_input(self) -> bool:
        """Accepted follow-ups remain owed until an actual request submits them."""
        return bool(self.pending or self.queued_messages)

    @property
    def queued_messages(self) -> dict[int, str]:
        """Messages in history that are still awaiting request submission.

        Draining a queue preserves tool-result ordering, but context preparation
        or a stop can still prevent delivery. Indices also survive journal resume.
        """
        queued: dict[int, str] = self.registry.services.setdefault("queued_messages", {})
        return queued

    def _queued_messages_sent(self) -> None:
        input_queue(self).sent(len(self.queued_messages))
        delivered = list(self.queued_messages.values())
        self.queued_messages.clear()
        if delivered:
            self._persist()
            for text in delivered:
                self._emit(AgentEvent(kind="user_message_sent", text=text))

    async def _queued_context(self, specs: list[ToolSpec], step: int, *,
                              repair: str = "", budget_fraction: float | None = None) -> list[Message]:
        # Context preparation can yield to input. Rebuild if more prompts arrive
        # so every message waiting at submission joins the same request.
        while True:
            self._drain_pending()
            queue = input_queue(self)
            signature = queue.signature()
            context = await self._context_view(specs, step, repair=repair, budget_fraction=budget_fraction)
            if not self.pending and signature == queue.signature():
                queue.prepared = signature
                return context

    def _emit(self, event: AgentEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)

    async def run(self, prompt: str, *, continue_run: bool = False) -> str:
        """Run a task; continue_run retains limits after a settings pause."""
        if self.running:
            raise RuntimeError("agent is already running")
        # Corrections queued before an API/protocol failure are older than this
        # prompt, even when the prior run did not finish through /stop.
        self._drain_pending()
        self.stop_requested = False
        self.stopped = False
        if not continue_run:
            self._requests = 0
        self._stop_event().clear()
        if not continue_run:
            self._loop_guard().reset()
        self.registry.context_notes.pop("progress", None)
        self.running = True
        try:
            return await self._run(prompt)
        finally:
            try:
                if self.on_boundary is not None:
                    await self.on_boundary()
            finally:
                self.running = False
                self._notify_jobs()
                if self.stop_requested:
                    self.stopped = True
                self._persist()

    async def _run(self, prompt: str) -> str:
        if prompt:
            self.messages.append(Message.user(prompt))
        self._persist()
        for run_step in range(self._requests + 1, self.max_steps + 1):
            if self.on_boundary is not None:
                await self.on_boundary()
            if self.stop_requested:
                return self._stop_notice(len(self.history.steps) + 1)
            try:
                answer = await self._step(run_step)
            except ContextStepLimit:
                break
            if answer is not None:
                return answer

        self._emit(AgentEvent(kind="warning", step=len(self.history.steps) + 1, text=STEP_LIMIT_NOTICE))
        return STEP_LIMIT_NOTICE

    def set_planning(self, enabled: bool) -> None:
        """Change plan availability without discarding the retained plan."""
        from .tools.planning import UpdatePlanTool
        self.planning = enabled
        if enabled:
            if "update_plan" not in self.registry:
                self.registry.register(UpdatePlanTool())
        else:
            self.registry.unregister("update_plan")

    def _filter_specs(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        if self.exposed_tools is None:
            return specs
        allowed = set(self.exposed_tools) | {"recall_history"}
        if "load_tools" in self.registry:
            allowed.add("load_tools")
        if self.planning:
            allowed.add("update_plan")
        else:
            allowed.discard("update_plan")
        missing = allowed - set(self.registry.names)
        if missing:
            raise ValueError(f"Unavailable tools in allowlist: {', '.join(sorted(missing))}")
        return [spec for spec in specs if spec.name in allowed]

    def _visible_specs(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        from .calllabels import labeled_spec
        return [labeled_spec(spec) for spec in self._selected_specs(specs)]

    def _selected_specs(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        from .tools.discovery import tool_catalog
        catalog = tool_catalog(self.registry)
        filtered = self._filter_specs(specs)
        if catalog is None:
            return filtered
        names = catalog.select(self._filter_specs(self.registry.specs()), self.messages)
        return [spec for spec in filtered if spec.name in names]

    async def _request_response(self, specs: list[ToolSpec], step: int,
                                run_step: int) -> tuple[RequestAccepted | RequestStopped, RequestProfile | None, list[ToolSpec]]:
        from .requests import RequestDriver, RequestHooks, RequestSettings
        context = await self._queued_context(specs, step)
        capabilities = await self.client.model_capabilities(self.model) if isinstance(self.client, APIClient) else None
        native_tools = self._uses_native_tools(capabilities)
        self.registry.services["request_failure_recorder"] = self._record_request_failure
        driver = RequestDriver(
            self.client, self.registry, self._budget(),
            RequestSettings(self.model, self.temperature, self.max_tokens, self.session_id),
            RequestHooks(
                context=lambda repair, fraction: self._queued_context(
                    specs, len(self.history.steps) + 1, repair=repair, budget_fraction=fraction),
                reserve=self._reserve_request, exhausted=lambda: self._requests >= self.max_steps,
                has_pending=lambda: bool(self.pending) or input_queue(self).changed(), delivered=self._queued_messages_sent,
                account=self._account_usage, persist=self._persist, emit=self._emit,
                stop_event=self._stop_event, stop_requested=lambda: self.stop_requested,
                stop_notice=self._stop_notice,
            ),
        )
        visible_specs = self._visible_specs(specs)
        return await driver.run(context, visible_specs, step, run_step, capabilities, native_tools), capabilities, visible_specs

    async def _step(self, step: int) -> str | None:
        """One response and its complete tool batch; behavior reloads between steps."""
        prepare_model = self.registry.services.get("prepare_model_request")
        if prepare_model is not None:
            await prepare_model()
        # Run budgets restart; invocation identity follows the archived conversation.
        step = len(self.history.steps) + 1
        self._notify_jobs()
        servers = self.registry.resources.servers
        if servers is not None:
            await servers.refresh()
        specs = self._filter_specs(self.registry.specs())
        self._emit(AgentEvent(kind="step_start", step=step))

        while True:
            try:
                outcome, capabilities, visible_specs = await self._request_response(
                    specs, len(self.history.steps) + 1, self._requests + 1)
            except ContextStopped:
                return self._stop_notice(step)
            except (APIError, ContextError) as exc:
                failure: Exception = exc
            else:
                if not isinstance(outcome, RequestStopped) or not outcome.loop_stopped:
                    break
                failure = ContextError(outcome.text)
                fallback = self.registry.services.get("model_fallback")
                if (self.stop_requested or self._requests >= self.max_steps or fallback is None
                        or not await fallback(failure)):
                    break
                continue
            fallback = self.registry.services.get("model_fallback")
            if (self.stop_requested or self._requests >= self.max_steps or fallback is None
                    or not await fallback(failure)):
                raise failure
        for delivered in ("background_commands", "resume"):
            self.registry.context_notes.pop(delivered, None)
        step = len(self.history.steps) + 1
        if isinstance(outcome, RequestStopped):
            if outcome.loop_stopped:
                self.stopped = True
            return outcome.text
        completion, record, step_usage = outcome.completion, outcome.record, outcome.usage

        for history_step in self.history.steps:
            if isinstance(history_step, CompletedStep):
                history_step.observed = True
        text = record.text.strip()
        self._emit(AgentEvent(kind="stream_end", step=step))
        if text:
            self._emit(AgentEvent(kind="assistant_text", step=step, text=text))

        if completion.finish_reason == "length":
            self._emit(
                AgentEvent(
                    kind="warning",
                    step=step,
                    text=(
                        "Response hit the token limit and may be incomplete."
                    ),
                )
            )

        message = Message.assistant(record.text or None, record.calls)
        message.reasoning = completion.message.reasoning
        message.reasoning_details = completion.message.reasoning_details
        message.reasoning_model = self.model
        self.messages.append(message)
        self._record_edit_delivery(step, self.registry.services.pop("accepted_user_edits", []))
        await self._name_session(step, getattr(record, "session_title", ""))
        if not record.calls:
            await self._archive_step(capabilities)
            if not self.has_queued_input and not self.stop_requested:
                for suggestion in getattr(record, "suggestions", []):
                    self._emit(AgentEvent(kind="suggestion", step=step, text=suggestion))
            self._emit(AgentEvent(kind="step_end", step=step, usage=step_usage))
            if self.stop_requested:
                self._stop_notice(step)
            # Text typed while this step ran is still owed a reply, so keep
            # it queued for the next step rather than dropping it.
            return text

        self._persist()

        from .toolbatch import ToolBatch

        batch = ToolBatch(
            self.registry, step, self.messages, emit=self._emit, persist=self._persist,
            archive_interrupted=lambda: self._archive_step(capabilities),
            checkpoints=self._checkpoints(),
            allowed_tools={spec.name for spec in visible_specs},
        )
        batch_results = await batch.run(message.tool_calls or [])
        await self._archive_step(capabilities)

        # Anything typed mid-step joins here, after every tool result, so
        # the model sees it as a new instruction in a well-formed history.
        guidance, repeated = "", False
        only_answers = bool(batch_results) and all(call.name == "answer" and not result.is_error for call, result in batch_results)
        if not self.pending and not only_answers:
            guidance, repeated = self._loop_guard().observe(batch_results, self.registry, len(self.history.steps))
        if guidance:
            self.registry.context_notes["progress"] = guidance
            if repeated:
                self._emit(AgentEvent(kind="warning", step=step, text=guidance))
        else:
            self.registry.context_notes.pop("progress", None)
        self._drain_pending()

        self._emit(AgentEvent(kind="step_end", step=step, usage=step_usage))
        if only_answers:
            if not self.has_queued_input and not self.stop_requested:
                for suggestion in getattr(record, "suggestions", []):
                    self._emit(AgentEvent(kind="suggestion", step=step, text=suggestion))
            return "\n\n".join(result.content["text"] for _, result in batch_results)
        if repeated:
            self.stopped = True
            return guidance
        if self.stop_requested:
            return self._stop_notice(step)
        return None
