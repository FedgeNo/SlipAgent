"""The agent loop.

Owns the cycle of asking, validating actions, executing a complete tool batch,
and supplying observations. Replies may be ordinary text or JSON envelopes.
Completed steps are summarized separately without blocking the next action.
Events leave presentation and lifecycle controls with the CLI.
"""

from __future__ import annotations

from .data_text import render_data

import uuid
import asyncio
import json
import sys
import copy
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .api import APIClient
from .config import DEFAULT_MAX_STEPS, DEFAULT_CONTEXT_STEPS
from .context import (
    DEFAULT_CONTEXT_LENGTH,
    ContextError, ContextStopped, ConversationHistory,
    RecallHistoryTool, CompletedStep,
    tool_history_as_text,
    _visible_response,
)
from .protocol import ResponseFormatError, parse_agent_response, agent_response_format
from .compaction import StepCompactor
from .activity import command_output
from .progress import LoopGuard, StreamLoopGuard, StreamLoopError
from .budget import ContextBudget
from .repomap import RepositoryMap
from .checks import check_edit_batch
from .checkpoints import FileCheckpoints, current_checkpoint
from .prompts import PromptSections, load_prompt
from .batching import run_batch
from .api import APIError, APIContextError, APIResponseError, APITransportError, RETRYABLE_STATUS
from .mcp import MCPTool
from .capabilities import RequestProfile
from .tools.base import ToolRegistry, ToolResult, current_invocation
from .types import Message, ToolCall, ToolSpec, Usage

EventKind = Literal[
    "context",
    "step_start",
    "assistant_text",
    "assistant_delta",
    "reasoning_delta",
    "stream_end",
    "retry",
    "retry_wait",
    "user_message",
    "user_message_sent",
    "user_queue_reset",
    "tool_start",
    "tool_output",
    "tool_end",
    "step_end",
    "warning",
    "notice",
    "error",
]


@dataclass(slots=True)
class AgentEvent:
    kind: EventKind
    step: int = 0
    text: str = ""
    tool_call: ToolCall | None = None
    result: ToolResult | None = None
    usage: Usage | None = None


EventHandler = Callable[[AgentEvent], None]


def build_system_prompt(workspace: str, *, project_instructions: str = "") -> str:
    """Render the default system prompt for a given workspace root."""
    prompt = load_prompt("system-prompt.md").format(
        workspace=workspace, interpreter=sys.executable
    )
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
MAX_MEMORY_ATTEMPTS = 3


class ContextStepLimit(Exception):
    """The current run has exhausted its model-request budget."""


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
        jobs = self.registry.services.get("command_jobs")
        if jobs is not None:
            # Idle is a safe display boundary, but must never launch model work.
            # Resolve the current method at delivery so behavior reloads apply.
            jobs.on_completion = lambda: self._notify_jobs() if not self.running else None

    def _compactor(self) -> StepCompactor:
        # A behavior reload can introduce this service into an existing session
        # without running Agent.__post_init__ again. The shared registry owns it.
        if "step_compactor" not in self.registry.services:
            self.registry.services["step_compactor"] = StepCompactor(
                self._compaction_usage,
                lambda text: self._emit(AgentEvent(kind="warning", text=text)),
            )
        result: StepCompactor = self.registry.services["step_compactor"]
        result.on_step = self._persist_step
        return result

    def _persist(self) -> None:
        journal = self.registry.services.get("session_journal")
        if journal is not None:
            journal.record(self)

    def _persist_step(self, step: CompletedStep) -> None:
        journal = self.registry.services.get("session_journal")
        if journal is not None:
            journal.step(step)

    def _compaction_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        self._persist()

    def _loop_guard(self) -> LoopGuard:
        if "loop_guard" not in self.registry.services:
            self.registry.services["loop_guard"] = LoopGuard()
        guard: LoopGuard = self.registry.services["loop_guard"]
        return guard

    def _checkpoints(self) -> FileCheckpoints | None:
        workspace = self.registry.services.get("workspace")
        if workspace is None:
            return None
        if "file_checkpoints" not in self.registry.services:
            self.registry.services["file_checkpoints"] = FileCheckpoints(workspace)
        checkpoints: FileCheckpoints = self.registry.services["file_checkpoints"]
        return checkpoints

    def _budget(self) -> ContextBudget:
        if "context_budget" not in self.registry.services:
            self.registry.services["context_budget"] = ContextBudget()
        budget: ContextBudget = self.registry.services["context_budget"]
        return budget

    async def wait_for_compaction(self) -> None:
        """Drain completed-step summaries when a caller needs settled memory."""
        await self._compactor().wait()

    async def _archive_step(self, capabilities: RequestProfile | None) -> None:
        self.history.sync(self.messages)
        step = self.history.steps[-1]
        if isinstance(step, CompletedStep):
            self._persist()
            if self.registry.services.get("interrupt_requested"):
                return  # Preserve interrupted results without starting new work.
            self._compactor().submit(
                step, self.client, self.model, capabilities, self.max_tokens,
                history=self.history.steps[:-1][-25:],
            )
            # Start the isolated request now without waiting for its response.
            await asyncio.sleep(0)

    def reset(self, *, new_session: bool = True) -> None:
        """Clear the conversation, keeping the system prompt."""
        jobs = self.registry.services.get("command_jobs")
        if jobs is not None:
            jobs.clear()
        diagnostics = self.registry.services.get("request_diagnostics")
        if diagnostics is not None:
            diagnostics.clear()
        self._compactor().reset()
        prefix = self.messages[:1] if self.messages and self.messages[0].role == "system" else []
        self.messages = list(prefix)
        self.usage = Usage()
        self.pending.clear()
        self.queued_messages.clear()
        self._emit(AgentEvent(kind="user_queue_reset"))
        self.stop_requested = False
        self.stopped = False
        self.history.clear()
        archive = self.registry.services.get("command_archive")
        if archive is not None:
            archive.clear()
        self._summary_progress.clear()
        self._loop_guard().reset()
        checkpoints = self.registry.services.get("file_checkpoints")
        if checkpoints is not None:
            checkpoints.clear()
        self.registry.context_notes.pop("file_rewind", None)
        self.registry.context_notes.pop("progress", None)
        self.registry.context_notes.pop("resume", None)
        self.registry.context_notes.pop("background_commands", None)
        self.registry.context_notes.pop("request_diagnostics", None)
        journal = self.registry.services.get("session_journal")
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
        jobs = self.registry.services.get("command_jobs")
        if jobs is None:
            return
        completed = jobs.completions()
        if completed:
            self.registry.context_notes["background_commands"] = {
                "completed": completed, "guidance": load_prompt('background-command-results.md'),
            }
            for job in completed:
                self._emit(AgentEvent(kind="notice", text=f"Background command {job['job_id']} {job['state']} (exit {job['returncode']})."))

    def extend(self, messages: Sequence[Message]) -> None:
        """Append pre-built messages (used for resuming saved conversations)."""
        self.messages.extend(messages)
        self.history.sync(self.messages)

    async def _context_length(self) -> int:
        if isinstance(self.client, APIClient):
            capabilities = await self.client.model_capabilities(self.model)
            if capabilities is not None:
                length = capabilities.context_length or DEFAULT_CONTEXT_LENGTH
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

    def _reserve_request(self) -> None:
        if self._requests >= self.max_steps:
            raise ContextStepLimit
        self._requests += 1

    async def _context_view(self, specs: list[ToolSpec], step: int, *,
                            capabilities: RequestProfile | None = None, preview: bool = False,
                            repair: str = "", budget_fraction: float | None = None) -> list[Message]:
        """Assemble the exact model-visible state within its endpoint limits.

        Selection previews must not register input or archive new steps on
        the active history. Runtime diagnostics and service metadata are explicit
        input because the model cannot see terminal notices or local state.
        """
        specs = self._filter_specs(specs)
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
        sections = PromptSections()
        if self.planning:
            saved_plan = next(((prior.id, message.content['content'])
                         for prior in reversed(self.history.steps)
                         for message in reversed(prior.messages)
                         if message.role == 'tool' and isinstance(message.content, dict)
                         and message.content.get('tool') == 'update_plan'
                         and message.content.get('status') == 'success'), None)
            plan_record = load_prompt('working-plan-empty.md')
            if saved_plan is not None:
                plan_record = load_prompt('working-plan-record.md', step_id=saved_plan[0],
                                          plan=render_data(saved_plan[1]))
            sections.add('plan', 'Working Plan', load_prompt('working-plan.md',
                         plan_record=plan_record.strip()), 45)
        workspace = self.registry.services.get("workspace")
        if workspace is not None and self.registry.services.get("editable_prompts"):
            self.system_prompt = build_system_prompt(str(workspace.root))
            if self.messages and self.messages[0].role == "system":
                self.messages[0] = Message.system(self.system_prompt)
        if workspace is not None:
            access = load_prompt('workspace-access-off.md')
            if workspace.access.danger:
                access = (load_prompt('workspace-access-on.md'))
            sections.add("access", "Workspace Access",
                         load_prompt('workspace-access.md', access=access.strip(), workspace=workspace.root),
                         15, owner="workspace")
        instructions = self.registry.services.get("project_instructions")
        if instructions is not None:
            snapshot = instructions.snapshot()
            sections.add("project", "Project Instructions (Current)", instructions.render(snapshot), 20, owner="instructions")
            if not preview:
                self.registry.services["instruction_snapshot"] = snapshot
        environment = self.registry.services.get("project_environment")
        if environment is not None:
            sections.add("environment", "Project Python Environment", load_prompt('environment-snapshot.md', environment=render_data(await environment.snapshot())), 30, owner="environment")
        native_tools = capabilities is None or capabilities.native_tools
        if capabilities is not None and (capabilities.format == "json_schema" or not native_tools):
            sections.add("schema", "Response Schema",
                load_prompt('response-json.md') + '\n\n'
                + (load_prompt('response-native-json.md')
                   if native_tools else load_prompt('response-embedded-json.md')),
                0, owner="protocol", dynamic=False,
            )
        elif native_tools:
            sections.add("schema", "Reply Format",
                         load_prompt('response-native-text.md'),
                         0, owner="protocol", dynamic=False)
        if capabilities is not None and not native_tools:
            sections.add("tools", "Available Tool Definitions", load_prompt('tool-definitions.md', definitions=render_data([spec.to_api() for spec in specs])), 10, owner="tools", dynamic=False)
        servers = {tool.client.spec.name: tool.client.instructions for tool in self.registry.tools
                   if isinstance(tool, MCPTool) and tool.client.connected and tool.client.instructions}
        if servers:
            sections.add("mcp", "Connected MCP Server Guidance",
                load_prompt('mcp-guidance.md', servers=render_data(servers)),
                40, owner="mcp", dynamic=False,
            )
        if self.registry.context_notes:
            sections.add("state", "Current Harness State", load_prompt('harness-state.md', state=render_data(self.registry.context_notes)), 50)
        sections.add("repair", "Response Correction", repair.strip(), 60)
        extra_instructions = sections.render()
        budget = copy.copy(self._budget()) if preview else self._budget()
        token_scale = budget.select(self.model, capabilities)
        if budget_fraction is None:
            budget_fraction = budget.fraction
        history = replace(self.history, steps=list(self.history.steps), task=copy.deepcopy(self.history.task)) if preview else self.history
        repository_map = ""
        if environment is not None:
            if "repository_map" not in self.registry.services:
                self.registry.services["repository_map"] = RepositoryMap(environment.workspace)
            query = next((m.content or "" for m in reversed(self.messages) if m.role == "user"), "")
            repository_map = await self.registry.services["repository_map"].snapshot(query, min(8000, int(length * budget_fraction * .03)))
        view_options: dict[str, Any] = dict(
            user_corrections="\n\n".join(part.strip() for part in (
                self.registry.context_notes.get("progress", ""), repair,
            ) if part.strip()),
            keep_steps=self.context_steps,
            overthinking=self.overthinking,
            reasoning_history_steps=self.reasoning_history_steps,
            filtered_thoughts=self.filtered_thoughts,
            context_length=int(length * budget_fraction),
            max_output=self.max_tokens or min(8192, max(256, length // 8)),
            native_tools=native_tools,
            text_tool_history=capabilities is not None and not native_tools,
            schema=agent_response_format(native_tools=native_tools) if capabilities is not None and capabilities.format == "json_schema" else None,
            token_scale=token_scale,
        )
        try:
            supplement = "\n\n".join(part.strip() for part in (extra_instructions, repository_map) if part.strip())
            context = await history.view(self.messages, specs, extra_instructions=supplement, **view_options)
        except ContextError:
            if not repository_map:
                raise
            # Orientation is optional; it must never displace the current task.
            context = await history.view(self.messages, specs, extra_instructions=extra_instructions, **view_options)
        if capabilities is not None and not native_tools:
            context = tool_history_as_text(context)
        return context

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
            self.pending.append(message)
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
    def queued_messages(self) -> dict[int, str]:
        """Messages in history that are still awaiting request submission.

        Draining a queue preserves tool-result ordering, but context preparation
        or a stop can still prevent delivery. Indices also survive journal resume.
        """
        queued: dict[int, str] = self.registry.services.setdefault("queued_messages", {})
        return queued

    def _queued_messages_sent(self) -> None:
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
            context = await self._context_view(specs, step, repair=repair, budget_fraction=budget_fraction)
            if not self.pending:
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
        for step in range(self._requests + 1, self.max_steps + 1):
            if self.on_boundary is not None:
                await self.on_boundary()
            if self.stop_requested:
                return self._stop_notice(step)
            try:
                answer = await self._step(step)
            except ContextStepLimit:
                break
            if answer is not None:
                return answer

        self._emit(AgentEvent(kind="warning", step=self.max_steps, text=STEP_LIMIT_NOTICE))
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
        if self.planning:
            allowed.add("update_plan")
        else:
            allowed.discard("update_plan")
        missing = allowed - set(self.registry.names)
        if missing:
            raise ValueError(f"Unavailable tools in allowlist: {', '.join(sorted(missing))}")
        return [spec for spec in specs if spec.name in allowed]

    async def _step(self, step: int) -> str | None:
        """One response and its complete tool batch; behavior reloads between steps."""
        self._notify_jobs()
        servers = self.registry.services.get("language_servers")
        if servers is not None:
            await servers.refresh()
        specs = self._filter_specs(self.registry.specs())
        self._emit(AgentEvent(kind="step_start", step=step))

        try:
            context = await self._queued_context(specs, step)
        except ContextStopped:
            return self._stop_notice(step)

        capabilities = await self.client.model_capabilities(self.model) if isinstance(self.client, APIClient) else None
        native_tools = capabilities is None or capabilities.native_tools
        extra_body: dict[str, Any] = {}
        if capabilities is not None and capabilities.format == "json_schema":
            extra_body["response_format"] = agent_response_format(native_tools=native_tools)
        elif not native_tools:
            extra_body["response_format"] = {"type": "json_object"}
        step_usage = Usage()
        attempts = 0
        overflows = 0
        transport_retries = 0
        budget_fraction = self._budget().fraction
        repair = ""
        while True:
            if self.pending:
                context = await self._queued_context(specs, step, repair=repair, budget_fraction=budget_fraction)
            self._reserve_request()
            request_body: dict[str, Any] = {}
            diagnostic_id: int | None = None
            diagnostics = self.registry.services.get("request_diagnostics")
            def finish_attempt(outcome: str, *, detail: str = "", response: str = "", usage: Usage | None = None) -> None:
                if diagnostics is not None:
                    diagnostics.finish(diagnostic_id, outcome, detail=detail, response=response,
                                       usage=asdict(usage) if usage is not None else None)
                    if diagnostics.error and self.registry.context_notes.get("request_diagnostics") != diagnostics.error:
                        self.registry.context_notes["request_diagnostics"] = diagnostics.error
                        self._emit(AgentEvent(kind="warning", text=f"Request diagnostics: {diagnostics.error}"))
            def request_sent(request: dict[str, Any]) -> None:
                nonlocal request_body, diagnostic_id
                request_body = request
                self._queued_messages_sent()
                if diagnostics is not None:
                    diagnostic_id = diagnostics.begin(request, step=step, step_id=len(self.history.steps) + 1)
                instructions = self.registry.services.get("project_instructions")
                if instructions is not None:
                    instructions.presented(self.registry.services.get("instruction_snapshot", {}))
                self._emit(AgentEvent(kind="context", step=step, text=json.dumps(request, ensure_ascii=False)))
            def delta(kind: str, chunk: str) -> None:
                stream_guard.feed(kind, chunk)
                if kind == "reasoning" and chunk:
                    self._emit(AgentEvent(kind="reasoning_delta", step=step, text=chunk))
            stream_guard = StreamLoopGuard()
            options: dict[str, Any] = {
                "on_delta": delta,
                "on_request": request_sent,
                "single_attempt": True,
            } if isinstance(self.client, APIClient) else {}
            try:
                if not options:
                    self._queued_messages_sent()
                    instructions = self.registry.services.get("project_instructions")
                    if instructions is not None:
                        instructions.presented(self.registry.services.get("instruction_snapshot", {}))
                completion = await self.client.chat(
                    model=self.model,
                    messages=context,
                    tools=(specs or None) if capabilities is None or native_tools else None,
                    extra_body=extra_body,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    session_id=self.session_id,
                    **options,
                )
                try:
                    if not options:
                        stream_guard.feed("reasoning", completion.message.reasoning or "")
                        stream_guard.feed("content", completion.text)
                    stream_guard.finish()
                except StreamLoopError as exc:
                    exc.usage = completion.usage
                    exc.partial_response = json.dumps(completion.message.to_api(), ensure_ascii=False)
                    raise
            except StreamLoopError as exc:
                self.usage = self.usage + exc.usage
                finish_attempt("loop_stopped", detail=str(exc), response=exc.partial_response, usage=exc.usage)
                self._emit(AgentEvent(kind="stream_end", step=step))
                self._emit(AgentEvent(kind="warning", step=step, text=str(exc)))
                self.stopped = True
                return str(exc)
            except APIContextError as exc:
                finish_attempt("context_overflow", detail=str(exc))
                self._emit(AgentEvent(kind="stream_end", step=step))
                if self.stop_requested:
                    return self._stop_notice(step)
                if overflows >= 2:
                    raise
                overflows += 1
                budget_fraction *= .65
                smaller = await self._queued_context(specs, step, repair=repair, budget_fraction=budget_fraction)
                if [m.to_api() for m in smaller] == [m.to_api() for m in context]:
                    raise ContextError("The provider rejected the current input even without removable history. Originals are preserved; shorten the input or select a larger-context model.")
                context = smaller
                self._budget().fraction = budget_fraction
                self._emit(AgentEvent(kind="retry", step=step, text=f"Provider context limit reached; retrying with less history ({overflows}/2). Originals are preserved."))
                continue
            except APIResponseError as exc:
                self.usage = self.usage + exc.usage
                step_usage = step_usage + exc.usage
                finish_attempt("request_error", detail=str(exc), response=exc.partial_response or exc.body or "", usage=exc.usage)
                self._emit(AgentEvent(kind="stream_end", step=step))
                if self.stop_requested:
                    return self._stop_notice(step)
                transient = isinstance(exc, APITransportError) or exc.status_code in RETRYABLE_STATUS
                if not transient:
                    raise
                if self._requests >= self.max_steps:
                    raise ContextStepLimit from exc
                def countdown(error: str, seconds: int) -> None:
                    self._emit(AgentEvent(kind="retry_wait", step=seconds, text=error))
                retry = await self.client.wait_retry(transport_retries, exc, stop=self._stop_event(), on_retry=countdown)
                if self._stop_event().is_set():
                    return self._stop_notice(step)
                if not retry:
                    raise
                transport_retries += 1
                continue
            except BaseException as exc:
                finish_attempt("cancelled" if isinstance(exc, asyncio.CancelledError) else "request_error", detail=str(exc))
                self._emit(AgentEvent(kind="stream_end", step=step))
                raise
            attempts += 1
            self._budget().observe(request_body, completion.usage.prompt_tokens)
            self.usage = self.usage + completion.usage
            self._persist()
            step_usage = step_usage + completion.usage
            try:
                if completion.response_error:
                    raise ResponseFormatError(completion.response_error, excerpt=completion.response_excerpt)
                record = parse_agent_response(completion.text, completion.tool_calls, tools=specs)
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
                self._emit(AgentEvent(kind="stream_end", step=step))
                self._emit(AgentEvent(kind="warning", step=step, text=str(exc)))
                self.stopped = True
                return str(exc)
            except ResponseFormatError as exc:
                finish_attempt("rejected", detail=str(exc), response=completion.response_excerpt or json.dumps(completion.message.to_api(), ensure_ascii=False), usage=completion.usage)
                rejection = str(exc)
                rejected_excerpt = exc.excerpt
            else:
                finish_attempt("accepted", response=json.dumps(completion.message.to_api(), ensure_ascii=False), usage=completion.usage)
                break
            self._emit(AgentEvent(kind="stream_end", step=step))
            if self.stop_requested:
                return self._stop_notice(step)
            if self._requests >= self.max_steps:
                raise ContextStepLimit
            if attempts >= MAX_MEMORY_ATTEMPTS:
                raise ContextError(f"The model returned invalid structured output in 3 consecutive responses: {rejection} No tools from these responses were executed; accepted history is preserved.")
            self._emit(AgentEvent(kind="retry", step=step, text=(
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
            context = await self._queued_context(specs, step, repair=repair, budget_fraction=budget_fraction)

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
        if not record.calls:
            self.messages.append(message)
            await self._archive_step(capabilities)
            self._emit(AgentEvent(kind="step_end", step=step, usage=step_usage))
            if self.stop_requested:
                self._stop_notice(step)
            # Text typed while this step ran is still owed a reply, so keep
            # it queued for the next step rather than dropping it.
            return text

        self.messages.append(message)
        self._persist()

        calls = message.tool_calls or []
        batch_results: list[tuple[ToolCall, ToolResult]] = []
        async def invoke_call(tool_call: ToolCall) -> ToolResult:
            journal = self.registry.services.get("session_journal")
            if journal is not None:
                journal.tool_started(len(self.history.steps) + 1, tool_call.id)
            self._emit(AgentEvent(kind="tool_start", step=step, tool_call=tool_call))
            invocation_token = current_invocation.set((len(self.history.steps) + 1, tool_call.id))
            output_token = command_output.set(
                lambda chunk: self._emit(AgentEvent(kind="tool_output", step=step, text=chunk, tool_call=tool_call))
            )
            try:
                if self.exposed_tools is not None and tool_call.name not in {spec.name for spec in specs}:
                    result = ToolResult.error("This tool is not available in the current request.")
                else:
                    result = await self.registry.invoke(tool_call.name, tool_call.arguments)
            except Exception as exc:
                result = ToolResult.error(f"Tool dispatch failed: {type(exc).__name__}: {exc}. Effects may be partial; inspect before retrying.")
            finally:
                command_output.reset(output_token)
                current_invocation.reset(invocation_token)
            return result

        def commit_call(tool_call: ToolCall, result: ToolResult) -> None:
            self.messages.append(_tool_message(tool_call, result))
            self._persist()
            batch_results.append((tool_call, result))
            self._emit(
                AgentEvent(kind="tool_end", step=step, tool_call=tool_call, result=result)
            )

        checkpoints = self._checkpoints()
        if checkpoints is not None:
            checkpoints.begin(len(self.history.steps) + 1)
        checkpoint_token = current_checkpoint.set(checkpoints)
        try:
            await run_batch(calls, self.registry, invoke_call, commit_call)
        except BaseException:
            await self._archive_step(capabilities)
            raise
        finally:
            current_checkpoint.reset(checkpoint_token)
            if checkpoints is not None:
                checkpoints.finish()

        diagnostics = await check_edit_batch(self.registry, batch_results, len(self.history.steps) + 1)
        if diagnostics:
            call, result = batch_results[-1]
            content = (result.content + "\n\nHarness Checks After This Complete Tool Batch:\n" + diagnostics
                       if isinstance(result.content, str) else {"result": result.content, "harness_checks": diagnostics})
            checked = ToolResult(content, result.is_error)
            self.messages[-1] = _tool_message(call, checked)
            batch_results[-1] = call, checked
            visible_diagnostics = "\n".join(line for line in diagnostics.splitlines()
                                            if "SKIPPED — no selected project Python." not in line)
            if any(word in visible_diagnostics for word in ("FAILED", "TIMED OUT", "SKIPPED", "unavailable")):
                self._emit(AgentEvent(kind="warning", step=step, text=visible_diagnostics))
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
            return "\n\n".join(result.content["text"] for _, result in batch_results)
        if repeated:
            self.stopped = True
            return guidance
        if self.stop_requested:
            return self._stop_notice(step)
        return None


def _tool_message(call: ToolCall, result: ToolResult) -> Message:
    """Keep execution status in the archived observation, not just the UI event."""
    return Message.tool_result(call.id, {
        "tool": call.name, "call_id": call.id,
        "status": "error" if result.is_error else "success", "content": result.content,
    })
