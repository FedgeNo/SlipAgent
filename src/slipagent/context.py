"""Keep originals in session memory; compact only the view sent to the model.

A step stores its active prompt, assistant response, and complete tool batch.
Recent steps include originals and available compressed analysis. Older steps
use summaries when smaller than the original representation; budgets may omit steps.
Original user prompts are retained independently of history selection. Compatibility readers
below support records from sessions using earlier inline-memory formats.
"""

from __future__ import annotations

from .prompts import load_prompt

import json
import re
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .api import APIError
from .tools.base import Tool, ToolResult
from .types import Message, ToolCall, ToolSpec, content_text
from .protocol import COMPRESSED_FIELDS, ResponseRecord, response_format
from .task import TaskMemory
from .records import record_message, step_record
from .data_text import MessageSections, render_data

DEFAULT_CONTEXT_LENGTH = 1_000_000
MIN_FULL_STEPS = 5
MAX_CONTEXT_SUMMARIES = 100
SUMMARY_MAX_CHARS = 6_000
SUMMARY_START = "<slipagent_context>"
SUMMARY_END = "</slipagent_context>"
TOOL_SUMMARY_KEY = "_slipagent_context"
TOOL_PREVIOUS_KEY = "_slipagent_previous"
HISTORY_PART_NAMES = ("prompt", "response", "reasoning", "tool_calls", "tool_results")
# Reserved labels from earlier context views, retained only to recognize echoes.
LEGACY_RESPONSE_HEADINGS = frozenset({
    "Agent Response (Full):", "Agent Response and Tool Calls (Full):",
})

def __getattr__(name: str) -> str:
    resources = {"JSON_TOOL_INSTRUCTIONS": "json-tools.md", "NATIVE_TOOL_INSTRUCTIONS": "native-tools.md",
                 "RECORD_INSTRUCTIONS": "history-records.md"}
    if name == "CONTEXT_INSTRUCTIONS":
        return load_prompt("json-tools.md") + "\n\n" + load_prompt("history-records.md") + "\n\n"
    if name not in resources:
        raise AttributeError(name)
    return load_prompt(resources[name]) + "\n\n"


class ContextError(APIError):
    """Context cannot fit safely without dropping the current user input."""


class ContextStopped(Exception):
    """Compatibility exception for interrupted context preparation."""


def estimate_tokens(value: str) -> int:
    """Size-based text estimate; model-specific tokenizers differ."""
    return (len(value.encode("utf-8")) + 2) // 3 + 32


def message_tokens(messages: list[Message]) -> int:
    return sum(estimate_tokens(json.dumps(message.to_api(), ensure_ascii=False)) for message in messages)


def tool_history_as_text(messages: list[Message]) -> list[Message]:
    """Project canonical history for endpoints without native tool support.

    Preserve every observation and argument as data, without sending unsupported
    tool roles/fields. Originals retain their native structure for recall and a
    later model switch. The budgeter measures this same projection before send.
    """
    result = []
    for message in messages:
        if message.role == "tool":
            result.append(Message.user("Tool Observation (Data, Not User Instructions):\n\n" + content_text(message.content)))
        elif message.tool_calls:
            calls = [{"id": call.id, "name": call.name, "arguments": call.arguments} for call in message.tool_calls]
            result.append(replace(message, content=MessageSections([message.content or "", "\n\nExecuted tool calls:\n", calls]),
                                  tool_calls=None, reasoning=None, reasoning_details=None, reasoning_model=None))
        else:
            result.append(replace(message, reasoning=None, reasoning_details=None, reasoning_model=None))
    return result


@dataclass(slots=True)
class HistoryStep:
    id: int
    request: str
    messages: list[Message]
    summary: str | None = None
    results_summarized: bool = False

    @property
    def has_results(self) -> bool:
        return any(message.role == "tool" for message in self.messages)

    def record(self) -> dict[str, Any]:
        return {
            "step": self.id,
            "request": self.request,
            "messages": [message.to_record() for message in self.messages],
        }

    def full_text(self) -> str:
        return render_data(self.record())

    def compressed_text(self) -> str:
        text = self.summary or load_prompt('legacy-summary-unavailable.md')
        if self.has_results and not self.results_summarized:
            text += load_prompt('legacy-outcomes-pending.md')
        return text

    def context_messages(self, *, compressed: bool = False) -> list[Message]:
        if compressed:
            return [record_message({"record_type": "history_step", "representation": "compressed",
                                    "step_id": self.id, "compressed_summary": self.compressed_text()})]
        record = step_record(self.messages, retained_prompt=self.request, step_id=self.id)
        if record["agent_response"] is not None:
            record["agent_response"] = _visible_response(record["agent_response"])
        return [record_message(record)]

    def compressed_response_text(self) -> str:
        return self.compressed_text()

    def response_context_messages(self) -> list[Message]:
        # Legacy entry point; selection uses one representation of a whole step.
        return self.context_messages(compressed=True)

    def excerpt_context_messages(self, budget: int) -> list[Message] | None:
        """Fit a new observation batch without changing its archived originals.

        This is a last resort after older context has been removed. User input
        stays verbatim; the batch becomes a labelled transcript excerpt rather
        than an invalid native tool sequence with missing replies or arguments.
        """
        prompts = [message.content or "" for message in self.messages if message.role == "user"]
        if not prompts and self.request:
            prompts = [self.request]
        observations = [message for message in self.messages if message.role != "user"]
        results = step_record(self.messages)["tool_results"]

        def clipped(text: str, length: int) -> dict[str, Any]:
            if len(text) <= length:
                return {"text": text, "omitted_characters": 0}
            half = length // 2
            return {"beginning": text[:half], "ending": text[-(length - half):] if length else "",
                    "omitted_characters": len(text) - length}

        def candidate(length: int, count: int) -> list[Message]:
            entries = []
            result_iterator = iter(results)
            for message in observations[:count]:
                entry: dict[str, Any] = {"role": message.role, "content_excerpt": clipped(content_text(message.content), length)}
                if message.role == "tool":
                    result = next(result_iterator)
                    entry.update({key: result[key] for key in ("call_id", "tool_name", "status")})
                    entry["content_excerpt"] = clipped(content_text(result["content"]), length)
                if message.tool_calls:
                    entry["calls_excerpt"] = clipped(render_data([call.to_record() for call in message.tool_calls]), length)
                entries.append(entry)
            note = (
                load_prompt('history-excerpt.md', shown_messages=count, total_messages=len(observations)) + '\n'
            )
            return [record_message({"record_type": "history_step", "representation": "excerpt",
                                    "step_id": self.id, "user_prompt": prompts, "messages": entries,
                                    "omitted_messages": len(observations) - count, "recall_instructions": note})]

        count = len(observations)
        while count and message_tokens(candidate(0, count)) > budget:
            count //= 2
        minimum = candidate(0, count)
        if message_tokens(minimum) > budget:
            return None
        low, high = 0, max(0, budget * 3)
        best = minimum
        while low <= high:
            middle = (low + high) // 2
            proposed = candidate(middle, count)
            if message_tokens(proposed) <= budget:
                best = proposed
                low = middle + 1
            else:
                high = middle - 1
        return best


@dataclass(slots=True)
class ResponseMemory:
    text: str
    summary: str | None = None
    previous: str | None = None


class StructuredStep(HistoryStep):
    """Separate compressed fields without changing the live step layout."""

    previous_tool_responses_compressed: str
    user_prompt_compressed: str
    agent_response_compressed: str
    _following_record: StructuredStep | None
    @property
    def summary(self) -> str | None:
        fields = {field: getattr(self, field, None) for field in COMPRESSED_FIELDS}
        if all(isinstance(value, str) for value in fields.values()):
            # The previous batch belongs to its originating step, where the
            # following record supplies its result summary exactly once.
            del fields["previous_tool_responses_compressed"]
            following = getattr(self, "_following_record", None)
            if isinstance(following, StructuredStep):
                fields["tool_responses_compressed"] = following.previous_tool_responses_compressed
            return render_data(fields)
        legacy = getattr(self, "_legacy_summary", None)
        return legacy if isinstance(legacy, str) else None

    @summary.setter
    def summary(self, value: str | None) -> None:
        self._legacy_summary = value

    def compressed_response_text(self) -> str:
        response = getattr(self, "agent_response_compressed", None)
        if not isinstance(response, str):
            return super().compressed_response_text()
        fields = {"agent_response_compressed": response}
        following = getattr(self, "_following_record", None)
        if self.has_results and isinstance(following, StructuredStep):
            fields["tool_responses_compressed"] = following.previous_tool_responses_compressed
        return render_data(fields)


class CompletedStep(HistoryStep):
    """One complete step with independently retrievable originals and summary.

    Messages remain the canonical ordered conversation. Separate part references
    reuse immutable text rather than copying it. Reasoning is one string from
    this response; compaction keeps factual memory and filtered thoughts separate.
    """

    def __init__(self, step_id: int, request: str, messages: list[Message]) -> None:
        super().__init__(step_id, request, messages)
        self.user_prompt = request
        assistant = next(message for message in messages if message.role == "assistant")
        self.agent_response = assistant.content or ""
        self.reasoning = assistant.reasoning or ""
        self.tool_calls = assistant.tool_calls or []
        self.tool_results = [message for message in messages if message.role == "tool"]
        self.compaction_status = "pending"
        self.compaction_error: str | None = None
        self.reasoning_summary: str | None = None
        self.observed = not self.has_results

    def parts(self) -> dict[str, Any]:
        return {"prompt": self.user_prompt, "response": self.agent_response,
                "reasoning": getattr(self, "reasoning", ""),
                "tool_calls": [call.to_record() for call in self.tool_calls],
                "tool_results": [{"call_id": result.tool_call_id, "content": result.content} for result in self.tool_results]}

    def compaction_input(self) -> dict[str, Any]:
        parts = self.parts()
        return {"user_prompt": parts["prompt"],
                           "agent_response": parts["response"], "tool_calls": parts["tool_calls"],
                "tool_results": parts["tool_results"], "reasoning": parts["reasoning"]}

    def record(self) -> dict[str, Any]:
        return {"step": self.id, **self.parts()}

    def compressed_text(self) -> str:
        if self.summary is not None:
            return self.summary
        return load_prompt("summary-unavailable.md", status=self.compaction_status)

    def context_messages(self, *, compressed: bool = False) -> list[Message]:
        return super().context_messages(compressed=compressed)


def memory_specs(specs: list[ToolSpec]) -> list[ToolSpec]:
    """Legacy schema adapter, retained for imported fixtures; not used by Agent."""
    result = []
    for spec in specs:
        properties = {**spec.parameters.get("properties", {}),
                      TOOL_SUMMARY_KEY: {"type": "string", "minLength": 1, "maxLength": SUMMARY_MAX_CHARS,
                                         "description": load_prompt("legacy-tool-summary.md")},
                      TOOL_PREVIOUS_KEY: {"type": "string", "maxLength": SUMMARY_MAX_CHARS,
                                          "description": load_prompt("legacy-tool-previous-summary.md")}}
        parameters = {**spec.parameters, "properties": properties,
                      "required": list(dict.fromkeys([*spec.parameters.get("required", []), TOOL_SUMMARY_KEY]))}
        result.append(ToolSpec(spec.name, spec.description, parameters))
    return result


def tool_response_memory(text: str, calls: list[ToolCall], step_id: int, previous_id: int | None) -> tuple[ResponseMemory, list[ToolCall]]:
    memory = response_memory(text, step_id, previous_id)
    cleaned = []
    for call in calls:
        summary = _valid_summary(call.arguments.get(TOOL_SUMMARY_KEY))
        previous = _valid_summary(call.arguments.get(TOOL_PREVIOUS_KEY))
        if memory.summary is None and summary is not None:
            memory.summary = summary
        if previous_id is not None and memory.previous is None and previous is not None:
            memory.previous = previous
        cleaned.append(ToolCall(call.id, call.name, {key: value for key, value in call.arguments.items()
                                                   if key not in {TOOL_SUMMARY_KEY, TOOL_PREVIOUS_KEY}}))
    return memory, cleaned


class VisibleStream:
    """Legacy prose-stream filter; current replies wait for action validation."""

    def __init__(self) -> None:
        self.content = ""
        self.emitted = 0

    def push(self, text: str) -> str:
        self.content += text
        markers = (SUMMARY_START, "[Harness context metadata]", "Previous step's current summary:")
        end = len(self.content)
        for marker in markers:
            index = self.content.find(marker)
            if index >= 0:
                end = min(end, index)
            for size in range(1, len(marker)):
                if self.content[:end].endswith(marker[:size]):
                    end = min(end, len(self.content[:end]) - size)
        end = len(self.content[:end].rstrip())
        chunk = self.content[self.emitted:end]
        self.emitted = max(self.emitted, end)
        return chunk

    def finish(self, visible: str) -> str:
        chunk = visible[self.emitted:] if visible.startswith(self.content[:self.emitted]) else ""
        self.emitted += len(chunk)
        return chunk


def response_memory(text: str, step_id: int, previous_id: int | None) -> ResponseMemory:
    """Hide the memory envelope and echoed bookkeeping; preserve prose/examples."""
    fallback = text
    # The summary itself may quote the marker. Try all candidates so a marker
    # inside a JSON string cannot hide or corrupt the actual outer envelope.
    for match in reversed(list(re.finditer(re.escape(SUMMARY_START), text))):
        start = match.start()
        fenced = False
        for line in text[:start].splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
        if fenced:
            continue
        body = text[match.end():].strip()
        if not body.startswith("{"):
            continue
        visible = text[:start].rstrip()
        if not body.endswith(SUMMARY_END):
            if SUMMARY_END not in body:
                fallback = visible
            continue
        fallback = visible
        try:
            raw = json.loads(body[:-len(SUMMARY_END)])
        except ValueError:
            continue
        if not isinstance(raw, dict) or type(raw.get("step")) is not int or raw["step"] != step_id:
            continue
        summary = _valid_summary(raw.get("summary"))
        previous = raw.get("previous")
        previous_summary = None
        if isinstance(previous, dict) and type(previous.get("step")) is int and previous["step"] == previous_id:
            previous_summary = _valid_summary(previous.get("summary"))
        return ResponseMemory(_visible_response(visible), summary, previous_summary)
    return ResponseMemory(_visible_response(fallback))


def _visible_response(text: str) -> str:
    """Remove reserved bookkeeping lines without hiding numbers in normal output."""
    lines = []
    fenced = False
    metadata = False
    changed = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            metadata = False
        elif not fenced:
            if stripped in LEGACY_RESPONSE_HEADINGS:
                changed = True
                continue
            if stripped == "[Harness context metadata]":
                metadata = True
                changed = True
                continue
            if stripped.startswith("Previous step's current summary:"):
                metadata = True
                changed = True
                continue
            if metadata and (not stripped or re.fullmatch(r"\d+|none", stripped)):
                changed = True
                continue
            metadata = False
        lines.append(line)
    return "".join(lines).strip() if changed else text


def _valid_summary(value: Any) -> str | None:
    if isinstance(value, str) and value.strip() and len(value) <= SUMMARY_MAX_CHARS:
        return value.strip()
    return None


Summarizer = Callable[[str], Awaitable[str]]


@dataclass(slots=True)
class ConversationHistory:
    steps: list[HistoryStep] = field(default_factory=list)
    cursor: int = 0
    digest: str = ""
    digest_through: int = 0
    _request: str = ""
    task: TaskMemory = field(default_factory=TaskMemory)

    def clear(self) -> None:
        self.steps.clear()
        self.cursor = 0
        self.digest = ""
        self.digest_through = 0
        self._request = ""
        self.task = TaskMemory()

    def sync(self, messages: list[Message]) -> None:
        """Archive complete batches and register the user sources they belong to.

        Stop at an incomplete assistant/tool sequence. The cursor advances only
        after every declared call has its matching result, so later recovery can
        finish that step without duplicating earlier accepted messages.
        """
        while self.cursor < len(messages) and messages[self.cursor].role == "system":
            self.cursor += 1
        start = self.cursor
        index = start
        while index < len(messages):
            message = messages[index]
            if message.role == "user":
                self.task.note_user(index, len(self.steps) + 1, message.content or "")
            if message.role != "assistant":
                index += 1
                continue
            end = index + 1
            calls = message.tool_calls or []
            for call in calls:
                if end >= len(messages) or messages[end].role != "tool" or messages[end].tool_call_id != call.id:
                    return
                end += 1
            segment = messages[start:end]
            requests = [entry.content or "" for entry in segment if entry.role == "user"]
            if requests:
                self._request = "\n".join(requests)
            self.steps.append(CompletedStep(len(self.steps) + 1, self._request, list(segment)))
            self.cursor = end
            start = end
            index = end

    def save_response(self, memory: ResponseMemory, previous_id: int | None) -> None:
        """Attach a legacy memory envelope; unused by the current agent loop."""
        step = self.steps[-1]
        step.summary = memory.summary
        step.results_summarized = not step.has_results
        if previous_id is not None and memory.previous is not None:
            previous = self.steps[previous_id - 1]
            previous.summary = memory.previous
            previous.results_summarized = True
        if step.summary is None:
            raise ContextError(
                "The model did not provide a valid compressed record for this response. "
                "The full record is preserved. No extra summary request was made."
            )

    def instructions(self, *, native_tools: bool = False) -> str:
        return ""

    def save_record(self, record: ResponseRecord, previous_id: int | None) -> None:
        """Attach the older inline-memory format to a StructuredStep.

        This batch's newly executed tools have not been read by the model yet;
        its successor supplies their compressed outcomes.
        Current CompletedSteps receive one isolated background summary instead.
        """
        step = self.steps[-1]
        if not isinstance(step, StructuredStep):
            raise ContextError("Cannot attach a structured record to a legacy history step.")
        for field in COMPRESSED_FIELDS:
            setattr(step, field, getattr(record, field))
        step.results_summarized = not step.has_results
        if previous_id is not None:
            previous = self.steps[previous_id - 1]
            if isinstance(previous, StructuredStep):
                previous._following_record = step
            elif previous.summary is not None:
                previous.summary += load_prompt('legacy-results.md', results=record.previous_tool_responses_compressed)
            previous.results_summarized = True

    async def view(
        self, messages: list[Message], specs: list[ToolSpec], *,
        keep_steps: int, context_length: int,
        max_output: int, summarize: Summarizer | None = None,
        extra_instructions: str = "",
        user_corrections: str = "",
        native_tools: bool = False,
        text_tool_history: bool = False,
        schema: dict[str, Any] | None = None,
        token_scale: float = 1.0,
        overthinking: bool = True,
        reasoning_history_steps: int = 25,
        filtered_thoughts: bool = True,
    ) -> list[Message]:
        """Select history and attach available analysis without altering originals.

        A newly returned batch must reach the working model before it can leave
        the full window, even if its independent summary already finished.
        Oversized new observations use explicit excerpts as a last resort.
        """
        self.sync(messages)
        def tokens(part: list[Message]) -> int:
            return math.ceil(message_tokens(tool_history_as_text(part) if text_tool_history else part) * token_scale)
        sections = [load_prompt("native-tools.md" if native_tools else "json-tools.md"), load_prompt("history-records.md"),
                    load_prompt("system-history.md")]
        if overthinking and reasoning_history_steps:
            sections.append(
                load_prompt('overthinking-on.md', steps=reasoning_history_steps)
            )
        else:
            sections.append(load_prompt('overthinking-off.md'))
        # Stable response/tool guidance precedes changing step IDs and task
        # state. Keep one system-message prefix for provider compatibility.
        sections.extend([extra_instructions, self.instructions(native_tools=native_tools)])
        if self.task.current_prompt_step is not None:
            has_new_user_input = any(message.role == "user" for message in messages[self.cursor:])
            input_guidance = (
                load_prompt('new-user-request.md')
                if has_new_user_input else
                load_prompt('tool-result-response.md')
            )
            sections.append(
                load_prompt('user-request.md', step_id=self.task.current_prompt_step, input_guidance=input_guidance, user_messages=render_data(self.task.sources[self.task.current_prompt_step]))
            )
        instructions = "\n" + "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"
        pinned = [message for message in messages if message.role == "system"]
        if pinned:
            pinned = [*pinned[:-1], Message.system((pinned[-1].content or "").rstrip() + "\n" + instructions)]
        else:
            pinned = [Message.system("\n" + instructions)]
        history_begin = load_prompt('history-opening.md').strip()
        history_end = load_prompt('history-closing.md').strip()
        history_prefix = (pinned[-1].content or "").rstrip() + "\n\n" + history_begin + "\n\n"
        history_suffix = "\n\n" + history_end + "\n"
        pinned[-1] = Message.system(MessageSections([history_prefix, [], history_suffix]))

        def assemble(selected: list[Message], current: list[Message]) -> list[Message]:
            # Only the request projection changes; archives retain their roles.
            return [*pinned[:-1], Message.system(MessageSections([
                history_prefix, [message.content for message in selected], history_suffix,
            ])), *current]

        tail_messages = [message for message in messages[self.cursor:] if message.role != "system"]
        tail = [record_message(step_record(tail_messages, current=True, step_id=len(self.steps) + 1))]
        overhead = tokens(pinned)
        if not text_tool_history:
            overhead += math.ceil(estimate_tokens(json.dumps([spec.to_api() for spec in specs])) * token_scale)
        if schema is not None:
            overhead += math.ceil(estimate_tokens(json.dumps(schema)) * token_scale)
        available = int(context_length * 0.85) - max_output - overhead
        rendered_limit = available + tokens(pinned)
        tail_tokens = tokens(tail)
        if tail_tokens + 256 >= available:
            raise ContextError("Current input and instructions exceed the context budget. Shorten the input or reference a file; conversation history is preserved.")
        boundary = max(0, len(self.steps) - max(MIN_FULL_STEPS, keep_steps))
        initial_boundary = boundary
        memory_start = max(0, boundary - MAX_CONTEXT_SUMMARIES)
        # Older originals remain archived indefinitely. Materialize only the
        # recent window and the bounded older candidates needed by this request.
        def context_part(index: int, *, compressed: bool = False,
                         include_summary: bool = True) -> list[Message]:
            step = self.steps[index]
            part = step.context_messages(compressed=compressed)
            if not compressed and include_summary and step.summary is not None:
                part = [record_message({**part[0].content, "compressed_summary": step.compressed_text()})]
            if overthinking and index >= len(self.steps) - reasoning_history_steps:
                reasoning = next((message.reasoning for message in step.messages
                                  if message.role == "assistant"), None)
                filtered = getattr(step, "reasoning_summary", None) if filtered_thoughts else None
                if filtered is not None:
                    reasoning = filtered
                if reasoning:
                    record = dict(part[0].content)
                    record["reasoning"] = reasoning
                    record["reasoning_representation"] = "filtered" if filtered is not None else "original"
                    part = [record_message(record)]
            return part

        parts = [context_part(index) for index in range(boundary, len(self.steps))]
        sizes = [tokens(part) for part in parts]
        pending = bool(self.steps and self.steps[-1].has_results and not (
            self.steps[-1].observed if isinstance(self.steps[-1], CompletedStep) else self.steps[-1].results_summarized))
        # A continuation still needs its active user prompt even at tiny budgets.
        required_last = pending or bool(self.steps and not tail_messages)
        max_boundary = len(self.steps) - int(required_last)
        older_parts: dict[int, list[Message]] = {}

        def older_context(start: int, end: int) -> list[Message]:
            result: list[Message] = []
            for index in range(start, end):
                if index not in older_parts:
                    original = parts[index - initial_boundary] if index >= initial_boundary else context_part(index)
                    compressed = context_part(index, compressed=True)
                    # Choose the base representation before attaching analysis;
                    # budget the combined record actually sent below.
                    raw = context_part(index, include_summary=False)
                    older_parts[index] = compressed if tokens(compressed) < tokens(raw) else original
                result.extend(older_parts[index])
            return result

        def current_input(selected: list[Message]) -> list[Message]:
            current = self.task.prompt_supplement(selected + tail) or tail
            if user_corrections.strip():
                record = dict(current[-1].content)
                record["user_prompt"] = [*record["user_prompt"], load_prompt('tool-use-correction.md', correction=user_corrections.strip()).strip()]
                current = [*current[:-1], record_message(record)]
            return current

        while True:
            memory_start = max(memory_start, boundary - MAX_CONTEXT_SUMMARIES)
            older = older_context(memory_start, boundary)
            offset = boundary - initial_boundary
            full = [message for part in parts[offset:] for message in part]
            # Prompt retention is the harness's responsibility. Even when its
            # original step is compressed, supply its exact text and source ID.
            current = current_input(older + full)
            budget = available - tokens(older) - tokens(current)
            if sum(sizes[offset:]) <= budget:
                candidate = assemble(older + full, current)
                if tokens(candidate) <= rendered_limit:
                    return candidate
            # Drop older records before shortening the full window. Start from
            # the normal boundaries on every request so records return when a
            # large step ages out; omission never changes the stored history.
            if memory_start < boundary:
                memory_start += 1
                continue
            if boundary < max_boundary:
                boundary += 1
                continue
            if required_last:
                excerpt_budget = budget
                while excerpt_budget > 0:
                    excerpt = self.steps[-1].excerpt_context_messages(int(excerpt_budget / token_scale))
                    if excerpt is None:
                        break
                    current = current_input(excerpt)
                    candidate = assemble(older + excerpt, current)
                    cost = tokens(candidate)
                    if cost <= rendered_limit:
                        return candidate
                    excerpt_budget -= max(1, cost - rendered_limit)
            raise ContextError("Current input and latest tool results exceed the context budget even without older history; originals and summaries are preserved.")


class RecallHistoryTool(Tool):
    name = "recall_history"
    description_prompt = 'tools/recall-history.md'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {
            "step_id": {"type": "integer", "minimum": 1},
            "call_id": {"type": "string"},
            "section": {"type": "string", "enum": ["all", "user", *HISTORY_PART_NAMES]},
            "sections": {"type": "array", "minItems": 1, "uniqueItems": True,
                         "items": {"type": "string", "enum": list(HISTORY_PART_NAMES)}},
            "query": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 16000},
        },
    }

    def __init__(self, history: ConversationHistory) -> None:
        self.history = history

    async def run(self, *, step_id: int | None = None, call_id: str | None = None,
                  query: str = "", offset: int = 0, limit: int = 8000, section: str | None = None,
                  sections: list[str] | None = None) -> ToolResult:
        if sections is not None and (section is not None or call_id is not None):
            return ToolResult.error("sections cannot be combined with section or call_id.")
        if sections is not None and (not sections or len(sections) != len(set(sections))):
            return ToolResult.error("sections must contain at least one part, without duplicates.")
        # Direct callers bypass schema validation; use the same names here so
        # unknown selectors cannot raise KeyError or silently select the full step.
        if section is not None and section not in ("all", "user", *HISTORY_PART_NAMES):
            return ToolResult.error(
                f"Unknown history section {section!r}. Choose from: all, user, {', '.join(HISTORY_PART_NAMES)}."
            )
        if sections is not None:
            for name in sections:
                if name not in HISTORY_PART_NAMES:
                    return ToolResult.error(
                        f"Unknown history part {name!r}. Choose from: {', '.join(HISTORY_PART_NAMES)}."
                    )
        section = section or "all"
        if (sections is not None or section != "all") and step_id is None:
            return ToolResult.error("Selecting original parts requires step_id.")
        if call_id is not None and section not in {"all", "tool_calls", "tool_results"}:
            return ToolResult.error("call_id requires section all, tool_calls, or tool_results.")
        if section == "user" and (step_id is None or call_id is not None):
            return ToolResult.error("section='user' requires step_id and cannot be combined with call_id.")
        if call_id is not None and step_id is None:
            return ToolResult.error("call_id requires step_id; call IDs are unique only within a batch.")
        structured: Any = None
        if step_id is not None:
            if step_id > len(self.history.steps):
                return ToolResult.error(f"No history step {step_id}. Available steps: 1–{len(self.history.steps)}.")
            step = self.history.steps[step_id - 1]
            structured = step.record()
            text = step.full_text()
            if section == "user":
                structured = [message.content or "" for message in self.history.steps[step_id - 1].messages
                              if message.role == "user"]
                text = render_data(structured)
            if section in HISTORY_PART_NAMES or sections is not None:
                original = step if isinstance(step, CompletedStep) else CompletedStep(step.id, step.request, step.messages)
                parts = original.parts()
                if call_id is not None:
                    key = "id" if section == "tool_calls" else "call_id"
                    parts[section] = [value for value in parts[section] if value[key] == call_id]
                    if not parts[section]:
                        return ToolResult.error(f"No {section} with call_id {call_id!r} in step {step_id}.")
                value = {name: parts[name] for name in sections} if sections is not None else parts[section]
                structured = value
                text = content_text(value)
            elif call_id is not None:
                observation = next((message for message in self.history.steps[step_id - 1].messages
                                    if message.role == "tool" and message.tool_call_id == call_id), None)
                if observation is None:
                    return ToolResult.error(f"No tool result with call_id {call_id!r} in step {step_id}.")
                structured = observation.content
                text = content_text(structured)
        else:
            needle = query.casefold()
            matches = []
            for step in self.history.steps:
                if needle and needle not in step.full_text().casefold() and needle not in (step.summary or "").casefold():
                    continue
                preview = (step.summary or step.request).replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:199] + "…"
                matches.append(f"Step {step.id}: {preview}")
            text = "\n".join(matches) or "No matching history steps."
        end = min(len(text), offset + limit)
        page: dict[str, Any] = {
            "step_id": step_id, "offset": offset, "next_offset": end if end < len(text) else None,
            "section": section,
            "sections": sections,
            "total_characters": len(text),
            "content": structured if offset == 0 and end == len(text) and structured is not None else text[offset:end],
        }
        if step_id is None:
            page["total_matches"] = len(matches)
        return ToolResult.ok(page)
