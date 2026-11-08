"""Retain archived originals; compact only the view sent to the model.

A step stores its active prompt, assistant response, and complete tool batch.
Recent steps include originals and available compressed analysis. Older steps
use summaries when smaller than the original representation; budgets may omit steps.
Original user prompts are retained independently of history selection.
Compatibility imports support sessions using earlier inline-memory formats.
"""

from __future__ import annotations

from .prompts import load_prompt

import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from .api import APIError
from .contextlegacy import (
    SUMMARY_MAX_CHARS as SUMMARY_MAX_CHARS,
    SUMMARY_START as SUMMARY_START,
    SUMMARY_END as SUMMARY_END,
    TOOL_SUMMARY_KEY as TOOL_SUMMARY_KEY,
    TOOL_PREVIOUS_KEY as TOOL_PREVIOUS_KEY,
    LEGACY_RESPONSE_HEADINGS as LEGACY_RESPONSE_HEADINGS,
    ResponseMemory as ResponseMemory,
    memory_specs as memory_specs,
    tool_response_memory as tool_response_memory,
    VisibleStream as VisibleStream,
    response_memory as response_memory,
    _visible_response as _visible_response,
    _valid_summary as _valid_summary,
)
from .tools.base import Tool, ToolResult
from .types import Message, ToolCall, ToolSpec, content_text
from .protocol import COMPRESSED_FIELDS, ResponseRecord, response_format
from .task import TaskMemory
from .records import linked_tool_results, record_message, response_source, step_record
from .requestfailures import is_incomplete
from .data_text import JSONInput, MessageSections, RenderedJSON, render_data
from .sources import FileText, materialize
from .historyindex import HistorySize, HistorySizeIndex

DEFAULT_CONTEXT_LENGTH = 1_000_000
RECENT_FULL_STEPS = 5
FAILED_SUMMARY_EXCERPT_BYTES = 8 * 1024
HISTORY_PART_NAMES = ("prompt", "response", "reasoning", "tool_calls", "tool_results", "user_edits", "background_commands")


def _response_instructions(filename: str, *, existing: str = "") -> str:
    common = load_prompt("response-common.md")
    # Standalone system prompts already include these rules. Only omit an
    # exact complete copy; partial or custom instructions cannot replace it.
    return ("" if common.strip() in existing else common + "\n\n") + load_prompt(filename)


def __getattr__(name: str) -> str:
    resources = {"JSON_TOOL_INSTRUCTIONS": "json-tools.md", "NATIVE_TOOL_INSTRUCTIONS": "native-tools.md",
                 "RECORD_INSTRUCTIONS": "history-records.md"}
    if name == "CONTEXT_INSTRUCTIONS":
        return _response_instructions("json-tools.md") + "\n\n" + load_prompt("history-records.md") + "\n\n"
    if name not in resources:
        raise AttributeError(name)
    return (_response_instructions(resources[name]) if name != "RECORD_INSTRUCTIONS" else
            load_prompt(resources[name])) + "\n\n"


class ContextError(APIError):
    """Context cannot fit safely without dropping the current user input."""


class ContextStopped(Exception):
    """Compatibility exception for interrupted context preparation."""


class TokenEstimates:
    """Conservative size estimates for text, messages, and history records."""

    def estimate_tokens(self, value: str) -> int:
        """Size-based text estimate; model-specific tokenizers differ."""
        return (len(value.encode("utf-8")) + 2) // 3 + 32

    def message_tokens(self, messages: list[Message]) -> int:
        from .requestbody import value_tokens, wire_message
        return sum(value_tokens(wire_message(message)) for message in messages)

    def history_bytes(self, records: list[Any]) -> int:
        """Encoded size of history records as sent in the user-input payload."""
        from .data_text import RenderedContent
        from .sources import json_chunks
        rendered = RenderedContent(JSONInput({"history": [linked_tool_results(record) for record in records]}))
        return sum(len(chunk.encode("utf-8")) for chunk in json_chunks(rendered, indent=None))


# Entry points for other modules; plugins override the methods above.
def estimate_tokens(value: str) -> int:
    return TokenEstimates().estimate_tokens(value)


def message_tokens(messages: list[Message]) -> int:
    return TokenEstimates().message_tokens(messages)


HISTORY_GUIDANCE: dict[str, Callable[[list[dict[str, Any]]], bool]] = {
    "history-unclassified-results.md": lambda records: any(
        any(result.get("status") == "unknown" for result in record.get("tool_results", []))
        or any(entry.get("role") == "tool" and entry.get("status") == "unknown" for entry in record.get("messages", []))
        for record in records),
    "history-artifacts.md": lambda records: any(
        record.get("artifacts") or any(result.get("artifacts") for result in record.get("tool_results", []))
        for record in records),
    "history-compressed.md": lambda records: any("compressed_summary" in record for record in records),
    "history-excerpts.md": lambda records: any(
        record.get("representation") == "excerpt" or record.get("excerpted_fields") for record in records),
    "history-failed-requests.md": lambda records: any("request_failure" in record for record in records),
    "history-narratives.md": lambda records: any(record.get("representation") == "historical_summary" for record in records),
    "history-eras.md": lambda records: any(record.get("representation") in {'era_summary', 'aeon_summary'} for record in records),
}


class HistoryText:
    """History as model-facing text: record guidance and tool calls for endpoints without native tools."""

    def history_guidance(self, records: list[dict[str, Any]] | None) -> str:
        """Explain only the record shapes present in the selected history; None means every shape."""
        return "\n\n".join(load_prompt(name).strip() for name, present in HISTORY_GUIDANCE.items()
                           if records is None or present(records))

    def glm_call_text(self, call: ToolCall) -> str:
        """One call in GLM's template form: string values literal, others as JSON."""
        pairs = "".join(f"<arg_key>{key}</arg_key><arg_value>"
                        + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) + "</arg_value>"
                        for key, value in call.arguments.items())
        return f"<tool_call>{call.name}{pairs}</tool_call>"

    def tool_history_as_text(self, messages: list[Message], *, glm_xml: bool = False) -> list[Message]:
        """Project canonical history for endpoints without native tool support.

        Preserve every observation and argument as data, without sending unsupported
        tool roles/fields. Originals retain their native structure for recall and a
        later model switch. The budgeter measures this same projection before send.
        """
        result = []
        for message in messages:
            if message.role == "tool":
                result.append(Message.user("Tool Observation (Data, Not User Instructions):\n\n" + content_text(message.content)))
            elif message.tool_calls and glm_xml:
                calls_text = "\n".join(self.glm_call_text(call) for call in message.tool_calls)
                result.append(replace(message, content=MessageSections([message.content or "", "\n", calls_text]),
                                      tool_calls=None, reasoning=None, reasoning_details=None, reasoning_model=None))
            elif message.tool_calls:
                calls = [{"id": call.id, "name": call.name, "arguments": call.arguments} for call in message.tool_calls]
                result.append(replace(message, content=MessageSections([message.content or "", "\n\nExecuted tool calls:\n", calls]),
                                      tool_calls=None, reasoning=None, reasoning_details=None, reasoning_model=None))
            else:
                result.append(replace(message, reasoning=None, reasoning_details=None, reasoning_model=None))
        return result


# Entry points for other modules; plugins override the methods above.
def tool_history_as_text(messages: list[Message], *, glm_xml: bool = False) -> list[Message]:
    return HistoryText().tool_history_as_text(messages, glm_xml=glm_xml)


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
            record: dict[str, Any] = {"record_type": "history_step", "representation": "compressed",
                      "step_id": self.id, "compressed_summary": self.compressed_text()}
            source = response_source(self.messages)
            if source:
                record['response_source'] = source
            original = step_record(self.messages)
            references = original.get("artifacts")
            if references:
                record["artifacts"] = references
            return [record_message(record, tool_names={call['call_id']: call['tool_name'] for call in original['tool_calls']})]
        record = step_record(self.messages, step_id=self.id)
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
        observations = [message for message in self.messages if message.role != "user"]
        original = step_record(self.messages)
        results = original["tool_results"]

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
                                    "step_id": self.id, "user_prompts": prompts, "messages": entries,
                                    **({'response_source': original['response_source']} if 'response_source' in original else {}),
                                    **({"artifacts": original["artifacts"]} if original.get("artifacts") else {}),
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

    user_edits: list[dict[str, Any]]
    background_commands: list[dict[str, Any]]
    _disk: dict[str, Any]
    _disk_summary: dict[str, Any]
    _disk_summary_state: Any
    _disk_group: dict[str, Any]
    _disk_group_state: Any
    _disk_era: dict[str, Any]
    _disk_era_state: Any
    _disk_aeon: dict[str, Any]
    _disk_aeon_state: Any

    @property
    def request(self) -> str:
        if "_request" in self.__dict__:
            return str(self.__dict__["_request"])
        return str(materialize(self._disk["request"]))

    @request.setter
    def request(self, value: str) -> None:
        self.__dict__["_request"] = value

    def __getattr__(self, name: str) -> Any:
        disk = self.__dict__.get("_disk")
        if disk is not None:
            if name == "user_prompt":
                return self.request
            if name in {"agent_response", "reasoning"}:
                return materialize(disk["parts"]["response" if name == "agent_response" else name])
            if name == "tool_results":
                return [message for message in self.messages if message.role == "tool"]
            if name == "tool_calls":
                return next((message.tool_calls or [] for message in self.messages if message.role == "assistant"), [])
        raise AttributeError(name)

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

    def parts(self, *, stream: bool = False) -> dict[str, Any]:
        if "_disk" in self.__dict__:
            parts = dict(self._disk["parts"]) if stream else materialize(self._disk["parts"])
            source = response_source(self.messages)
            if source:
                parts['response_source'] = source
            return parts
        parts = {"prompt": self.user_prompt, "response": self.agent_response,
                "reasoning": getattr(self, "reasoning", ""),
                "tool_calls": [call.to_record() for call in self.tool_calls],
                "tool_results": [{"call_id": result.tool_call_id, "content": result.content} for result in self.tool_results]}
        source = response_source(self.messages)
        if source:
            parts['response_source'] = source
        failure = getattr(self, "request_failure", None)
        if failure is not None:
            parts.update(request_failure=failure, response=self.agent_response or None)
            if is_incomplete(failure):
                parts["incomplete"] = True
        if getattr(self, "user_edits", None):
            parts["user_edits"] = self.user_edits
        if getattr(self, "background_commands", None):
            parts["background_commands"] = self.background_commands
        return parts

    def compaction_input(self) -> dict[str, Any]:
        parts = self.parts(stream=True)
        source = {"user_prompt": parts["prompt"],
                "agent_response": parts["response"] or "", "tool_calls": parts["tool_calls"],
                "tool_results": parts["tool_results"], "reasoning": parts["reasoning"]}
        if "request_failure" in parts:
            source["request_failure"] = parts["request_failure"]
            if parts.get("incomplete"):
                source["incomplete"] = True
        if "user_edits" in parts:
            source["user_edits"] = parts["user_edits"]
        if "background_commands" in parts:
            source["background_commands"] = parts["background_commands"]
        if 'response_source' in parts:
            source['response_source'] = parts['response_source']
        return source

    def record(self) -> dict[str, Any]:
        return {"step": self.id, **self.parts()}

    def compressed_text(self) -> str:
        if self.summary is not None:
            return self.summary
        return load_prompt("summary-unavailable.md", status=self.compaction_status)

    def context_messages(self, *, compressed: bool = False) -> list[Message]:
        if "_disk" in self.__dict__:
            original = self._disk["model"]
            source = response_source(self.messages)
            attribution = {'response_source': source} if source else {}
            if not compressed:
                return [record_message({**original, **attribution})]
            names = {call['call_id']: call['tool_name'] for call in original['tool_calls']}
            state = (self.summary, getattr(self, "reasoning_summary", None), self.compaction_status, self.compaction_error)
            if self.__dict__.get("_disk_summary_state") == state:
                return [record_message({**self._disk_summary["model"], **attribution}, tool_names=names)]
            record: dict[str, Any] = {"record_type": "history_step", "representation": "compressed",
                      "step_id": self.id, "compressed_summary": self.compressed_text(), **attribution}
            for name in ("artifacts", "user_edits", "background_commands", "request_failure", "incomplete"):
                if name in original:
                    record[name] = original[name]
            if "user_edits" in record:
                record["user_edits"] = [{key: value for key, value in edit.items() if key != "diff"}
                                         for edit in record["user_edits"]]
            if "request_failure" in record:
                record["request_failure"] = {key: value for key, value in record["request_failure"].items()
                                             if key != "response_excerpt"}
            return [record_message(record, tool_names=names)]
        part = super().context_messages(compressed=compressed)
        if getattr(self, "background_commands", None):
            part = [record_message({**part[0].content, "background_commands": self.background_commands})]
        edits = getattr(self, "user_edits", None)
        if edits:
            record = dict(part[0].content)
            record["user_edits"] = [{key: value for key, value in edit.items() if not compressed or key != "diff"}
                                    for edit in edits]
            record["artifacts"] = [*record.get("artifacts", []),
                                   *(edit["artifact"] for edit in edits if "artifact" in edit)]
            part = [record_message(record)]
        failure = getattr(self, "request_failure", None)
        if failure is not None:
            record = dict(part[0].content)
            record["request_failure"] = ({key: value for key, value in failure.items() if key != "response_excerpt"}
                                         if compressed else failure)
            if is_incomplete(failure):
                record["incomplete"] = True
            return [record_message(record)]
        return part

    def excerpt_context_messages(self, budget: int) -> list[Message] | None:
        metadata: dict[str, Any] = {}
        if getattr(self, "background_commands", None):
            metadata["background_commands"] = self.background_commands
        edits = getattr(self, "user_edits", None)
        if edits:
            metadata["user_edits"] = [{key: value for key, value in edit.items() if key != "diff"}
                                      for edit in edits]
            metadata["artifacts"] = [edit["artifact"] for edit in edits if "artifact" in edit]
        failure = getattr(self, "request_failure", None)
        if failure is not None:
            metadata["request_failure"] = {key: value for key, value in failure.items() if key != "response_excerpt"}
            if is_incomplete(failure):
                metadata["incomplete"] = True
        overhead = message_tokens([record_message(metadata)]) if metadata else 0
        part = super().excerpt_context_messages(budget - overhead)
        if part is not None and metadata:
            record = dict(part[0].content)
            references = [*record.get("artifacts", []), *metadata.pop("artifacts", [])]
            record.update(metadata)
            if references:
                record["artifacts"] = references
            part = [record_message(record)]
        return part


Summarizer = Callable[[str], Awaitable[str]]


@dataclass(slots=True)
class ConversationHistory:
    steps: list[HistoryStep] = field(default_factory=list)
    cursor: int = 0
    digest: str = ""
    digest_through: int = 0
    _request: str = ""
    task: TaskMemory = field(default_factory=TaskMemory)
    store: Any = None
    compact_through: int = 0

    def clear(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        self.steps.clear()
        self.cursor = 0
        self.digest = ""
        self.digest_through = 0
        self._request = ""
        self.task = TaskMemory()
        self.compact_through = 0

    def sync(self, messages: list[Message], *, on_step: Callable[[CompletedStep], None] | None = None) -> None:
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
            completed = CompletedStep(len(self.steps) + 1, self._request, list(segment))
            self.steps.append(completed)
            self.cursor = end
            if on_step is not None:
                on_step(completed)
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
        glm_xml: bool = False,
        schema: dict[str, Any] | None = None,
        token_scale: float = 1.0,
        overthinking: bool = True,
        reasoning_history_steps: int = 25,
        filtered_thoughts: bool = True,
        user_edits: list[dict[str, Any]] | None = None,
    ) -> list[Message]:
        """Select history and attach available analysis without altering originals.

        A newly returned batch must reach the working model before it can leave
        the full window, even if its independent summary already finished.
        Oversized new observations use explicit excerpts as a last resort.
        """
        self.sync(messages)
        from .requestfailures import retry_state
        recovery = retry_state(self.steps)
        has_new_user_input = any(message.role == "user" for message in messages[self.cursor:])
        # The step whose tool results this turn responds to: the newest step,
        # or the batch before a run of failed attempts.
        results_step = len(self.steps) - 1 - (recovery["consecutive_failures"] if recovery else 0)
        has_results = results_step >= 0 and self.steps[results_step].has_results
        continuing_tools = not has_new_user_input and has_results
        def tokens(part: list[Message]) -> int:
            return math.ceil(message_tokens(tool_history_as_text(part, glm_xml=glm_xml) if text_tool_history else part) * token_scale)
        replies = "native-tools.md" if native_tools else "response-glm-xml.md" if glm_xml else "json-tools.md"
        # Stable response/tool guidance precedes changing step IDs and task
        # state. Keep one system-message prefix for provider compatibility.
        later_sections = [extra_instructions, self.instructions(native_tools=native_tools)]
        tail_sections: list[str] = []
        if self.task.current_prompt_step is not None:
            input_guidance = (
                load_prompt('new-user-request.md')
                if has_new_user_input else
                load_prompt('tool-result-response.md' if continuing_tools else 'continuation.md')
            )
            tail_sections.append(
                load_prompt('user-request.md', step_id=self.task.current_prompt_step, input_guidance=input_guidance)
            )
        if recovery and not has_new_user_input:
            tail_sections.append(load_prompt('request-retry.md'))
        if self.compact_through:
            tail_sections.append(load_prompt('manual-compaction.md', step_id=self.compact_through))
        pinned_sources = [message for message in messages if message.role == "system"]
        leading = pinned_sources[:-1]
        pinned_base = (pinned_sources[-1].content or "").rstrip() + "\n" if pinned_sources else "\n"
        # Custom or absent system prompts still receive the complete working
        # rules, independently of the channel and selected history records.
        head_sections = [_response_instructions(replies, existing="\n".join(
            content_text(source.content) for source in pinned_sources)), load_prompt("history-records.md")]

        def system_prefix_for(guidance: str, thoughts: bool) -> str:
            # History-shape guidance follows the record format and covers only the selected records.
            # The model learns about Overthinking Mode only when thoughts accompany its history.
            overthinking_guidance = load_prompt('overthinking-on.md', steps=reasoning_history_steps) if thoughts else ""
            sections = [*head_sections, guidance, load_prompt("system-history.md"), overthinking_guidance,
                        *later_sections, *tail_sections]
            instructions = "\n" + "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"
            return (pinned_base + instructions).rstrip() + "\n\n"

        # Budget with every guidance section; each assembled candidate includes only applicable ones.
        system_prefix = system_prefix_for(HistoryText().history_guidance(None), overthinking)
        pinned = [*leading, Message.system(system_prefix.rstrip() + "\n")]
        from .historymerge import next_run, run_numbers
        runs = run_numbers(self.steps)
        metadata: dict[str, Any] = {"step_id": len(self.steps) + 1, "run": next_run(self.steps, runs, has_new_user_input),
                                    "is_tool_result_response": has_results}
        if self.compact_through:
            metadata["compact_through"] = self.compact_through
        if recovery:
            metadata["request_retry"] = recovery
        edit_guidance = load_prompt('user-edits.md') if user_edits or any(getattr(step, 'user_edits', None) for step in self.steps) else ''
        current_turn_guidance = load_prompt("current-turn.md")
        if metadata["is_tool_result_response"]:
            current_turn_guidance += load_prompt("current-turn-tool-results.md")
        # Budget the optional guidance conservatively before history selection.
        pinned[-1] = Message.system(MessageSections([system_prefix + edit_guidance, current_turn_guidance, metadata]))

        edit_view: list[dict[str, Any]] = []
        edits_omitted = len(user_edits or [])

        def assemble(selected: list[Message]) -> list[Message]:
            # Only the request projection changes; archives retain their roles.
            user_text = [message.content for message in tail_messages if message.role == "user"]
            source_step = self.task.current_prompt_step
            payload: dict[str, Any] = {
                "user_prompts": user_text,
                # Every continuation carries the active request here; history
                # records list only the messages actually sent in their step.
                "retained_user_prompts": self.task.sources[source_step] if source_step is not None and not has_new_user_input else [],
                "history": [linked_tool_results(message.content) for message in selected],
                "errors": [user_corrections] if user_corrections.strip() else [],
            }
            if edit_view:
                payload["user_edits"] = edit_view
            if edits_omitted:
                payload["user_edits_omitted"] = edits_omitted
            has_edits = bool(edit_view or edits_omitted or any(
                isinstance(message.content, dict) and message.content.get("user_edits") for message in selected))
            prefix = system_prefix_for(
                HistoryText().history_guidance([message.content for message in selected]),
                any(isinstance(message.content, dict) and message.content.get("reasoning") for message in selected))
            current_system = Message.system(MessageSections([
                prefix + (edit_guidance if has_edits else ''), current_turn_guidance, metadata]))
            return [*leading, current_system, Message.user(JSONInput(payload))]

        tail_messages = [message for message in messages[self.cursor:] if message.role != "system"]
        # Budget the same input fields that will be sent, including the retained
        # request and corrections even when no new user message was supplied.
        overhead = tokens(pinned)
        if not text_tool_history:
            overhead += math.ceil(estimate_tokens(json.dumps([spec.to_api() for spec in specs])) * token_scale)
        if schema is not None:
            overhead += math.ceil(estimate_tokens(json.dumps(schema)) * token_scale)
        available = int(context_length * 0.85) - max_output - overhead
        # History adds bytes to one user message. Apply its rounding/message
        # allowance once, after reserving all request-wide instructions and tools.
        from .requestbody import wire_message
        from .sources import json_chunks
        available = (math.floor(available / token_scale) - 32) * 3
        reserve = math.ceil(256 * 3 / token_scale)

        def input_bytes() -> int:
            return sum(len(chunk.encode('utf-8')) for chunk in json_chunks(wire_message(assemble([])[-1]), indent=None))

        tail_bytes = input_bytes()
        if tail_bytes + reserve >= available:
            raise ContextError("Current input and instructions exceed the context budget. Shorten the input or reference a file; conversation history is preserved.")
        if user_edits:
            from .editdelivery import bounded_notifications
            edit_budget = min(2048, max(256, int(available * token_scale / 3 * .2)),
                              math.floor((available - tail_bytes - reserve) * token_scale / 3))
            edit_view, edits_omitted = bounded_notifications(user_edits, edit_budget,
                lambda edits, omitted: tokens([Message.user(RenderedJSON({"user_edits": edits, "user_edits_omitted": omitted}))]))
            tail_bytes = input_bytes()
        def context_part(index: int, *, compressed: bool = False) -> list[Message]:
            step = self.steps[index]
            part = step.context_messages(compressed=compressed)
            part = [record_message({**part[0].content, "run": runs[index]})]
            if overthinking and index >= len(self.steps) - reasoning_history_steps:
                reasoning = (step._disk["parts"].get("reasoning") if isinstance(step, CompletedStep) and "_disk" in step.__dict__
                             else next((message.reasoning for message in step.messages if message.role == "assistant"), None))
                filtered = getattr(step, "reasoning_summary", None) if filtered_thoughts else None
                if filtered is not None:
                    reasoning = filtered
                elif compressed and step.id <= self.compact_through:
                    reasoning = None
                if reasoning:
                    record = dict(part[0].content)
                    record["reasoning"] = reasoning
                    record["reasoning_representation"] = "filtered" if filtered is not None else "original"
                    part = [record_message(record)]
            return part

        pending = bool(self.steps and (self.steps[-1].has_results or recovery is not None) and not (
            self.steps[-1].observed if isinstance(self.steps[-1], CompletedStep) else self.steps[-1].results_summarized))
        # A continuation still needs its active user prompt even at tiny budgets.
        required_last = pending or bool(self.steps and not tail_messages)
        unseen_results: int | None = None
        if recovery:
            previous = len(self.steps) - recovery["consecutive_failures"] - 1
            if previous >= 0:
                batch = self.steps[previous]
                if batch.has_results and not (batch.observed if isinstance(batch, CompletedStep) else batch.results_summarized):
                    unseen_results = previous
        required = {len(self.steps) - 1} if required_last else set()
        if unseen_results is not None:
            required.add(unseen_results)
        from .historymerge import aeon_records, era_records, historical_records
        parts: dict[tuple[int, bool], list[Message]] = {}
        costs: dict[tuple[int, bool], HistorySize] = {}
        sizing = HistorySizeIndex(TokenEstimates().history_bytes)
        indexes: dict[int, HistorySizeIndex] = {}

        def part_cost(index: int, compressed: bool, part: list[Message]) -> HistorySize:
            step = self.steps[index]
            disk = step.__dict__.get("_disk") if isinstance(step, CompletedStep) else None
            attached = overthinking and index >= len(self.steps) - reasoning_history_steps
            state = (step.summary, getattr(step, "reasoning_summary", None),
                     getattr(step, "compaction_status", None), getattr(step, "compaction_error", None))
            if index not in indexes:
                previous = step.__dict__.get('_size_index') if disk is not None else None
                if disk is not None and previous is not None and previous[0] is disk and previous[1] == state:
                    indexes[index] = previous[2]
                else:
                    indexes[index] = HistorySizeIndex(TokenEstimates().history_bytes)
                    if disk is not None:
                        step.__dict__['_size_index'] = (disk, state, indexes[index])
            key = (compressed, runs[index], attached, filtered_thoughts, step.id <= self.compact_through, 'artifact_metadata')
            return indexes[index].get(key, part[0].content)

        def narrative_cost(record: dict[str, Any], cache: str = "_narrative_size") -> HistorySize:
            anchor = self.steps[record["step_id"] - 1]
            entry: tuple[Any, HistorySize] | None = (
                anchor.__dict__.get(cache) if isinstance(anchor, CompletedStep) else None)
            if entry is not None and entry[0] is record:
                size = entry[1]
            else:
                size = sizing.measure_record(record)
                if isinstance(anchor, CompletedStep):
                    anchor.__dict__[cache] = (record, size)
            indexes[record["step_id"] - 1].remember((cache,), size)
            return size

        def supplied(index: int, compressed: bool) -> tuple[list[Message], HistorySize]:
            key = (index, compressed)
            if key not in parts:
                if compressed:
                    original, original_cost = supplied(index, False)
                else:
                    original = context_part(index)
                    original_cost = part_cost(index, False, original)
                chosen, cost = original, original_cost
                # An unavailable summary is not memory. Retain the original
                # rather than replacing evidence with a failure notice.
                if compressed and self.steps[index].summary is not None:
                    compact = context_part(index, compressed=True)
                    compact_cost = part_cost(index, True, compact)
                    if sizing.record_bytes(compact_cost) < sizing.record_bytes(original_cost):
                        chosen, cost = compact, compact_cost
                parts[key], costs[key] = chosen, cost
            return parts[key], costs[key]

        # Selection starts from full originals on every request and compresses
        # only what the budget requires, oldest turns first; the newest five
        # remain the last resort. keep_steps is compatibility only.
        count = len(self.steps)
        recent_start = max(0, count - RECENT_FULL_STEPS)
        FULL, STEP, NARRATIVE, ERA, AEON, PARTIAL, DROPPED = range(7)
        level = [FULL] * count
        unit_records: dict[int, dict[str, Any]] = {}
        unit_costs = [supplied(index, False)[1] for index in range(count)]
        record_costs: dict[int, HistorySize] = {}
        aggregate_sources: dict[tuple[int, int], dict[str, Any]] = {}
        partials: dict[int, list[Message]] = {}
        partial_compressed: set[int] = set()
        removed: list[tuple[list[int], int, HistorySize]] = []
        content_bytes = sum(cost.content_bytes for cost in unit_costs)
        block_count = sum(cost.blocks for cost in unit_costs)
        record_count = count

        def change_cost(before: list[HistorySize], after: list[HistorySize]) -> None:
            nonlocal content_bytes, block_count, record_count
            content_bytes += sum(cost.content_bytes for cost in after) - sum(cost.content_bytes for cost in before)
            block_count += sum(cost.blocks for cost in after) - sum(cost.blocks for cost in before)
            record_count += len(after) - len(before)

        def total_bytes(extra: list[HistorySize] | None = None) -> int:
            addition = extra or []
            return tail_bytes + sizing.total_bytes(
                content_bytes + sum(cost.content_bytes for cost in addition),
                block_count + sum(cost.blocks for cost in addition), record_count + len(addition))

        def finish(selected: list[Message]) -> list[Message]:
            if self.store is not None:
                for index, entry in indexes.items():
                    if entry.dirty:
                        self.store.write(f'{self.steps[index].id:08d}.sizes.json', entry.snapshot())
                        entry.dirty = False
            return assemble(selected)

        def compressible(index: int) -> bool:
            step = self.steps[index]
            unseen = (step.id <= self.compact_through and step.has_results
                      and isinstance(step, CompletedStep) and not step.observed)
            return not unseen and index != unseen_results and not (pending and index == count - 1)

        def summarize_step(index: int) -> bool:
            if level[index] != FULL or not compressible(index):
                return False
            cost = supplied(index, True)[1]
            if sizing.record_bytes(cost) >= sizing.record_bytes(unit_costs[index]):
                return False  # Missing or larger summaries keep the original evidence.
            change_cost([unit_costs[index]], [cost])
            level[index], unit_costs[index] = STEP, cost
            return True

        def merge_into(record: dict[str, Any], form: int, cache: str, cap: int) -> bool:
            # One record replaces every unit that covers its source steps.
            members = [step_id - 1 for step_id in record["step_ids"]]
            if any(index >= cap or not compressible(index) or level[index] >= form for index in members):
                return False
            current: dict[int, HistorySize] = {}
            for index in members:
                if level[index] in (NARRATIVE, ERA, AEON):
                    covering = unit_records[index]
                    current[-id(covering)] = record_costs[id(covering)]
                else:
                    current[index] = unit_costs[index]
            cost = narrative_cost(record, cache)
            old = list(current.values())
            if sizing.record_bytes(cost) >= sizing.total_bytes(
                    sum(item.content_bytes for item in old), sum(item.blocks for item in old), len(old)):
                return False
            change_cost(old, [cost])
            for index in members:
                level[index] = form
                unit_records[index] = record
            record_costs[id(record)] = cost
            aggregate_sources[(form, record['step_id'])] = record
            return True

        def drop_step(index: int) -> bool:
            if index in required or level[index] == DROPPED:
                return False
            if level[index] in (NARRATIVE, ERA, AEON):
                record = unit_records[index]
                members = [member for member in range(count)
                           if level[member] == level[index] and unit_records.get(member) is record]
                cost = record_costs[id(record)]
            else:
                members = [index]
                cost = unit_costs[index]
            removed.append((members, level[index], cost))
            change_cost([cost], [])
            for member in members:
                level[member] = DROPPED
            return True

        def excerpt_required(index: int) -> bool:
            target = max(0, sizing.record_bytes(unit_costs[index]) - max(1, total_bytes() - available))
            part = self.steps[index].excerpt_context_messages(max(0, target // 3 - 32))
            if part is None:
                return False
            record = {**part[0].content, "run": runs[index]}
            cost = sizing.measure_record(record)
            if sizing.record_bytes(cost) >= sizing.record_bytes(unit_costs[index]):
                return False
            change_cost([unit_costs[index]], [cost])
            level[index], unit_costs[index] = PARTIAL, cost
            partials[index] = [record_message(record)]
            return True

        async def excerpt_step(index: int, *, results_only: bool, budget: int | None = None) -> bool:
            if level[index] == DROPPED or level[index] == PARTIAL and results_only:
                return False
            from .historyexcerpts import excerpt_fields
            from .tools.blocking import run_blocking
            # Reduce the selected form: compressed records can still carry
            # oversized reasoning. Unseen observations keep their original form.
            grouped = level[index] in (NARRATIVE, ERA, AEON)
            compressed = level[index] == STEP or index in partial_compressed
            if grouped:
                current = unit_records[index]
                original = aggregate_sources[(level[index], current['step_id'])]
                previous_cost = record_costs[id(current)]
            else:
                original = supplied(index, compressed)[0][0].content
                previous_cost = unit_costs[index]
            target = (max(0, sizing.record_bytes(previous_cost) - max(1, total_bytes() - available))
                      if budget is None else budget)
            def measure(record: dict[str, Any]) -> int:
                return sizing.record_bytes(sizing.measure_record(record))
            # Keep a fitting newest batch ahead of its own optional prose.
            without_prose = {key: value for key, value in original.items()
                             if key not in ('reasoning', 'agent_response')}
            prefer_results = index == count - 1 and measure(without_prose) <= target
            from .requestbody import value_bytes
            reasoning_bytes = value_bytes(original.get('reasoning'))
            largest_result = max(previous_cost.result_bytes, default=0)
            # Keep recent evidence ahead of thoughts. An older result that
            # cannot fit by itself may yield before much shorter reasoning.
            oversized_old_result = (index < recent_start
                                    and largest_result > max(target, 4 * reasoning_bytes))
            reasoning_first = not oversized_old_result
            if results_only and (prefer_results or reasoning_first):
                return False
            clipped = await run_blocking(excerpt_fields, original, target, measure,
                                         results_only=results_only, prefer_results=prefer_results,
                                         reasoning_first=reasoning_first)
            if clipped is None:
                if measure(original) >= sizing.record_bytes(previous_cost):
                    return False
                clipped = original
            cost = sizing.measure_record(clipped)
            if sizing.record_bytes(cost) >= sizing.record_bytes(previous_cost):
                return False
            change_cost([previous_cost], [cost])
            if grouped:
                for member in range(count):
                    if unit_records.get(member) is current:
                        unit_records[member] = clipped
                record_costs[id(clipped)] = cost
            else:
                if compressed:
                    partial_compressed.add(index)
                level[index], unit_costs[index] = PARTIAL, cost
                partials[index] = [record_message(clipped)]
            return True

        async def reductions() -> AsyncIterator[None]:
            # Walk outward from the oldest turn. Each turn first gives up its
            # original for its own summary; once a narrative, era or aeon is
            # complete at that turn, it is tried, finest first, before any newer
            # turn is reduced. A small shortfall therefore costs only the oldest
            # turns their originals, and a larger one coarsens the old end level
            # by level while recent turns keep theirs.
            aggregates: dict[int, list[tuple[dict[str, Any], int, str]]] = {}
            for records, form, cache in ((historical_records(self.steps), NARRATIVE, '_narrative_size'),
                                         (era_records(self.steps), ERA, '_era_size'),
                                         (aeon_records(self.steps), AEON, '_aeon_size')):
                for record in records:
                    aggregates.setdefault(record['step_id'] - 1, []).append((record, form, cache))
            for index in range(recent_start):
                if summarize_step(index):
                    yield
                elif (level[index] == FULL and compressible(index)
                      and self.steps[index].summary is None):
                    # A failed gap has no AI memory. Keep labelled evidence,
                    # rather than letting its original displace newer history.
                    target = max(FAILED_SUMMARY_EXCERPT_BYTES,
                                 sizing.record_bytes(unit_costs[index]) - max(1, total_bytes() - available))
                    if await excerpt_step(index, results_only=False, budget=target):
                        yield
                for record, form, cache in aggregates.get(index, []):
                    if merge_into(record, form, cache, index + 1):
                        yield
            def unit_bytes(index: int) -> int:
                if level[index] in (NARRATIVE, ERA, AEON):
                    return sizing.record_bytes(record_costs[id(unit_records[index])])
                return sizing.record_bytes(unit_costs[index])

            # Exhaust each older turn before touching a newer one. The newest
            # five stay original until reductions outside that window cannot fit.
            # A unit that cannot close the deficit on its own is dropped intact,
            # so backfill can restore it whole rather than as a stub.
            for index in range(count):
                if level[index] == DROPPED or index in required:
                    continue
                if index >= recent_start and summarize_step(index):
                    yield
                if total_bytes() - available < unit_bytes(index):
                    for results_only in (True, False):
                        if await excerpt_step(index, results_only=results_only):
                            yield
                if drop_step(index):
                    yield
            # Required observations and the latest failure share the same
            # selection and backfill accounting; they can shrink but cannot drop.
            changed = True
            while changed:
                changed = False
                for index in sorted(required):
                    if summarize_step(index):
                        changed = True
                        yield
                    for results_only in (True, False):
                        if await excerpt_step(index, results_only=results_only):
                            changed = True
                            yield
                    if excerpt_required(index):
                        changed = True
                        yield

        def selection() -> list[Message]:
            entries: list[tuple[int, list[Message]]] = []
            emitted: set[int] = set()
            for index in range(count):
                if level[index] in (FULL, STEP):
                    entries.append((self.steps[index].id, supplied(index, level[index] == STEP)[0]))
                elif level[index] == PARTIAL:
                    entries.append((self.steps[index].id, partials[index]))
                elif level[index] in (NARRATIVE, ERA, AEON):
                    record = unit_records[index]
                    if record["step_id"] not in emitted:
                        emitted.add(record["step_id"])
                        entries.append((record["step_id"], [record_message(record)]))
            entries.sort(key=lambda entry: entry[0])
            return [message for _, part in entries for message in part]

        def backfill() -> list[Message]:
            # Newest first, each in the best form that fits: a clipped unit's
            # intact summary, original or aggregate comes back before its excerpt.
            for members, form, cost in reversed(removed):
                options: list[tuple[int, HistorySize, dict[str, Any] | None]] = [(form, cost, None)]
                if form == PARTIAL and len(members) == 1:
                    compressed = members[0] in partial_compressed
                    options.insert(0, (STEP if compressed else FULL, supplied(members[0], compressed)[1], None))
                elif form in (NARRATIVE, ERA, AEON):
                    current = unit_records[members[0]]
                    original = aggregate_sources[(form, current['step_id'])]
                    if original is not current:
                        options.insert(0, (form, record_costs[id(original)], original))
                for candidate, candidate_cost, record in options:
                    if total_bytes([candidate_cost]) <= available:
                        for member in members:
                            level[member] = candidate
                            if record is not None:
                                unit_records[member] = record
                        change_cost([], [candidate_cost])
                        break
            return finish(selection())

        # A manual boundary chooses economical compressed forms even when the
        # full history fits. Later steps still follow normal budget selection.
        boundary = min(self.compact_through, count)
        if boundary:
            for record in aeon_records(self.steps):
                merge_into(record, AEON, '_aeon_size', boundary)
            for record in era_records(self.steps):
                merge_into(record, ERA, "_era_size", boundary)
            for record in historical_records(self.steps):
                merge_into(record, NARRATIVE, "_narrative_size", boundary)
            for index in range(boundary):
                summarize_step(index)

        if total_bytes() <= available:
            return finish(selection())
        async for _ in reductions():
            if total_bytes() <= available:
                return backfill()
        raise ContextError("Current input and latest tool results exceed the context budget even without older history; originals and summaries are preserved.")


class RecallHistoryTool(Tool):
    name = "recall_history"
    parameter_prompts = "tools/recall-history-parameters.json"
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
        },
    }

    def __init__(self, history: ConversationHistory) -> None:
        self.history = history

    async def run(self, *, step_id: int | None = None, call_id: str | None = None,
                  query: str = "", section: str | None = None,
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
            if step_id < 1 or step_id > len(self.history.steps):
                return ToolResult.error(f"No history step {step_id}. Available steps: 1–{len(self.history.steps)}.")
            step = self.history.steps[step_id - 1]
            if section == "all" and sections is None and call_id is None:
                structured = step.record()
            if section == "user":
                structured = [message.content or "" for message in self.history.steps[step_id - 1].messages
                              if message.role == "user"]
            if section in HISTORY_PART_NAMES or sections is not None:
                original = step if isinstance(step, CompletedStep) else CompletedStep(step.id, step.request, step.messages)
                parts = original.parts()
                parts.setdefault("user_edits", [])
                parts.setdefault("background_commands", [])
                if call_id is not None:
                    key = "id" if section == "tool_calls" else "call_id"
                    parts[section] = [value for value in parts[section] if value[key] == call_id]
                    if not parts[section]:
                        return ToolResult.error(f"No {section} with call_id {call_id!r} in step {step_id}.")
                value = {name: parts[name] for name in sections} if sections is not None else parts[section]
                structured = value
            elif call_id is not None:
                observation = next((message for message in self.history.steps[step_id - 1].messages
                                    if message.role == "tool" and message.tool_call_id == call_id), None)
                if observation is None:
                    return ToolResult.error(f"No tool result with call_id {call_id!r} in step {step_id}.")
                structured = observation.content
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
        result: dict[str, Any] = {
            "step_id": step_id,
            "section": section,
            "sections": sections,
            "content": structured if step_id is not None else text,
        }
        if step_id is None:
            result["total_matches"] = len(matches)
        return ToolResult.ok(result)
