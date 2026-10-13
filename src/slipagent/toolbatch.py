"""Execute one tool batch with ordered observations and explicit resource owners."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from .activity import command_output
from .batching import run_batch
from .checkpoints import FileCheckpoints, current_checkpoint
from .checks import check_edit_batch
from .tools.base import ToolRegistry, ToolResult, current_invocation, validate_arguments, _binds_arguments
from .types import Message, ToolCall

from .events import AgentEvent


class ToolBatch:
    def __init__(
        self, registry: ToolRegistry, step: int, messages: list[Message], *,
        emit: Callable[[AgentEvent], None], persist: Callable[[], None],
        archive_interrupted: Callable[[], Awaitable[None]],
        checkpoints: FileCheckpoints | None, allowed_tools: set[str] | None,
    ) -> None:
        self.registry = registry
        self.step = step
        self.messages = messages
        self.emit = emit
        self.persist = persist
        self.archive_interrupted = archive_interrupted
        self.checkpoints = checkpoints
        self.allowed_tools = allowed_tools
        self.results: list[tuple[ToolCall, ToolResult]] = []

    async def invoke_call(self, tool_call: ToolCall) -> ToolResult:
        journal = self.registry.resources.journal
        if journal is not None:
            journal.tool_started(self.step, tool_call.id)
        self.emit(AgentEvent(kind="tool_start", step=self.step, tool_call=tool_call))
        invocation_token = current_invocation.set((self.step, tool_call.id))
        output_token = command_output.set(
            lambda chunk: self.emit(AgentEvent(kind="tool_output", step=self.step, text=chunk, tool_call=tool_call))
        )
        try:
            result = await self.registry.invoke(tool_call.name, tool_call.arguments)
        except Exception as exc:
            result = ToolResult.error(f"Tool dispatch failed: {type(exc).__name__}: {exc}. Effects may be partial; inspect before retrying.")
        finally:
            command_output.reset(output_token)
            current_invocation.reset(invocation_token)
        return result

    def commit_call(self, tool_call: ToolCall, result: ToolResult) -> None:
        artifacts = self.registry.resources.artifacts
        references = artifacts.references(self.step, tool_call.id) if artifacts is not None else []
        observation = tool_message(tool_call, result)
        if references:
            observation.content["artifacts"] = references
        self.messages.append(observation)
        self.persist()
        self.results.append((tool_call, result))
        self.emit(
            AgentEvent(kind="tool_end", step=self.step, tool_call=tool_call, result=result)
        )
        for reference in references:
            self.emit(AgentEvent(kind="artifact", step=self.step, tool_call=tool_call,
                                  result=ToolResult.ok(reference)))

    async def run(self, calls: list[ToolCall]) -> list[tuple[ToolCall, ToolResult]]:
        unavailable = sorted({call.name for call in calls
                              if self.allowed_tools is not None and call.name not in self.allowed_tools})
        if unavailable:
            # Validate the whole response before dispatch so earlier calls cannot have effects.
            error = ("No tools were executed. The entire batch was rejected because these tools are not available "
                     f"in the current request: {', '.join(unavailable)}. "
                     "Only call tools whose full definitions are supplied in the current request. "
                     "For deferred catalog tools, call load_tools in a separate response and wait for the "
                     "next request before using them. Tools excluded by session restrictions cannot be loaded.")
            for call in calls:
                self.emit(AgentEvent(kind="tool_start", step=self.step, tool_call=call))
                self.commit_call(call, ToolResult.error(error))
            return self.results
        problems = []
        for call in calls:
            tool = self.registry.get(call.name)
            problem = (validate_arguments(tool.parameters, call.arguments, allow_unknown=not tool.strict_arguments)
                       or _binds_arguments(tool.run, call.arguments)) if tool is not None else "unknown tool"
            if problem:
                problems.append(f"{call.name}: {problem}")
        if problems:
            error = "No tools were executed. The entire batch was rejected because of invalid arguments: " + "; ".join(problems)
            for call in calls:
                self.emit(AgentEvent(kind="tool_start", step=self.step, tool_call=call))
                self.commit_call(call, ToolResult.error(error))
            return self.results
        checkpoints = self.checkpoints
        if checkpoints is not None:
            checkpoints.begin(self.step)
        checkpoint_token = current_checkpoint.set(checkpoints)
        try:
            await run_batch(calls, self.registry, self.invoke_call, self.commit_call)
        except BaseException:
            await self.archive_interrupted()
            raise
        finally:
            current_checkpoint.reset(checkpoint_token)
            if checkpoints is not None:
                checkpoints.finish()

        check_report = await check_edit_batch(self.registry, self.results, self.step)
        if check_report:
            call, result = self.results[-1]
            content = (result.content + "\n\nHarness Checks After This Complete Tool Batch:\n" + check_report
                       if isinstance(result.content, str) else {"result": result.content, "harness_checks": check_report})
            checked = ToolResult(content, result.is_error)
            artifacts = self.messages[-1].content.get("artifacts", [])
            self.messages[-1] = tool_message(call, checked)
            if artifacts:
                self.messages[-1].content["artifacts"] = artifacts
            self.results[-1] = call, checked
            visible_diagnostics = "\n".join(line for line in check_report.splitlines()
                                            if "SKIPPED — no selected project Python." not in line)
            if any(word in visible_diagnostics for word in ("FAILED", "TIMED OUT", "SKIPPED", "unavailable")):
                self.emit(AgentEvent(kind="warning", step=self.step, text=visible_diagnostics))
        return self.results


def tool_message(call: ToolCall, result: ToolResult) -> Message:
    """Keep execution status in the archived observation, not just the UI event."""
    return Message.tool_result(call.id, {
        "tool": call.name, "call_id": call.id,
        "status": "error" if result.is_error else "success", "content": result.content,
    })
