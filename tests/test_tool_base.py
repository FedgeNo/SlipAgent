"""Argument validation, error-to-result conversion, and registry dispatch."""

from __future__ import annotations

from typing import Any

import pytest

from slipagent.tools.base import Tool, ToolRegistry, ToolResult, validate_arguments
from slipagent.tools.shell import RunCommandTool
from slipagent.types import ToolCall
from slipagent.workspace import Workspace


class EchoTool(Tool):
    name = "echo"
    description = "Echo the arguments back."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "times": {"type": "integer", "minimum": 1},
            "mode": {"type": "string", "enum": ["fast", "slow"]},
            "flag": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["text"],
    }

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> ToolResult:
        self.seen.append(kwargs)
        return ToolResult.ok(f"echo: {kwargs}")


class BoomTool(Tool):
    name = "boom"
    description = "Always explodes."
    parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("kaboom")


class TypeErrorTool(Tool):
    name = "typeerror"
    description = "Raises TypeError from inside its own body."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def run(self, text: str) -> ToolResult:
        return ToolResult.ok(str(len(None)))


class NarrowTool(Tool):
    name = "narrow"
    description = "Takes one fixed keyword and no extras."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
    strict_arguments = False

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def run(self, text: str = "") -> ToolResult:
        self.seen.append({"text": text})
        return ToolResult.ok(f"narrow: {text}")


# --------------------------------------------------------------------------- #
# validate_arguments
# --------------------------------------------------------------------------- #


def test_validates_accepts_good_input() -> None:
    schema = EchoTool.parameters
    assert validate_arguments(schema, {"text": "a", "times": 2}) is None


def test_validates_rejects_missing_required() -> None:
    problem = validate_arguments(EchoTool.parameters, {})
    assert problem is not None and "missing required argument" in problem


def test_validates_rejects_unknown_argument() -> None:
    problem = validate_arguments(EchoTool.parameters, {"text": "a", "nope": 1})
    assert problem is not None and "unexpected argument" in problem


def test_validates_rejects_wrong_type() -> None:
    problem = validate_arguments(EchoTool.parameters, {"text": 5})
    assert problem is not None and "must be string" in problem


def test_validates_rejects_bool_as_integer() -> None:
    problem = validate_arguments(EchoTool.parameters, {"text": "a", "times": True})
    assert problem is not None and "got boolean" in problem


def test_validates_enforces_enum() -> None:
    problem = validate_arguments(EchoTool.parameters, {"text": "a", "mode": "sideways"})
    assert problem is not None and "must be one of" in problem


def test_validates_enforces_minimum() -> None:
    problem = validate_arguments(EchoTool.parameters, {"text": "a", "times": 0})
    assert problem is not None and "must be >= 1" in problem


def test_validates_checks_array_items() -> None:
    problem = validate_arguments(EchoTool.parameters, {"text": "a", "tags": [1]})
    assert problem is not None and "tags[0]" in problem


# --------------------------------------------------------------------------- #
# Tool.invoke
# --------------------------------------------------------------------------- #


async def test_invoke_accepts_json_string() -> None:
    tool = EchoTool()
    result = await tool.invoke('{"text": "hi"}')

    assert not result.is_error
    assert tool.seen == [{"text": "hi"}]


async def test_invoke_accepts_dict() -> None:
    tool = EchoTool()
    result = await tool.invoke({"text": "hi"})

    assert not result.is_error


async def test_invoke_treats_empty_string_as_no_arguments() -> None:
    tool = EchoTool()
    result = await tool.invoke("")

    # Missing the required `text`, reported as a validation error rather than
    # a crash.
    assert result.is_error
    assert "missing required argument" in result.content


async def test_invoke_reports_invalid_json() -> None:
    result = await EchoTool().invoke("{not json")

    assert result.is_error
    assert "Invalid JSON" in result.content


async def test_invoke_reports_non_object_json() -> None:
    result = await EchoTool().invoke("[1, 2, 3]")

    assert result.is_error
    assert "expected a JSON object" in result.content


async def test_invoke_converts_exception_to_error_result() -> None:
    result = await BoomTool().invoke({})

    assert result.is_error
    assert "RuntimeError: kaboom" in result.content


async def test_invoke_reports_a_type_error_from_inside_run_as_a_tool_fault() -> None:
    """A TypeError from the tool's own body is a tool bug, not a bad call.

    Reporting it as "invalid arguments" would send the model off to correct
    arguments that are already correct, and it would retry forever.
    """
    result = await TypeErrorTool().invoke({"text": "hi"})

    assert result.is_error
    assert "TypeError:" in result.content
    assert "Invalid arguments" not in result.content


async def test_invoke_reports_arguments_that_cannot_bind_to_run() -> None:
    """A permissive tool still rejects arguments its signature cannot accept."""
    tool = NarrowTool()

    result = await tool.invoke({"not_a_parameter": 1})

    assert result.is_error
    assert "Invalid arguments" in result.content
    assert tool.seen == []


async def test_invoke_runs_a_permissive_tool_with_extra_arguments() -> None:
    """`**kwargs` tools keep accepting unknown keys."""
    tool = EchoTool()
    tool.strict_arguments = False
    tool.parameters = {"type": "object", "additionalProperties": True}

    result = await tool.invoke({"text": "hi", "extra": 2})

    assert not result.is_error
    assert tool.seen == [{"text": "hi", "extra": 2}]


async def test_unparseable_arguments_become_a_tool_error() -> None:
    """A model emitting malformed JSON arguments should get a usable error."""
    registry = ToolRegistry([EchoTool()])
    raw_call = {"id": "c1", "function": {"name": "echo", "arguments": "{oops"}}

    parsed = ToolCall.from_api(raw_call)
    assert parsed.arguments == {"__invalid_arguments__": "{oops"}

    result = await registry.invoke(parsed.name, parsed.arguments)
    assert result.is_error


# --------------------------------------------------------------------------- #
# ToolRegistry
# --------------------------------------------------------------------------- #


async def test_registry_lists_names_and_specs() -> None:
    registry = ToolRegistry([EchoTool(), BoomTool()])

    assert registry.names == ["boom", "echo"]
    assert len(registry.specs()) == 2
    assert registry.get("echo") is not None
    assert "echo" in registry
    assert len(registry) == 2


async def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


async def test_registry_reports_unknown_tool() -> None:
    registry = ToolRegistry([EchoTool()])
    result = await registry.invoke("nope", {})

    assert result.is_error
    assert "Unknown tool 'nope'" in result.content
    assert "echo" in result.content


# --------------------------------------------------------------------------- #
# run_command
# --------------------------------------------------------------------------- #


async def test_command_runs_in_workspace_root(project: Workspace) -> None:
    result = await RunCommandTool(project).invoke({"command": "pwd"})

    assert not result.is_error
    assert str(project.root) in result.content
    assert "exit code: 0" in result.content


async def test_command_failure_is_an_error_result(project: Workspace) -> None:
    result = await RunCommandTool(project).invoke({"command": "exit 3"})

    assert result.is_error
    assert "exit code: 3" in result.content


async def test_command_captures_stdout_and_stderr(project: Workspace) -> None:
    result = await RunCommandTool(project).invoke(
        {"command": "echo out; echo err >&2"}
    )

    assert "--- stdout ---\nout" in result.content
    assert "--- stderr ---\nerr" in result.content


async def test_command_can_write_inside_workspace(project: Workspace) -> None:
    result = await RunCommandTool(project).invoke(
        {"command": "echo hi > made.txt"}
    )

    assert not result.is_error
    assert (project.root / "made.txt").read_text().strip() == "hi"


async def test_command_timeout_is_reported(project: Workspace) -> None:
    result = await RunCommandTool(project).invoke(
        {"command": "sleep 5", "timeout": 0.2}
    )

    assert result.is_error
    assert "timed out" in result.content


async def test_timeout_keeps_partial_stdout_and_stderr(project):
    result = await RunCommandTool(project).invoke({
        "command": "echo reached-checkpoint; echo failed-checkpoint >&2; sleep 5", "timeout": .1,
    })
    assert result.is_error and "timed out" in result.content
    assert "reached-checkpoint" in result.content and "failed-checkpoint" in result.content
    assert "partial" in result.content


async def test_command_rejects_blank(project: Workspace) -> None:
    result = await RunCommandTool(project).invoke({"command": "   "})

    assert result.is_error
    assert "must not be empty" in result.content


async def test_command_output_is_truncated(project: Workspace) -> None:
    result = await RunCommandTool(project).invoke(
        {"command": "python3 -c \"print('x' * 100000)\""}
    )

    assert "[output truncated at 30000 chars]" in result.content


async def test_large_command_preserves_beginning_and_final_diagnostic(workspace):
    import shlex
    import sys
    code = "print('BEGINNING'); print('x' * 100000); print('FINAL DIAGNOSTIC')"
    result = await RunCommandTool(workspace).invoke({"command": shlex.join([sys.executable, "-c", code])})
    stdout = result.content.split("--- stdout ---\n", 1)[1]
    assert "BEGINNING" in stdout and "FINAL DIAGNOSTIC" in stdout
    assert "middle not archived" in stdout


async def test_timeout_kills_descendants(workspace: Workspace) -> None:
    import asyncio
    import shlex
    import sys

    code = "import time; from pathlib import Path; time.sleep(.6); Path('escaped.txt').touch()"
    result = await asyncio.wait_for(RunCommandTool(workspace).invoke({
        "command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} & wait",
        "timeout": .1,
    }), timeout=2)
    await asyncio.sleep(.7)
    assert result.is_error
    assert not (workspace.root / "escaped.txt").exists()


async def test_cancellation_kills_command(workspace: Workspace) -> None:
    import asyncio
    import shlex
    import sys

    code = "import time; from pathlib import Path; time.sleep(.6); Path('cancelled.txt').touch()"
    task = asyncio.create_task(RunCommandTool(workspace).invoke({
        "command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    }))
    await asyncio.sleep(.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(.7)
    assert not (workspace.root / "cancelled.txt").exists()


def test_schema_allows_additional_properties_when_requested() -> None:
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert validate_arguments(schema, {"count": 3}) is None
    assert validate_arguments(schema, {"count": "wrong"}) is not None


def test_schema_validates_nested_objects() -> None:
    schema = {"properties": {"options": {"type": "object", "properties": {
        "count": {"type": "integer"}}, "required": ["count"]}}}
    assert validate_arguments(schema, {"options": {}}) is not None
    assert validate_arguments(schema, {"options": {"count": "wrong"}}) is not None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_schema_rejects_nonfinite_numbers(value: float) -> None:
    assert validate_arguments({"properties": {"n": {"type": "number"}}}, {"n": value})


def test_schema_supports_nullable_values_and_maximum() -> None:
    schema = {"properties": {"n": {"type": ["integer", "null"], "maximum": 3}}}
    assert validate_arguments(schema, {"n": None}) is None
    assert validate_arguments(schema, {"n": "wrong"}) is not None
    assert validate_arguments(schema, {"n": 4}) is not None


async def test_registry_closes_remaining_tools_after_close_error() -> None:
    closed = []
    class ClosingTool(EchoTool):
        def __init__(self, name):
            self.name = name
        async def aclose(self):
            closed.append(self.name)
            if self.name == "a":
                raise RuntimeError("failed close")
    registry = ToolRegistry([ClosingTool("a"), ClosingTool("b")])
    with pytest.raises(RuntimeError, match="failed close"):
        await registry.aclose()
    assert closed == ["a", "b"]


async def test_invalid_argument_marker_never_runs_permissive_tool() -> None:
    tool = EchoTool()
    tool.parameters = {"type": "object", "additionalProperties": True}
    result = await tool.invoke({"__invalid_arguments__": "{broken"})
    assert result.is_error
    assert tool.seen == []
