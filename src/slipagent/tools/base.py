"""Tool primitives: the `Tool` base class, results, and the registry.

Ordinary tool failures become results at the agent loop boundary. Bad arguments,
unknown tools, and unexpected exceptions are converted into an error result
carrying an explanatory message, so the model gets a chance to correct itself
instead of the run dying. Cancellation deliberately propagates so the owner
can stop subprocesses and finish the interrupted batch's history consistently.
"""

from __future__ import annotations

import inspect
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Any

from ..types import ToolSpec, content_text, decode_json_content
from ..lifecycle import Lifetime

# A call ID is unique only within one numbered history step. Context-local
# provenance also works when callers invoke independent tools concurrently.
current_invocation: ContextVar[tuple[int, str] | None] = ContextVar("slipagent_invocation", default=None)

_PYTHON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


@dataclass(slots=True)
class ToolResult:
    """What a tool hands back to the model."""

    content: Any
    is_error: bool = False

    def __post_init__(self) -> None:
        self.content = decode_json_content(self.content)

    def text(self) -> str:
        return content_text(self.content)

    @classmethod
    def ok(cls, content: Any) -> ToolResult:
        return cls(content=content)

    @classmethod
    def error(cls, content: Any) -> ToolResult:
        return cls(content=content, is_error=True)


class Tool(ABC):
    """Base class for every tool exposed to the model.

    Subclasses set `name`, `description`, and a JSON Schema `parameters` object
    as class attributes, then implement `run`. Deliberately not a dataclass:
    subclasses assign plain class attributes. A tool whose name is computed per
    instance (an MCP server's remote tools, for example) may instead assign
    these in `__init__`, which the non-ClassVar annotations allow.
    """

    name: str
    description: str
    description_prompt: str | None = None
    parameter_prompts: str | None = None
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    strict_arguments: bool = True
    concurrent_safe: bool = False
    instruction_path: str | None = None
    mutates_workspace: bool = False

    def allows_concurrency(self, arguments: dict[str, Any]) -> bool:
        """Opt in only when these arguments cannot mutate shared project state."""
        return self.concurrent_safe

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the tool. Raise only for genuinely unexpected faults.

        Subclasses narrow this to their own keyword signature; the `*args` on
        the base keeps that a valid override for the type checker.
        """

    @property
    def spec(self) -> ToolSpec:
        from ..prompts import load_prompt
        parameters = self.parameters
        if self.parameter_prompts:
            import copy
            parameters = copy.deepcopy(parameters)
            descriptions = json.loads(load_prompt(self.parameter_prompts))
            for path, description in descriptions.items():
                schema = parameters
                for part in path.split("/") if path else []:
                    schema = schema[part]
                schema["description"] = "\n" + description.strip() + "\n"
        return ToolSpec(
            name=self.name,
            description=load_prompt(self.description_prompt) if self.description_prompt else self.description,
            parameters=parameters,
        )

    async def invoke(self, raw_arguments: str | dict[str, Any] | None) -> ToolResult:
        """Parse, validate, and run; convert ordinary errors, propagate cancellation."""
        try:
            return await self._invoke(raw_arguments)
        except Exception as exc:
            return ToolResult.error(f"{type(exc).__name__}: {exc}")

    async def _invoke(self, raw_arguments: str | dict[str, Any] | None) -> ToolResult:
        if isinstance(raw_arguments, str):
            if not raw_arguments.strip():
                arguments: dict[str, Any] = {}
            else:
                try:
                    parsed = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    return ToolResult.error(
                        f"Invalid JSON for tool '{self.name}': {exc}. "
                        f"Arguments must be a JSON object."
                    )
                if not isinstance(parsed, dict):
                    return ToolResult.error(
                        f"Invalid arguments for tool '{self.name}': expected a JSON "
                        f"object, got {type(parsed).__name__}."
                    )
                arguments = parsed
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = {}

        if "__invalid_arguments__" in arguments:
            return ToolResult.error(f"Invalid JSON arguments for tool '{self.name}': expected a JSON object.")

        problem = validate_arguments(self.parameters, arguments, allow_unknown=not self.strict_arguments)
        if problem is not None:
            return ToolResult.error(f"Invalid arguments for tool '{self.name}': {problem}")

        problem = _binds_arguments(self.run, arguments)
        if problem is not None:
            return ToolResult.error(f"Invalid arguments for tool '{self.name}': {problem}")

        return await self.run(**arguments)


def _binds_arguments(run: Callable[..., Any], arguments: dict[str, Any]) -> str | None:
    """Report why `arguments` cannot bind to `run`, or None if they can.

    Arity is checked by inspecting the signature instead of by catching the
    TypeError a bad call raises. A TypeError raised *inside* `run` is a bug in
    the tool, not a mistake in the call, and reporting it as "invalid
    arguments" would send the model off to correct arguments that are already
    correct while the tool stays broken.
    """
    try:
        signature = inspect.signature(run)
        signature.bind(**arguments)
    except (TypeError, ValueError) as exc:
        # ValueError means the signature itself is uninspectable, which is a
        # property of the tool rather than a reason to reject the call.
        return None if isinstance(exc, ValueError) else str(exc)
    return None


def validate_schema_shape(schema: Any, path: str = "inputSchema") -> None:
    """Check schema structure before our argument validator consumes it.

    Remote servers remain responsible for full JSON Schema semantics ($ref,
    combinators, and extensions). Accept boolean subschemas and validate the
    standard nested shapes instead of silently replacing broken metadata.
    """
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must be a schema object or boolean")
    expected = schema.get("type")
    if expected is not None:
        choices = expected if isinstance(expected, list) else [expected]
        if not choices or any(not isinstance(item, str) or item not in _PYTHON_TYPES for item in choices):
            raise ValueError(f"{path}.type has invalid JSON Schema types")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
        raise ValueError(f"{path}.required must be an array of property names")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}.enum must be an array")
    for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        children = schema.get(keyword, {})
        if not isinstance(children, dict):
            raise ValueError(f"{path}.{keyword} must be an object")
        for key, child in children.items():
            validate_schema_shape(child, f"{path}.{keyword}.{key}")
    for keyword in ("items", "additionalItems", "additionalProperties", "contains", "not", "if", "then", "else",
                    "propertyNames", "unevaluatedProperties", "unevaluatedItems"):
        if keyword in schema:
            children = schema[keyword] if keyword == "items" and isinstance(schema[keyword], list) else [schema[keyword]]
            for child in children:
                validate_schema_shape(child, f"{path}.{keyword}")
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if keyword in schema:
            if not isinstance(schema[keyword], list):
                raise ValueError(f"{path}.{keyword} must be an array of schemas")
            for child in schema[keyword]:
                validate_schema_shape(child, f"{path}.{keyword}")


def validate_arguments(
    schema: dict[str, Any], arguments: dict[str, Any], *, allow_unknown: bool = False
) -> str | None:
    """Return a human-readable problem with `arguments`, or None if valid."""
    properties: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = schema.get("required") or []

    missing = [key for key in required if key not in arguments]
    if missing:
        return f"missing required argument(s): {', '.join(sorted(missing))}"

    unknown = [key for key in arguments if key not in properties]
    additional = schema.get("additionalProperties", allow_unknown)
    if unknown and additional is False:
        allowed = ", ".join(sorted(properties)) or "(none)"
        return f"unexpected argument(s): {', '.join(sorted(unknown))}. Allowed: {allowed}"

    for key, value in arguments.items():
        value_schema = properties.get(key, additional)
        if not isinstance(value_schema, dict):
            continue
        problem = _check_value(key, value_schema, value)
        if problem is not None:
            return problem
    return None


def _check_value(key: str, schema: dict[str, Any], value: Any) -> str | None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if all(_check_value(key, {**schema, "type": option}, value) is not None for option in expected):
            return f"'{key}' must satisfy one of the types: {', '.join(expected)}"
        return None
    if isinstance(expected, str) and expected in _PYTHON_TYPES:
        python_type = _PYTHON_TYPES[expected]
        # bool is a subclass of int; never let it satisfy an int/number schema.
        if expected in {"integer", "number"} and isinstance(value, bool):
            return f"'{key}' must be {expected}, got boolean"
        if not isinstance(value, python_type):
            return f"'{key}' must be {expected}, got {type(value).__name__}"

    if isinstance(value, float) and not math.isfinite(value):
        return f"'{key}' must be a finite number"

    choices = schema.get("enum")
    if choices is not None and value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        return f"'{key}' must be one of: {allowed}"

    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and isinstance(value, (int, float)):
        if value < minimum:
            return f"'{key}' must be >= {minimum}, got {value}"

    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and isinstance(value, (int, float)):
        if value > maximum:
            return f"'{key}' must be <= {maximum}, got {value}"

    if isinstance(value, dict):
        problem = validate_arguments(schema, value, allow_unknown=True)
        if problem is not None:
            return f"'{key}': {problem}"

    if schema.get("type") == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                problem = _check_value(f"{key}[{index}]", item_schema, item)
                if problem is not None:
                    return problem
    return None


class ToolRegistry:
    """Tools plus explicit ownership of session services.

    The live registry owns services. Reload candidates borrow them while building
    replacement tools and must not close them if staging fails. Tool-owned clients
    are separate resources and are closed when those tool instances retire.
    """

    def __init__(self, tools: Iterable[Tool] | None = None, *,
                 services: dict[str, Any] | None = None, owns_services: bool = True) -> None:
        self._tools: dict[str, Tool] = {}
        self.builtin_names: frozenset[str] = frozenset()
        self.context_notes: dict[str, Any] = {}
        self.services = services if services is not None else {}
        self.owns_services = owns_services
        self._closing: Lifetime | None = None
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if self._closing is not None:
            raise RuntimeError("Tool registry is closing or closed")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        validate_schema_shape(tool.parameters, f"{tool.name}.parameters")
        self._tools[tool.name] = tool

    def replace(self, tools: Iterable[Tool]) -> None:
        """Swap a validated collection without changing the registry's identity."""
        replacement: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in replacement:
                raise ValueError(f"tool already registered: {tool.name}")
            validate_schema_shape(tool.parameters, f"{tool.name}.parameters")
            replacement[tool.name] = tool
        self._tools = replacement

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns False if it was not registered.

        Used when an MCP server disconnects and the tools it contributed are
        no longer callable.
        """
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in self.names]

    async def invoke(
        self, name: str, raw_arguments: str | dict[str, Any] | None
    ) -> ToolResult:
        if self._closing is not None:
            return ToolResult.error("Tool registry is closing or closed")
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.names) or "(none)"
            return ToolResult.error(
                f"Unknown tool '{name}'. Available tools: {available}."
            )
        try:
            instructions = self.services.get("project_instructions")
            if instructions is not None:
                # Resolve the accepted helper at dispatch: this registry is a
                # stable contract, while its I/O runner is reloadable behavior.
                from .blocking import run_blocking
                problem = await run_blocking(instructions.guard, tool, raw_arguments)
                if problem is not None:
                    return ToolResult.error(problem)
            result = await tool.invoke(raw_arguments)
            if not isinstance(result, ToolResult):
                raise TypeError(f"Tool '{name}' did not return a ToolResult")
            return result
        except Exception as exc:
            return ToolResult.error(f"{type(exc).__name__}: {exc}")

    @property
    def tools(self) -> list[Tool]:
        return [self._tools[name] for name in self.names]

    async def aclose(self) -> None:
        """Drain resources exactly once, including when shutdown is cancelled."""
        if self._closing is None:
            self._closing = Lifetime("tool registry")
            owners = [*self.tools, *(reversed(list(self.services.values())) if self.owns_services else [])]
            seen: set[int] = set()
            for owner in reversed(owners):
                closer = getattr(owner, "aclose", None)
                if closer is not None and id(owner) not in seen:
                    seen.add(id(owner))
                    self._closing.defer(closer)
        await self._closing.aclose()
