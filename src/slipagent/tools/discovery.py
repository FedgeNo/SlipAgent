"""Session-retained tool definitions with a compact discovery catalog."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..sources import TextSource, materialize
from ..turnstore import resident_content
from ..types import Message, ToolSpec
from .base import Tool, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from ..context import ConversationHistory

CORE_TOOLS = frozenset({"read_file", "write_file", "list_dir", "grep", "edit_file", "run_command", "run_python", "load_tools", "update_plan"})
UNAVAILABLE_BATCH = "No tools were executed. The entire batch was rejected because these tools are not available "


class ToolCatalog:
    def __init__(self) -> None:
        self.available: dict[str, ToolSpec] = {}
        self.loaded: set[str] = set()
        self._steps: list[Any] | None = None
        self._processed = 0
        self._last_step: Any = None

    def select(self, specs: list[ToolSpec], messages: list[Message], *,
               history: ConversationHistory | None = None) -> set[str]:
        self.available = {spec.name: spec for spec in specs}
        if history is None:
            self.loaded.update(self._restore(messages))
            return set((CORE_TOOLS | self.loaded) & self.available.keys())
        steps = history.steps
        if (self._steps is not steps or self._processed > len(steps)
                or (self._processed and steps[self._processed - 1] is not self._last_step)):
            if self._steps is not None:
                self.loaded.clear()
            self._steps = steps
            self._processed = 0
        for index in range(self._processed, len(steps)):
            self.loaded.update(self._restore(steps[index].messages))
        self._processed = len(steps)
        self._last_step = steps[-1] if steps else None
        # Unfinished batches may still receive rejection results. Revisit only
        # this tail instead of permanently caching their provisional tool use.
        tail = self._restore(messages[history.cursor:])
        return set((CORE_TOOLS | self.loaded | tail) & self.available.keys())

    @staticmethod
    def _restore(messages: list[Message]) -> set[str]:
        loaded: set[str] = set()
        # Ordinary history restores discovery after resume, fork, or rewind.
        # Call IDs are local to one assistant batch, including older sessions
        # whose executed tools restore loading without a load_tools result.
        rejected: set[str | None] = set()
        for message in reversed(messages):
            if message.role == "assistant":
                loaded.update(call.name for call in message.tool_calls or [] if call.id not in rejected)
                rejected.clear()
            if message.role != "tool":
                continue
            # Archived results stay on disk; only their small envelope is read here.
            observation = resident_content(message)
            if not isinstance(observation, dict):
                continue
            if observation.get("status") == "error" and _starts_with(observation.get("content", ""), UNAVAILABLE_BATCH):
                rejected.add(message.tool_call_id)
            if observation.get("tool") == "load_tools":
                content = materialize(observation.get("content"))
                if isinstance(content, dict) and isinstance(content.get("loaded"), list):
                    loaded.update(name for name in content["loaded"] if isinstance(name, str))
        return loaded

    def deferred(self) -> list[dict[str, str]]:
        result = []
        for name, spec in self.available.items():
            if name in CORE_TOOLS or name in self.loaded:
                continue
            purpose = next((line.strip() for line in spec.description.splitlines()
                            if line.strip() and not line.lstrip().startswith("#")), "")
            result.append({"name": name, "purpose": purpose if len(purpose) <= 180 else purpose[:179] + "…"})
        return result


def _starts_with(value: Any, prefix: str) -> bool:
    if not isinstance(value, TextSource):
        return str(value).startswith(prefix)
    text = ""
    for chunk in value.chunks():
        text += chunk
        if len(text) >= len(prefix):
            break
    return text.startswith(prefix)


def tool_catalog(registry: ToolRegistry) -> ToolCatalog | None:
    if "load_tools" not in registry:
        return None
    catalog = registry.services.get("tool_catalog")
    if catalog is None:
        catalog = ToolCatalog()
        registry.services["tool_catalog"] = catalog
    if not isinstance(catalog, ToolCatalog):
        raise TypeError("Invalid tool catalog service")
    return catalog


class LoadToolsTool(Tool):
    name = "load_tools"
    description = ""
    description_prompt = "tools/load-tools.md"
    parameter_prompts = "tools/load-tools-parameters.json"
    parameters = {"type": "object", "properties": {
        "names": {"type": "array", "items": {"type": "string"}},
    }, "required": ["names"]}

    def __init__(self, services: dict[str, Any]) -> None:
        self.services = services

    async def run(self, names: list[str]) -> ToolResult:
        catalog = self.services.get("tool_catalog")
        if catalog is None:
            return ToolResult.error("The tool catalog has not been supplied in a model request yet.")
        loaded, definitions, errors = [], [], {}
        for name in dict.fromkeys(names):
            spec = catalog.available.get(name)
            if spec is None:
                errors[name] = "Unknown or unavailable tool; choose a name from the supplied catalog."
                continue
            catalog.loaded.add(name)
            loaded.append(name)
            if name not in self.services.get("request_only_definitions", ()):
                definitions.append(spec.to_api())
        return ToolResult({"loaded": loaded, "definitions": definitions, "errors": errors}, bool(errors))
