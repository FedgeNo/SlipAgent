"""Session-retained tool definitions with a compact discovery catalog."""

from __future__ import annotations

from typing import Any

from ..sources import TextSource, materialize
from ..turnstore import resident_content
from ..types import Message, ToolSpec
from .base import Tool, ToolRegistry, ToolResult

CORE_TOOLS = frozenset({"read_file", "write_file", "list_dir", "grep", "edit_file", "run_command", "load_tools", "update_plan"})
UNAVAILABLE_BATCH = "No tools were executed. The entire batch was rejected because these tools are not available "


class ToolCatalog:
    def __init__(self) -> None:
        self.available: dict[str, ToolSpec] = {}
        self.loaded: set[str] = set()

    def select(self, specs: list[ToolSpec], messages: list[Message]) -> set[str]:
        self.available = {spec.name: spec for spec in specs}
        # Ordinary history restores discovery after resume, fork, or rewind.
        # Call IDs are local to one assistant batch, including older sessions
        # whose executed tools restore loading without a load_tools result.
        rejected: set[str | None] = set()
        for message in reversed(messages):
            if message.role == "assistant":
                self.loaded.update(call.name for call in message.tool_calls or [] if call.id not in rejected)
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
                    self.loaded.update(name for name in content["loaded"] if isinstance(name, str))
        return set((CORE_TOOLS | self.loaded) & self.available.keys())

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
