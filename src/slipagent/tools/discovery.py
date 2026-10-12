"""Session-retained tool definitions with a compact discovery catalog."""

from __future__ import annotations

from typing import Any

from ..types import Message, ToolSpec
from .base import Tool, ToolRegistry, ToolResult

CORE_TOOLS = frozenset({"read_file", "write_file", "list_dir", "grep", "edit_file", "run_command", "load_tools"})


class ToolCatalog:
    def __init__(self) -> None:
        self.available: dict[str, ToolSpec] = {}
        self.loaded: set[str] = set()

    def select(self, specs: list[ToolSpec], messages: list[Message]) -> set[str]:
        self.available = {spec.name: spec for spec in specs}
        # Ordinary history restores discovery after resume, fork, or rewind.
        for message in messages:
            if message.role == "assistant":
                self.loaded.update(call.name for call in message.tool_calls or [])
            if (message.role == "tool" and isinstance(message.content, dict)
                    and message.content.get("tool") == "load_tools"):
                content = message.content.get("content")
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
            definitions.append(spec.to_api())
        return ToolResult({"loaded": loaded, "definitions": definitions, "errors": errors}, bool(errors))
