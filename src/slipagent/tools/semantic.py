"""Deferred code discovery using the session's local semantic index."""

from __future__ import annotations

from typing import Any

from ..workspace import Workspace
from .base import Tool, ToolResult
from .blocking import run_blocking


class SemanticSearchTool(Tool):
    name = "semantic_search"
    description = ""
    description_prompt = "tools/semantic-search.md"
    parameter_prompts = "tools/semantic-search-parameters.json"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(self, workspace: Workspace, services: dict[str, Any]) -> None:
        self.workspace = workspace
        self.services = services

    async def run(self, query: str, limit: int = 5) -> ToolResult:
        if not query.strip():
            return ToolResult.error("Supply a short description of the code or behavior to find.")
        from ..semantic.service import SemanticSearch
        service = self.services.get("semantic_search")
        if service is None:
            service = SemanticSearch(self.workspace)
            self.services["semantic_search"] = service
        try:
            result = await run_blocking(service.lookup, query, limit, on_cancel=service.interrupt)
            return ToolResult.ok(result)
        except Exception as error:
            detail = str(error)
            if "database is locked" in detail:
                detail = "another SlipAgent session is indexing this project; retry shortly"
            return ToolResult.error(f"Semantic search unavailable: {detail}. Use grep, glob, and read_file to continue.")
