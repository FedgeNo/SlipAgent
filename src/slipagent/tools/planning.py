"""A bounded plan recorded as an ordinary tool result for journal-safe continuity."""

from typing import Any

from .base import Tool, ToolResult


class UpdatePlanTool(Tool):
    name = "update_plan"
    description = ""
    description_prompt = "tools/update-plan.md"
    parameter_prompts = "tools/update-plan-parameters.json"
    parameters = {
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "stages": {"type": "array", "items": {
                "type": "object", "properties": {
                    "task": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "complete"]},
                    "evidence": {"type": "string"},
                }, "required": ["task", "status", "evidence"], "additionalProperties": False,
            }},
        },
        "required": ["objective", "stages"], "additionalProperties": False,
    }

    async def run(self, objective: str, stages: list[dict[str, Any]]) -> ToolResult:
        if not objective.strip() or len(objective) > 500 or not 1 <= len(stages) <= 8:
            return ToolResult.error("Use a nonempty objective of at most 500 characters and 1–8 stages.")
        if any(not stage['task'].strip() or len(stage['task']) > 500 or len(stage['evidence']) > 1000
               for stage in stages):
            return ToolResult.error("Each stage needs a nonempty task (at most 500 characters) and evidence of at most 1000 characters.")
        if sum(stage['status'] == 'in_progress' for stage in stages) > 1:
            return ToolResult.error("At most one stage may be in progress.")
        return ToolResult.ok({"objective": objective, "stages": stages})
