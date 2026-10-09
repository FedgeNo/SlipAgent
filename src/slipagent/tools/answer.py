"""Return user-facing text; a successful answer-only batch ends the run."""

from .base import Tool, ToolResult
from ..prompts import load_prompt


class AnswerTool(Tool):
    name = "answer"
    description_prompt = 'tools/answer.md'
    parameter_prompts = 'tools/answer-parameters.json'
    description = load_prompt(description_prompt)
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": ""}},
        "required": ["text"],
        "additionalProperties": False,
    }

    async def run(self, text: str) -> ToolResult:
        # A wrapper preserves JSON-looking answer text as text in stored results.
        return ToolResult.ok({"text": text})
