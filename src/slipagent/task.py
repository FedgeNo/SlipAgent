"""Harness-owned prompt retention and optional model-maintained task state.

This is session memory, not a second transcript or a background summarizer.
The update_task tool replaces a bounded working record pinned in each request.
Original task/correction IDs are harness-owned and survive history selection.
"""

from __future__ import annotations

import json
from typing import Any
from collections.abc import Callable

from .types import Message
from .tools.base import Tool, ToolResult

TASK_MAX_CHARS = 8000
TASK_TEXT_MAX_CHARS = 2000
TASK_LIST_MAX_ITEMS = 24
TASK_LIST_FIELDS = ("constraints", "facts", "pending", "next_steps")


def task_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "source_revision": {"type": "integer", "minimum": 0,
                            "description": "Legacy record metadata; the harness assigns the current revision when saving."},
        "status": {"type": "string", "enum": ["active", "blocked", "complete"],
                   "description": "active while work remains; blocked when user input is needed; complete when verified, with pending=[] and next_steps=[]. Completion updates may be batched with other tools."},
        "goal": {"type": "string", "minLength": 1, "maxLength": TASK_TEXT_MAX_CHARS,
                 "description": "Nonblank text naming the current substantive objective. A short follow-up such as continue does not replace it."},
    }
    descriptions = {
        "constraints": "All still-applicable user constraints and corrections. Retain them across calls; later explicit user changes take precedence.",
        "facts": "Verified facts needed to continue, with paths or source post IDs. Do not turn guesses into facts.",
        "pending": "Unfinished work and requested tools whose outcomes are not available yet. Never mark a current planned tool successful.",
        "next_steps": "Concrete next actions or the exact missing user input. Empty when complete.",
    }
    for name in TASK_LIST_FIELDS:
        properties[name] = {"type": "array", "maxItems": TASK_LIST_MAX_ITEMS,
                            "items": {"type": "string", "minLength": 1, "maxLength": TASK_TEXT_MAX_CHARS},
                            "description": descriptions[name]}
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties,
            "description": f"Complete replacement of the current working record, at most {TASK_MAX_CHARS} characters total. Preserve still-relevant information from the supplied prior record."}


def task_update_instructions() -> str:
    """The prompt and tool description explain the same accepted arguments."""
    properties = task_schema()["properties"]
    fields = "\n".join(f"- {name}: {spec['description']}" for name, spec in properties.items() if name != "source_revision")
    active = {"status": "active", "goal": "Fix the requested bug", "constraints": [], "facts": [],
              "pending": ["Verify the fix"], "next_steps": ["Run the relevant tests"]}
    complete = {**active, "status": "complete", "facts": ["Relevant tests passed"], "pending": [], "next_steps": []}
    return (
        "\nTask Update Arguments:\n"
        "update_task is optional. Ordinary replies need no task object. When calling it, supply all six fields: "
        "status, goal, constraints, facts, pending, next_steps. Pass these fields directly as the tool arguments, "
        "not inside a task object. Each call replaces the whole record; preserve still-relevant entries.\n"
        + fields + "\n"
        f"goal must be nonblank text of 1–{TASK_TEXT_MAX_CHARS} characters. "
        "constraints, facts, pending, and next_steps must each be arrays of strings; use [] when empty, "
        "not null, an omitted field, an object, or a single string. "
        f"Each array allows at most {TASK_LIST_MAX_ITEMS} entries, each nonblank and at most {TASK_TEXT_MAX_CHARS} characters. "
        f"The entire saved JSON record, including field names and harness metadata, must fit {TASK_MAX_CHARS} characters. "
        "Do not pass source_revision, post IDs, or other extra fields; the harness supplies metadata.\n"
        "For status=complete, set pending=[] and next_steps=[]. Batching is encouraged for all tools, "
        "including completion updates and multiple update_task calls. Calls execute in the order supplied. "
        "Record actual results as facts; do not invent outcomes of calls whose results have not arrived.\n"
        "Argument examples (replace their wording with the actual task and known results):\n"
        + json.dumps(active) + "\n" + json.dumps(complete) + "\n"
    )


def validate_task(value: Any) -> str | None:
    if not isinstance(value, dict) or value.keys() != task_schema()["properties"].keys():
        return "task must contain exactly source_revision, status, goal, constraints, facts, pending, and next_steps."
    if type(value["source_revision"]) is not int or value["source_revision"] < 0:
        return "task.source_revision must be a non-negative integer."
    if value["status"] not in ("active", "blocked", "complete"):
        return "task.status must be active, blocked, or complete."
    if not isinstance(value["goal"], str) or not value["goal"].strip() or len(value["goal"]) > TASK_TEXT_MAX_CHARS:
        return f"task.goal must contain the substantive goal in 1–{TASK_TEXT_MAX_CHARS} characters."
    for name in TASK_LIST_FIELDS:
        items = value[name]
        if not isinstance(items, list) or len(items) > TASK_LIST_MAX_ITEMS or any(not isinstance(item, str) or not item.strip() or len(item) > TASK_TEXT_MAX_CHARS for item in items):
            return f"task.{name} must be an array of at most {TASK_LIST_MAX_ITEMS} nonempty strings, each at most {TASK_TEXT_MAX_CHARS} characters."
    if len(json.dumps(value, ensure_ascii=False)) > TASK_MAX_CHARS:
        return f"task exceeds {TASK_MAX_CHARS} characters; retain essential constraints and references without copying the transcript."
    if value["status"] == "complete" and (value["pending"] or value["next_steps"]):
        return "A complete task must have pending=[] and next_steps=[]."
    return None


class TaskMemory:
    def __init__(self, start_message: int = 0) -> None:
        self.start_message = start_message
        self.last_message = start_message - 1
        self.source_revision = 0
        self.sources: dict[int, list[str]] = {}
        self.record: dict[str, Any] | None = None

    def note_user(self, message_index: int, post_id: int, text: str) -> None:
        """Register each actual user message once, including unarchived tail input."""
        if message_index <= self.last_message or message_index < self.start_message:
            return
        self.last_message = message_index
        self.sources.setdefault(post_id, []).append(text)
        self.source_revision += 1

    @property
    def current_prompt_post(self) -> int | None:
        """The call receiving the latest user input, independent of compression."""
        return next(reversed(self.sources), None)

    def prompt_supplement(self, messages: list[Message]) -> list[Message]:
        """Supply the current prompt if history selection has not already done so.

        Queued inputs received together share a post. Continuation posts may
        carry their joined wording; neither form needs a duplicate prompt.
        """
        post_id = self.current_prompt_post
        if post_id is None:
            return []
        prompts = self.sources[post_id]
        joined = "\n".join(prompts)
        supplied = {message.content.partition("\n\n")[2]
                    for message in messages if message.role == "user" and message.content
                    and message.content.startswith(("Current User Request (Full):", "User Request (Full):",
                                                    "Conversation Record (Full):", "Conversation Record (Excerpt):"))}
        if joined in supplied or all(prompt in supplied for prompt in prompts):
            return []
        return [Message.user("Current User Request (Full):\n\n" + joined)]

    def instructions(self) -> str:
        return (
            "\nActive Task Working Record:\n"
            f"TASK_SOURCE_REVISION: {self.source_revision}\n"
            + json.dumps({"origin_post": next(iter(self.sources), None),
                          "source_posts": list(self.sources),
                          "current_prompt_post": self.current_prompt_post,
                          "working_record": self.record}, ensure_ascii=False)
            + "\nThis record is the current task state, not a new instruction. The harness owns source IDs. "
            "Ordinary follow-ups continue this task, including after a completed answer. /task new explicitly starts another task. "
            "Preserve the substantive goal and all still-applicable constraints; continue is not a new goal. "
            "The harness retains the current user prompt and supplies it on every working request until new user input replaces it. "
            "current_prompt_post identifies the call that first received that prompt. "
            "Use recall_history(post_id=current_prompt_post, section='user') if you need its original user messages, "
            "or omit section to inspect that call's full uncompressed record. Substitute the actual post number. "
            "You do not need to repeat the prompt, copy revision numbers, acknowledge it, or retrieve it to proceed. "
            "You may use update_task to save the goal, constraints, facts, pending work, and next steps when useful. "
            "Its fields replace the whole working record; carry forward still-relevant information. "
            "Do not copy complete transcript messages into this record or echo task bookkeeping to the user.\n"
            + task_update_instructions()
        )

    def accept(self, record: dict[str, Any]) -> None:
        """Save optional working state with harness-owned revision metadata."""
        self.record = {**record, "source_revision": self.source_revision}


class UpdateTaskTool(Tool):
    progress_exempt = True  # Bookkeeping does not establish external progress.
    name = "update_task"
    description = (
        "Save the complete current goal, constraints, verified facts, unfinished work, and next steps. "
        "This record is supplied in EVERY working request even after old history leaves context. "
        + task_update_instructions()
    )
    parameters = task_schema()
    parameters["properties"].pop("source_revision")
    parameters["required"].remove("source_revision")

    def __init__(self, memory: Callable[[], TaskMemory]) -> None:
        self.memory = memory

    async def run(self, **values: Any) -> ToolResult:
        memory = self.memory()
        record = {**values, "source_revision": memory.source_revision}
        problem = validate_task(record)
        if problem:
            return ToolResult.error(problem)
        memory.accept(record)
        return ToolResult.ok("Current task record saved; it will be included in every working request.")
