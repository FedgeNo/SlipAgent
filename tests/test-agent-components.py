"""Pipeline components preserve ownership without needing a running Agent."""

import asyncio

import pytest

from data_text_reader import current_input
from slipagent.budget import ContextBudget
from slipagent.context import ConversationHistory
from slipagent.contextbuilder import ContextBuilder, ContextSettings
from slipagent.toolbatch import ToolBatch
from slipagent.tools.base import Tool, ToolRegistry, ToolResult, current_invocation
from slipagent.types import Message, ToolCall


async def test_context_preview_preserves_history_and_budget_owners():
    history = ConversationHistory()
    budget = ContextBudget()
    messages = [Message.system("System guidance."), Message.user("Inspect the project.")]
    builder = ContextBuilder(
        ToolRegistry(), history, messages, budget,
        ContextSettings("test/model", False, 5, False, 25, True, None),
    )
    preview = await builder.build([], 100_000, True, preview=True)
    assert current_input(preview)["user_prompts"] == ["Inspect the project."]
    assert history.cursor == 0 and history.steps == []
    assert budget.model == ""
    actual = await builder.build([], 100_000, True)
    assert [message.to_api() for message in actual] == [message.to_api() for message in preview]
    assert budget.model == "test/model"


async def test_interrupted_batch_archives_all_observations_before_propagating():
    ready = asyncio.Event()
    provenance = []

    class WaitingTool(Tool):
        name = "wait"
        description = "Wait for the test to cancel this invocation."

        async def run(self):
            provenance.append(current_invocation.get())
            ready.set()
            await asyncio.Event().wait()
            return ToolResult.ok("finished")

    messages, events, archives, persisted = [], [], [], []

    async def archive():
        archives.append(list(messages))

    batch = ToolBatch(
        ToolRegistry([WaitingTool()]), 7, messages, emit=events.append,
        persist=lambda: persisted.append(len(messages)), archive_interrupted=archive,
        checkpoints=None, allowed_tools=None,
    )
    task = asyncio.create_task(batch.run([ToolCall("first", "wait"), ToolCall("second", "wait")]))
    await asyncio.wait_for(ready.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provenance == [(7, "first")]
    assert [message.tool_call_id for message in messages] == ["first", "second"]
    assert all(message.content["status"] == "error" for message in messages)
    assert "not run" in messages[1].content["content"]
    assert archives == [messages] and persisted == [1, 2]
    assert [event.step for event in events] == [7, 7, 7]
    assert current_invocation.get() is None
