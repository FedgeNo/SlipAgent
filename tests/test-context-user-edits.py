"""Saved-edit guidance is present only alongside actual edit observations."""

import pytest

from slipagent.context import ConversationHistory
from slipagent.types import Message


async def view(history, messages, **options):
    return await history.view(messages, [], keep_steps=10, context_length=32_000,
                              max_output=1000, **options)


def system_text(messages):
    return "\n".join(message.to_api()["content"] for message in messages if message.role == "system")


@pytest.mark.parametrize("edits", [None, []])
async def test_no_saved_edits_means_no_saved_edit_notice(edits):
    history = ConversationHistory()
    request = await view(history, [Message.user("Inspect the project")], user_edits=edits)
    assert "user_edits" not in system_text(request)
    assert "Saved User File Changes" not in system_text(request)
    assert "user_edits" not in request[-1].content.data


async def test_saved_edits_include_guidance_and_survive_into_selected_history():
    history = ConversationHistory()
    messages = [Message.user("Inspect the project")]
    edits = [{"edit_id": "file", "revision": "one", "path": "example.py", "diff": "+changed", "source": "user"}]
    request = await view(history, messages, user_edits=edits)
    assert "Saved User File Changes" in system_text(request)
    assert request[-1].content.data["user_edits"][0]["path"] == "example.py"
    messages.append(Message.assistant("Done"))
    history.sync(messages)
    history.steps[0].user_edits = edits
    request = await view(history, messages)
    assert "Saved User File Changes" in system_text(request)
    assert "user_edits" not in request[-1].content.data
    assert request[-1].content.data["history"][0]["user_edits"]


async def test_omitted_old_edits_do_not_trigger_current_request_notice():
    history = ConversationHistory()
    messages = [Message.user("Old task"), Message.assistant("Old details " * 20_000)]
    history.sync(messages)
    history.steps[0].user_edits = [{"edit_id": "old", "path": "old.py", "diff": "+old", "source": "user"}]
    for number in range(12):
        messages.extend([Message.user(f"Task {number}"), Message.assistant("Done")])
    request = await view(history, messages)
    assert all(not record.get("user_edits") for record in request[-1].content.data["history"])
    assert "Saved User File Changes" not in system_text(request)
    assert "user_edits" not in system_text(request)
