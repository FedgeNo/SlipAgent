"""The model names an untitled session once, from the user's first request."""

import json

import pytest

from slipagent.agent import Agent
from slipagent.protocol import agent_response_format, parse_agent_response
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry
from slipagent.types import Completion, Message, content_text

ASKS_FOR_TITLE = "This conversation has no title yet."


@pytest.mark.parametrize(("value", "title"), [
    ("Align Editor Menu Columns", "Align Editor Menu Columns"),
    ("  Fix\nthe\tParser  ", "Fix the Parser"),
    ("x" * 80, "x" * 59 + "…"),
    (42, ""),
    ("   ", ""),
])
def test_session_title_is_optional_cleaned_response_data(value, title):
    result = parse_agent_response(json.dumps({"response": "Done.", "session_title": value}), [])
    assert result.text == "Done." and result.session_title == title
    assert parse_agent_response("Done.", []).session_title == ""


def test_session_title_is_not_required_by_either_schema():
    for native in (False, True):
        schema = agent_response_format(native_tools=native)["json_schema"]["schema"]
        assert "session_title" in schema["properties"]
        assert "session_title" not in schema["required"]


def agent_with_journal(tmp_path, events, prompts):
    class Client:
        async def chat(self, **kwargs):
            text = "\n".join(content_text(message.content) for message in kwargs["messages"])
            if text.lstrip().startswith("# Summarizing the Previous Turn"):
                return Completion(Message.assistant('{"summary":"Answered.","reasoning_summary":""}'), "stub")
            prompts.append(text)
            return Completion(Message.assistant(json.dumps({"response": "Done.", "session_title": "Model Title"})), "stub")

    registry = ToolRegistry([])
    journal = SessionJournal(str(tmp_path / "project"), tmp_path / "store")
    registry.services["session_journal"] = journal
    return Agent(Client(), registry, "stub", on_event=events.append), journal


async def test_untitled_session_takes_the_model_title_and_stops_asking(tmp_path):
    events, prompts = [], []
    agent, journal = agent_with_journal(tmp_path, events, prompts)
    assert await agent.run("Fix the menu.") == "Done."
    await agent.wait_for_compaction()
    assert journal.title == "Model Title | SlipAgent"
    assert [event.text for event in events if event.kind == "session_title"] == ["Model Title | SlipAgent"]
    assert journal.listing()[0]["title"] == "Model Title | SlipAgent"
    assert ASKS_FOR_TITLE in prompts[0]
    assert await agent.run("Thanks.") == "Done."
    await agent.wait_for_compaction()
    assert ASKS_FOR_TITLE not in prompts[1]
    assert sum(event.kind == "session_title" for event in events) == 1


async def test_a_name_the_user_chose_is_never_replaced(tmp_path):
    events, prompts = [], []
    agent, journal = agent_with_journal(tmp_path, events, prompts)
    journal.rename("Mine")
    assert await agent.run("Fix the menu.") == "Done."
    await agent.wait_for_compaction()
    assert journal.title == "Mine | SlipAgent"
    assert ASKS_FOR_TITLE not in prompts[0]
    assert not any(event.kind == "session_title" for event in events)
