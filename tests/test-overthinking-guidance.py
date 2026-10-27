"""Overthinking instructions reach the model only when its thoughts do."""

from dataclasses import replace

from test_agent import RecordingTool, build_agent, call, completion
from slipagent.types import content_text

MARKER = "archived `reasoning`"


def with_thoughts(response, thoughts):
    message = replace(response.message, reasoning=thoughts)
    return replace(response, message=message)


def instructions(request):
    return "\n".join(content_text(message.content) for message in request["messages"] if message.role == "system")


async def run_two_turns(*, overthinking=True, thoughts="Considering option A."):
    agent, client = build_agent(
        [with_thoughts(completion(tool_calls=[call(value="A")]), thoughts), completion("Done")], tools=[RecordingTool("ok")])
    agent.overthinking = overthinking
    await agent.run("Do A")
    return client.calls


async def test_first_request_has_no_thoughts_and_no_overthinking_instructions():
    calls = await run_two_turns()
    assert MARKER not in instructions(calls[0]) and "archived `reasoning`" not in instructions(calls[0])


async def test_instructions_follow_the_thoughts_into_the_request():
    calls = await run_two_turns()
    assert "Considering option A." in str([content_text(m.content) for m in calls[1]["messages"]])
    assert MARKER in instructions(calls[1])


async def test_disabled_mode_sends_no_thoughts_and_no_instructions():
    calls = await run_two_turns(overthinking=False)
    assert "Considering option A." not in str([content_text(m.content) for m in calls[1]["messages"]])
    assert MARKER not in instructions(calls[1])


async def test_steps_without_thoughts_bring_no_instructions():
    calls = await run_two_turns(thoughts=None)
    assert MARKER not in instructions(calls[1])


def test_the_fixed_system_prompt_never_mentions_the_mode():
    from slipagent.prompts import load_prompt
    assert "verthinking" not in load_prompt("system-prompt.md")
