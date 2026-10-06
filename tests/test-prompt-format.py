"""Model requests retain paragraph and structured-data boundaries without wrapping."""

import json
import sys
from pathlib import Path

import httpx
import pytest

from slipagent.agent import Agent, build_system_prompt
from slipagent.capabilities import ModelCapabilities
from slipagent.context import JSON_TOOL_INSTRUCTIONS
from slipagent.openrouter import OpenRouterClient
from slipagent.tools.base import ToolRegistry
from test_agent import summary_response


def assert_unwrapped_paragraphs(text):
    for paragraph in text.strip().split("\n\n"):
        lines = paragraph.splitlines()
        assert len(lines) == 1, paragraph


@pytest.mark.parametrize("native", [False, True])
async def test_working_and_summary_requests_have_unwrapped_paragraphs(native):
    requests, summaries = [], []
    user = "User paragraph line one\nline two\n\n- first\n- second"

    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            summaries.append(body)
            return httpx.Response(200, json=summary)
        requests.append(body)
        text = "Done" if native else json.dumps({"response": "Done", "tool_calls": []})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

    registry = ToolRegistry()
    parameters = ["tools"] if native else ["response_format"]
    profile = ModelCapabilities({}, [{"tag": "stub", "supported_parameters": parameters, "context_length": 1000000}])
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities("test", profile)
        agent = Agent(client, registry, "test", system_prompt=build_system_prompt("/project"))
        assert await agent.run(user) == "Done"
        await agent.wait_for_compaction()
    assert len(requests) == len(summaries) == 1
    assert_unwrapped_paragraphs(requests[0]["messages"][0]["content"])
    assert_unwrapped_paragraphs(summaries[0]["messages"][0]["content"])
    assert json.loads(requests[0]["messages"][1]["content"])["user_prompt"] == [user]
    assert json.loads(summaries[0]["messages"][1]["content"])["user_prompt"] == user
    definitions = requests[0].get("tools")
    if not native:
        definitions = json.loads(requests[0]["messages"][0]["content"].split("Available Tool Definitions:\n\n", 1)[1].split("\n\n", 1)[0])
    for definition in definitions:
        assert_unwrapped_paragraphs(definition["function"]["description"])
    await registry.aclose()


def test_editable_prompt_file_is_the_rendered_template():
    import slipagent.prompts as prompts

    template = Path(prompts.__file__).with_name("system-prompt.txt").read_text(encoding="utf-8")
    assert build_system_prompt("/project") == template.format(workspace="/project", interpreter=sys.executable).rstrip("\n")
    assert_unwrapped_paragraphs(template)


def test_unwrapped_json_call_example_is_valid():
    example = next(line.removeprefix("Example: ") for line in JSON_TOOL_INSTRUCTIONS.splitlines() if line.startswith("Example: "))
    value = json.loads(example)
    assert json.loads(value["tool_calls"][0]["arguments"]) == {"path": "README.md"}
