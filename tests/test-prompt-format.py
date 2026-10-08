"""Model requests retain paragraph and structured-data boundaries without wrapping."""

import json
import sys
import shutil
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
        paragraph = paragraph.strip()
        if not paragraph:
            continue
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
        definitions = json.loads(requests[0]["messages"][0]["content"].split("============================= BEGIN AVAILABLE TOOL DEFINITIONS ==============================\n\n", 1)[1].split("\n\n", 1)[0])
    for definition in definitions:
        assert_unwrapped_paragraphs(definition["function"]["description"])
    await registry.aclose()


def test_editable_prompt_file_is_the_rendered_template():
    import slipagent.prompts as prompts

    template = (prompts.prompt_directory() / "system-prompt.txt").read_text(encoding="utf-8")
    assert build_system_prompt("/project") == "\n" + template.format(workspace="/project", interpreter=sys.executable).strip() + "\n"
    assert_unwrapped_paragraphs(template)


def test_unwrapped_json_call_example_is_valid():
    example = next(line.removeprefix("Example: ") for line in JSON_TOOL_INSTRUCTIONS.splitlines() if line.startswith("Example: "))
    value = json.loads(example)
    assert json.loads(value["tool_calls"][0]["arguments"]) == {"path": "README.md"}


@pytest.fixture
def editable_prompts(tmp_path, monkeypatch):
    import slipagent.prompts as prompts
    directory = tmp_path / "prompts"
    shutil.copytree(prompts.prompt_directory(), directory)
    monkeypatch.setattr(prompts, "prompt_directory", lambda: directory)
    return directory


async def test_request_rereads_system_and_protocol_prompts_without_reload(editable_prompts, workspace):
    from test_agent import StubClient
    registry = ToolRegistry()
    registry.services.update(workspace=workspace, editable_prompts=True)
    agent = Agent(StubClient([]), registry, "test", system_prompt=build_system_prompt(str(workspace.root)))
    before = await agent._context_view([], 1)
    path = editable_prompts / "system-prompt.txt"
    path.write_text(path.read_text() + "\n\nFresh system guidance.\n")
    path = editable_prompts / "native-tools.txt"
    path.write_text(path.read_text() + "\n\nFresh protocol guidance.\n")
    after = await agent._context_view([], 1)
    assert "Fresh system guidance." not in before[0].content
    assert "Fresh system guidance." in after[0].content
    assert "Fresh protocol guidance." in after[0].content


def test_tool_spec_rereads_description_and_argument_guidance(editable_prompts, workspace):
    from slipagent.tools.files import ReadFileTool
    tool = ReadFileTool(workspace)
    before = tool.spec
    (editable_prompts / tool.description_prompt).write_text("Fresh read guidance.\n")
    path = editable_prompts / tool.parameter_prompts
    descriptions = json.loads(path.read_text())
    descriptions["properties/path"] = "Fresh path guidance."
    path.write_text(json.dumps(descriptions))
    after = tool.spec
    assert before.description != after.description == "\nFresh read guidance.\n"
    assert after.parameters["properties"]["path"]["description"] == "\nFresh path guidance.\n"
    assert before.parameters["properties"]["path"]["description"] != "Fresh path guidance."


async def test_background_summary_rereads_prompt_at_request_time(editable_prompts):
    summaries = []
    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            summaries.append(body)
            return httpx.Response(200, json=summary)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"}}]})
    registry = ToolRegistry()
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        profile = ModelCapabilities({}, [{"tag": "stub", "supported_parameters": ["tools"], "context_length": 1000000}])
        client.cache_capabilities("test", profile)
        agent = Agent(client, registry, "test")
        await agent.run("First request")
        await agent.wait_for_compaction()
        path = editable_prompts / "background-summary-prompt.txt"
        path.write_text(path.read_text() + "\n\nFresh summary guidance.\n")
        await agent.run("Second request")
        await agent.wait_for_compaction()
    assert len(summaries) == 2
    assert "Fresh summary guidance." not in summaries[0]["messages"][0]["content"]
    assert "Fresh summary guidance." in summaries[1]["messages"][0]["content"]
    await registry.aclose()


def test_packaged_prompt_directory_uses_bundled_files(tmp_path, monkeypatch):
    import slipagent.prompts as prompts
    package = tmp_path / "site-packages" / "slipagent"
    (package / "prompts").mkdir(parents=True)
    (package / "prompts" / "example.txt").write_text("Packaged guidance.\n")
    monkeypatch.setattr(prompts, "__file__", str(package / "prompts.py"))
    assert prompts.load_prompt("example.txt") == "\nPackaged guidance.\n"


def test_prompt_whitespace_boundaries(editable_prompts):
    from slipagent.prompts import load_prompt
    (editable_prompts / "example.txt").write_text(" \t\n\n First paragraph.\n\nSecond paragraph. \t\n")
    assert load_prompt("example.txt") == "\nFirst paragraph.\n\nSecond paragraph.\n"
    assert "before" + load_prompt("example.txt") + "after" == "before\nFirst paragraph.\n\nSecond paragraph.\nafter"


def test_major_sections_use_uppercase_dividers():
    from slipagent.prompts import PromptSections, section_divider
    sections = PromptSections()
    sections.add("project", "Project Instructions (Current)", "Project guidance.", 1)
    divider = section_divider("Project Instructions (Current)")
    assert divider == "============================= PROJECT INSTRUCTIONS (CURRENT) =============================="
    assert sections.render() == section_divider("BEGIN Project Instructions (Current)") + "\n\nProject guidance.\n\n" + section_divider("END Project Instructions (Current)") + "\n"


def test_missing_prompt_reports_path(editable_prompts):
    from slipagent.prompts import PromptError, load_prompt
    with pytest.raises(PromptError, match="Missing prompt file:") as error:
        load_prompt("missing.txt")
    assert str(editable_prompts / "missing.txt") in str(error.value)


def test_cli_reports_missing_prompt_without_traceback(monkeypatch, capsys):
    from slipagent import cli
    from slipagent.prompts import PromptError

    async def dispatch(args, prompt):
        raise PromptError("Missing prompt file: /example/prompts/missing.txt. Restore this file in the prompts folder.")

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    assert cli.main([]) == 2
    assert capsys.readouterr().err == "slipagent: Missing prompt file: /example/prompts/missing.txt. Restore this file in the prompts folder.\n"
