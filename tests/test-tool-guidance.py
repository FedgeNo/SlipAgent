"""Tool contracts reach both request profiles without relying on the main prompt."""

import json
from data_text_reader import read_data

import httpx
import pytest

from slipagent.agent import Agent, build_system_prompt
from slipagent.capabilities import ModelCapabilities
from slipagent.lsp import NavigateCodeTool
from slipagent.openrouter import OpenRouterClient
from slipagent.tools import build_default_registry


@pytest.mark.parametrize("native", [False, True])
async def test_tool_guidance_reaches_actual_request_body(workspace, native):
    captured = []

    def transport(request):
        assert request.method == "POST"
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Done"},
                                                      "finish_reason": "stop"}]})

    registry = build_default_registry(workspace)
    registry.register(NavigateCodeTool(registry.services["language_servers"]))
    profile = ModelCapabilities({}, [{"tag": "stub", "supported_parameters": ["tools"] if native else ["response_format"],
                                      "context_length": 1000000}])
    main = build_system_prompt(str(workspace.root))
    assert all(name not in main for name in ("edit_file", "read_file", "run_command", "read_command_output",
                                             "navigate_code", "command_jobs", "old_string", "next_offset"))
    try:
        async with OpenRouterClient("dummy", transport=httpx.MockTransport(transport)) as client:
            agent = Agent(client, registry, "test", system_prompt=main)
            view = await agent._context_view(registry.specs(), 0, capabilities=profile)
            await client.chat(model="test", messages=view, tools=registry.specs() if native else None,
                              request_profile=profile)
        body = captured[0]
        system = body["messages"][0]["content"]
        system_definitions = read_data(system.split("# Available Tool Definitions\n\n```text\n", 1)[1].split("\n```", 1)[0])
        assert all("# Available Tool Definitions" not in message["content"] for message in body["messages"] if message["role"] == "user")
        if native:
            definitions = body["tools"]
            assert system_definitions == definitions
        else:
            assert "tools" not in body
            definitions = system_definitions
        descriptions = {item["function"]["name"]: item["function"]["description"] for item in definitions}
        assert "substring" in descriptions["edit_file"] and "`old_string`" in descriptions["edit_file"]
        assert "same read batch" in descriptions["read_file"]
        assert "whole file" in descriptions["write_file"]
        assert "`poll=true`" in descriptions["run_command"]
        assert "`background=true`" in descriptions["run_command"]
        assert "`next_offset`" in descriptions["read_command_output"]
        assert "30 seconds" in descriptions["command_jobs"]
        assert "Unicode character positions" in descriptions["navigate_code"]
        for name in ("git_status", "git_diff", "git_log", "git_add", "git_commit"):
            assert "`read_command_output`" in descriptions[name]
    finally:
        await registry.aclose()
