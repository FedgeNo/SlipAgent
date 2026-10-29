"""Replies cut off by the output token limit are reported as truncation."""

import pytest

from slipagent.context import ContextError
from slipagent.types import Completion, Message, Usage, content_text
from test_agent import StubClient, build_agent


class RawClient(StubClient):
    """Return working replies unchanged, so an empty reply stays empty."""

    async def chat(self, **kwargs):
        system = content_text(kwargs["messages"][0].content).lstrip()
        if system.startswith(("# Merging Older Conversation", "# Summarizing the Previous Turn")):
            return await super().chat(**kwargs)
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.mark.parametrize("error", ["tool_calls[0].arguments: Invalid JSON", None])
async def test_truncated_reply_names_the_output_limit(error):
    responses = [Completion(message=Message.assistant(None), model="test/model", finish_reason="length",
                            usage=Usage(), response_error=error) for _ in range(3)]
    agent, _ = build_agent([])
    agent.client = RawClient(responses)
    with pytest.raises(ContextError, match="output token limit"):
        await agent.run("go")


def test_anthropic_payload_without_profile_keeps_a_default_budget():
    from slipagent.anthropic import AnthropicClient

    payload: dict = {}
    AnthropicClient.prepare_payload(object.__new__(AnthropicClient), payload, None)
    assert payload["max_tokens"] == 8192
