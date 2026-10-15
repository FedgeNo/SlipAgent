"""Provider thought wrappers are reasoning, never response JSON or tool input."""

import json

import httpx
import pytest

from slipagent.api import APIResponseError
from slipagent.google import GoogleClient
from slipagent.protocol import parse_agent_response
from slipagent.thoughts import ThoughtStream
from slipagent.types import Message, ToolSpec


ANSWER = json.dumps({"response": "Ready", "tool_calls": []})
WIRE = ' \n<thought>Check {"tool_calls": "not executable"}.</thought>\n<thought>Then act.</thought>' + ANSWER


@pytest.mark.parametrize("boundary", range(len(WIRE) + 1))
def test_every_chunk_boundary(boundary):
    parts = {"reasoning": [], "content": []}
    parser = ThoughtStream(lambda kind, text: parts[kind].append(text))
    parser.add(WIRE[:boundary])
    parser.add(WIRE[boundary:])
    parser.finish()
    assert "".join(parts["reasoning"]) == 'Check {"tool_calls": "not executable"}.Then act.'
    assert "".join(parts["content"]) == ANSWER


@pytest.mark.parametrize("text", ['{"response":"<thought>literal</thought>"}',
                                 'Example: <thought>literal</thought>', '<thoughtful>ordinary</thoughtful>'])
def test_tags_inside_answers_are_preserved(text):
    parts = []
    parser = ThoughtStream(lambda kind, chunk: parts.append((kind, chunk)))
    for char in text:
        parser.add(char)
    parser.finish()
    assert all(kind == "content" for kind, _ in parts)
    assert "".join(chunk for _, chunk in parts) == text


@pytest.mark.parametrize("mode", ["stream", "json", "json-fallback"])
async def test_provider_thoughts_are_separated_before_json_parsing(mode):
    captured = []
    def handle(request):
        if mode == "stream":
            chunks = [{"choices": [{"delta": {"content": char,
                       "extra_content": {"google": {"thought": True}}}}]} for char in WIRE]
            chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                                  + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": WIRE},
                                                      "finish_reason": "stop"}]})
    async with GoogleClient("dummy", transport=httpx.MockTransport(handle)) as client:
        result = await client.chat(model="gemini-3.6-flash", messages=[Message.user("Inspect")],
                                   on_delta=(lambda kind, text: captured.append((kind, text))) if mode != "json" else None)
    assert json.loads(result.message.content) == {"response": "Ready", "tool_calls": []}
    assert result.message.reasoning == 'Check {"tool_calls": "not executable"}.Then act.'
    if mode != "json":
        assert "".join(text for kind, text in captured if kind == "reasoning") == result.message.reasoning
        assert all("<thought>" not in text and "</thought>" not in text for _, text in captured)


@pytest.mark.parametrize("text", ["<thought>", '<thought>{"response":"not an answer"}'])
async def test_unclosed_thought_is_rejected(text):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [
        {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}))
    async with GoogleClient("dummy", transport=transport) as client:
        with pytest.raises(APIResponseError, match="Unclosed"):
            await client.chat(model="gemini-3.6-flash", messages=[Message.user("Inspect")])


async def test_only_post_thought_tool_call_reaches_agent_parser():
    hidden = json.dumps({"response": "Thinking", "tool_calls": [{"name": "record", "arguments": {"value": "hidden"}}]})
    answer = json.dumps({"response": "Ready", "tool_calls": [{"name": "record", "arguments": {"value": "accepted"}}]})
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{
        "message": {"role": "assistant", "content": "<thought>" + hidden + "</thought>" + answer},
        "finish_reason": "stop"}]}))
    async with GoogleClient("dummy", transport=transport) as client:
        completion = await client.chat(model="gemini-3.6-flash", messages=[Message.user("Inspect")])
    response = parse_agent_response(completion.message.content, [], native_tools=False,
                                    tools=[ToolSpec("record", "", {"properties": {"value": {"type": "string"}}})])
    assert response.text == "Ready"
    assert [call.arguments for call in response.calls] == [{"value": "accepted"}]
    assert completion.message.reasoning == hidden
