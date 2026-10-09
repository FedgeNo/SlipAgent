"""Answer displays user text and finishes only answer-only successful batches."""

import io
import json

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.capabilities import RequestProfile
from slipagent.cli import Renderer, Style
from slipagent.openrouter import OpenRouterClient
from slipagent.protocol import parse_agent_response
from slipagent.tools import AnswerTool, build_default_registry
from slipagent.tools.base import ToolRegistry
from slipagent.types import ToolCall
from slipagent.workspace import Workspace
from test_agent import RecordingTool, summary_response


TEXT = 'Task fulfilled.\nReport saved. &amp; <literal>'


async def run_replies(replies, tools):
    requests, events = [], []
    sink = io.StringIO()
    renderer = Renderer(Style(False), sink, False)
    def emit(event):
        events.append(event)
        renderer.handle(event)
    def handle(request):
        body = json.loads(request.content)
        summary = summary_response(body)
        if summary is not None:
            return httpx.Response(200, json=summary)
        requests.append(body)
        assert replies, 'Answer must not cause an extra request'
        reply = replies.pop(0)
        finish = reply.pop('finish', 'stop')
        return httpx.Response(200, json={'choices': [{'message': reply, 'finish_reason': finish}]})
    registry = ToolRegistry(tools)
    async with OpenRouterClient('dummy', transport=httpx.MockTransport(handle)) as client:
        client.cache_capabilities('test', RequestProfile(parameters={'tools'}, context_length=1000000))
        agent = Agent(client, registry, 'test', on_event=emit)
        answer = await agent.run('Answer the question')
        await agent.wait_for_compaction()
    await registry.aclose()
    return answer, requests, events, sink.getvalue(), agent


@pytest.mark.parametrize('carrier', ['native', 'json', 'qwen', 'xml', 'xml_open'])
async def test_answer_only_displays_and_stops_without_another_request(carrier):
    call = ToolCall('answer-1', 'answer', {'text': TEXT})
    if carrier == 'native':
        reply = {'tool_calls': [call.to_api()]}
    elif carrier == 'json':
        reply = {'content': json.dumps({'response': '', 'tool_calls': [call.to_record()]})}
    elif carrier == 'qwen':
        reply = {'content': '<tool_call><function=answer><parameter=text>\n' + TEXT + '\n</parameter></function></tool_call>'}
    else:
        reply = {'content': '<tool_call>\n<answer>\n' + TEXT + '\n</answer>' + ('\n</tool_call>' if carrier == 'xml' else '')}
    answer, requests, events, output, agent = await run_replies([reply], [AnswerTool()])
    assert answer == TEXT and len(requests) == 1
    assert output.count('Task fulfilled.') == 1
    assert 'answer(' not in output
    assert agent.messages[-1].role == 'tool'
    assert agent.messages[-1].content['content'] == {'text': TEXT}


async def test_answer_with_another_tool_displays_and_continues():
    recorder = RecordingTool()
    replies = [{'tool_calls': [ToolCall('a', 'answer', {'text': 'Working.'}).to_api(),
                               ToolCall('r', 'record', {'value': 'A'}).to_api()]},
               {'tool_calls': [ToolCall('b', 'answer', {'text': 'Finished.'}).to_api()]}]
    answer, requests, events, output, _ = await run_replies(replies, [AnswerTool(), recorder])
    assert answer == 'Finished.' and len(requests) == 2
    assert recorder.seen == [{'value': 'A'}]
    assert output.count('Working.') == output.count('Finished.') == 1


async def test_invalid_answer_arguments_do_not_finish_the_run():
    replies = [{'tool_calls': [ToolCall('bad', 'answer', {'text': 42}).to_api()]},
               {'tool_calls': [ToolCall('good', 'answer', {'text': 'Corrected.'}).to_api()]}]
    answer, requests, events, output, _ = await run_replies(replies, [AnswerTool()])
    assert answer == 'Corrected.' and len(requests) == 2
    assert any(event.kind == 'tool_end' and event.result.is_error for event in events)


async def test_multiple_answers_without_other_tools_end_the_run():
    replies = [{'tool_calls': [ToolCall('a', 'answer', {'text': 'First.'}).to_api(),
                               ToolCall('b', 'answer', {'text': 'Second.'}).to_api()]}]
    answer, requests, events, output, _ = await run_replies(replies, [AnswerTool()])
    assert answer == 'First.\n\nSecond.' and len(requests) == 1
    assert output.count('First.') == output.count('Second.') == 1


@pytest.mark.parametrize('ending', ['', '</answer>', '</answer></tool_call>', '</tool_call>'])
def test_xml_answer_accepts_omitted_closing_tags(ending):
    result = parse_agent_response('<tool_call><answer>Done.' + ending, [])
    assert len(result.calls) == 1
    assert result.calls[0].name == 'answer'
    assert result.calls[0].arguments == {'text': 'Done.'}


@pytest.mark.parametrize('ending', ['', '</parameter>', '</parameter></function>', '</parameter></function></tool_call>'])
def test_qwen_accepts_missing_trailing_closing_tags(ending):
    result = parse_agent_response('<tool_call><function=answer><parameter=text>Done.' + ending, [], tools=[AnswerTool().spec])
    assert result.calls[0].arguments == {'text': 'Done.'}


def test_json_tag_accepts_missing_outer_close_after_complete_arguments():
    result = parse_agent_response('{"response":"Reading"}\n<tool_call>{"name":"answer","arguments":{"text":"Done."}}', [])
    assert result.calls[0].arguments == {'text': 'Done.'}


async def test_one_shot_cli_prints_answer_once_and_exits(tmp_path, monkeypatch):
    from test_cli_e2e import StubOpenRouter, run_cli
    monkeypatch.setattr('test_cli_e2e.structured_message', lambda message, *args, **kwargs: message)
    script = [{'choices': [{'message': {'role': 'assistant', 'content': '<tool_call>\n<answer>Task fulfilled.</answer>'}, 'finish_reason': 'stop'}]}]
    with StubOpenRouter(script, include_memory=False) as stub:
        result = run_cli('--no-mcp', '--no-reload', '-p', 'Answer', '--base-url', stub.base_url, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'Task fulfilled.'
    assert 'Task fulfilled.' not in result.stderr
    assert len(stub.requests) == 1


async def test_restored_answer_displays_as_text_without_tool_json():
    _, _, _, _, agent = await run_replies([{'content': '<tool_call><answer>Restored answer.</answer>'}], [AnswerTool()])
    output = io.StringIO()
    renderer = Renderer(Style(False), output, False)
    await renderer.restore_transcript(agent.messages, [])
    assert output.getvalue().count('Restored answer.') == 1
    assert '"text":' not in output.getvalue()
    assert 'answer(' not in output.getvalue()


async def test_provider_truncation_still_rejects_answer():
    replies = [{'content': '<tool_call><answer>Incomplete', 'finish': 'length'},
               {'content': '<tool_call><answer>Complete</answer>'}]
    answer, requests, events, output, _ = await run_replies(replies, [AnswerTool()])
    assert answer == 'Complete' and len(requests) == 2
    assert 'Incomplete' not in output
    assert len([event for event in events if event.kind == 'tool_start']) == 1


async def test_default_registry_advertises_answer_and_preserves_json_text(tmp_path):
    registry = build_default_registry(Workspace(tmp_path))
    try:
        assert 'answer' in registry.builtin_names
        assert (await registry.invoke('answer', {'text': '{"hello": 1}'})).content == {'text': '{"hello": 1}'}
    finally:
        await registry.aclose()
