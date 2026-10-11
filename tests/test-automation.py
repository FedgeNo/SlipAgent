"""Automation records preserve values across stdout encodings and carriers."""

import io
import json

from slipagent.automation import JsonEventStream
from slipagent.events import AgentEvent
from slipagent.tools.base import ToolResult
from slipagent.types import ToolCall


def test_structured_event_round_trips_through_ascii_stdout():
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding='ascii')
    events = JsonEventStream(stream)
    value = {'text': "é 日本語 can't \\ escape\r\n", 'nested': [False, None, {'number': 7}]}
    events.event(AgentEvent(kind='tool_end', step=3,
                           tool_call=ToolCall('call', 'example', {'path': 'é.txt'}),
                           result=ToolResult(value)))
    # The record is available before the stream closes, including on a pipe.
    record = json.loads(output.getvalue())
    assert record['result']['content'] == value
    assert record['tool_call']['arguments'] == {'path': 'é.txt'}
    assert record['sequence'] == 1


def test_fragments_are_omitted_and_context_remains_an_object():
    stream = io.StringIO()
    events = JsonEventStream(stream)
    for kind in ('reasoning_delta', 'assistant_delta', 'tool_output'):
        events.event(AgentEvent(kind=kind, text='fragment'))
    request = {'messages': [{'role': 'user', 'content': 'Literal text'}]}
    events.event(AgentEvent(kind='context', text=json.dumps(request)))
    record = json.loads(stream.getvalue())
    assert record['request'] == request
    assert record['sequence'] == 1
    assert 'text' not in record
