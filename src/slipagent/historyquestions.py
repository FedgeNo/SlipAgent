"""Validated references to original history, carried through the ordinary input queue."""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING
from urllib.parse import unquote

from wcwidth import iter_graphemes

from .context import CompletedStep, HISTORY_PART_NAMES
from .prompts import load_prompt
from .turnstore import resident_content

if TYPE_CHECKING:
    from .agent import Agent
    from .context import HistoryStep

QUESTION_PREFIX = '{"slipagent_history_question":'
QUESTION_LIMIT = 131072


def _step(agent: Agent, step_id: int) -> HistoryStep:
    steps = agent.history.steps
    if not 1 <= step_id <= len(steps) or steps[step_id - 1].id != step_id:
        raise ValueError('This history step is no longer available.')
    return steps[step_id - 1]


def _parts(step: HistoryStep) -> dict[str, Any]:
    if isinstance(step, CompletedStep):
        return step.parts(stream=True)
    return CompletedStep(step.id, step.request, step.messages).parts(stream=True)


def _walk(value: Any, path: list[str | int]) -> Any:
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, (list, tuple)) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise ValueError('This history field is no longer available.')
    return value


def _selection(agent: Agent, step_id: int, section: str, call_id: str | None) -> Any:
    step = _step(agent, step_id)
    if section == 'user':
        value: Any = [resident_content(message) for message in step.messages if message.role == 'user']
    elif section == 'all':
        value = {'step': step.id, **_parts(step)} if isinstance(step, CompletedStep) else step.record()
    elif section in HISTORY_PART_NAMES:
        value = _parts(step).get(section, [] if section in ('user_edits', 'background_commands') else None)
    else:
        raise ValueError('Choose a retrievable history section.')
    if call_id is not None:
        if section not in ('tool_calls', 'tool_results'):
            raise ValueError('A call reference requires tool calls or results.')
        key = 'id' if section == 'tool_calls' else 'call_id'
        value = [item for item in value if item.get(key) == call_id]
        if not value:
            raise ValueError('This history call is no longer available.')
    return value


def reference_label(agent: Agent, target: dict[str, Any]) -> str:
    section, path = target['section'], target['fieldPath']
    label = f"Step {target['stepId']}"
    names = {'prompt': 'Prompt', 'response': 'Response', 'reasoning': 'Reasoning', 'user': 'User Message',
             'tool_calls': 'Tool Calls', 'tool_results': 'Tool Results',
             'user_edits': 'User Edits', 'background_commands': 'Background Commands'}
    if section != 'all':
        label += ' · ' + names[section]
    if target['callId'] is not None:
        original = _selection(agent, target['stepId'], section, None)
        key = 'id' if section == 'tool_calls' else 'call_id'
        index = next(index for index, item in enumerate(original) if item.get(key) == target['callId'])
        label = f"Step {target['stepId']} · {'Tool Call' if section == 'tool_calls' else 'Tool Result'} {index + 1}"
        path = path[1:] if path and path[0] == 0 else path
    if path:
        label += ' · ' + ' › '.join(str(key) for key in path)
    clusters = list(iter_graphemes(' '.join(re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', label).split())))
    return ''.join(clusters) if len(clusters) <= 160 else ''.join(clusters[:159]) + '…'


def history_reference(agent: Agent, node: str) -> dict[str, Any] | None:
    segments = [unquote(part) for part in node.strip('/').split('/')]
    if len(segments) < 2 or segments[0] not in ('turns', 'messages'):
        return None
    index, tail = int(segments[1]), segments[2:]
    if segments[0] == 'turns':
        step = _step(agent, index)
    else:
        if not 0 <= index < len(agent.messages):
            raise ValueError('This history message is no longer available.')
        message = agent.messages[index]
        found = next((item for item in agent.history.steps if any(prior is message for prior in item.messages)), None)
        if found is None:
            return None  # Pending input and system instructions have no completed history index.
        step = found
        if message.role == 'user':
            if tail not in ([], ['content']):
                return None
            prompts = [item for item in step.messages if item.role == 'user']
            target = {'stepId': step.id, 'section': 'user', 'callId': None,
                      'fieldPath': [next(index for index, item in enumerate(prompts) if item is message)]}
            return {'sessionId': agent.session_id, 'label': reference_label(agent, target), 'target': target}
        if message.role == 'tool':
            if tail and tail[0] != 'content':
                return None
            results = [item for item in step.messages if item.role == 'tool']
            result_index = next(index for index, item in enumerate(results) if item is message)
            tail = ['tool_results', str(result_index), *tail]
        elif message.role == 'assistant':
            if tail and tail[0] not in ('content', 'tool_calls', 'reasoning', 'response_source'):
                return None
            tail = ['response' if part == 'content' and index == 0 else part for index, part in enumerate(tail)]
        else:
            return None
    section = tail[0] if tail and tail[0] in HISTORY_PART_NAMES else 'all'
    parts = tail[1:] if section != 'all' else tail
    value = _selection(agent, step.id, section, None)
    path: list[str | int] = []
    for part in parts:
        if isinstance(value, dict) and part not in value:
            return None  # Some message metadata has no counterpart in recall_history's original parts.
        key: str | int = int(part) if isinstance(value, (list, tuple)) else part
        value = _walk(value, [key])
        path.append(key)
    call_id = None
    if section in ('tool_calls', 'tool_results') and path and type(path[0]) is int:
        item = _selection(agent, step.id, section, None)[path[0]]
        call_id = item['id' if section == 'tool_calls' else 'call_id']
        path[0] = 0  # recall_history returns a one-element list when selecting call_id.
    target = {'stepId': step.id, 'section': section, 'callId': call_id, 'fieldPath': path}
    return {'sessionId': agent.session_id, 'label': reference_label(agent, target), 'target': target}


def history_question(agent: Agent, question: str, *, step_id: int, section: str,
                     call_id: str | None, field_path: list[str | int]) -> str:
    _walk(_selection(agent, step_id, section, call_id), field_path)
    target = {'stepId': step_id, 'section': section, 'callId': call_id, 'fieldPath': field_path}
    arguments: dict[str, Any] = {'step_id': step_id, 'section': section}
    if call_id is not None:
        arguments['call_id'] = call_id
    return json.dumps({'slipagent_history_question': 1, 'question': question,
        'history_reference': {'label': reference_label(agent, target), 'tool': 'recall_history',
                              'arguments': arguments, 'selected_field_path': field_path},
        'guidance': load_prompt('history-question.md').strip()}, ensure_ascii=False)


def question_preview(text: str) -> dict[str, Any] | None:
    """Recognize the owned input envelope for presentation; originals stay unchanged."""
    if not text.startswith(QUESTION_PREFIX) or len(text) > QUESTION_LIMIT:
        return None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if (not isinstance(value, dict) or set(value) != {'slipagent_history_question', 'question', 'history_reference', 'guidance'}
            or type(value['slipagent_history_question']) is not int or value['slipagent_history_question'] != 1
            or not isinstance(value['question'], str) or not isinstance(value['guidance'], str)):
        return None
    reference = value['history_reference']
    if (not isinstance(reference, dict) or reference.get('tool') != 'recall_history'
            or not isinstance(reference.get('label'), str) or not isinstance(reference.get('arguments'), dict)
            or not isinstance(reference.get('selected_field_path'), list)):
        return None
    return value


def display_question(text: str) -> str:
    value = question_preview(text)
    return f"{value['question']}\n\n[About {value['history_reference']['label']}]" if value is not None else text
