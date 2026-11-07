"""Translate native Messages responses into SlipAgent's shared completion channel."""

from __future__ import annotations

import json
from typing import Any

from .api import APIResponseError, APITransportError, _StreamCompletion
from .data_text import MessageSections, RenderedContent
from .types import Completion


def usage_data(usage: dict[str, Any]) -> dict[str, Any]:
    cached = usage.get('cache_read_input_tokens', 0)
    prompt = usage.get('input_tokens', 0) + cached + usage.get('cache_creation_input_tokens', 0)
    return {'prompt_tokens': prompt, 'completion_tokens': usage.get('output_tokens', 0),
            'prompt_tokens_details': {'cached_tokens': cached}}


def finish_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    return {'end_turn': 'stop', 'stop_sequence': 'stop', 'tool_use': 'tool_calls',
            'max_tokens': 'length', 'refusal': 'stop'}.get(reason, reason)


def completion_data(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw.get('content'), list):
        raise APIResponseError('Anthropic returned malformed message content.')
    text, thoughts, calls = [], [], []
    for block in raw['content']:
        kind = block.get('type')
        if kind == 'text':
            text.append(block['text'])
        elif kind == 'thinking':
            thoughts.append(block.get('thinking', ''))
        elif kind == 'tool_use':
            calls.append({'id': block['id'], 'type': 'function',
                          'function': {'name': block['name'], 'arguments': block['input']}})
        elif kind != 'redacted_thinking':
            raise APIResponseError(f'Unsupported Anthropic content block: {kind}.')
    return {'model': raw.get('model', ''), 'usage': usage_data(raw.get('usage') or {}),
            'choices': [{'finish_reason': finish_reason(raw.get('stop_reason')),
                         'message': {'role': 'assistant', 'content': ''.join(text),
                                     'reasoning': ''.join(thoughts), 'tool_calls': calls}}]}


def message_payload(payload: dict[str, Any]) -> None:
    systems: list[Any] = []
    messages: list[dict[str, Any]] = []
    for message in payload['messages']:
        role, content = message['role'], message.get('content')
        if role in {'system', 'developer'}:
            if content:
                systems.append(content)
            continue
        blocks: list[dict[str, Any]] = []
        if role == 'tool':
            role = 'user'
            blocks.append({'type': 'tool_result', 'tool_use_id': message['tool_call_id'],
                           'content': content or ''})
        else:
            if content:
                blocks.append({'type': 'text', 'text': content})
            for call in message.get('tool_calls') or []:
                args = call['function']['arguments']
                if isinstance(args, str):
                    args = json.loads(args)
                blocks.append({'type': 'tool_use', 'id': call['id'],
                               'name': call['function']['name'], 'input': args})
        if messages and messages[-1]['role'] == role:
            messages[-1]['content'].extend(blocks)
        else:
            messages.append({'role': role, 'content': blocks})
    payload['messages'] = messages
    if systems:
        sections: list[Any] = []
        for system in systems:
            if sections:
                sections.append('\n\n')
            sections.append(system)
        payload['system'] = RenderedContent(MessageSections(sections))
    if 'tools' in payload:
        payload['tools'] = [{'name': tool['function']['name'],
                             'description': tool['function'].get('description', ''),
                             'input_schema': tool['function']['parameters']} for tool in payload['tools']]
    if 'tool_choice' in payload:
        choice = payload['tool_choice']
        payload['tool_choice'] = ({'type': 'tool', 'name': choice['function']['name']}
                                  if isinstance(choice, dict) else
                                  {'type': 'any' if choice == 'required' else choice})
    payload.pop('stream_options', None)


class AnthropicStream(_StreamCompletion):
    """Retain shared tool assembly, deadlines, partial diagnostics and usage accounting."""

    def __init__(self, on_delta: Any) -> None:
        super().__init__(on_delta)
        self.native_usage: dict[str, Any] = {}
        self.stopped = False

    def add(self, data: str) -> None:
        try:
            raw = json.loads(data)
            kind = raw.get('type')
            event: dict[str, Any] = {'choices': []}
            delta: dict[str, Any] = {}
            if kind == 'error':
                error = raw.get('error') or {}
                code = {'overloaded_error': 529, 'rate_limit_error': 429,
                        'api_error': 500}.get(error.get('type', ''))
                super().add(json.dumps({'error': {**error, 'code': code}}))
                return
            if kind == 'message_start':
                message = raw['message']
                event['model'] = message.get('model', '')
                self.native_usage.update(message.get('usage') or {})
            elif kind == 'content_block_start':
                block = raw['content_block']
                if block['type'] == 'tool_use':
                    delta['tool_calls'] = [{'index': raw['index'], 'id': block['id'],
                                            'function': {'name': block['name'], 'arguments': ''}}]
                elif block['type'] == 'text':
                    delta['content'] = block.get('text', '')
                elif block['type'] == 'thinking':
                    delta['reasoning'] = block.get('thinking', '')
                elif block['type'] != 'redacted_thinking':
                    raise ValueError('unsupported content block')
            elif kind == 'content_block_delta':
                part = raw['delta']
                if part['type'] == 'text_delta':
                    delta['content'] = part['text']
                elif part['type'] == 'thinking_delta':
                    delta['reasoning'] = part['thinking']
                elif part['type'] == 'input_json_delta':
                    delta['tool_calls'] = [{'index': raw['index'], 'function': {'arguments': part['partial_json']}}]
            elif kind == 'message_delta':
                self.native_usage.update(raw.get('usage') or {})
                event['choices'] = [{'delta': {}, 'finish_reason': finish_reason(raw['delta'].get('stop_reason'))}]
            elif kind == 'message_stop':
                self.stopped = True
            if delta:
                event['choices'] = [{'delta': delta}]
            event['usage'] = usage_data(self.native_usage)
            super().add(json.dumps(event))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise APIResponseError(f'Malformed Anthropic stream: {exc}') from exc

    def completion(self) -> Completion:
        if not self.stopped:
            raise APITransportError('Anthropic stream ended before message_stop')
        return super().completion()
