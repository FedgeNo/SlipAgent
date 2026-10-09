"""Read literal model-input records in wire-level tests, without unescaping text."""

import re


def read_data(text):
    lines = text.split("\n")
    cursor = 0

    def node(header, indent):
        nonlocal cursor
        match = re.fullmatch(r"text \((\d+) characters\):", header)
        if match:
            length = int(match[1])
            parts = []
            while True:
                prefix = " " * (indent + 2)
                assert lines[cursor].startswith(prefix), lines[cursor]
                parts.append(lines[cursor][len(prefix):])
                cursor += 1
                result = "\n".join(parts)
                if len(result) >= length:
                    assert len(result) == length
                    return result
        match = re.fullmatch(r"object \((\d+) fields\):", header)
        if match:
            result = {}
            for _ in range(int(match[1])):
                line = lines[cursor][indent + 2:]
                cursor += 1
                if line.startswith("? "):
                    key = node(line[2:], indent + 2)
                    line = lines[cursor][indent + 2:]
                    cursor += 1
                    assert line.startswith(": ")
                    value = node(line[2:], indent + 2)
                else:
                    key, header = line.split(": ", 1)
                    value = node(header, indent + 2)
                result[key] = value
            return result
        match = re.fullmatch(r"list \((\d+) items\):", header)
        if match:
            result = []
            for _ in range(int(match[1])):
                line = lines[cursor][indent + 2:]
                cursor += 1
                assert line.startswith("- ")
                result.append(node(line[2:], indent + 2))
            return result
        if header == "null":
            return None
        if header in {"true", "false"}:
            return header == "true"
        return float(header) if any(char in header for char in ".eE") else int(header)

    cursor = 1
    return node(lines[0], 0)


def input_data(value):
    return read_data(value) if isinstance(value, str) else value


def current_input(messages):
    from slipagent.prompts import load_prompt
    from slipagent.types import Message, content_text
    for message in messages:
        role = message.role if isinstance(message, Message) else message['role']
        content = message.content if isinstance(message, Message) else message['content']
        if role == 'system':
            _, marker, body = content_text(content).partition(load_prompt('current-turn.md'))
            if marker:
                metadata = read_data(body)
                payload = request_input(messages)
                prompts = ([payload['user_message']] if payload['user_message'] else []) + payload['additional_user_messages']
                supplied = [text for record in payload['history'] for text in record.get('user_prompt', [])]
                if not prompts and '\n'.join(payload['retained_user_request']) not in supplied and not all(text in supplied for text in payload['retained_user_request']):
                    prompts = list(payload['retained_user_request'])
                prompts += [load_prompt('tool-use-correction.md', correction=error).strip() for error in payload['errors']]
                return {'record_type': 'current_step', 'representation': 'full',
                        'user_prompt': prompts, 'agent_response': None, 'tool_calls': [], 'tool_results': [], **metadata}
    raise AssertionError('Current Turn Input missing from system prompt')


def request_input(messages):
    import json
    from slipagent.types import Message
    from slipagent.data_text import JSONInput
    for message in messages:
        role = message.role if isinstance(message, Message) else message['role']
        content = message.content if isinstance(message, Message) else message['content']
        if role == 'user':
            return content.data if isinstance(content, JSONInput) else json.loads(content)
    raise AssertionError('JSON user input missing')


def history_records(payload):
    """Restore the archived shape for existing behavioral assertions."""
    for original in payload['history']:
        if original['representation'] == 'excerpt':
            record = {key: value for key, value in original.items() if key not in ('tool_calls', 'response_excerpts')}
            entries = list(original['response_excerpts'])
            for call in original['tool_calls']:
                for key, status in [('result_excerpt', 'success'), ('error_excerpt', 'error'), ('unclassified_result_excerpt', 'unknown')]:
                    if key in call:
                        entries.append({'role': 'tool', 'content_excerpt': call[key], 'call_id': call['call_id'], 'tool_name': call['tool_name'], 'status': status})
            record['messages'] = entries
            yield record
            continue
        if original['representation'] != 'full':
            yield original
            continue
        record = dict(original)
        calls, results = [], []
        for source in original['tool_calls']:
            call = dict(source)
            for key, status in [('result', 'success'), ('error', 'error'), ('unclassified_result', 'unknown')]:
                if key in call:
                    results.append({'call_id': call['call_id'], 'tool_name': call['tool_name'], 'status': status, 'content': call.pop(key)})
            calls.append(call)
        record.update(tool_calls=calls, tool_results=results)
        yield record
