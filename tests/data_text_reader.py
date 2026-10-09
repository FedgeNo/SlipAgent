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
                return read_data(body)
    raise AssertionError('Current Turn Input missing from system prompt')
