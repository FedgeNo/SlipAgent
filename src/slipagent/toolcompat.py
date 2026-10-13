"""Bounded syntax and schema conversions for explicit, complete tool calls."""

from __future__ import annotations

import json
import re
from typing import Any


def compatible_json(text: str) -> str:
    """Escape literal string bytes and omit trailing commas, without completing JSON.

    Valid escapes retain their meaning. Invalid escapes retain their backslash
    as literal data. Missing quotes, values, and brackets remain parse errors.
    """
    result: list[str] = []
    quoted = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            quoted = not quoted
        elif quoted and char == '\\':
            following = text[index + 1:index + 2]
            unicode_escape = following == 'u' and re.fullmatch(r'[0-9a-fA-F]{4}', text[index + 2:index + 6])
            if following and (following in '"\\/bfnrt' or unicode_escape):
                size = 6 if unicode_escape else 2
                result.append(text[index:index + size])
                index += size
                continue
            result.append('\\\\')
            index += 1
            continue
        elif quoted and ord(char) < 32:
            result.append(json.dumps(char)[1:-1])
            index += 1
            continue
        elif not quoted and char == ',':
            following_index = index + 1
            while following_index < len(text) and text[following_index].isspace():
                following_index += 1
            previous_index = len(result) - 1
            while previous_index >= 0 and result[previous_index].isspace():
                previous_index -= 1
            if (following_index < len(text) and text[following_index] in '}]'
                    and previous_index >= 0 and result[previous_index] not in '{[,:'):
                index += 1
                continue
        result.append(char)
        index += 1
    return ''.join(result)


def tool_name(name: str, names: list[str]) -> str:
    """Prefer exact names; accept only unambiguous namespace/case variants."""
    if name in names:
        return name
    candidate = name.strip()
    for prefix in ('functions.', 'tools.'):
        candidate = candidate.removeprefix(prefix)
    candidates = {name.strip(), candidate, candidate.replace('.', '__')}
    exact = set(names).intersection(candidates)
    if len(exact) == 1:
        return exact.pop()
    if exact:
        return name
    matches = {registered for registered in names if registered.casefold() in {c.casefold() for c in candidates}}
    if len(matches) == 1:
        return matches.pop()
    if matches:
        return name
    suffixes = {registered for registered in names if any(registered.endswith('__' + c) for c in candidates)}
    return suffixes.pop() if len(suffixes) == 1 else name


def _has_type(value: Any, kind: Any) -> bool:
    if isinstance(kind, list):
        return any(_has_type(value, option) for option in kind)
    if not isinstance(kind, str):
        return False
    if kind == 'integer':
        return type(value) is int
    if kind == 'number':
        return type(value) in (int, float)
    return {'string': isinstance(value, str), 'object': isinstance(value, dict),
            'array': isinstance(value, list), 'boolean': type(value) is bool,
            'null': value is None}.get(kind, False)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate argument field: ' + key)
        result[key] = value
    return result


def schema_arguments(value: Any, schema: dict[str, Any], *, root: dict[str, Any] | None = None, depth: int = 0) -> Any:
    """Convert only types selected by the schema, retaining unknown fields and strings."""
    if depth > 12:
        return value
    root = schema if root is None else root
    reference = schema.get('$ref')
    if isinstance(reference, str) and reference.startswith('#/'):
        target: Any = root
        for key in reference[2:].split('/'):
            target = target.get(key.replace('~1', '/').replace('~0', '~')) if isinstance(target, dict) else None
        if isinstance(target, dict):
            return schema_arguments(value, target, root=root, depth=depth + 1)
        return value
    kind = schema.get('type')
    if isinstance(kind, list):
        matching = [k for k in kind if _has_type(value, k)]
        remaining = [k for k in kind if k != 'null']
        kind = matching[0] if matching else remaining[0] if len(remaining) == 1 else None
    if kind is None:
        branches = schema.get('anyOf', schema.get('oneOf'))
        if isinstance(branches, list):
            # Preserve an existing accepted primitive rather than coerce it
            # into a different branch. Ambiguous conversions remain errors.
            if any(isinstance(branch, dict) and _has_type(value, branch.get('type', '')) for branch in branches):
                return value
            candidates = [schema_arguments(value, branch, root=root, depth=depth + 1)
                          for branch in branches if isinstance(branch, dict) and branch.get('type') != 'null']
            changed = [candidate for candidate in candidates if type(candidate) is not type(value)]
            return changed[0] if len(changed) == 1 else value
        if 'properties' in schema:
            kind = 'object'
    if isinstance(value, str):
        if kind in ('object', 'array', 'integer', 'number', 'boolean'):
            try:
                try:
                    decoded = json.loads(value, object_pairs_hook=_unique_object)
                except json.JSONDecodeError:
                    decoded = json.loads(compatible_json(value), object_pairs_hook=_unique_object)
            except (ValueError, RecursionError):
                return value
            if _has_type(decoded, kind):
                value = decoded
    if kind == 'object' and isinstance(value, dict):
        properties = schema.get('properties', {})
        additional = schema.get('additionalProperties')
        return {key: schema_arguments(item, properties.get(key, additional), root=root, depth=depth + 1)
                if isinstance(properties.get(key, additional), dict) else item for key, item in value.items()}
    if kind == 'array' and isinstance(value, list):
        items = schema.get('items')
        prefix = schema.get('prefixItems', [])
        converted = []
        for i, item in enumerate(value):
            item_schema = prefix[i] if i < len(prefix) else items
            converted.append(schema_arguments(item, item_schema, root=root, depth=depth + 1)
                             if isinstance(item_schema, dict) else item)
        return converted
    return value
