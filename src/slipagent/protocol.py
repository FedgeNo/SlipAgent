"""One source for the response schema and strict local acceptance boundary.

The provider's structured-output mode is assistance, not validation. Local
checks run before reply acceptance or tool effects for every provider. The harness
supplies the active user prompt; responses need no task-memory acknowledgment.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any

from .capabilities import RequestProfile
from .types import ToolCall, ToolSpec
from .prompts import load_prompt, model_prompt_profile

RECORD_MAX_CHARS = 6000
COMPRESSED_FIELDS = (
    "previous_tool_responses_compressed",
    "user_prompt_compressed",
    "agent_response_compressed",
)
XML_CALL_TAGS = ("<function_calls>", "<tool_calls>")


class ResponseFormatError(ValueError):
    """The response cannot be displayed or executed."""

    def __init__(self, message: str, *, excerpt: str = "") -> None:
        super().__init__(message)
        self.excerpt = excerpt


class ResponseRecord:
    """Accepted legacy reply/actions with independent summaries."""

    def __init__(self, text: str, calls: list[ToolCall], previous: str, user: str, agent: str) -> None:
        self.text = text
        self.calls = calls
        self.previous_tool_responses_compressed = previous
        self.user_prompt_compressed = user
        self.agent_response_compressed = agent


class AgentResponse:
    """Reply and actions, independent of background compaction."""

    def __init__(self, text: str, calls: list[ToolCall], suggestions: list[str] | None = None) -> None:
        self.text = text
        self.calls = calls
        self.suggestions = suggestions or []


def agent_response_format(*, native_tools: bool) -> dict[str, Any]:
    """A small optional schema for endpoints that actually enforce schemas."""
    result = response_format(False, native_tools=native_tools)
    schema = result["json_schema"]["schema"]
    schema["properties"] = {key: value for key, value in schema["properties"].items()
                            if key in {"response", "tool_calls"}}
    schema["required"] = list(schema["properties"])
    schema["properties"]["suggestions"] = {
        "type": "array", "maxItems": 3, "items": {"type": "string", "minLength": 1},
        "description": json.loads(load_prompt("response-fields.json"))["suggestions"],
    }
    # Strict schemas require every property; suggestions must remain omittable.
    result["json_schema"]["strict"] = False
    return result


def model_response_format(model: str, capabilities: RequestProfile | None, *,
                          native_tools: bool) -> dict[str, Any] | None:
    """Select the working reply constraint for both prompt and API construction."""
    if not model_prompt_profile(model).get("structured_output", True):
        return None
    if capabilities is not None and capabilities.format == "json_schema":
        return agent_response_format(native_tools=native_tools)
    if not native_tools:
        return {"type": "json_object"}
    return None


def parse_agent_response(text: str, native_calls: list[ToolCall], *, native_tools: bool = True,
                         json_response: bool = False, tools: list[ToolSpec] | None = None,
                         xml_tool_calls: bool = False) -> AgentResponse:
    """Detect each response independently; request hints do not restrict carriers.

    The legacy keyword arguments remain accepted for reload compatibility.
    Tool schemas describe XML argument types, never a model's response format.
    """
    result = _detect_response(text, tools or [], xml_tool_calls=xml_tool_calls)
    result.calls = combine_calls(normalize_calls([call.to_record() for call in native_calls]), result.calls)
    if not isinstance(result.text, str):
        raise ResponseFormatError("response must be a string.")
    if not result.text.strip() and not result.calls:
        raise ResponseFormatError("Provide a nonempty response when no tool calls are requested.")
    return result


def combine_calls(*sources: list[ToolCall]) -> list[ToolCall]:
    """Combine carriers in order, preserving repeated calls within each carrier.

    Mirrored calls match by ID or by name and exact JSON arguments. Multiplicity
    is preserved, so two intentional equal calls in a batch still execute twice.
    """
    result: list[ToolCall] = []
    def signature(call: ToolCall) -> tuple[str, str]:
        return call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=True)
    for source in sources:
        unmatched = list(result)
        additions = []
        for call in source:
            same_id = next((prior for prior in result if prior.id == call.id), None)
            if same_id is not None and signature(same_id) != signature(call):
                raise ResponseFormatError("Conflicting tool calls share the same ID.")
            prior = next((prior for prior in unmatched if prior.id == call.id), None)
            if prior is None:
                prior = next((prior for prior in unmatched if signature(prior) == signature(call)), None)
            if prior is not None:
                unmatched.remove(prior)
            else:
                additions.append(call)
        result.extend(additions)
    return normalize_calls([call.to_record() for call in result])


def _detect_response(text: str, tools: list[ToolSpec], *, xml_tool_calls: bool = False) -> AgentResponse:
    text = text.removeprefix("\ufeff")
    candidate = text.strip().removeprefix("\ufeff").strip()
    if xml_tool_calls:
        xml_fence = re.fullmatch(r"```xml[ \t]*\r?\n(.*)\r?\n```", candidate, re.DOTALL | re.IGNORECASE)
        if xml_fence:
            candidate = xml_fence[1].strip()
            if candidate.startswith(XML_CALL_TAGS + ("<tool_call>", "<function_call>")):
                return AgentResponse("", _tagged_calls(candidate, tools))
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n```", candidate, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence[1].strip()
    # Decode envelopes before inspecting tags, so code strings stay inert.
    if candidate.startswith("{") or re.match(r"```(?:json)?[ \t]*\r?\n\s*\{", candidate, re.IGNORECASE):
        value, tagged = _response_object(candidate, tools=tools, xml_tool_calls=xml_tool_calls)
        if isinstance(value, dict):
            if set(value) & {"response", "tool_calls", "function_call", *COMPRESSED_FIELDS}:
                allowed = {"response", "tool_calls", "function_call", "suggestions", *COMPRESSED_FIELDS}
                suggestions = value.get("suggestions", [])
                if not isinstance(suggestions, list) or len(suggestions) > 3 or any(
                        not isinstance(item, str) or not item.strip() for item in suggestions):
                    raise ResponseFormatError("suggestions must be an array of up to three nonempty prompt strings.")
                if all(field in value for field in COMPRESSED_FIELDS):
                    allowed.add("task")
                if set(value) - allowed:
                    raise ResponseFormatError("Unexpected response envelope fields.")
                if any(field in value for field in COMPRESSED_FIELDS):
                    calls = combine_calls(normalize_calls(value.get("tool_calls")),
                                          normalize_calls(value.get("function_call")), tagged)
                    legacy_value = {key: item for key, item in value.items() if key not in {"function_call", "suggestions"}}
                    legacy_value["tool_calls"] = [call.to_record() for call in calls]
                    legacy = _legacy_response(legacy_value, bool(value.get(COMPRESSED_FIELDS[0])), [], [])
                    return AgentResponse(legacy.text, legacy.calls, suggestions)
                reply = value.get("response", "")
                if xml_tool_calls and isinstance(reply, str):
                    embedded = _detect_response(reply, tools, xml_tool_calls=True)
                    reply = embedded.text
                    tagged = combine_calls(tagged, embedded.calls)
                calls = combine_calls(normalize_calls(value.get("tool_calls")),
                                      normalize_calls(value.get("function_call")), tagged)
                return AgentResponse(reply, calls, suggestions)
    # Markdown examples are not call carriers. Only top-level tags are parsed.
    if not fence:
        offset = 0
        in_fence = ""
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip(" \t")
            marker = re.match(r"(`{3,}|~{3,})", stripped)
            if marker:
                if not in_fence:
                    in_fence = marker[1]
                elif marker[1][0] == in_fence[0] and len(marker[1]) >= len(in_fence):
                    in_fence = ""
            elif (not in_fence and len(line) - len(stripped) < 4 and "\t" not in line[:len(line) - len(stripped)]
                  and stripped.startswith(("<tool_call>", "<function_call>") + (XML_CALL_TAGS if xml_tool_calls else ()))):
                return AgentResponse(text[:offset].rstrip(), _tagged_calls(text[offset:], tools))
            offset += len(line)
    return AgentResponse(text, [])


def _tagged_calls(text: str, tools: list[ToolSpec]) -> list[ToolCall]:
    remaining = text.strip()
    raw: list[dict[str, Any]] = []
    decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant)
    while remaining:
        xml_tag = next((tag for tag in XML_CALL_TAGS if remaining.startswith(tag)), None)
        if xml_tag is not None:
            closing_tag = "</" + xml_tag[1:]
            block, closing, remaining = remaining.partition(closing_tag)
            if not closing:
                raise ResponseFormatError(f"Missing closing {closing_tag}.")
            raw.extend(_xml_calls(block + closing, tools))
            remaining = remaining.lstrip()
            continue
        opening = re.match(r"<(tool_call|function_call)>\s*", remaining)
        if opening is None:
            raise ResponseFormatError("Expected a complete tagged tool call without trailing prose.")
        remaining = remaining[opening.end():]
        fence = re.match(r"```(?:json)?[ \t]*\r?\n", remaining, re.IGNORECASE)
        if fence:
            remaining = remaining[fence.end():].lstrip()
        if remaining.startswith("<answer>"):
            answer, closing, remaining = remaining[len("<answer>"):].partition("</answer>")
            if not closing:
                boundary = re.search(r"</(?:tool_call|function_call)>|<(?:tool_call|function_call)>", answer)
                remaining = answer[boundary.start():] if boundary else ""
                answer = answer[:boundary.start()] if boundary else answer
            answer = answer[2:] if answer.startswith("\r\n") else answer.removeprefix("\n")
            answer = answer[:-2] if answer.endswith("\r\n") else answer.removesuffix("\n")
            raw.append({"name": "answer", "arguments": {"text": answer}})
            remaining = remaining.lstrip()
        elif remaining.startswith("<function="):
            call, remaining = _qwen_call(remaining, tools)
            raw.append(call)
        else:
            try:
                _, end = decoder.raw_decode(remaining)
            except (ValueError, RecursionError) as exc:
                raise ResponseFormatError("Expected complete JSON inside tool-call tags.") from exc
            calls = normalize_calls(_decode(remaining[:end]))
            raw.extend(call.to_record() for call in calls)
            remaining = remaining[end:].lstrip()
        if fence:
            if not remaining.startswith("```"):
                raise ResponseFormatError("Missing closing code fence in tagged call.")
            remaining = remaining[3:].lstrip()
        closing = f"</{opening[1]}>"
        if remaining.startswith(closing):
            remaining = remaining[len(closing):].lstrip()
        elif remaining and not remaining.startswith(("<tool_call>", "<function_call>")):
            raise ResponseFormatError(f"Missing closing {closing}.")
    return normalize_calls(raw)


def _xml_calls(text: str, tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Decode invoke/parameter XML and tool_calls/call wrappers without repair.

    A call wrapper contains exactly one invoke; missing tool names or broken
    nesting reject the complete batch before any native or embedded calls run.
    Parameter strings retain their whitespace and decoded XML entities.
    """
    from xml.etree import ElementTree
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ResponseFormatError(f"Invalid tool-call XML: {exc}") from exc
    calls = []
    for element in root:
        invoke = element
        if root.tag == "tool_calls":
            if element.tag != "call" or len(element) != 1 or (element.text or "").strip():
                raise ResponseFormatError("Expected each call in tool_calls to contain exactly one invoke element.")
            invoke = element[0]
            if (invoke.tail or "").strip():
                raise ResponseFormatError("Unexpected text after invoke in call.")
        if invoke.tag != "invoke":
            raise ResponseFormatError(f"Expected an invoke element in {root.tag}.")
        name = invoke.get("name")
        properties: dict[str, Any] = next((tool.parameters.get("properties", {}) for tool in tools if tool.name == name), {})
        arguments = {}
        for parameter in invoke:
            key = parameter.get("name")
            if parameter.tag != "parameter" or not key or key in arguments or len(parameter):
                raise ResponseFormatError("Expected named, nonduplicated text parameters in invoke.")
            value = parameter.text or ""
            kind = properties.get(key, {}).get("type")
            if kind == "string" or isinstance(kind, list) and "string" in kind:
                arguments[key] = value
            else:
                try:
                    arguments[key] = _decode(value)
                except ResponseFormatError:
                    arguments[key] = value
        calls.append({"id": invoke.get("id"), "name": name, "arguments": arguments})
    return calls


def _qwen_call(text: str, tools: list[ToolSpec]) -> tuple[dict[str, Any], str]:
    function = re.match(r"<function=([^<>\s]+)>\s*", text)
    if function is None:
        raise ResponseFormatError("Invalid Qwen3-Coder function tag.")
    name = function[1]
    properties: dict[str, Any] = next((tool.parameters.get("properties", {}) for tool in tools if tool.name == name), {})
    remaining = text[function.end():]
    arguments: dict[str, Any] = {}
    while remaining and not remaining.startswith(("</function>", "</tool_call>", "</function_call>", "<tool_call>", "<function_call>")):
        parameter = re.match(r"<parameter=([^<>\s]+)>", remaining)
        if parameter is None or parameter[1] in arguments:
            raise ResponseFormatError("Invalid or duplicate Qwen3-Coder parameter.")
        value, closing, remaining = remaining[parameter.end():].partition("</parameter>")
        if not closing:
            boundary = re.search(r"</(?:function|tool_call|function_call)>|<(?:tool_call|function_call)>", value)
            remaining = value[boundary.start():] if boundary else ""
            value = value[:boundary.start()] if boundary else value
        # Qwen's framing adds one newline at each edge; preserve code whitespace.
        value = value[2:] if value.startswith("\r\n") else value.removeprefix("\n")
        value = value[:-2] if value.endswith("\r\n") else value.removesuffix("\n")
        schema = properties.get(parameter[1], {})
        kind = schema.get("type", "string")
        arguments[parameter[1]] = value if kind == "string" or isinstance(kind, list) and "string" in kind else _decode(value)
        remaining = remaining.lstrip()
    return {"name": name, "arguments": arguments}, remaining.removeprefix("</function>").lstrip()


def response_format(has_previous_results: bool, *, native_tools: bool = False) -> dict[str, Any]:
    """Describe legacy reply, actions, and three summaries."""
    descriptions = json.loads(load_prompt("response-fields.json"))
    properties: dict[str, Any] = {
        "response": {
            "type": "string",
            "description": (
                descriptions["response"]
            ),
        },
        "tool_calls": {
            "type": "array", "description": descriptions["tool_calls"],
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "name", "arguments"],
                "properties": {
                    "id": {"type": "string", "minLength": 1, "description": descriptions["call_id"]},
                    "name": {"type": "string", "minLength": 1, "description": descriptions["tool_name"]},
                    "arguments": {"type": "string", "description": descriptions["arguments"]},
                },
            },
        },
        "previous_tool_responses_compressed": {
            "type": "string", "maxLength": RECORD_MAX_CHARS,
            "description": descriptions["previous_results"],
            **({"minLength": 1} if has_previous_results else {"enum": [""]}),
        },
        "user_prompt_compressed": {
            "type": "string", "minLength": 1, "maxLength": RECORD_MAX_CHARS,
            "description": descriptions["user_request"],
        },
        "agent_response_compressed": {
            "type": "string", "minLength": 1, "maxLength": RECORD_MAX_CHARS,
            "description": descriptions["current_step"],
        },
    }
    if native_tools:
        del properties["tool_calls"]
    return {"type": "json_schema", "json_schema": {
        "name": "slipagent_response", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "required": list(properties), "properties": properties},
    }}


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResponseFormatError(f"Duplicate JSON field: {key}.")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise ResponseFormatError(f"Non-finite JSON value: {value}.")


def _decode(text: str) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                try:
                    item.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ResponseFormatError("JSON strings must contain valid Unicode, without unpaired surrogates.") from exc
            elif isinstance(item, float) and not math.isfinite(item):
                raise ResponseFormatError("JSON numbers must be finite.")
            elif isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        return value
    except json.JSONDecodeError as exc:
        start, end = max(0, exc.pos - 120), min(len(text), exc.pos + 120)
        raise ResponseFormatError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno} (character {exc.pos}): {exc.msg}. "
            "Expected complete valid JSON.",
            excerpt=json.dumps({"character_offset": start, "invalid_text": text[start:end]}, indent=2),
        ) from exc
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, ResponseFormatError):
            raise
        raise ResponseFormatError("Expected complete valid JSON.") from exc


def normalize_calls(raw: Any) -> list[ToolCall]:
    """Normalize explicit calls, never infer actions from natural-language text.

    Native OpenAI calls, the historical flat envelope, and tool_use blocks share
    one acceptance boundary. Argument strings are decoded exactly; repairing
    truncated JSON or guessing quotes could change code, paths, or commands.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ResponseFormatError("tool_calls must be an array of calls.")
    calls = []
    ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ResponseFormatError(f"tool_calls[{index}] must be an object.")
        function = item.get("function")
        if not isinstance(function, dict):
            function = item
        name = function.get("name") or item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ResponseFormatError("Tool name must be a nonempty string.")
        _decode(json.dumps(name, ensure_ascii=True))
        # Prefer nested arguments, then flat aliases; provider metadata is not
        # part of the executable call contract.
        arguments: Any = next((source[key] for source in (function, item)
                          for key in ("arguments", "input", "parameters")
                          if source.get(key) is not None), {})
        try:
            arguments = _decode(arguments) if isinstance(arguments, str) and arguments.strip() else arguments
            if arguments is None or isinstance(arguments, str) and not arguments.strip():
                arguments = {}
            if not isinstance(arguments, dict):
                raise ResponseFormatError("Tool arguments must decode to an object.")
            # Preserve valid JSON values without modifying tool payloads.
            _decode(json.dumps(arguments, ensure_ascii=True))
        except ResponseFormatError as exc:
            raise ResponseFormatError(f"tool_calls[{index}].arguments: {exc}", excerpt=exc.excerpt) from exc
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
            call_id = "call_" + uuid.uuid4().hex
        _decode(json.dumps(call_id, ensure_ascii=True))
        ids.add(call_id)
        calls.append(ToolCall(call_id, name, arguments))
    return calls


def merge_call_sources(*sources: list[ToolCall]) -> list[ToolCall]:
    """Legacy merger: accept one batch or identical alternate representations.

    Native IDs take precedence when a provider repeats its batch in content.
    Sequence and multiplicity matter: two intentional identical calls in one
    batch stay two calls. Disagreeing batches are ambiguous and must be retried.
    """
    nonempty = [calls for calls in sources if calls]
    if not nonempty:
        return []
    first = nonempty[0]
    def signature(calls: list[ToolCall]) -> list[tuple[str, str]]:
        # Python considers True == 1; JSON arguments do not. Compare their
        # canonical encodings rather than silently merging different values.
        return [(call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=True)) for call in calls]
    if any(signature(calls) != signature(first) for calls in nonempty[1:]):
        raise ResponseFormatError("Conflicting tool-call batches in separate response fields; return one consistent batch.")
    return first


def _response_object(text: str, *, tools: list[ToolSpec] | None = None, xml_tool_calls: bool = False) -> tuple[Any, list[ToolCall]]:
    """Read a complete record plus optional explicitly delimited JSON calls.

    Only whole code fences are unwrapped. We do not scan prose, the response
    string, reasoning, or quoted examples for something that looks executable.
    JSONDecoder finds object boundaries, so braces/tags in code strings cannot
    terminate a record or a call early.
    """
    remaining = text.strip().removeprefix("\ufeff").strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", remaining, re.DOTALL | re.IGNORECASE)
    if fence:
        remaining = fence[1].strip()
    decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant)
    record: Any = None
    calls: list[ToolCall] = []
    while remaining:
        if xml_tool_calls and remaining.startswith(XML_CALL_TAGS):
            calls.extend(_tagged_calls(remaining, tools or []))
            remaining = ""
            break
        if remaining.startswith(("<tool_call>", "<function_call>")) and any(tag in remaining for tag in ("<function=", "<answer>")):
            calls.extend(_tagged_calls(remaining, tools or []))
            remaining = ""
            break
        tag = next((tag for tag in ("tool_call", "function_call") if remaining.startswith(f"<{tag}>")), None)
        if tag:
            remaining = remaining[len(tag) + 2:].lstrip()
        block_fence = re.match(r"```(?:json)?[ \t]*\r?\n", remaining, re.IGNORECASE)
        if block_fence:
            remaining = remaining[block_fence.end():].lstrip()
        try:
            _, end = decoder.raw_decode(remaining)
        except (ValueError, RecursionError):
            # Use the shared decoder for a bounded, precise diagnostic.
            _decode(remaining)
            raise ResponseFormatError("Expected a complete response JSON object.")
        value = _decode(remaining[:end])
        remaining = remaining[end:].lstrip()
        if block_fence:
            if not remaining.startswith("```"):
                raise ResponseFormatError("Missing closing code fence after response JSON.")
            remaining = remaining[3:].lstrip()
        if tag:
            closing = f"</{tag}>"
            if remaining.startswith(closing):
                remaining = remaining[len(closing):].lstrip()
            elif remaining and not remaining.startswith(("<tool_call>", "<function_call>")):
                raise ResponseFormatError(f"Missing closing {closing} after tool-call JSON.")
            calls.extend(normalize_calls(value))
        elif record is None:
            if not isinstance(value, dict):
                raise ResponseFormatError("Response must be one JSON object.")
            record = value
        else:
            raise ResponseFormatError("Multiple response JSON objects are ambiguous; return one record.")
    if record is None:
        raise ResponseFormatError("Missing response JSON object with the required compressed fields and task.")
    # IDs must also be unique across separately delimited calls.
    if len({call.id for call in calls}) != len(calls):
        raise ResponseFormatError("Tool IDs must be unique within the response.")
    return record, calls


def parse_response(text: str, has_previous_results: bool, *, native_calls: list[ToolCall] | None = None) -> ResponseRecord:
    try:
        return _parse_response(text, has_previous_results, native_calls or [])
    except ResponseFormatError as exc:
        if not exc.excerpt:
            # A stateless retry needs evidence of the rejected output. This is
            # bounded diagnostic data, never an executable assistant message.
            exc.excerpt = json.dumps({"invalid_text_prefix": text[:1500], "total_characters": len(text)}, indent=2)
        raise


def _parse_response(text: str, has_previous_results: bool, native_calls: list[ToolCall]) -> ResponseRecord:
    value, tagged_calls = _response_object(text)
    return _legacy_response(value, has_previous_results, native_calls, tagged_calls)


def _legacy_response(
    value: Any, has_previous_results: bool, native_calls: list[ToolCall], tagged_calls: list[ToolCall],
) -> ResponseRecord:
    """Adapt a decoded earlier response envelope without serializing it again."""
    expected = {"response", *COMPRESSED_FIELDS}
    if not isinstance(value, dict):
        raise ResponseFormatError("Response must be one JSON object.")
    missing = expected - value.keys()
    if missing:
        raise ResponseFormatError("Missing required field(s): " + ", ".join(sorted(missing)) + ".")
    if value.keys() - {"tool_calls", "function_call", "task"} != expected:
        raise ResponseFormatError("Unexpected response fields; follow the supplied schema.")
    if not isinstance(value["response"], str):
        raise ResponseFormatError("response must be a string.")
    for field in COMPRESSED_FIELDS:
        summary = value[field]
        if not isinstance(summary, str) or len(summary) > RECORD_MAX_CHARS:
            raise ResponseFormatError(f"{field} must be a string under {RECORD_MAX_CHARS + 1} characters.")
        if (field != "previous_tool_responses_compressed" or has_previous_results) and not summary.strip():
            raise ResponseFormatError(f"{field} must be nonempty.")
    if not has_previous_results and value["previous_tool_responses_compressed"] != "":
        raise ResponseFormatError("previous_tool_responses_compressed must be empty when no previous results were supplied.")
    if sum(len(value[field]) for field in COMPRESSED_FIELDS) > RECORD_MAX_CHARS:
        raise ResponseFormatError(f"Keep the three compressed fields within {RECORD_MAX_CHARS} characters in total.")
    calls = merge_call_sources(
        normalize_calls([call.to_record() for call in native_calls]),
        normalize_calls(value.get("tool_calls")), normalize_calls(value.get("function_call")), tagged_calls,
    )
    if not calls and not value["response"].strip():
        raise ResponseFormatError("A final response must contain an answer.")
    return ResponseRecord(value["response"], calls, value["previous_tool_responses_compressed"],
                          value["user_prompt_compressed"], value["agent_response_compressed"])
