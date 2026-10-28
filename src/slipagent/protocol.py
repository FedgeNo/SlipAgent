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
from .toolcompat import compatible_json, schema_arguments, tool_name
from wcwidth import iter_graphemes

RECORD_MAX_CHARS = 6000
COMPRESSED_FIELDS = (
    "previous_tool_responses_compressed",
    "user_prompt_compressed",
    "agent_response_compressed",
)
XML_CALL_TAGS = ("<function_calls>", "<tool_calls>")
TOOL_MARKERS = ("[TOOL_CALLS]", "<｜tool▁calls▁begin｜>", "<|tool_calls_section_begin|>")
CALL_OPENING = re.compile(r"<(tool_call|function_call|function_calls|tool_calls)(?=[\s>])")
FUNCTION_OPENING = "<function="
GLM_CALL_NAME = re.compile(r"([A-Za-z_][\w.\-]*)\s*(?=<arg_key>|</(?:tool_call|function_call)\s*>)")
# Call fragments inside prose: a reply carrying one was meant to act.
BROKEN_CALL = re.compile(r'<(?:tool_call|function_call)(?=[\s>])|"tool_calls"\s*:\s*\[\s*\{')


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


def session_title_value(value: Any) -> str:
    """A displayable title; an unusable one is dropped rather than failing a cosmetic field."""
    if not isinstance(value, str):
        return ""
    clusters = list(iter_graphemes(" ".join(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value).split())))
    return "".join(clusters) if len(clusters) <= 60 else "".join(clusters[:59]).rstrip() + "…"


class AgentResponse:
    """Reply and actions, independent of background compaction."""

    def __init__(self, text: str, calls: list[ToolCall], suggestions: list[str] | None = None,
                 session_title: str = "") -> None:
        self.text = text
        self.calls = calls
        self.suggestions = suggestions or []
        self.session_title = session_title


class ResponseFormats:
    """Response schemas sent to providers and described in prompts."""

    def agent_response_format(self, *, native_tools: bool) -> dict[str, Any]:
        """A small optional schema for endpoints that actually enforce schemas."""
        result = self.response_format(False, native_tools=native_tools)
        schema = result["json_schema"]["schema"]
        schema["properties"] = {key: value for key, value in schema["properties"].items()
                                if key in {"response", "tool_calls"}}
        schema["required"] = list(schema["properties"])
        descriptions = json.loads(load_prompt("response-fields.json"))
        schema["properties"]["suggestions"] = {
            "type": "array", "maxItems": 3, "items": {"type": "string", "minLength": 1},
            "description": descriptions["suggestions"],
        }
        schema["properties"]["session_title"] = {"type": "string", "description": descriptions["session_title"]}
        # Strict schemas require every property; suggestions and the title must remain omittable.
        result["json_schema"]["strict"] = False
        return result

    def model_response_format(self, model: str, capabilities: RequestProfile | None, *,
                              native_tools: bool) -> dict[str, Any] | None:
        """Select the working reply constraint for both prompt and API construction."""
        profile = model_prompt_profile(model)
        if not profile.get("structured_output", True) or profile.get("tool_channel") == "glm_xml":
            return None
        if capabilities is not None and capabilities.format == "json_schema":
            return self.agent_response_format(native_tools=native_tools)
        if not native_tools and (capabilities is None or capabilities.format == "json_object"):
            return {"type": "json_object"}
        return None

    def response_format(self, has_previous_results: bool, *, native_tools: bool = False) -> dict[str, Any]:
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


# Entry points for other modules; plugins override the methods above.
def agent_response_format(*, native_tools: bool) -> dict[str, Any]:
    return ResponseFormats().agent_response_format(native_tools=native_tools)


def model_response_format(model: str, capabilities: RequestProfile | None, *,
                          native_tools: bool) -> dict[str, Any] | None:
    return ResponseFormats().model_response_format(model, capabilities, native_tools=native_tools)


def response_format(has_previous_results: bool, *, native_tools: bool = False) -> dict[str, Any]:
    return ResponseFormats().response_format(has_previous_results, native_tools=native_tools)


class ResponseParser:
    """Detect replies and tool calls in every supported carrier and validate them; each format is one method."""

    def parse_agent_response(self, text: str, native_calls: list[ToolCall], *, native_tools: bool = True,
                             json_response: bool = False, tools: list[ToolSpec] | None = None,
                             xml_tool_calls: bool = False) -> AgentResponse:
        """Detect each response independently; request hints do not restrict carriers.

        The legacy keyword arguments remain accepted for reload compatibility.
        Tool schemas describe XML argument types, never a model's response format.
        """
        result = self._detect_response(text, tools or [], xml_tool_calls=xml_tool_calls)
        available = {tool.name: tool for tool in tools or []}
        def compatible(calls: list[ToolCall]) -> list[ToolCall]:
            converted = []
            for call in calls:
                name = tool_name(call.name, list(available))
                arguments = schema_arguments(call.arguments, available[name].parameters) if name in available else call.arguments
                converted.append(ToolCall(call.id, name, arguments))
            return converted
        combined = self.combine_calls(compatible(self.normalize_calls([call.to_record() for call in native_calls])), compatible(result.calls))
        # Name and schema conversion can change values; validate the final batch once.
        result.calls = self.normalize_calls([call.to_record() for call in combined])
        if not isinstance(result.text, str):
            raise ResponseFormatError("response must be a string.")
        if not result.text.strip() and not result.calls:
            raise ResponseFormatError("Provide a nonempty response when no tool calls are requested.")
        return result

    def combine_calls(self, *sources: list[ToolCall]) -> list[ToolCall]:
        """Combine carriers in order, preserving repeated calls within each carrier.

        Mirrored calls match by ID or by name and exact JSON arguments. Multiplicity
        is preserved, so two intentional equal calls in a batch still execute twice.
        Sources are already normalized; only IDs need to stay unique across them.
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
        unique: list[ToolCall] = []
        ids: set[str] = set()
        for call in result:
            if call.id in ids:
                call = ToolCall("call_" + uuid.uuid4().hex, call.name, call.arguments)
            ids.add(call.id)
            unique.append(call)
        return unique

    def _detect_response(self, text: str, tools: list[ToolSpec], *, xml_tool_calls: bool = False) -> AgentResponse:
        text = text.removeprefix("\ufeff")
        # Reasoning is not an executable carrier. Only a complete leading block
        # can be separated from the answer; unfinished thoughts remain terminal.
        original = text
        if text.lstrip().startswith("</think>") and not text.startswith(("    ", "\t")):
            text = text.lstrip()[len("</think>"):].lstrip("\r\n")
            if not text.strip():
                return AgentResponse(original, [])
        thought_body = text.lstrip()
        while thought_body.startswith("<think>") and not text.startswith(("    ", "\t")):
            _, closing, following = thought_body.partition("</think>")
            if not closing or not following.strip():
                return AgentResponse(original, [])
            text = following.lstrip("\r\n")
            thought_body = text.lstrip()
        candidate = text.strip().removeprefix("\ufeff").strip()
        if xml_tool_calls:
            xml_fence = re.fullmatch(r"```xml[ \t]*\r?\n(.*)\r?\n```", candidate, re.DOTALL | re.IGNORECASE)
            if xml_fence:
                candidate = xml_fence[1].strip()
                if CALL_OPENING.match(candidate):
                    return AgentResponse("", self._tagged_calls(candidate, tools))
        fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n```", candidate, re.DOTALL | re.IGNORECASE)
        if fence:
            candidate = fence[1].strip()
        if candidate.startswith(TOOL_MARKERS) and (fence or not text.startswith(("    ", "\t"))):
            return AgentResponse("", self._marker_calls(candidate))
        if candidate.startswith("[") and self._json_call_start(candidate) and (fence or not text.startswith(("    ", "\t"))):
            return AgentResponse("", self.normalize_calls(self._decode(candidate)))
        # Decode envelopes before inspecting tags, so code strings stay inert.
        # A reply that opens with a code block and continues with prose carries an
        # envelope only if that block is one; otherwise the block is an example.
        leading_block = fence is None and re.match(r"```(?:json)?[ \t]*\r?\n\s*\{", candidate, re.IGNORECASE) is not None
        if leading_block and not self._leading_block_is_envelope(candidate):
            leading_block = False
        if (candidate.startswith("{") and (fence or not text.startswith(("    ", "\t")))) or leading_block:
            value, tagged = self._response_object(candidate, tools=tools, xml_tool_calls=xml_tool_calls)
            if self._call_value(value):
                return AgentResponse("", self.combine_calls(self.normalize_calls(value), tagged))
            if isinstance(value, dict):
                if set(value) & {"response", "tool_calls", "function_call", *COMPRESSED_FIELDS}:
                    allowed = {"response", "tool_calls", "function_call", "suggestions", "session_title", *COMPRESSED_FIELDS}
                    title = session_title_value(value.get("session_title"))
                    suggestions = value.get("suggestions", [])
                    if not isinstance(suggestions, list) or len(suggestions) > 3 or any(
                            not isinstance(item, str) or not item.strip() for item in suggestions):
                        raise ResponseFormatError("suggestions must be an array of up to three nonempty prompt strings.")
                    if all(field in value for field in COMPRESSED_FIELDS):
                        allowed.add("task")
                    if set(value) - allowed:
                        raise ResponseFormatError("Unexpected response envelope fields.")
                    if any(field in value for field in COMPRESSED_FIELDS):
                        calls = self.combine_calls(self.normalize_calls(value.get("tool_calls")),
                                              self.normalize_calls(value.get("function_call")), tagged)
                        legacy_value = {key: item for key, item in value.items()
                                        if key not in {"function_call", "suggestions", "session_title"}}
                        legacy_value["tool_calls"] = [call.to_record() for call in calls]
                        legacy = self._legacy_response(legacy_value, bool(value.get(COMPRESSED_FIELDS[0])), [], [])
                        return AgentResponse(legacy.text, legacy.calls, suggestions, title)
                    reply = value.get("response", "")
                    if isinstance(reply, str) and (xml_tool_calls or reply.lstrip().startswith(TOOL_MARKERS)):
                        embedded = self._detect_response(reply, tools, xml_tool_calls=xml_tool_calls)
                        reply = embedded.text
                        tagged = self.combine_calls(tagged, embedded.calls)
                    calls = self.combine_calls(self.normalize_calls(value.get("tool_calls")),
                                          self.normalize_calls(value.get("function_call")), tagged)
                    return AgentResponse(reply, calls, suggestions, title)
                if not tagged:
                    return AgentResponse(text, [])
        # Markdown examples are not call carriers. Only top-level tags are parsed.
        if not fence:
            offset = 0
            in_fence = ""
            literal_until = 0
            for line in text.splitlines(keepends=True):
                if offset < literal_until:
                    offset += len(line)
                    continue
                stripped = line.lstrip(" \t")
                marker = re.match(r"(`{3,}|~{3,})", stripped)
                if marker:
                    if not in_fence:
                        trailing = self._trailing_fence_calls(text[offset:], tools, xml_tool_calls=xml_tool_calls)
                        if trailing is not None:
                            reply = "\n".join(part for part in (text[:offset].rstrip(), trailing.text) if part)
                            return AgentResponse(reply, trailing.calls, trailing.suggestions, trailing.session_title)
                        in_fence = marker[1]
                    elif marker[1][0] == in_fence[0] and len(marker[1]) >= len(in_fence):
                        in_fence = ""
                elif not in_fence and len(line) - len(stripped) < 4 and "\t" not in line[:len(line) - len(stripped)]:
                    if not stripped.startswith(">") and BROKEN_CALL.search(re.sub(r"`[^`\n]*`", "", stripped)) and not (
                            CALL_OPENING.match(stripped) or stripped.startswith(("{", "[", FUNCTION_OPENING, *TOOL_MARKERS))):
                        inline = re.search(r"<tool_call(?=[\s>])", line)
                        # An odd count of backticks before the tag places it inside inline code.
                        if inline is not None and line[:inline.start()].count("`") % 2 == 0:
                            start = offset + inline.start()
                            try:
                                return AgentResponse(text[:start].rstrip(), self._tagged_calls(text[start:], tools))
                            except ResponseFormatError:
                                pass
                        raise ResponseFormatError(
                            "The reply contains a tool call that could not be parsed. Resend the complete "
                            "tool calls in the required response format; no tools from this reply ran.")
                    opening = CALL_OPENING.match(stripped)
                    if stripped.startswith(FUNCTION_OPENING):
                        functions = []
                        remaining = text[offset:].strip()
                        while remaining.startswith(FUNCTION_OPENING):
                            call, remaining = self._qwen_call(remaining, tools)
                            functions.append(call)
                        if remaining:
                            raise ResponseFormatError("Unexpected text after function calls.")
                        return AgentResponse(text[:offset].rstrip(), self.normalize_calls(functions))
                    if opening and (xml_tool_calls or opening[1] in {"tool_call", "function_call"}):
                        return AgentResponse(text[:offset].rstrip(), self._tagged_calls(text[offset:], tools))
                    if stripped.startswith(TOOL_MARKERS):
                        return AgentResponse(text[:offset].rstrip(), self._marker_calls(text[offset:].strip()))
                    if offset and stripped.startswith(("{", "[")) and self._json_call_start(text[offset:].strip()):
                        embedded = self._detect_response(text[offset:], tools, xml_tool_calls=xml_tool_calls)
                        reply = "\n".join(part for part in (text[:offset].rstrip(), embedded.text) if part)
                        return AgentResponse(reply, embedded.calls, embedded.suggestions, embedded.session_title)
                    if stripped.startswith(("{", "[")):
                        # A literal JSON value is one block. Never inspect its
                        # nested objects or arrays as independent executable calls.
                        tail = text[offset:].lstrip()
                        try:
                            _, end = self._json_prefix(tail)
                        except (ValueError, RecursionError):
                            pass
                        else:
                            literal_until = len(text) - len(tail) + end
                offset += len(line)
        return AgentResponse(text, [])

    def _leading_block_is_envelope(self, text: str) -> bool:
        """Whether the code block opening a longer reply is a response record or a call."""
        block = re.match(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```", text, re.DOTALL | re.IGNORECASE)
        if block is None:
            return True  # An unterminated block keeps the strict envelope path and its diagnostic.
        return self._json_call_start(block[1].strip())

    def _trailing_fence_calls(self, text: str, tools: list[ToolSpec], *, xml_tool_calls: bool) -> AgentResponse | None:
        """Calls in a JSON code block that ends the reply, naming at least one registered tool.

        Models without a native tool channel often explain an action, then write
        its call in a closing code block. Blocks followed by more prose stay examples.
        """
        block = re.fullmatch(r"(`{3,}|~{3,})[ \t]*(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*\1[ \t]*",
                             text.strip(), re.DOTALL | re.IGNORECASE)
        if block is None:
            return None
        body = block[2].strip()
        if not body.startswith(("{", "[")) or not self._json_call_start(body):
            return None
        found = self._detect_response(body, tools, xml_tool_calls=xml_tool_calls)
        names = [tool.name for tool in tools]
        if not found.calls or names and not any(tool_name(call.name, names) in names for call in found.calls):
            return None
        return found

    def _call_value(self, value: Any) -> bool:
        """Require explicit call fields before treating bare JSON as actions."""
        if isinstance(value, list):
            return any(self._call_value(item) for item in value)
        if not isinstance(value, dict):
            return False
        if set(value) - {"id", "type", "index", "metadata", "function", "name", "arguments", "input", "parameters"}:
            return False
        function = value.get("function", value)
        return (isinstance(function, dict) and "name" in function
                and any(key in function for key in ("arguments", "input", "parameters")))

    def _json_prefix(self, text: str) -> tuple[Any, int]:
        """Locate a complete JSON structure in original text before encoding repairs."""
        try:
            return json.JSONDecoder(object_pairs_hook=self._object, parse_constant=self._constant).raw_decode(text)
        except json.JSONDecodeError:
            pass
        quoted = False
        escaped = False
        stack: list[str] = []
        if not text.startswith(('{', '[', '"')):
            raise ResponseFormatError("Expected JSON arguments.")
        for index, char in enumerate(text):
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                    if text.startswith('"') and not stack:
                        return self._decode(text[:index + 1]), index + 1
            elif char == '"':
                quoted = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                if not stack or stack.pop() != (
                        "[" if char == "]" else "{"):
                    break
                if not stack:
                    return self._decode(text[:index + 1]), index + 1
        raise ResponseFormatError("Expected a complete JSON object or array.")

    def _json_call_start(self, text: str) -> bool:
        try:
            value, _ = self._json_prefix(text)
        except (ValueError, RecursionError):
            # Recognize broken explicit carriers so an intended call cannot silently
            # become a final reply. Strict decoding still owns the diagnostic.
            return bool(re.match(r'(?:\[\s*)?\{\s*"(?:response|tool_calls|function_call|name|function)"\s*:', text))
        return self._call_value(value) or isinstance(value, dict) and bool(set(value) & {"response", "tool_calls", "function_call"})

    def _marker_calls(self, text: str) -> list[ToolCall]:
        """Decode documented Mistral, DeepSeek V3.1, and Kimi tool markers.

        JSON boundaries are decoded before control tokens so markers inside code
        strings remain arguments. No names, parameters, or missing ends are guessed.
        """
        remaining = text.strip()
        raw: list[dict[str, Any]] = []
        while remaining:
            if remaining.startswith("[TOOL_CALLS]"):
                body = remaining[len("[TOOL_CALLS]"):].lstrip()
                name = None
                if not body.startswith("["):
                    name, separator, body = body.partition("[ARGS]")
                    if not separator or not name.strip() or re.search(r"\s", name.strip()):
                        raise ResponseFormatError("Expected a tool name and [ARGS] after [TOOL_CALLS].")
                    body = body.lstrip()
                try:
                    _, end = self._json_prefix(body)
                except (ValueError, RecursionError):
                    self._decode(body)
                    raise ResponseFormatError("Expected complete JSON after tool markers.")
                value = self._decode(body[:end])
                raw.extend(call.to_record() for call in self.normalize_calls(value if name is None else {"name": name.strip(), "arguments": value}))
                remaining = body[end:].strip()
                if remaining == "</s>":
                    remaining = ""
            elif remaining.startswith("<｜tool▁calls▁begin｜>"):
                remaining = remaining[len("<｜tool▁calls▁begin｜>"):].lstrip()
                while remaining.startswith("<｜tool▁call▁begin｜>"):
                    body = remaining[len("<｜tool▁call▁begin｜>"):]
                    name, separator, body = body.partition("<｜tool▁sep｜>")
                    if not separator:
                        raise ResponseFormatError("Missing DeepSeek tool separator.")
                    body = body.lstrip()
                    try:
                        _, end = self._json_prefix(body)
                    except (ValueError, RecursionError):
                        self._decode(body)
                        raise ResponseFormatError("Expected complete DeepSeek tool arguments.")
                    raw.append({"name": name.strip(), "arguments": self._decode(body[:end])})
                    remaining = body[end:].lstrip()
                    if not remaining.startswith("<｜tool▁call▁end｜>"):
                        raise ResponseFormatError("Missing DeepSeek tool-call end marker.")
                    remaining = remaining[len("<｜tool▁call▁end｜>"):].lstrip()
                if not remaining.startswith("<｜tool▁calls▁end｜>"):
                    raise ResponseFormatError("Missing DeepSeek tool-batch end marker.")
                remaining = remaining[len("<｜tool▁calls▁end｜>"):].strip()
                if remaining == "<｜end▁of▁sentence｜>":
                    remaining = ""
            elif remaining.startswith("<|tool_calls_section_begin|>"):
                remaining = remaining[len("<|tool_calls_section_begin|>"):].lstrip()
                while remaining.startswith("<|tool_call_begin|>"):
                    body = remaining[len("<|tool_call_begin|>"):]
                    identity, separator, body = body.partition("<|tool_call_argument_begin|>")
                    match = re.fullmatch(r"functions\.([^\s<>:]+):([0-9]+)", identity.strip())
                    if not separator or match is None:
                        raise ResponseFormatError("Expected a Kimi function identity and argument marker.")
                    body = body.lstrip()
                    value, end = self._json_prefix(body)
                    remaining = body[end:].lstrip()
                    if remaining.startswith("<|tool_call_argument_end|>"):
                        remaining = remaining[len("<|tool_call_argument_end|>"):].lstrip()
                    if not remaining.startswith("<|tool_call_end|>"):
                        raise ResponseFormatError("Missing Kimi tool-call end marker.")
                    remaining = remaining[len("<|tool_call_end|>"):].lstrip()
                    raw.append({"name": match[1], "arguments": value})
                if not remaining.startswith("<|tool_calls_section_end|>"):
                    raise ResponseFormatError("Missing Kimi tool-batch end marker.")
                remaining = remaining[len("<|tool_calls_section_end|>"):].strip()
            else:
                raise ResponseFormatError("Unexpected text after tool markers; return one complete call batch.")
        return self.normalize_calls(raw)

    def _tagged_calls(self, text: str, tools: list[ToolSpec]) -> list[ToolCall]:
        remaining = text.strip()
        raw: list[dict[str, Any]] = []
        while remaining:
            xml_opening = CALL_OPENING.match(remaining)
            opening_end = remaining.find(">")
            if xml_opening and (xml_opening[1] in {"function_calls", "tool_calls"}
                                or re.match(r"\s*<invoke(?=[\s>])", remaining[opening_end + 1:])):
                xml_closing = re.search(r"</" + xml_opening[1] + r"\s*>", remaining)
                if not xml_closing:
                    raise ResponseFormatError(f"Missing closing </{xml_opening[1]}>.")
                raw.extend(self._xml_calls(remaining[:xml_closing.end()], tools))
                remaining = remaining[xml_closing.end():]
                remaining = remaining.lstrip()
                continue
            opening = re.match(r"<(tool_call|function_call)(?:\s+[^<>]*)?>\s*", remaining)
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
                call, remaining = self._qwen_call(remaining, tools)
                raw.append(call)
            elif glm := GLM_CALL_NAME.match(remaining):
                call, remaining = self._glm_call(glm, remaining, tools)
                raw.append(call)
            else:
                try:
                    _, end = self._json_prefix(remaining)
                except (ValueError, RecursionError) as exc:
                    raise ResponseFormatError("Expected complete JSON inside tool-call tags.") from exc
                calls = self.normalize_calls(self._decode(remaining[:end]))
                raw.extend(call.to_record() for call in calls)
                remaining = remaining[end:].lstrip()
            if fence:
                if not remaining.startswith("```"):
                    raise ResponseFormatError("Missing closing code fence in tagged call.")
                remaining = remaining[3:].lstrip()
            closing_match = re.match(r"</" + opening[1] + r"\s*>", remaining)
            if closing_match:
                remaining = remaining[closing_match.end():].lstrip()
            elif remaining and not CALL_OPENING.match(remaining):
                raise ResponseFormatError(f"Missing closing </{opening[1]}>.")
        return self.normalize_calls(raw)

    def _xml_calls(self, text: str, tools: list[ToolSpec]) -> list[dict[str, Any]]:
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
        if root.tag in {"tool_call", "function_call"} and len(root) != 1:
            raise ResponseFormatError("Expected exactly one invoke in a singular tool-call wrapper.")
        for element in root:
            invoke = element
            if root.tag == "tool_calls" and element.tag == "call":
                if len(element) != 1 or (element.text or "").strip():
                    raise ResponseFormatError("Expected each call in tool_calls to contain exactly one invoke element.")
                invoke = element[0]
                if (invoke.tail or "").strip():
                    raise ResponseFormatError("Unexpected text after invoke in call.")
            if invoke.tag != "invoke":
                raise ResponseFormatError(f"Expected an invoke element in {root.tag}.")
            name = invoke.get("name")
            if isinstance(name, str):
                name = tool_name(name, [tool.name for tool in tools])
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
                        arguments[key] = self._decode(value)
                    except ResponseFormatError:
                        arguments[key] = value
            calls.append({"id": invoke.get("id"), "name": name, "arguments": arguments})
        return calls

    def _glm_call(self, name: re.Match[str], text: str, tools: list[ToolSpec]) -> tuple[dict[str, Any], str]:
        """Decode GLM's `name<arg_key>k</arg_key><arg_value>v</arg_value>` call body.

        Its template writes string values as text and other values as JSON, so
        schema string parameters stay literal and the rest decode when valid.
        """
        resolved = tool_name(name[1], [tool.name for tool in tools])
        properties: dict[str, Any] = next((tool.parameters.get("properties", {}) for tool in tools if tool.name == resolved), {})
        remaining = text[name.end():]
        arguments: dict[str, Any] = {}
        while (pair := re.match(r"\s*<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", remaining, re.DOTALL)):
            key, value = pair[1].strip(), pair[2]
            if not key or key in arguments:
                raise ResponseFormatError("Expected named, nonduplicated GLM call arguments.")
            kind = properties.get(key, {}).get("type")
            if kind == "string" or isinstance(kind, list) and "string" in kind:
                arguments[key] = value
            else:
                try:
                    arguments[key] = self._decode(value)
                except ResponseFormatError:
                    arguments[key] = value
            remaining = remaining[pair.end():]
        remaining = remaining.lstrip()
        if not re.match(r"</(?:tool_call|function_call)\s*>", remaining):
            raise ResponseFormatError("Expected </tool_call> after GLM call arguments.")
        return {"name": resolved, "arguments": arguments}, remaining

    def _qwen_call(self, text: str, tools: list[ToolSpec]) -> tuple[dict[str, Any], str]:
        function = re.match(r"<function=([^<>\s]+)>\s*", text)
        if function is None:
            raise ResponseFormatError("Invalid Qwen3-Coder function tag.")
        name = tool_name(function[1], [tool.name for tool in tools])
        properties: dict[str, Any] = next((tool.parameters.get("properties", {}) for tool in tools if tool.name == name), {})
        remaining = text[function.end():]
        arguments: dict[str, Any] = {}
        while remaining and not remaining.startswith(("</function>", "</tool_call>", "</function_call>", "<tool_call>", "<function_call>")):
            parameter = re.match(r"<parameter=([^<>\s=]+)>", remaining)
            malformed = re.match(r"<parameter=([A-Za-z0-9_]+)=([^<]*)</parameter>", remaining)
            if parameter is None and malformed is not None:
                remaining = '<parameter=' + malformed[1] + '>' + malformed[2] + remaining[malformed.end(2):]
                parameter = re.match(r"<parameter=([^<>\s=]+)>", remaining)
            if parameter is None or parameter[1] in arguments:
                raise ResponseFormatError("Invalid or duplicate Qwen3-Coder parameter.")
            value, closing, remaining = remaining[parameter.end():].partition("</parameter>")
            if not closing:
                # The legacy answer carrier ends in literal display text. Preserve
                # its trailing-tag tolerance without extending it to executable tools.
                if name != "answer" or parameter[1] != "text":
                    raise ResponseFormatError("Missing Qwen3-Coder parameter closing tag.")
            # Qwen's framing adds one newline at each edge; preserve code whitespace.
            value = value[2:] if value.startswith("\r\n") else value.removeprefix("\n")
            value = value[:-2] if value.endswith("\r\n") else value.removesuffix("\n")
            schema = properties.get(parameter[1], {})
            kind = schema.get("type", "string")
            arguments[parameter[1]] = value if kind == "string" or isinstance(kind, list) and "string" in kind else self._decode(value)
            remaining = remaining.lstrip()
        if not remaining.startswith("</function>"):
            if name == "answer" and set(arguments) == {"text"} and not remaining:
                return {"name": name, "arguments": arguments}, ""
            raise ResponseFormatError("Missing Qwen3-Coder function closing tag.")
        return {"name": name, "arguments": arguments}, remaining[len("</function>"):].lstrip()

    def _object(self, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ResponseFormatError(f"Duplicate JSON field: {key}.")
            result[key] = value
        return result

    def _constant(self, value: str) -> Any:
        raise ResponseFormatError(f"Non-finite JSON value: {value}.")

    def _decode(self, text: str) -> Any:
        try:
            try:
                value = json.loads(text, object_pairs_hook=self._object, parse_constant=self._constant)
            except json.JSONDecodeError as original_error:
                try:
                    value = json.loads(compatible_json(text), object_pairs_hook=self._object, parse_constant=self._constant)
                except json.JSONDecodeError:
                    raise original_error
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

    def normalize_calls(self, raw: Any) -> list[ToolCall]:
        """Normalize explicit calls, never infer actions from natural-language text.

        Native OpenAI calls, the historical flat envelope, and tool_use blocks share
        one acceptance boundary. Complete argument objects tolerate encoding noise;
        truncated structures and guessed quotes could change code, paths, or commands.
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
            self._decode(json.dumps(name, ensure_ascii=True))
            # Prefer nested arguments, then flat aliases; provider metadata is not
            # part of the executable call contract.
            arguments: Any = next((source[key] for source in (function, item)
                              for key in ("arguments", "input", "parameters")
                              if source.get(key) is not None), {})
            try:
                arguments = self._decode(arguments) if isinstance(arguments, str) and arguments.strip() else arguments
                if isinstance(arguments, str) and arguments.lstrip().startswith("{"):
                    arguments = self._decode(arguments)
                if arguments is None or isinstance(arguments, str) and not arguments.strip():
                    arguments = {}
                if not isinstance(arguments, dict):
                    raise ResponseFormatError("Tool arguments must decode to an object.")
                # Preserve valid JSON values without modifying tool payloads.
                self._decode(json.dumps(arguments, ensure_ascii=True))
            except ResponseFormatError as exc:
                raise ResponseFormatError(f"tool_calls[{index}].arguments: {exc}", excerpt=exc.excerpt) from exc
            call_id = item.get("id")
            if type(call_id) is int:
                call_id = str(call_id)
            if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
                call_id = "call_" + uuid.uuid4().hex
            self._decode(json.dumps(call_id, ensure_ascii=True))
            ids.add(call_id)
            calls.append(ToolCall(call_id, name, arguments))
        return calls

    def merge_call_sources(self, *sources: list[ToolCall]) -> list[ToolCall]:
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

    def _response_object(self, text: str, *, tools: list[ToolSpec] | None = None, xml_tool_calls: bool = False) -> tuple[Any, list[ToolCall]]:
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
        record: Any = None
        calls: list[ToolCall] = []
        while remaining:
            xml_opening = CALL_OPENING.match(remaining)
            if xml_tool_calls and xml_opening:
                calls.extend(self._tagged_calls(remaining, tools or []))
                remaining = ""
                break
            if remaining.startswith(("<tool_call>", "<function_call>")) and any(tag in remaining for tag in ("<function=", "<answer>")):
                calls.extend(self._tagged_calls(remaining, tools or []))
                remaining = ""
                break
            tag = next((tag for tag in ("tool_call", "function_call") if remaining.startswith(f"<{tag}>")), None)
            if tag:
                remaining = remaining[len(tag) + 2:].lstrip()
            block_fence = re.match(r"```(?:json)?[ \t]*\r?\n", remaining, re.IGNORECASE)
            if block_fence:
                remaining = remaining[block_fence.end():].lstrip()
            try:
                _, end = self._json_prefix(remaining)
            except (ValueError, RecursionError):
                # Use the shared decoder for a bounded, precise diagnostic.
                self._decode(remaining)
                raise ResponseFormatError("Expected a complete response JSON object.")
            value = self._decode(remaining[:end])
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
                calls.extend(self.normalize_calls(value))
            elif record is None:
                if not isinstance(value, dict):
                    raise ResponseFormatError("Response must be one JSON object.")
                record = value
            else:
                raise ResponseFormatError("Multiple response JSON objects are ambiguous; return one record.")
        if record is None:
            raise ResponseFormatError("Missing response JSON object.")
        # IDs must also be unique across separately delimited calls.
        if len({call.id for call in calls}) != len(calls):
            raise ResponseFormatError("Tool IDs must be unique within the response.")
        return record, calls

    def parse_response(self, text: str, has_previous_results: bool, *, native_calls: list[ToolCall] | None = None) -> ResponseRecord:
        try:
            return self._parse_response(text, has_previous_results, native_calls or [])
        except ResponseFormatError as exc:
            if not exc.excerpt:
                # A stateless retry needs evidence of the rejected output. This is
                # bounded diagnostic data, never an executable assistant message.
                exc.excerpt = json.dumps({"invalid_text_prefix": text[:1500], "total_characters": len(text)}, indent=2)
            raise

    def _parse_response(self, text: str, has_previous_results: bool, native_calls: list[ToolCall]) -> ResponseRecord:
        value, tagged_calls = self._response_object(text)
        return self._legacy_response(value, has_previous_results, native_calls, tagged_calls)

    def _legacy_response(
        self, value: Any, has_previous_results: bool, native_calls: list[ToolCall], tagged_calls: list[ToolCall],
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
        calls = self.merge_call_sources(
            self.normalize_calls([call.to_record() for call in native_calls]),
            self.normalize_calls(value.get("tool_calls")), self.normalize_calls(value.get("function_call")), tagged_calls,
        )
        if not calls and not value["response"].strip():
            raise ResponseFormatError("A final response must contain an answer.")
        return ResponseRecord(value["response"], calls, value["previous_tool_responses_compressed"],
                              value["user_prompt_compressed"], value["agent_response_compressed"])


# Entry points for other modules; plugins override the methods above.
def parse_agent_response(text: str, native_calls: list[ToolCall], *, native_tools: bool = True,
                         json_response: bool = False, tools: list[ToolSpec] | None = None,
                         xml_tool_calls: bool = False) -> AgentResponse:
    return ResponseParser().parse_agent_response(text, native_calls, native_tools=native_tools, json_response=json_response,
                                                 tools=tools, xml_tool_calls=xml_tool_calls)


def combine_calls(*sources: list[ToolCall]) -> list[ToolCall]:
    return ResponseParser().combine_calls(*sources)


def normalize_calls(raw: Any) -> list[ToolCall]:
    return ResponseParser().normalize_calls(raw)


def parse_response(text: str, has_previous_results: bool, *, native_calls: list[ToolCall] | None = None) -> ResponseRecord:
    return ResponseParser().parse_response(text, has_previous_results, native_calls=native_calls)


