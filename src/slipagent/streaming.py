"""Decode provisional reply text without accepting or executing response actions."""

from __future__ import annotations

import re

from .contextlegacy import LEGACY_RESPONSE_HEADINGS
from .protocol import CALL_OPENING, FUNCTION_OPENING, TOOL_MARKERS, XML_CALL_TAGS


class ResponseStream:
    """Preview plain prose or a top-level JSON response string in one pass.

    The complete response still goes through protocol validation. Unsupported
    carriers remain buffered for that boundary rather than being repaired here.
    """

    def __init__(self, *, xml_tool_calls: bool = False) -> None:
        self._xml = xml_tool_calls
        self._prefix = ""
        self._mode = "start"
        self._json = _JSONResponseStream()
        self._nested: ResponseStream | None = None
        self._visible = _VisibleTextStream(tagged=True)
        self._chunks: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    def push(self, chunk: str) -> str:
        if self._mode == "buffer":
            return ""
        if self._mode == "start":
            self._prefix += chunk
            candidate = self._prefix.lstrip(" \t\r\n\ufeff")
            if not candidate:
                return ""
            if "<think>".startswith(candidate):
                return ""
            if "</think>".startswith(candidate):
                return ""
            if candidate.startswith("</think>"):
                candidate = candidate[len("</think>"):].lstrip()
                self._prefix = candidate
                if not candidate:
                    return ""
            while candidate.startswith("<think>"):
                _, closing, following = candidate.partition("</think>")
                if not closing:
                    return ""
                candidate = following.lstrip()
                self._prefix = candidate
                if not candidate or "<think>".startswith(candidate):
                    return ""
            if candidate.startswith("["):
                self._mode = "buffer"
                self._prefix = ""
                return ""
            if candidate.startswith("`"):
                if self._xml and re.match(r"```xml[ \t]*\r?\n", candidate, re.IGNORECASE):
                    self._mode = "buffer"
                    self._prefix = ""
                    return ""
                opening = re.match(r"```(?:json)?[ \t]*\r?\n", candidate, re.IGNORECASE)
                if opening:
                    body = candidate[opening.end():].lstrip()
                    if not body:
                        return ""
                    if body.startswith("["):
                        self._mode = "buffer"
                        self._prefix = ""
                        return ""
                    if body.startswith("{"):
                        candidate = body
                    else:
                        candidate = self._prefix
                else:
                    prefixes = ("```json", "```xml") if self._xml else ("```json",)
                    if (any(prefix.startswith(candidate.lower()) for prefix in prefixes)
                            or re.fullmatch(r"```(?:json|xml)?[ \t]*\r?", candidate, re.IGNORECASE)):
                        return ""
            self._mode = "json" if candidate.startswith("{") else "plain"
            if self._mode == "json":
                self._visible = _VisibleTextStream(tagged=self._xml)
                if self._xml:
                    self._nested = ResponseStream(xml_tool_calls=True)
                chunk = candidate
            else:
                chunk = self._prefix
            self._prefix = ""
        decoded = self._json.push(chunk) if self._mode == "json" else chunk
        visible = self._nested.push(decoded) if self._nested is not None else self._visible.push(decoded)
        if visible:
            self._chunks.append(visible)
        return visible

    def finish(self, accepted: str) -> str:
        """Flush only text confirmed by the complete, validated response."""
        shown = self.text
        remaining = accepted[len(shown):] if shown and accepted.startswith(shown) else ""
        if remaining:
            self._chunks.append(remaining)
        return remaining


class _JSONResponseStream:
    """Lex strings and nesting; emit only the first root response value."""

    def __init__(self) -> None:
        self._depth = 0
        self._stage = "key"
        self._key = ""
        self._string: str | None = None
        self._escape = ""
        self._surrogate: int | None = None
        self._done = False

    def push(self, chunk: str) -> str:
        output: list[str] = []
        escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
        for char in chunk:
            if self._done:
                break
            if self._string is not None:
                decoded = ""
                if self._escape.startswith("\\u"):
                    if char not in "0123456789abcdefABCDEF":
                        self._done = True
                        break
                    self._escape += char
                    if len(self._escape) < 6:
                        continue
                    code = int(self._escape[2:], 16)
                    self._escape = ""
                    if self._surrogate is not None:
                        if not 0xDC00 <= code <= 0xDFFF:
                            self._done = True
                            break
                        decoded = chr(0x10000 + ((self._surrogate - 0xD800) << 10) + code - 0xDC00)
                        self._surrogate = None
                    elif 0xD800 <= code <= 0xDBFF:
                        self._surrogate = code
                    elif 0xDC00 <= code <= 0xDFFF:
                        self._done = True
                        break
                    else:
                        decoded = chr(code)
                elif self._escape:
                    if char == "u":
                        self._escape += char
                        continue
                    if char not in escapes or self._surrogate is not None:
                        self._done = True
                        break
                    decoded = escapes[char]
                    self._escape = ""
                elif char == "\\":
                    self._escape = char
                    continue
                elif self._surrogate is not None or ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF:
                    self._done = True
                    break
                elif char == '"':
                    if self._string == "response":
                        self._done = True
                    elif self._string == "key":
                        self._stage = "colon"
                    self._string = None
                    continue
                else:
                    decoded = char
                if self._string == "response":
                    output.append(decoded)
                elif self._string == "key" and len(self._key) <= len("response"):
                    self._key += decoded
            elif char == '"':
                if self._depth == 1 and self._stage == "key":
                    self._key = ""
                    self._string = "key"
                else:
                    self._string = "response" if self._depth == 1 and self._stage == "value" and self._key == "response" else "skip"
                    if self._depth == 1:
                        self._stage = "after"
            elif char in "{[":
                self._depth += 1
            elif char in "}]":
                self._depth -= 1
                if self._depth <= 0:
                    self._done = True
            elif self._depth == 1:
                if char == ":" and self._stage == "colon":
                    self._stage = "value"
                elif char == ",":
                    self._stage = "key"
                elif not char.isspace():
                    self._stage = "after"
        return "".join(output)


INLINE_TAGS = ("<tool_call>", "<tool_call ", "<function_call>", "<function_call ")


class _VisibleTextStream:
    """Hold ambiguous line prefixes, bookkeeping, and trailing whitespace."""

    def __init__(self, *, tagged: bool) -> None:
        self._tagged = tagged
        self._line = ""
        self._passing = False
        self._fenced = False
        self._metadata = False
        self._stopped = False
        self._space = ""
        self._started = False
        # A call tag can also start mid-line after prose, outside inline code.
        self._inline = ""
        self._ticks = 0

    def _classify(self, *, final: bool) -> str:
        stripped = self._line.strip()
        if not stripped:
            return "skip" if final and self._metadata else "pass" if final else "hold"
        if any(marker.startswith(stripped) for marker in ("```", "~~~")) and not final:
            return "hold"
        if stripped.startswith(("```", "~~~")):
            self._fenced = not self._fenced
            self._metadata = False
            return "pass"
        if self._fenced:
            return "pass"
        tags = ("<tool_call>", "<function_call>", FUNCTION_OPENING) + XML_CALL_TAGS if self._tagged else ()
        leading = self._line[:len(self._line) - len(self._line.lstrip(" \t"))]
        if len(leading) < 4 and "\t" not in leading:
            if (any(stripped.startswith(marker) for marker in TOOL_MARKERS)
                    or self._tagged and (CALL_OPENING.match(stripped) or stripped.startswith(FUNCTION_OPENING))
                    or self._tagged and stripped.startswith(("{", "["))):
                self._stopped = True
                return "skip"
            if not final and any(tag.startswith(stripped) for tag in (*tags, *TOOL_MARKERS)):
                return "hold"
        markers = (*LEGACY_RESPONSE_HEADINGS, "[Harness context metadata]", "Previous step's current summary:")
        if stripped.startswith("Previous step's current summary:"):
            self._metadata = True
            return "skip"
        if stripped in markers:
            if not final:
                return "hold"
            if stripped == "[Harness context metadata]":
                self._metadata = True
            return "skip"
        if not final and any(marker.startswith(stripped) for marker in markers):
            return "hold"
        if self._metadata and (stripped.isdecimal() or stripped == "none"):
            return "skip" if final else "hold"
        if self._metadata and not final and "none".startswith(stripped):
            return "hold"
        self._metadata = False
        return "pass"

    def push(self, chunk: str) -> str:
        output: list[str] = []
        for char in chunk:
            if self._stopped:
                break
            visible = ""
            if self._passing:
                if self._tagged and (self._inline or char == "<" and self._ticks % 2 == 0):
                    self._inline += char
                    if any(self._inline.startswith(tag) for tag in INLINE_TAGS):
                        self._stopped = True
                        break
                    if not any(tag.startswith(self._inline) for tag in INLINE_TAGS):
                        visible, self._inline = self._inline, ""
                else:
                    visible = char
            elif char == "\n":
                if self._classify(final=True) == "pass":
                    visible = self._line + char
            else:
                self._line += char
                if self._classify(final=False) == "pass":
                    visible = self._line
                    self._line = ""
                    self._passing = True
            if char == "`":
                self._ticks += 1
            if char == "\n":
                self._line = ""
                self._passing = False
                self._ticks = 0
            for value in visible:
                if value.isspace():
                    if self._started:
                        self._space += value
                else:
                    output.extend((self._space, value))
                    self._space = ""
                    self._started = True
        return "".join(output)
