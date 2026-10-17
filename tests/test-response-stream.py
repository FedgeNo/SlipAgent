"""Provisional reply decoding across arbitrary provider chunk boundaries."""

import json

import pytest

from slipagent.streaming import ResponseStream


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 1000])
@pytest.mark.parametrize("fence", [False, True])
def test_response_string_streams_without_envelope_or_other_fields(chunk_size, fence):
    text = 'Hello "λ".\nPath: C:\\work\t😀 done.'
    wire = json.dumps({"tool_calls": [{"arguments": {"response": "hidden"}}],
                       "response": text, "suggestions": ["hidden"]})
    if fence:
        wire = "```json\n" + wire + "\n```"
    stream = ResponseStream()
    chunks = [stream.push(wire[i:i + chunk_size]) for i in range(0, len(wire), chunk_size)]
    assert "".join(chunks) == text
    assert stream.finish(text) == ""


def test_open_response_string_is_visible_before_closing_quote_or_object():
    stream = ResponseStream()
    assert stream.push('{"response":"Already working') == "Already working"
    assert stream.push('\\nNext \\uD83D') == "\nNext"
    assert stream.push('\\uDE00') == " 😀"
    assert stream.push('","tool_calls":[') == ""


@pytest.mark.parametrize("wire", [
    '{"response":"before\\qafter"}',
    '{"response":"before\\uD800Xafter"}',
    '{"response":"before\\uDC00after"}',
    '{"response":"before\\uZZZZafter"}',
])
def test_invalid_escape_stops_preview_without_repairing_json(wire):
    stream = ResponseStream()
    assert "".join(stream.push(char) for char in wire) == "before"


@pytest.mark.parametrize("wire, expected", [
    ("  Plain reply.  ", "Plain reply."),
    ('{"other":{"response":"hidden"},"response":"Visible"}', "Visible"),
    ('{"response":123,"tool_calls":[]}', ""),
    ('{"response":"First","response":"Second"}', "First"),
    ("```json\nExample, not an envelope.\n```", "```json\nExample, not an envelope.\n```"),
    ("Example:\n```\n[Harness context metadata]\n```", "Example:\n```\n[Harness context metadata]\n```"),
    ("Visible.\n[Harness context metadata]\n123\nnone\nNext.", "Visible.\nNext."),
    ("Visible.\nAgent Response (Full):\nNext.", "Visible.\nNext."),
    ('Visible.\n<tool_call>{"name":"hidden"}</tool_call>', "Visible."),
    ('Example:\n```xml\n<tool_call>literal</tool_call>\n```', 'Example:\n```xml\n<tool_call>literal</tool_call>\n```'),
])
def test_visible_prose_and_call_boundaries(wire, expected):
    stream = ResponseStream()
    assert "".join(stream.push(char) for char in wire) + stream.finish(expected) == expected


def test_response_tags_are_literal_except_for_xml_profile():
    text = '<tool_call>example</tool_call>'
    wire = json.dumps({"response": text})
    assert ResponseStream().push(wire) == text
    assert ResponseStream(xml_tool_calls=True).push(wire) == ""


@pytest.mark.parametrize("envelope", [False, True])
def test_xml_call_fences_do_not_expose_arguments_in_preview(envelope):
    wire = '```xml\n<tool_call><answer>Hidden until validated</answer></tool_call>\n```'
    if envelope:
        wire = json.dumps({"response": wire})
    stream = ResponseStream(xml_tool_calls=True)
    assert "".join(stream.push(char) for char in wire) == ""


@pytest.mark.parametrize("size", [1, 7, 1000])
def test_mid_line_call_tags_are_never_previewed(size):
    # GLM writes its call right after a sentence; only the prose may stream.
    prose = "I'll load `list_rules` first, then run a demo."
    text = prose + '<tool_call>load_tools<arg_key>names</arg_key><arg_value>["a"]</arg_value></tool_call>'
    stream = ResponseStream()
    shown = "".join(stream.push(text[i:i + size]) for i in range(0, len(text), size))
    assert shown == prose
    assert stream.finish(prose) == ""


@pytest.mark.parametrize("text", ["Use `<tool_call>` tags.", "Keep a < b and <b>bold</b>.", "A <tool_callx> stays."])
def test_inline_code_and_other_tags_still_preview(text):
    stream = ResponseStream()
    assert "".join(stream.push(char) for char in text) == text
