"""Regressions for the completed source audit; all resources are disposable."""

import asyncio
import io
import json
from types import SimpleNamespace

import httpx
import pytest

from slipagent.api import APIResponseError, _StreamCompletion, _parse_completion
from slipagent.cli import ModelPicker, Style, build_parser
from slipagent.lsp import LanguageServer, LSPError
from slipagent.openrouter import OpenRouterClient
from slipagent.plugins import fingerprint
from slipagent.plugintools import PluginTools
from slipagent.protocol import parse_agent_response
from slipagent.types import ModelInfo


@pytest.mark.parametrize("streamed", [False, True])
def test_incoming_ids_never_name_calls_and_early_labels_keep_local_ids(streamed):
    events = []
    calls = [{"index": index, "id": "same-incoming-id", "function": {
        "name": "record", "arguments": json.dumps({"call_label": f"Record {index}", "value": index})}}
        for index in range(2)]
    if streamed:
        stream = _StreamCompletion(lambda kind, text: events.append((kind, text)))
        stream.add(json.dumps({"choices": [{"delta": {"tool_calls": calls}}]}))
        first_ids = [stream.calls[index]["id"] for index in range(2)]
        stream.add(json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": index, "id": "changed-incoming-id"} for index in range(2)]},
            "finish_reason": "tool_calls"}]}))
        completion = stream.completion()
        assert [call.id for call in completion.tool_calls] == first_ids
        assert [json.loads(text)["id"] for kind, text in events if kind == "tool_label"] == first_ids
    else:
        completion = _parse_completion({"choices": [{"message": {"tool_calls": calls}}]})
    accepted = parse_agent_response("", completion.tool_calls)
    assert [call.id for call in accepted.calls] == [call.id for call in completion.tool_calls]
    assert len({call.id for call in accepted.calls}) == 2
    assert all(call.id.startswith("call_") for call in accepted.calls)
    embedded = parse_agent_response(json.dumps({"tool_calls": [
        {"id": "same-incoming-id", "name": "record", "arguments": {"value": index}}
        for index in range(2)]}), [])
    assert len({call.id for call in embedded.calls}) == 2
    assert all(call.id.startswith("call_") for call in embedded.calls)


@pytest.mark.parametrize("header", [
    'from __future__ import annotations\n',
    '#!/usr/bin/env python\n"""Plugin documentation."""\nfrom __future__ import annotations\n',
])
def test_plugin_import_insertion_preserves_future_imports_and_docstring(header):
    source = header + 'import slipagent.plugins as p\n@p.replaces("terminal.TerminalUI")\nclass Overlay:\n    pass\n'
    result = PluginTools().with_import(source, "overrides")
    compile(result, "plugin.py", "exec")
    assert result.startswith(header)
    assert result.index("from __future__") < result.index("from slipagent.plugins import overrides")


def test_fingerprint_tracks_property_setters_and_deleters():
    source = '''class Widget:
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value):
        self._value = value
    @value.deleter
    def value(self):
        del self._value
'''
    initial = fingerprint(source, "Widget", "value")
    assert fingerprint(source.replace("self._value = value", "self._value = value + 1"), "Widget", "value") != initial
    assert fingerprint(source.replace("del self._value", "self._value = None"), "Widget", "value") != initial


def test_malformed_reasoning_preserves_usage_and_response_evidence():
    with pytest.raises(APIResponseError) as caught:
        _parse_completion({"choices": [{"message": {"content": "Useful reply", "reasoning_details": "invalid"}}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": .01}})
    assert caught.value.usage.total_tokens == 120
    assert caught.value.usage.cost == .01
    assert "Useful reply" in caught.value.partial_response


async def test_failed_lsp_reader_retires_and_rejects_further_requests(workspace, monkeypatch):
    client = LanguageServer(workspace, {"command": ["unused"]})
    reader = asyncio.StreamReader()
    reader.feed_data(b"Content-Length: invalid\r\n\r\n")
    client.process = SimpleNamespace(stdout=reader)
    closed = asyncio.Event()

    async def close():
        closed.set()

    monkeypatch.setattr(client, "_close", close)
    await client._read()
    await client.aclose()
    assert closed.is_set()
    with pytest.raises(LSPError, match="connection failed"):
        await client.request("textDocument/hover", {})
    assert not client.pending


@pytest.mark.parametrize("tier,selected", [("free", True), ("paid", False)])
async def test_provider_default_uses_google_account_tier(monkeypatch, tier, selected):
    import slipagent.cli as cli

    model = ModelInfo("gemini-3.8-flash", provider="google", pricing={
        "prompt": "unknown", "completion": "unknown", "free_tier": True})
    choices = []

    async def catalog(**kwargs):
        return [model]

    async def select_model(entry):
        choices.append(entry)

    monkeypatch.setenv("GEMINI_BILLING_TIER", tier)
    monkeypatch.setattr(cli, "save_model_choice", lambda *_: None)
    session = SimpleNamespace(catalog=catalog, select_model=select_model)
    await ModelPicker(session, Style(False), io.StringIO()).provider_default("google")
    assert choices == ([model] if selected else [])


@pytest.mark.parametrize("limit", ["0", "-1", "1", "4095"])
def test_invalid_cli_context_caps_are_argument_errors(limit):
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["--context-tokens", limit])
    assert caught.value.code == 2


@pytest.mark.parametrize("row", [{"id": ""}, {"id": "model", "pricing": [1]}, {"id": "model", "pricing": "free"}])
async def test_malformed_catalog_rows_fail_at_the_api_boundary(row):
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [row]}))) as client:
        with pytest.raises(APIResponseError, match="malformed catalog"):
            await client.list_models()


def test_integral_float_tab_preferences_are_normalized(monkeypatch):
    import slipagent.uipreferences as preferences

    monkeypatch.setattr(preferences, "_read_preferences", lambda: {"inspection": {"tab_width": 4.0}})
    result = preferences.load_ui_preferences()
    assert type(result["tab_width"]) is int
    assert " " * result["tab_width"] == "    "


def test_scrollbar_drags_within_a_single_wrapped_line():
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.layout.controls import FormattedTextControl, UIContent
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from slipagent.uiwindows import TranscriptScrollbar, TranscriptWindow

    window = TranscriptWindow(FormattedTextControl(""))
    content = UIContent(get_line=lambda _: [("", "x" * 200)], line_count=1)
    window.render_info = SimpleNamespace(ui_content=content, window_width=10, window_height=5)
    bar = TranscriptScrollbar(lambda: window)
    drawn = bar.create_content(1, 10)
    assert any("┃" in fragment[1] for row in range(10) for fragment in drawn.get_line(row))
    bar.pointer(MouseEvent(Point(0, 5), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset()), 5)
    assert window.vertical_scroll == 0
    assert 0 < window.vertical_scroll_2 < 15
