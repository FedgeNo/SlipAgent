"""Request inspection preserves text without retaining full formatted copies."""

import gc
import json
import tracemalloc

import pytest

from slipagent.contextview import ContextView
from slipagent.budget import ContextBudget
from slipagent.requestbody import RequestBody


def test_context_rows_styles_and_resize():
    view = ContextView([("class:context-system", "abc def\n\nend"), ("", "next")])
    try:
        assert len(view.rows) == 0
        content = view.content(4)
        assert [content.get_line(n) for n in range(content.line_count)] == [
            [("class:context-system", "abc")], [("class:context-system", "def")],
            [("class:context-system", "")], [("class:context-system", "end")],
            [("", "")], [("", "next")], [("", "")],
        ]
        content = view.content(20)
        assert content.get_line(0) == [("class:context-system", "abc def")]
        assert list(view) == [("class:context-system", "abc def\n\nend"), ("", "next")]
    finally:
        view.close()
    assert view.blocks._file.closed and view.rows._file.closed


def test_context_preserves_message_metadata_and_request_fields():
    request = {"messages": [{"role": "system", "content": "Exact 🌕 text"},
                            {"role": "assistant", "content": None, "tool_calls": [{"id": "one"}]}],
               "tools": [{"name": "example"}], "stream": True}
    view = ContextView.from_request(json.dumps(request))
    try:
        assert list(view) == [
            ("", "Messages — sent order"), ("class:context-system", "[01] SYSTEM\nExact 🌕 text"),
            ("", '[02] ASSISTANT\nnull\n{\n  "tool_calls": [\n    {\n      "id": "one"\n    }\n  ]\n}'),
            ("", 'tools:\n[\n  {\n    "name": "example"\n  }\n]'), ("", "stream:\ntrue")]
    finally:
        view.close()


@pytest.mark.parametrize("content", ["", None, [], [{"type": "text", "text": "Hello"}]])
def test_streamed_request_inspection_matches_json_inspection(content):
    request = {"messages": [{"role": "user", "content": content}]}
    body = RequestBody(request)
    streamed = ContextView.from_body(body)
    direct = ContextView.from_request(json.dumps(request))
    try:
        assert list(streamed) == list(direct)
    finally:
        streamed.close()
        direct.close()
        body.temporary.cleanup()


def test_large_context_retains_only_indexes_and_a_bounded_row_cache():
    # Compare growth across 4 and 32 cache capacities. Fixed file buffers cancel
    # out, so retaining the input or every displayed row still fails at this size.
    measurements = []
    for lines in (512, 4096):
        request = json.dumps({"messages": [{"role": "user", "content": ("0123456789 " * 10 + "\n") * lines}]})
        tracemalloc.start()
        view = ContextView.from_request(request)
        try:
            gc.collect()
            hidden_bytes = tracemalloc.get_traced_memory()[0]
            content = view.content(120)
            for index in range(content.line_count):
                content.get_line(index)
            gc.collect()
            visible_bytes = tracemalloc.get_traced_memory()[0]
            assert hidden_bytes < 500_000
            assert visible_bytes < 1_000_000
            assert content.get_line(content.line_count - 2)[0][1] == ""
            assert len(view.cache) <= 128
            measurements.append((len(request), hidden_bytes, visible_bytes))
        finally:
            view.close()
            tracemalloc.stop()
    small, large = measurements
    growth = large[0] - small[0]
    assert large[1] - small[1] < growth // 4
    assert large[2] - small[2] < growth // 2


@pytest.mark.parametrize("usage", [0, 123, 456])
def test_scalar_budget_measurement_matches_request_measurement(usage):
    request = {"messages": [{"role": "user", "content": "Hello"}],
               "tools": [{"name": "example"}], "response_format": {"type": "json_object"}}
    original, scalar = ContextBudget(), ContextBudget()
    original.observe(request, usage)
    scalar.observe_estimate(scalar.estimate(request), usage)
    assert scalar.scale == original.scale
    assert scalar.measurements == original.measurements
