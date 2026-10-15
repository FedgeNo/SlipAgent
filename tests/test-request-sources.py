"""File-backed history and bounded, replayable provider requests."""

import json
import tracemalloc

import httpx
import pytest

from slipagent.context import ConversationHistory
from slipagent.contextview import ContextView
from slipagent.data_text import JSONInput
from slipagent.openrouter import OpenRouterClient
from slipagent.requestbody import RequestBody, wire_message
from slipagent.sources import FileText, json_chunks, materialize, write_indexed_json
from slipagent.turnstore import TurnStore
from slipagent.types import Message


def test_indexed_json_converts_compatible_mapping_keys(tmp_path):
    value = {2: "integer", 2.5: "float", False: "boolean", None: "null"}
    path = tmp_path / "keys.json"
    write_indexed_json(path, value)
    assert json.loads(path.read_text()) == json.loads(json.dumps(value))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_request_body_rejects_nonfinite_transport_numbers(value):
    with pytest.raises(ValueError):
        RequestBody({"messages": [], "temperature": value})


@pytest.mark.parametrize("indent", [None, 2])
def test_indexed_values_preserve_unicode_escaping_and_json_format(tmp_path, indent):
    original = {"body": ('é🌙\\"\n\t' * 3000), "values": [None, True, 2.5, ""]}
    indexed = write_indexed_json(tmp_path / "source.json", original)
    assert isinstance(indexed["body"], FileText)
    assert materialize(indexed) == original
    assert "".join(json_chunks(indexed, indent=indent)) == json.dumps(original, ensure_ascii=False, indent=indent)


def test_turn_store_keeps_raw_summary_and_group_separately(tmp_path):
    history = ConversationHistory()
    messages = [Message.user("question"), Message.assistant("original " * 1000)]
    history.sync(messages)
    step = history.steps[0]
    store = TurnStore(tmp_path)
    store.archive(step, messages)
    assert isinstance(step._disk["model"]["agent_response"], FileText)
    assert messages[1].content == "original " * 1000
    step.summary = "summary " * 100
    step.compaction_status = "complete"
    store.summarized(step)
    step.historical_summary = {"step_ids": [1], "compressed_summary": "grouped narrative"}
    store.summarized(step)
    assert json.loads((tmp_path / "00000001.raw.json").read_text())["parts"]["response"] == "original " * 1000
    assert json.loads((tmp_path / "00000001.compressed.json").read_text())["summary"] == step.summary
    assert json.loads((tmp_path / "00000001-00000001.grouped.json").read_text())["compressed_summary"] == "grouped narrative"


def test_request_snapshot_and_view_survive_source_replacement(tmp_path):
    path = tmp_path / "history.json"
    indexed = write_indexed_json(path, {"response": "original " * 1000})
    message = Message.user(JSONInput({"history": [indexed]}))
    expected = message.to_api()
    body = RequestBody({"messages": [wire_message(message)]})
    path.write_text("replaced")
    assert body.read()["messages"] == [expected]
    assert sum(map(len, body.chunks())) == body.size
    view = ContextView.from_body(body)
    try:
        assert expected["content"] in "\n".join(text for _, text in view)
    finally:
        view.close()


async def test_http_body_is_bounded_and_replays_after_retry(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("x" * (2 * 1024 * 1024))
    source = FileText(path, 0, path.stat().st_size, path.stat().st_size)
    attempts = []
    snapshots = []

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            import hashlib
            digest = hashlib.sha256()
            size = 0
            async for chunk in request.stream:
                assert len(chunk) <= 65536
                digest.update(chunk)
                size += len(chunk)
            assert size == int(request.headers["Content-Length"])
            attempts.append(digest.hexdigest())
            if len(attempts) == 1:
                path.write_text("changed during retry")
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]})

    async with OpenRouterClient("dummy-test-key", transport=Transport()) as client:
        tracemalloc.start()
        try:
            result = await client.chat(model="test", messages=[Message.user(source)], on_request_body=snapshots.append)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert result.text == "done"
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert snapshots[0].size > 2 * 1024 * 1024
    assert peak < 2 * 1024 * 1024


async def test_completed_inspector_call_releases_large_resident_values():
    import io
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from slipagent.inspection import field_snapshot
    from slipagent.terminal import TerminalUI
    from slipagent.types import ToolCall

    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: "status", io.StringIO(), input=pipe, output=DummyOutput())
        call = ToolCall("call", "write_file", {"path": "example.txt", "content": "payload " * 1000})
        try:
            ui.track_call(call, 1)
            ui.finish_call(call, 1, {"output": "result " * 1000}, False)
            record = ui._inspection_calls[(1, "call")]
            assert isinstance(record["arguments"]["content"], FileText)
            assert isinstance(record["result"]["output"], FileText)
            path = record["result"]["output"].path
            snapshot = field_snapshot(record)
            try:
                assert "result " * 1000 in "".join(snapshot)
            finally:
                snapshot.close()
        finally:
            ui.close()
        assert not path.exists()
