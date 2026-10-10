"""Inspection snapshots stay bounded and preserve decoded text."""

import json
import asyncio
import tracemalloc
from pathlib import Path

import pytest

from slipagent.artifacts import ArtifactStore
from slipagent.inspection import IndexedText, artifact_snapshot, field_snapshot


def test_large_single_line_uses_disk_indexes_and_bounded_memory(tmp_path):
    path = tmp_path / "large.txt"
    with path.open("w") as output:
        for _ in range(3072):
            output.write("x" * 4096)
        output.write("\nlast line\n")
    tracemalloc.start()
    try:
        snapshot = IndexedText.from_path(path)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 2_000_000
        assert len(snapshot[0]) == 4096
        assert snapshot[snapshot.line_index(2)] == "last line"
        assert snapshot[-1] == "last line"
    finally:
        tracemalloc.stop()
        snapshot.close()


def test_large_json_record_is_written_and_decoded_in_chunks(tmp_path):
    store = ArtifactStore(64_000_000)
    store.use_directory(tmp_path)
    text = 'quoted "text" \\ literal\\n 😀\n'
    tracemalloc.start()
    try:
        record = store.save("command", (text for _ in range(150_000)), kind="command_output",
                            call_id="id", step_id=1, error="stderr\nlast")
        snapshot = artifact_snapshot(store.record_path(record["record_id"]), record)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 2_000_000
        assert snapshot[0] == "command" and snapshot[1] == ""
        assert snapshot[2] == text.removesuffix("\n")
        assert snapshot[-1] == "last"
    finally:
        tracemalloc.stop()
        snapshot.close()


def test_json_escapes_cross_every_chunk_boundary(tmp_path, monkeypatch):
    import slipagent.inspection as inspection
    monkeypatch.setattr(inspection, "CHUNK_SIZE", 1)
    store = ArtifactStore(10000)
    store.use_directory(tmp_path)
    record = store.save("cmd", '😀\nquote" slash\\ tab\t', kind="command_output", step_id=1, call_id="a")
    path = store.record_path(record["record_id"])
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data, ensure_ascii=True))
    snapshot = artifact_snapshot(path, record)
    try:
        # Tiny display segments also retain the source-line mapping.
        assert "".join(snapshot[snapshot.line_index(3):]) == '😀quote" slash\\ tab\t'
    finally:
        snapshot.close()


@pytest.mark.parametrize("change", ["duplicate", "trailing", "wrong_metadata", "wrong_type"])
def test_streaming_record_validation(tmp_path, change):
    store = ArtifactStore(10000)
    store.use_directory(tmp_path)
    record = store.save("cmd", "output", kind="command_output", call_id=None, step_id=1)
    path = store.record_path(record["record_id"])
    raw = path.read_text()
    if change == "duplicate":
        raw = raw.replace('"output":', '"command": "cmd", "output":')
    elif change == "trailing":
        raw += "{}"
    elif change == "wrong_metadata":
        raw = raw.replace('"cmd"', '"other"')
    else:
        raw = raw.replace('"call_id": null', '"call_id": "null"')
    path.write_text(raw)
    with pytest.raises(ValueError):
        artifact_snapshot(path, record)


def test_call_snapshot_keeps_literal_strings():
    snapshot = field_snapshot({"result": 'first\n"quote" and \\n', "arguments": {"path": "a"}})
    try:
        assert snapshot[1] == "  first"
        assert snapshot[2] == '  "quote" and \\n'
    finally:
        snapshot.close()


@pytest.mark.parametrize("ending", [b"\r", b"\r\n", b"\n"])
def test_file_line_links_support_line_endings(tmp_path, ending):
    path = tmp_path / "lines.txt"
    path.write_bytes(ending.join([b"one", b"two", b"three"]))
    snapshot = IndexedText.from_path(path)
    try:
        assert snapshot[snapshot.line_index(2)] == "two"
        assert snapshot[-1] == "three"
    finally:
        snapshot.close()


def test_quota_failure_removes_partial_record(tmp_path):
    store = ArtifactStore(100)
    store.use_directory(tmp_path)
    with pytest.raises(OSError):
        store.save("cmd", ("x" * 4096 for _ in range(10)), kind="command_output")
    assert not list(tmp_path.glob("*.json"))
    assert not store.records and store.used_bytes == 0


async def test_cancelled_snapshot_waits_for_worker_and_closes_files():
    import threading
    from slipagent.inspection import build_snapshot
    started, release = threading.Event(), threading.Event()
    snapshot = IndexedText()
    def build():
        started.set()
        release.wait(5)
        snapshot.text.write("complete")
        return snapshot.finish()
    task = asyncio.create_task(build_snapshot(build))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert snapshot.text.closed and snapshot._index.closed and snapshot.logical.closed
