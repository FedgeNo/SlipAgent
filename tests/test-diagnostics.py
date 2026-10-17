import gzip
import json
import os

import httpx
import pytest

from slipagent.diagnostics import RequestDiagnostics
from slipagent.openrouter import OpenRouterClient
from slipagent.types import Message


@pytest.mark.parametrize("oversized", [False, True])
def test_rolling_logging_keeps_newest_complete_attempt(tmp_path, oversized):
    archive = RequestDiagnostics(1500)
    archive.use_directory(tmp_path)
    for index in range(1, 5):
        attempt = archive.begin({"messages": [{"role": "user", "content": str(index)}]}, step=index, step_id=index)
        assert attempt == index
        response = os.urandom(4096).hex() if oversized else "response " * 50
        body = {"choices": [{"message": {"content": response, "tool_calls": [
            {"id": "call", "function": {"name": "run_command", "arguments": '{"command":"pwd"}'}}
        ]}}]}
        archive.response(attempt, body)
        archive.finish(attempt, "accepted", response=response)
        assert not archive.error
        record = json.loads((tmp_path / f"{index:08d}.json").read_text())
        assert record["outcome"] == "accepted" and record["response_excerpt"] == response
        assert json.loads(gzip.decompress((tmp_path / record["provider_response"]).read_bytes())) == body
        assert archive.used == sum(path.stat().st_size for path in tmp_path.iterdir())
    assert [record['attempt'] for record in archive.listing()] == [4]
    assert len(list(tmp_path.iterdir())) == 3
    assert (archive.used > 1500) == oversized
    with pytest.raises(ValueError, match="no longer retained"):
        archive.read(1)


@pytest.mark.parametrize("operation", ["begin", "response", "finish"])
@pytest.mark.parametrize("quota", [None, 32 * 1024 * 1024])
def test_live_logging_recovers_from_legacy_default_quota(tmp_path, operation, quota):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    attempt = archive.begin({}, step=1, step_id=1)
    # A retained pre-reload owner has only the old constructor's fields.
    archive.__dict__.pop("_rolling_quota", None)
    archive.quota = quota
    if quota is not None:
        archive.error = f"Request diagnostic quota of {quota} bytes reached; existing records are preserved"
    if operation == "begin":
        assert archive.begin({}, step=2, step_id=2) == 2
    elif operation == "response":
        archive.response(attempt, {"choices": []})
        assert (tmp_path / "00000001-response.json.gz").is_file()
    else:
        archive.finish(attempt, "accepted", response="done")
        assert json.loads((tmp_path / "00000001.json").read_text())["outcome"] == "accepted"
    assert not archive.error


def test_live_migration_preserves_custom_quota_and_io_errors(tmp_path):
    archive = RequestDiagnostics(1000)
    archive.use_directory(tmp_path)
    archive.__dict__.pop("_rolling_quota", None)
    archive.error = "permission denied"
    assert archive.begin({}, step=1, step_id=1) is None
    assert archive.error == "permission denied"
    archive.error = ""
    attempt = archive.begin({}, step=1, step_id=1)
    archive.finish(attempt, "accepted", response="x" * 4000)
    assert not archive.error
    assert "x" * 4000 in archive.read(attempt)
    archive.begin({}, step=2, step_id=2)
    assert [record['attempt'] for record in archive.listing()] == [2]


def test_resume_preserves_attempts_and_counts_existing_storage(tmp_path):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    first = archive.begin({"messages": []}, step=1, step_id=1)
    archive.finish(first, "success")
    original = (tmp_path / "00000001.json").read_bytes()
    # Even an incomplete/corrupt record must not be replaced on resume.
    (tmp_path / "00000005.json").write_bytes(b"incomplete")
    resumed = RequestDiagnostics()
    resumed.use_directory(tmp_path)
    assert resumed.used == sum(path.stat().st_size for path in tmp_path.iterdir())
    assert resumed.begin({"messages": []}, step=2, step_id=2) == 6
    assert (tmp_path / "00000001.json").read_bytes() == original
    assert (tmp_path / "00000005.json").read_bytes() == b"incomplete"


def test_resume_rolls_existing_archive_forward(tmp_path):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    archive.begin({"messages": []}, step=1, step_id=1)
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    resumed = RequestDiagnostics(quota=sum(map(len, original.values())))
    resumed.use_directory(tmp_path)
    assert resumed.begin({"messages": []}, step=2, step_id=2) == 2
    assert not resumed.error
    assert [record['attempt'] for record in resumed.listing()] == [2]
    assert len(list(tmp_path.glob('*.gz'))) == 1
    assert '"messages": []' in resumed.read(2)


def test_exact_request_deduplication_and_failed_attempt_retention(tmp_path):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path / "requests")
    request = json.dumps({"messages": [{"role": "user", "content": "exact 日本語"}]}, ensure_ascii=False)
    first = archive.begin(request, step=1, step_id=1)
    archive.finish(first, "request_error", detail="Disconnected", response="partial thoughts")
    second = archive.begin(request, step=1, step_id=1)
    archive.finish(second, "accepted", response="done")
    files = list(archive.directory.glob("*.gz"))
    assert len(files) == 1 and gzip.decompress(files[0].read_bytes()).decode() == request
    assert [row["outcome"] for row in archive.listing()] == ["request_error", "accepted"]
    assert "partial thoughts" in archive.read(first)
    if os.name == "posix":
        assert files[0].stat().st_mode & 0o777 == 0o600
    directory = archive.directory
    archive.clear()
    assert files[0].exists() and directory.exists()


def test_rolling_removes_shared_snapshots_only_after_last_reference(tmp_path):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    first = archive.begin('shared request', step=1, step_id=1)
    archive.finish(first, 'accepted')
    second = archive.begin('shared request', step=2, step_id=2)
    archive.finish(second, 'accepted')
    shared = next(tmp_path.glob('*.gz'))
    archive.quota = archive.used + 50
    third = archive.begin('different request', step=3, step_id=3)
    assert not (tmp_path / '00000001.json').exists()
    assert 'shared request' in archive.read(second)
    assert shared.exists()
    archive.finish(third, 'accepted', response='x' * 4000)
    assert not shared.exists()
    assert 'different request' in archive.read(third)


def test_failed_write_preserves_prior_attempts(tmp_path, monkeypatch):
    archive = RequestDiagnostics(1000)
    archive.use_directory(tmp_path)
    first = archive.begin('first', step=1, step_id=1)
    archive.finish(first, 'accepted')
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr('slipagent.diagnostics.os.replace', fail)
    assert archive.begin('second', step=2, step_id=2) is None
    assert archive.error == 'disk full'
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_rolling_only_removes_owned_diagnostics(tmp_path):
    outside = tmp_path / 'retained.txt'
    outside.write_text('Keep this file')
    directory = tmp_path / 'requests'
    directory.mkdir()
    (directory / '00000001.json').write_text(json.dumps({'request': '../retained.txt'}))
    (directory / '00000002.json').write_text('incomplete')
    unknown = directory / 'unrelated.bin'
    unknown.write_bytes(b'x' * 2000)
    orphan = directory / ('a' * 64 + '.json.gz')
    orphan.write_bytes(b'interrupted snapshot')
    archive = RequestDiagnostics(1000)
    archive.use_directory(directory)
    assert archive.begin('latest', step=3, step_id=3) == 3
    assert not archive.error
    assert outside.read_text() == 'Keep this file'
    assert unknown.read_bytes() == b'x' * 2000
    assert not orphan.exists()
    assert 'latest' in archive.read(3)
    assert [record['attempt'] for record in archive.listing()] == [3]


@pytest.mark.parametrize("streamed", [False, True])
async def test_complete_response_saved_before_parsing_without_fragment_list(tmp_path, streamed):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    attempt = archive.begin({"messages": []}, step=1, step_id=1)
    thought = "some reasoning\n" * 2000
    message = {"role": "assistant", "reasoning": thought, "content": '```json\n{"response":"done"}\n```',
               "provider_extra": {"preserve": True},
               "tool_calls": [{"index": 0, "function": {"name": "record", "arguments": '{broken'}, "extra": "keep"}]}
    full = {"id": "response-id", "provider": "stub", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}]}
    def handle(request):
        if not streamed:
            return httpx.Response(200, json=full)
        events = [
            {"id": "response-id", "provider": "stub", "choices": [{"index": 0, "delta": {"reasoning": thought[:16000]}}]},
            {"choices": [{"index": 0, "delta": {**message, "reasoning": thought[16000:]}, "finish_reason": "tool_calls"}]},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=''.join('data: ' + json.dumps(event) + '\n\n' for event in events) + 'data: [DONE]\n\n')
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(handle)) as client:
        completion = await client.chat(model="test", messages=[Message.user("test")],
                                       on_delta=(lambda *args: None) if streamed else None,
                                       on_response=lambda body: archive.response(attempt, body))
    assert completion.response_error
    archive.finish(attempt, "rejected", response=thought)
    record = json.loads((tmp_path / "00000001.json").read_text())
    body = json.loads(gzip.decompress((tmp_path / record["provider_response"]).read_bytes()))
    assert body["provider"] == "stub"
    assert body["choices"][0]["delta" if streamed else "message"] == message
    assert record["response_excerpt"] == thought
    assert thought in archive.read(attempt).replace('\\n', '\n')


async def test_partial_response_is_logged_when_stream_breaks(tmp_path):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    attempt = archive.begin({}, step=1, step_id=1)
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning":"partial","tool_calls":[{"index":0,"function":{"arguments":"{unfinished"}}]}}]}\n\n'
            raise httpx.ReadError("disconnected")
    async with OpenRouterClient("dummy", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Broken())
    )) as client:
        with pytest.raises(Exception, match="interrupted"):
            await client.chat(model="test", messages=[Message.user("test")], single_attempt=True,
                              on_delta=lambda *args: None, on_response=lambda body: archive.response(attempt, body))
    body = json.loads(gzip.decompress((tmp_path / "00000001-response.json.gz").read_bytes()))
    assert body["choices"][0]["delta"]["reasoning"] == "partial"
    assert body["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "{unfinished"
