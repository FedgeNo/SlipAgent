"""Exact response bytes survive parsing, interruptions, and rolling retention."""

import asyncio
import gzip
import json
import os
import zlib

import httpx
import pytest

from slipagent.api import APIResponseError
from slipagent.diagnostics import RequestDiagnostics
from slipagent.openrouter import OpenRouterClient
from slipagent.types import Message


REPLY = b'data: {"choices":[{"delta":{"content":"Done."},"finish_reason":"stop"}]}\r\n\r\n'


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, failure=None):
        self.chunks = chunks
        self.failure = failure
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        self.started.set()
        if self.failure == 'disconnect':
            raise httpx.ReadError('disconnected')
        if self.failure == 'cancel':
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize('failure', [None, 'malformed', 'disconnect', 'cancel'])
async def test_raw_stream_preserves_every_received_byte(tmp_path, failure, monkeypatch):
    monkeypatch.setattr('slipagent.responsecapture.FLUSH_INTERVAL', 0.05)
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    attempt = archive.begin({}, step=1, step_id=1)
    chunks = [b': keepalive\r\n\r\n', b'event: completion\nid: 123\nretry: 500\n']
    if failure == 'malformed':
        chunks += [b'data: {broken\xff}\r\n\r\n']
    elif failure in ('disconnect', 'cancel'):
        chunks += [b'data: {"unfinished"']
    else:
        chunks += [REPLY[:13], REPLY[13:], b': last ping\n\ndata: [DONE]\n\n']
    stream = Stream(chunks, failure)
    captured = []
    async with OpenRouterClient('dummy', transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream', 'x-provider': 'stub'}, stream=stream)
    )) as client:
        task = asyncio.create_task(client.chat(
            model='test', messages=[Message.user('work')], on_delta=lambda *args: None,
            on_raw_response=lambda response: archive.capture_response(attempt, response),
            on_response=captured.append, single_attempt=True,
        ))
        if failure == 'cancel':
            try:
                await asyncio.wait_for(stream.started.wait(), 1)
                record = json.loads((tmp_path / '00000001.json').read_text())
                # A stalled stream becomes readable before its gzip footer is finalized.
                deadline = asyncio.get_running_loop().time() + 2
                while True:
                    compressed = (tmp_path / record['raw_response']).read_bytes()
                    received = zlib.decompressobj(31).decompress(compressed)
                    if received == b''.join(chunks) or asyncio.get_running_loop().time() > deadline:
                        break
                    await asyncio.sleep(0.01)
                assert received == b''.join(chunks)
            finally:
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif failure is not None:
            with pytest.raises(APIResponseError):
                await task
        else:
            assert (await task).text == 'Done.'
    assert stream.closed and captured
    assert not archive.error
    record = json.loads((tmp_path / '00000001.json').read_text())
    raw = tmp_path / record['raw_response']
    assert gzip.decompress(raw.read_bytes()) == b''.join(chunks)
    assert record['response_status'] == 200
    assert ['x-provider', 'stub'] in record['response_headers']
    assert archive.used == sum(path.stat().st_size for path in tmp_path.iterdir() if path.is_file())
    if os.name == 'posix':
        assert raw.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('streamed', [False, True])
async def test_raw_capture_precedes_http_content_decoding(tmp_path, streamed):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    attempt = archive.begin({}, step=1, step_id=1)
    body = json.dumps({'choices': [{'message': {'content': 'Done.'}, 'finish_reason': 'stop'}]}).encode()
    encoded = gzip.compress(body, mtime=0)
    stream = Stream([encoded[:12], encoded[12:]])
    async with OpenRouterClient('dummy', transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'application/json', 'content-encoding': 'gzip'}, stream=stream)
    )) as client:
        reply = await client.chat(model='test', messages=[Message.user('work')],
                                  on_delta=(lambda *args: None) if streamed else None,
                                  on_raw_response=lambda response: archive.capture_response(attempt, response))
    assert reply.text == 'Done.'
    record = json.loads((tmp_path / '00000001.json').read_text())
    assert gzip.decompress((tmp_path / record['raw_response']).read_bytes()) == encoded
    assert ['content-encoding', 'gzip'] in record['response_headers']
    assert not record['raw_response_decoded']


async def test_raw_capture_io_failure_does_not_interrupt_model_output(tmp_path, monkeypatch):
    archive = RequestDiagnostics()
    archive.use_directory(tmp_path)
    attempt = archive.begin({}, step=1, step_id=1)
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(gzip.GzipFile, 'write', fail)
    async with OpenRouterClient('dummy', transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Stream([REPLY, b'data: [DONE]\n\n']))
    )) as client:
        reply = await client.chat(model='test', messages=[Message.user('work')], on_delta=lambda *args: None,
                                  on_raw_response=lambda response: archive.capture_response(attempt, response))
    assert reply.text == 'Done.'
    assert archive.error == 'disk full'


async def test_rolling_eviction_includes_raw_stream_files(tmp_path):
    archive = RequestDiagnostics(1500)
    archive.use_directory(tmp_path)
    for number in (1, 2):
        attempt = archive.begin({}, step=number, step_id=number)
        stream = Stream([b': ' + os.urandom(3000).hex().encode() + b'\n\n', REPLY, b'data: [DONE]\n\n'])
        async with OpenRouterClient('dummy', transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
        )) as client:
            await client.chat(model='test', messages=[Message.user('work')], on_delta=lambda *args: None,
                              on_raw_response=lambda response: archive.capture_response(attempt, response))
        archive.finish(attempt, 'accepted')
    assert not archive.error
    records = archive.listing()
    assert [record['attempt'] for record in records] == [2]
    assert not list(tmp_path.glob('00000001*'))
    assert gzip.decompress((tmp_path / records[0]['raw_response']).read_bytes()) == b''.join(stream.chunks)
    assert archive.used == sum(path.stat().st_size for path in tmp_path.iterdir() if path.is_file())
