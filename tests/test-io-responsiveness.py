"""Request spooling and large journal records leave input processing responsive."""

import asyncio
import threading

import httpx
import pytest

from slipagent.agent import Agent
from slipagent.openrouter import OpenRouterClient
from slipagent.sessions import SessionJournal
from slipagent.tools.base import ToolRegistry
from slipagent.types import Message
from test_agent import StubClient, completion


async def test_request_spool_is_constructed_off_the_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    constructed = []
    original = OpenRouterClient.request_body

    def build(self, payload):
        constructed.append(threading.get_ident())
        return original(self, payload)

    monkeypatch.setattr(OpenRouterClient, 'request_body', build)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        'choices': [{'message': {'role': 'assistant', 'content': 'Done'}, 'finish_reason': 'stop'}]}))
    async with OpenRouterClient('dummy-test-key', transport=transport) as client:
        assert (await client.chat(model='test', messages=[Message.user('payload ' * 1024)])).text == 'Done'
    assert constructed and loop_thread not in constructed


async def test_large_journal_messages_are_serialized_off_the_event_loop(workspace, tmp_path, monkeypatch):
    from slipagent import sources
    loop_thread = threading.get_ident()
    serialized = []
    chunks = sources.json_chunks

    def observed(value, *args, **kwargs):
        if isinstance(value, dict) and value.get('type') == 'message':
            serialized.append(threading.get_ident())
        yield from chunks(value, *args, **kwargs)

    agent = Agent(StubClient([completion('Done')]), ToolRegistry(), 'test')
    journal = SessionJournal(str(workspace.root), tmp_path / 'saved')
    agent.registry.resources.journal = journal
    journal.begin(agent)
    monkeypatch.setattr(sources, 'json_chunks', observed)
    assert await agent.run('payload ' * 1024) == 'Done'
    assert serialized and loop_thread not in serialized
    assert any(message.content == 'payload ' * 1024 for message in journal.load(journal.session_id)['messages'])


async def test_cancelled_spool_settles_worker_and_removes_snapshot(monkeypatch):
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    snapshots, sent = [], []
    original = OpenRouterClient.request_body

    def build(self, payload):
        body = original(self, payload)
        snapshots.append(body)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return body

    monkeypatch.setattr(OpenRouterClient, 'request_body', build)
    async with OpenRouterClient('dummy-test-key', transport=httpx.MockTransport(sent.append)) as client:
        task = asyncio.create_task(client.chat(model='test', messages=[Message.user('payload')]))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not sent and not snapshots[0].path.exists()


async def test_deferred_journal_freezes_records_and_settles_cancelled_flush(workspace, tmp_path, monkeypatch):
    import json
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    agent = Agent(StubClient([]), ToolRegistry(), 'test')
    journal = SessionJournal(str(workspace.root), tmp_path / 'saved')
    journal.begin(agent)
    write = journal._write_record

    def gated(kind, **data):
        if data.get('note') == ['original']:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        write(kind, **data)

    monkeypatch.setattr(journal, '_write_record', gated)
    note = ['original']
    with journal.defer_appends():
        journal._append('state', note=note)
        note[0] = 'mutated after capture'
        journal._append('state', note=['second'])
        flushing = asyncio.create_task(journal.flush())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            journal._append('state', note=['third'])
            flushing.cancel()
            await asyncio.sleep(0)
            assert not flushing.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await flushing
    await journal.flush()
    records = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [record['note'] for record in records if 'note' in record] == [['original'], ['second'], ['third']]


async def test_failed_writer_cannot_leak_queued_records_into_new_session(workspace, tmp_path, monkeypatch):
    from slipagent.sessions import SessionError
    agent = Agent(StubClient([]), ToolRegistry(), 'test')
    journal = SessionJournal(str(workspace.root), tmp_path / 'saved')
    journal.begin(agent)
    parent = journal.path
    original = parent.read_bytes()

    def fail(kind, **data):
        raise OSError('simulated append failure')

    with monkeypatch.context() as patch:
        patch.setattr(journal, '_write_record', fail)
        with journal.defer_appends():
            journal._append('state', note='failed record')
            journal._append('state', note='old pending record')
            with pytest.raises(SessionError, match='append failure'):
                await journal.flush()
    assert journal.failed
    journal.begin(agent)
    assert journal.path != parent and not journal.failed
    assert parent.read_bytes() == original
    assert 'old pending record' not in journal.path.read_text()
