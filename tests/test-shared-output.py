"""Versioned command payloads preserve both readers and crash-safe migration."""

import io
import json
import os
import shutil
import tracemalloc
from importlib import import_module

import pytest

from slipagent.agent import Agent
from slipagent.artifacts import ArtifactStore, ReadArtifactTool
from slipagent.inspection import artifact_snapshot
from slipagent.sessions import SessionJournal
from slipagent.tools import build_default_registry
from slipagent.tools.output import CommandArchive, ReadCommandOutputTool
from test_agent import StubClient

run_copy = import_module('test-runtime').run_copy


@pytest.fixture
async def storage(tmp_path):
    store = ArtifactStore(8_000_000)
    store.use_directory(tmp_path / 'artifacts')
    archive = CommandArchive(8_000_000, directory=tmp_path)
    archive.artifacts = store
    try:
        yield archive, store
    finally:
        await archive.aclose()
        await store.aclose()


def legacy_log(archive, store, text='old output\n', *, saved=None):
    log = archive.start('legacy command', step_id=2, call_id='old-call')
    log.write('stdout', text)
    log.artifact_record = store.save(log.command, text if saved is None else saved,
                                    kind='command_output', step_id=2, call_id='old-call')
    return log


async def test_both_readers_share_one_payload_and_keep_offsets(storage):
    archive, store = storage
    log = archive.start('build', step_id=1, call_id='build')
    text = 'α\r\n{"path":"a\\b"}\n'
    log.write('stdout', text)
    old = log.streams['stdout'].path
    log.write('stderr', 'warning\n')
    log.finish(0, False)
    record = log.artifact_record
    assert record['format_version'] == 2 and not old.exists()
    assert log.streams['stdout'].path.parent == store.directory / 'payloads'
    header = json.loads(store.record_path(record['record_id']).read_text())
    assert header['format_version'] == 2 and header['output'] == record['payloads']['output']['sha256']
    assert store.read(record['record_id'])['output'] == text
    page = await ReadArtifactTool(store).run(record['record_id'], offset=1, limit=3)
    assert page.content['output'] == text[1:4] and page.content['next_offset'] == 4
    raw = await ReadCommandOutputTool(archive).run(log_id=log.id, offset=2, limit=3)
    assert raw.content['content'] == text[1:4] and raw.content['offset_unit'] == 'UTF-8 bytes'
    snapshot = artifact_snapshot(store.record_path(record['record_id']), record)
    try:
        snapshot.text.seek(0)
        displayed = snapshot.text.read()
        assert text in displayed and 'warning\n' in displayed
    finally:
        snapshot.close()


async def test_physical_storage_has_no_second_stream_copy(storage):
    archive, store = storage
    text = '{"escaped":"a\\b","status":"done"}\n' * 30000
    for _ in range(2):
        log = archive.start('same output')
        log.write('stdout', text)
        log.finish(0, False)
    payloads = list((store.directory / 'payloads').iterdir())
    assert len(payloads) == 1 and payloads[0].read_text() == text
    size = sum(path.stat().st_size for path in store.directory.rglob('*') if path.is_file())
    size += sum(path.stat().st_size for path in archive.directory.rglob('*') if path.is_file())
    assert size < len(text.encode()) * 1.01
    assert not list(archive.directory.iterdir())
    if os.name == 'posix':
        assert payloads[0].stat().st_mode & 0o077 == 0


async def test_shared_payload_paging_and_viewer_keep_memory_bounded(storage):
    archive, store = storage
    log = archive.start('large output')
    text = ('x' * 1000 + '\n') * 4000
    log.write('stdout', text)
    log.finish(0, False)
    record = log.artifact_record
    tracemalloc.start()
    snapshot = None
    try:
        page = store.read_page(record['record_id'], 'output', len(text) - 30, 30)
        assert page['output'] == text[-30:] and page['eof']
        snapshot = artifact_snapshot(store.record_path(record['record_id']), record)
        assert snapshot.logical_count == 4002
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 2_000_000
    finally:
        if snapshot is not None:
            snapshot.close()
        tracemalloc.stop()


async def test_migration_keeps_record_id_and_deletes_only_verified_copies(storage):
    archive, store = storage
    log = legacy_log(archive, store)
    old_record = log.artifact_record
    old_path = store.record_path(old_record['record_id'])
    old_stream = log.streams['stdout'].path
    log.finish(0, False)
    assert log.artifact_record['record_id'] == old_record['record_id']
    assert log.artifact_record['format_version'] == 2
    assert not old_path.exists() and not old_stream.exists()
    assert store.read(old_record['record_id'])['output'] == 'old output\n'


async def test_mismatched_legacy_output_is_preserved(storage):
    archive, store = storage
    log = legacy_log(archive, store, saved='different inspection output')
    raw_path = log.streams['stdout'].path
    log.finish(0, False)
    assert 'both copies were retained' in log.artifact_error
    assert 'format_version' not in log.artifact_record
    assert raw_path.read_text() == 'old output\n'
    assert store.read(log.artifact_record['record_id'])['output'] == 'different inspection output'


async def test_failed_index_publication_preserves_originals_and_can_retry(storage, monkeypatch):
    archive, store = storage
    log = legacy_log(archive, store)
    original = log.streams['stdout'].path
    old_record = dict(log.artifact_record)
    replace = os.replace
    def fail_index(source, target):
        if str(target).endswith('index.json'):
            raise OSError('simulated index failure')
        replace(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', fail_index)
        log.finish(0, False)
    assert original.exists() and store.records[old_record['record_id']] == old_record
    assert 'simulated index failure' in log.artifact_error
    assert not list((store.directory / 'payloads').iterdir())
    log.consolidate()
    log.remove_legacy_output()
    assert log.artifact_record['format_version'] == 2 and not original.exists()


async def test_failed_journal_does_not_delete_original_raw_stream(storage):
    archive, store = storage
    log = legacy_log(archive, store)
    original = log.streams['stdout'].path
    old_record_path = store.record_path(log.artifact_record['record_id'])
    def fail(log):
        raise OSError('journal unavailable')
    archive.on_log = fail
    with pytest.raises(OSError, match='journal unavailable'):
        log.finish(0, False)
    assert original.exists() and old_record_path.exists()
    assert store.read(log.artifact_record['record_id'])['output'] == 'old output\n'
    archive.on_log = None
    log.remove_legacy_output()
    assert not original.exists() and not old_record_path.exists()


async def test_retry_after_index_commit_durability_failure(storage, monkeypatch):
    import slipagent.payloads as payloads
    archive, store = storage
    log = legacy_log(archive, store)
    original = log.streams['stdout'].path
    sync = payloads.sync_directory
    def fail_after_index(directory):
        if directory == store.directory and any(record.get('format_version') == 2 for record in store.records.values()):
            raise OSError('directory durability failure')
        sync(directory)
    with monkeypatch.context() as patch:
        patch.setattr(payloads, 'sync_directory', fail_after_index)
        log.finish(0, False)
    record = store.records[log.artifact_record['record_id']]
    assert record['format_version'] == 2 and original.exists()
    assert store.read(record['record_id'])['output'] == 'old output\n'
    log.consolidate()
    log.remove_legacy_output()
    assert not original.exists() and not (store.directory / f"{record['record_id']}.json").exists()


async def test_retry_after_orphan_version_two_header(storage):
    archive, store = storage
    log = legacy_log(archive, store)
    old_record = dict(log.artifact_record)
    old_index = (store.directory / 'index.json').read_bytes()
    # Simulate a crash after the new header was written but before index publication.
    log.streams['stdout'].file.flush()
    store.save_command(log.command, {name: stream.path for name, stream in log.streams.items()},
                       kind='command_output', step_id=2, call_id='old-call', error_suffix='',
                       replace_id=old_record['record_id'])
    (store.directory / 'index.json').write_bytes(old_index)
    store.use_directory(store.directory)
    log.finish(0, False)
    assert log.artifact_record['format_version'] == 2
    assert store.read(old_record['record_id'])['output'] == 'old output\n'


async def test_recovery_cleans_verified_legacy_raw_files_after_journal_barrier(storage):
    archive, store = storage
    log = legacy_log(archive, store)
    log.streams['stdout'].file.close()
    log.streams['stdout'].file = None
    original = log.streams['stdout'].path
    log.finished = True
    log.consolidate()
    assert original.exists()
    # The journal has new payload paths but cleanup did not run before a crash.
    log.__dict__.pop('_obsolete_output')
    log.consolidate()
    log.remove_legacy_output()
    assert not original.exists()
    assert store.read(log.artifact_record['record_id'])['output'] == 'old output\n'


async def test_unknown_versions_and_payload_path_traversal_are_rejected(storage):
    archive, store = storage
    log = archive.start('version')
    log.write('stdout', 'text')
    log.finish(0, False)
    record = store.records[log.artifact_record['record_id']]
    index = store.directory / 'index.json'
    original = index.read_text()
    index.write_text(json.dumps([{**record, 'format_version': 3}]))
    with pytest.raises(ValueError, match='index'):
        ArtifactStore(100000).use_directory(store.directory)
    index.write_text(original)
    record['payloads']['output']['sha256'] = '../../outside'
    result = await ReadArtifactTool(store).run(record['record_id'])
    assert result.is_error


async def test_quota_failure_retains_log_and_reports_missing_inspection_record(storage):
    archive, store = storage
    store.quota_bytes = 10
    log = archive.start('quota')
    log.write('stdout', 'retained output')
    log.finish(0, False)
    assert 'quota' in log.artifact_error and not hasattr(log, 'artifact_record')
    assert (await ReadCommandOutputTool(archive).run(log_id=log.id)).content['content'] == 'retained output'
    assert not list((store.directory / 'payloads').iterdir())


async def test_payload_tampering_and_symlink_are_rejected(storage, tmp_path):
    archive, store = storage
    log = archive.start('integrity')
    log.write('stdout', 'original')
    log.finish(0, False)
    path = log.streams['stdout'].path
    path.write_text('changed!')
    result = await ReadArtifactTool(store).run(log.artifact_record['record_id'])
    assert result.is_error and 'integrity' in result.content
    path.unlink()
    secret = tmp_path / 'other-file'
    secret.write_text('do not read this')
    path.symlink_to(secret)
    assert (await ReadCommandOutputTool(archive).run(log_id=log.id)).is_error
    assert (await ReadArtifactTool(store).run(log.artifact_record['record_id'])).is_error


@pytest.mark.parametrize('legacy', [False, True])
async def test_resume_and_fork_preserve_readers_without_parent_dependency(workspace, tmp_path, legacy):
    registry = build_default_registry(workspace)
    agent = Agent(StubClient([]), registry, 'test', system_prompt='Test')
    journal = SessionJournal(str(workspace.root), tmp_path / 'sessions')
    registry.services['session_journal'] = journal
    journal.begin(agent)
    archive, store = registry.resources.archive, registry.resources.artifacts
    log = archive.start('saved command')
    log.write('stdout', 'persistent α output\n')
    if legacy:
        log.streams['stdout'].file.close()
        log.streams['stdout'].file = None
        log.finished, log.returncode = True, 0
        log.artifact_record = store.save(log.command, 'persistent α output\n', kind='command_output')
        journal.log(log)
    else:
        log.finish(0, False)
    parent = journal.session_id
    saved = journal.load(parent)
    identifier = log.artifact_record['record_id']
    try:
        for fork in (False, True):
            restored_registry = build_default_registry(workspace)
            restored = Agent(StubClient([]), restored_registry, 'test', system_prompt='Test')
            other = SessionJournal(str(workspace.root), tmp_path / 'sessions')
            restored_registry.services['session_journal'] = other
            try:
                other.restore(restored, saved, fork=fork)
                if fork:
                    shutil.rmtree(journal.directory / (parent + '-artifacts'))
                    shutil.rmtree(journal.directory / (parent + '-logs'))
                output = await ReadCommandOutputTool(restored_registry.resources.archive).run(log_id=log.id)
                record = await ReadArtifactTool(restored_registry.resources.artifacts).run(identifier)
                assert output.content['content'] == record.content['output'] == 'persistent α output\n'
                assert restored_registry.resources.artifacts.records[identifier]['format_version'] == 2
                assert not list(restored_registry.resources.archive.directory.iterdir())
            finally:
                await restored_registry.aclose()
    finally:
        await registry.aclose()


def test_active_log_survives_accepted_and_rejected_reload(run_copy):
    run_copy('''
        from slipagent.tools.output import ReadCommandOutputTool
        jobs = registry.resources.jobs
        archive = registry.resources.archive
        log = archive.start('temporary output producer')
        gate = asyncio.Event()
        async def produce():
            log.write('stdout', 'before reload\\n')
            await gate.wait()
            log.write('stdout', 'after reload\\n')
            return ToolResult.ok('done')
        job = jobs.start(log, produce)
        await asyncio.sleep(0)
        frame.request()
        await frame.checkpoint()
        assert frame.generation == 1, sink.getvalue()
        edit('payloads.py', 'CHUNK_BYTES = 65536', 'CHUNK_BYTES = (')
        await frame.checkpoint()
        assert frame.generation == 1
        gate.set()
        await job.task
        record = log.artifact_record
        assert record['format_version'] == 2
        assert registry.resources.artifacts.read(record['record_id'])['output'] == 'before reload\\nafter reload\\n'
        page = await ReadCommandOutputTool(archive).run(log_id=log.id)
        assert page.content['content'] == 'before reload\\nafter reload\\n'
        assert archive is session.registry.resources.archive
    ''')
