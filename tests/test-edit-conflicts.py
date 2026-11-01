"""Edits preserve external changes made before atomic publication."""

import os

import pytest

from slipagent.checkpoints import FileCheckpoints, current_checkpoint
from slipagent.tools import files, hashline


@pytest.mark.parametrize('anchored', [False, True])
@pytest.mark.parametrize('timing', ['planning', 'fsync'])
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('deleted', [False, True])
async def test_external_change_rejects_edit(workspace, monkeypatch, anchored, timing, checkpointed, deleted):
    path = workspace.root / 'example.txt'
    before = b'before\r\nuntouched\r\n'
    external = b'before\r\nexternal change\r\n'
    path.write_bytes(before)

    def change():
        if deleted:
            path.unlink()
        else:
            path.write_bytes(external)

    if timing == 'planning':
        owner = hashline if anchored else files
        name = 'apply_hashline_edits' if anchored else 'apply_edits'
        plan = getattr(owner, name)

        def changed_plan(*args):
            result = plan(*args)
            change()
            return result

        monkeypatch.setattr(owner, name, changed_plan)
    else:
        publish = files._atomic_write_bytes
        sync = os.fsync

        def changed_sync(descriptor):
            sync(descriptor)
            change()

        def changed_publish(*args, **kwargs):
            with monkeypatch.context() as patch:
                patch.setattr(os, 'fsync', changed_sync)
                return publish(*args, **kwargs)

        monkeypatch.setattr(files, '_atomic_write_bytes', changed_publish)

    store = FileCheckpoints(workspace)
    token = current_checkpoint.set(store if checkpointed else None)
    if checkpointed:
        store.begin(1)
    try:
        if anchored:
            view = hashline.anchored_view(before)
            result = await hashline.EditHashlinesTool(workspace).invoke({
                'path': path.name, 'revision': view['revision'],
                'edits': [{'op': 'replace', 'start': view['lines'][0]['anchor'], 'lines': ['after']}]})
        else:
            result = await files.EditFileTool(workspace).invoke({
                'path': path.name, 'old_string': 'before', 'new_string': 'after'})
        assert result.is_error
        if deleted:
            assert not path.exists()
        else:
            assert path.read_bytes() == external
        assert not list(workspace.root.glob('.slipagent-write-*'))
        assert all(entry['status'] != 'done' for batch in store.listing() for entry in batch['files'])
    finally:
        current_checkpoint.reset(token)
        store.finish()
        await store.aclose()
