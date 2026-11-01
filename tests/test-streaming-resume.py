"""Cold replay retains indexes and one record/batch rather than all payloads."""

import gc
import tracemalloc

import pytest

from slipagent.agent import Agent
from slipagent.sessions import SessionJournal
from slipagent.sources import FileText
from slipagent.tools.base import ToolRegistry
from slipagent.turnstore import resident_content
from slipagent.types import Message


def saved_history(workspace, directory, count):
    agent = Agent(None, ToolRegistry(), 'test')
    journal = SessionJournal(str(workspace.root), directory)
    journal.begin(agent)
    for index in range(count):
        journal._append('message', message={'role': 'user', 'content': f'Prompt {index}'})
        journal._append('message', message={'role': 'assistant', 'content': f'{index}:' + 'x' * (256 * 1024)})
    return journal


def test_replay_memory_does_not_retain_all_message_payloads(workspace, tmp_path):
    peaks = []
    for count in (8, 32):
        journal = saved_history(workspace, tmp_path / f'saved-{count}', count)
        tracemalloc.start()
        try:
            loaded = journal.load(journal.session_id)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        try:
            peaks.append(peak)
            assert loaded['messages'][1].content == '0:' + 'x' * (256 * 1024)
        finally:
            loaded['history'].clear()
    assert peaks[1] < 4 * 1024 * 1024
    assert peaks[1] - peaks[0] < 2 * 1024 * 1024


def test_loaded_message_list_keeps_its_sources_alive(workspace, tmp_path):
    journal = saved_history(workspace, tmp_path / 'saved', 2)
    messages = journal.load(journal.session_id)['messages']
    gc.collect()
    assert isinstance(resident_content(messages[1]), FileText)
    assert messages[1].content == '0:' + 'x' * (256 * 1024)


def test_repeated_restore_keeps_system_instructions_after_old_store_closes(workspace, tmp_path):
    instructions = 'Keep the original system instructions. ' * 1000
    original = Agent(None, ToolRegistry(), 'test')
    original.messages = [Message.system(instructions), Message.user('Prompt'), Message.assistant('Answer')]
    journal = SessionJournal(str(workspace.root), tmp_path / 'saved')
    journal.begin(original)
    restored = Agent(None, ToolRegistry(), 'test')
    restored.registry.resources.journal = journal
    for _ in range(2):
        journal.restore(restored, journal.load(journal.session_id))
        assert restored.messages[0].content == instructions
    restored.history.clear()


@pytest.mark.parametrize('pending', [False, True])
def test_fork_relocates_loaded_sources_into_child_storage(workspace, tmp_path, pending):
    journal = saved_history(workspace, tmp_path / 'saved', 2)
    if pending:
        journal._append('message', message={'role': 'user', 'content': 'Unanswered input ' * 20000})
    parent = journal.session_id
    loaded = journal.load(parent)
    agent = Agent(None, ToolRegistry(), 'test')
    agent.registry.resources.journal = journal
    journal.restore(agent, loaded, fork=True)
    child = journal.session_id
    del loaded
    gc.collect()
    content = resident_content(agent.messages[1])
    assert isinstance(content, FileText)
    assert content.path.is_relative_to(journal.directory / (child + '-turns'))
    if pending:
        assert resident_content(agent.messages[-1]).path.is_relative_to(journal.directory / (child + '-turns'))
    journal.delete_saved(parent)
    assert agent.messages[1].content == '0:' + 'x' * (256 * 1024)
    assert journal.load(child)['messages'][1].content == agent.messages[1].content
    if pending:
        assert journal.load(child)['messages'][-1].content == 'Unanswered input ' * 20000
    agent.history.clear()
