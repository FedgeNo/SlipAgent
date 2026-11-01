"""Fuzzy path selection preserves prompt state and rejects obsolete workers."""

import asyncio
import threading
import unicodedata
from importlib import import_module

import pytest
from prompt_toolkit.input import create_pipe_input

from slipagent.filepicker import MAX_RESULTS, PathMatch, match_path, scan_paths
from slipagent.terminal import TerminalUI
from slipagent.workspace import Workspace

terminal_tests = import_module('test-terminal')
display = terminal_tests.display
wait_until = terminal_tests.wait_until
send_keys = terminal_tests.send_keys


@pytest.mark.parametrize('path,query,expected', [
    ('src/agent.py', 'agpy', True), ('src/agent.py', 'AG PY', True),
    ('src/agent.py', 's/a', True), ('src/agent.py', 'missing', False),
    ('Straße.py', 'strasse', True), ('cafe\u0301.py', 'café', True),
    ('src/😀notes.py', '😀n', True),
])
def test_fuzzy_matching_uses_unicode_and_path_subsequences(path, query, expected):
    match = match_path(path, query)
    assert (match is not None) is expected
    if match is not None:
        from wcwidth import iter_graphemes
        clusters = list(iter_graphemes(path))
        highlighted = ''.join(clusters[index] for index in sorted(match.positions))
        assert unicodedata.normalize('NFC', highlighted).casefold()


def test_scan_is_bounded_ranked_and_respects_workspace_ignores_and_links(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'agent.py').write_text('top match')
    folder = root / 'long-folder'
    folder.mkdir()
    for index in range(80):
        (folder / f'agent_extra_{index}.py').touch()
    ignored = root / 'node_modules'
    ignored.mkdir()
    (ignored / 'agent.py').touch()
    outside = tmp_path / 'outside'
    outside.write_text('private')
    (root / 'agent-link.py').symlink_to(outside)
    progress = []
    scan_paths(Workspace(root), 'agent', threading.Event(), lambda *result: progress.append(result))
    matches, count, complete = progress[-1]
    assert complete and count == 81 and len(matches) == MAX_RESULTS
    assert matches[0].path == 'agent.py'
    assert all('node_modules' not in match.path and 'agent-link' not in match.path for match in matches)
    assert all(len(snapshot[0]) <= MAX_RESULTS for snapshot in progress)


async def test_picker_typing_opens_read_only_file_without_submitting_draft(display, tmp_path):
    sink, _, output, _, snapshot = display
    path = tmp_path / 'alpha_file.py'
    path.write_text('SELECTED_FILE_CONTENT\n')
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: 'status', sink, input=pipe, output=output, project_directory=tmp_path)
        running = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: any('Ready' in row for row in snapshot()))
            ui.input.text = 'Unsubmitted prompt'
            await send_keys(ui, pipe, '\x10')
            await wait_until(lambda: ui._file_picker.active)
            await send_keys(ui, pipe, 'afpy')
            await wait_until(lambda: ui._menu_options and ui._menu_options[0][0] == path.name)
            await send_keys(ui, pipe, '\r')
            await wait_until(lambda: ui._viewer_active)
            await wait_until(lambda: any('SELECTED_FILE_CONTENT' in row for row in snapshot()))
            assert ui.input.text == 'Unsubmitted prompt'
            assert ui._lines.empty() and not ui._editor_active
            assert path.read_text() == 'SELECTED_FILE_CONTENT\n'
        finally:
            ui.close()
            await running


async def test_picker_escape_preserves_draft_and_does_not_interrupt_agent(display, tmp_path):
    sink, _, output, _, snapshot = display
    (tmp_path / 'file.txt').touch()
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: 'status', sink, input=pipe, output=output, project_directory=tmp_path)
        interrupted = []
        ui.set_interrupt_handler(lambda: interrupted.append(True))
        running = asyncio.create_task(ui.run())
        try:
            await wait_until(lambda: any('Ready' in row for row in snapshot()))
            ui.input.text = 'Keep this draft'
            ui.set_working(True)
            await send_keys(ui, pipe, '\x10')
            await wait_until(lambda: ui._file_picker.active)
            await send_keys(ui, pipe, 'not-found')
            await wait_until(lambda: '0 matches' in ui._file_picker.status)
            await send_keys(ui, pipe, '\r')
            assert ui._file_picker.active
            await send_keys(ui, pipe, '\x1b')
            await wait_until(lambda: not ui._file_picker.active)
            assert not interrupted and ui.working
            assert ui.input.text == 'Keep this draft' and ui._lines.empty()
        finally:
            ui.close()
            await running


async def test_picker_rechecks_access_when_selected_path_changes(display, tmp_path):
    sink, _, output, _, snapshot = display
    root = tmp_path / 'project'
    root.mkdir()
    path = root / 'visible.txt'
    path.write_text('Original')
    outside = tmp_path / 'private.txt'
    outside.write_text('PRIVATE_CONTENT_MUST_NOT_OPEN')
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: 'status', sink, input=pipe, output=output, project_directory=root)
        running = asyncio.create_task(ui.run())
        choosing = None
        try:
            await wait_until(lambda: any('Ready' in row for row in snapshot()))
            choosing = asyncio.create_task(ui.find_file(Workspace(root), 'visible'))
            await wait_until(lambda: ui._menu_options and ui._menu_options[0][0] == 'visible.txt')
            path.unlink()
            path.symlink_to(outside)
            await send_keys(ui, pipe, '\r')
            await choosing
            assert not ui._viewer_active
            assert 'PRIVATE_CONTENT_MUST_NOT_OPEN' not in sink.getvalue()
        finally:
            if choosing is not None:
                choosing.cancel()
                await asyncio.gather(choosing, return_exceptions=True)
            ui.close()
            await running


async def test_old_query_results_cannot_replace_new_query(display, tmp_path, monkeypatch):
    from slipagent import filepicker
    sink, _, output, _, snapshot = display
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    threads = []

    def scan(workspace, query, stop, publish):
        threads.append(threading.get_ident())
        if query == 'old':
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        publish([PathMatch(query + '.py', 0, frozenset())], 1, True)

    monkeypatch.setattr(filepicker, 'scan_paths', scan)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: 'status', sink, input=pipe, output=output, project_directory=tmp_path)
        running = asyncio.create_task(ui.run())
        choosing = None
        try:
            await wait_until(lambda: any('Ready' in row for row in snapshot()))
            choosing = asyncio.create_task(ui._file_picker.choose(Workspace(tmp_path), 'old'))
            await asyncio.wait_for(entered.wait(), 5)
            ui._file_picker.field.text = 'new'
            assert not ui._menu_options
            release.set()
            await wait_until(lambda: ui._menu_options and ui._menu_options[0][0] == 'new.py')
            assert threading.get_ident() not in threads
            ui._finish_menu(None)
            await choosing
        finally:
            release.set()
            if choosing is not None:
                choosing.cancel()
                await asyncio.gather(choosing, return_exceptions=True)
            ui.close()
            await running


async def test_old_root_results_cannot_replace_reopened_picker(display, tmp_path, monkeypatch):
    from slipagent import filepicker
    sink, _, output, _, snapshot = display
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def scan(workspace, query, stop, publish):
        if workspace.root == first:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        publish([PathMatch(workspace.root.name + '.py', 0, frozenset())], 1, True)

    monkeypatch.setattr(filepicker, 'scan_paths', scan)
    with create_pipe_input() as pipe:
        ui = TerminalUI(lambda width: 'status', sink, input=pipe, output=output, project_directory=first)
        running = asyncio.create_task(ui.run())
        old = new = None
        try:
            await wait_until(lambda: any('Ready' in row for row in snapshot()))
            old = asyncio.create_task(ui._file_picker.choose(Workspace(first)))
            await asyncio.wait_for(entered.wait(), 5)
            ui._finish_menu(None)
            await wait_until(lambda: not ui._file_picker.active)
            new = asyncio.create_task(ui._file_picker.choose(Workspace(second)))
            await wait_until(lambda: ui._menu_options and ui._menu_options[0][0] == 'second.py')
            release.set()
            await old
            assert ui._file_picker.active and ui._menu_options[0][0] == 'second.py'
            ui._finish_menu(None)
            await new
        finally:
            release.set()
            for task in (old, new):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(task for task in (old, new) if task is not None), return_exceptions=True)
            ui.close()
            await running
