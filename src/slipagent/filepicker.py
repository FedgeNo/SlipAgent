"""Asynchronous, bounded fuzzy path selection over workspace enumeration."""

from __future__ import annotations

import asyncio
import bisect
import threading
import time
import unicodedata
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from prompt_toolkit.document import Document
from prompt_toolkit.widgets import TextArea
from wcwidth import iter_graphemes

from .lifecycle import finish_cleanup
from .links import CallbackLink
from .menulifecycle import MenuLifecycle
from .tools.blocking import run_blocking
from .workspace import Workspace

MAX_RESULTS = 50
MAX_QUERY = 256


@dataclass(frozen=True)
class PathMatch:
    path: str
    score: int
    positions: frozenset[int]


def match_path(path: str, query: str) -> PathMatch | None:
    """Match normalized subsequences, retaining original grapheme positions."""
    query = unicodedata.normalize('NFC', ''.join(query.split()).replace('\\', '/')).casefold()
    clusters = list(iter_graphemes(path))
    pieces = []
    owners = []
    for index, cluster in enumerate(clusters):
        text = unicodedata.normalize('NFC', cluster).casefold()
        pieces.append(text)
        owners.extend([index] * len(text))
    folded = ''.join(pieces)
    if not query:
        return PathMatch(path, -len(clusters), frozenset())
    basename = folded.rfind('/') + 1

    def locate(start: int) -> list[int] | None:
        positions = []
        for character in query:
            found = folded.find(character, start)
            if found < 0:
                return None
            positions.append(found)
            start = found + 1
        return positions

    positions = locate(basename)
    filename = positions is not None
    if positions is None:
        positions = locate(0)
    if positions is None:
        return None
    adjacent = sum(right == left + 1 for left, right in zip(positions, positions[1:]))
    boundaries = sum(position == 0 or folded[position - 1] in '/_.- ' for position in positions)
    gaps = positions[-1] - positions[0] + 1 - len(positions)
    score = 1000 * filename + 20 * adjacent + 10 * boundaries - gaps - len(clusters)
    return PathMatch(path, score, frozenset(owners[position] for position in positions))


def scan_paths(workspace: Workspace, query: str, stop: threading.Event,
               publish: Callable[[list[PathMatch], int, bool], None]) -> None:
    """Keep only the best results; never read file contents or retain the tree."""
    if stop.is_set():
        return
    best: list[tuple[int, str, str, PathMatch]] = []
    scanned = 0
    published = time.monotonic()
    for path in workspace.iter_files():
        if stop.is_set():
            return
        scanned += 1
        relative = path.relative_to(workspace.root).as_posix()
        match = match_path(relative, query)
        if match is not None:
            bisect.insort(best, (-match.score, relative.casefold(), relative, match))
            if len(best) > MAX_RESULTS:
                best.pop()
        now = time.monotonic()
        if now - published >= .1:
            publish([item[3] for item in best], scanned, False)
            published = now
    if not stop.is_set():
        publish([item[3] for item in best], scanned, True)


class FilePicker:
    def __init__(self, terminal: Any) -> None:
        self.terminal = terminal
        self.active = False
        self.action = 'view'
        self.status = 'Scanning…'
        self.revision = 0
        self.changed = asyncio.Event()
        self.stop: threading.Event | None = None
        self.workspace: Workspace | None = None
        self.task: asyncio.Task[None] | None = None
        self.field = TextArea(multiline=False, height=1, prompt=lambda: f'Files [{self.status}] › ',
                              accept_handler=self.accept, style='class:user')
        self.field.buffer.on_text_changed += self.query_changed

    def accept(self, buffer: Any) -> bool:
        ui = self.terminal
        if ui._menu_options:
            ui._finish_menu(ui._menu_options[ui._menu_index][0])
        return False

    def select(self, path: str, action: str) -> None:
        self.action = action
        self.terminal._finish_menu(path)

    def query_changed(self, buffer: Any) -> None:
        if not self.active:
            return
        self.revision += 1
        if self.stop is not None:
            self.stop.set()
        self.terminal._menu_options = []
        self.terminal._choice_panel.option_highlights = {}
        self.terminal._choice_panel.option_links = {}
        self.status = 'Scanning…'
        self.changed.set()
        self.terminal.app.invalidate()

    def publish(self, revision: int, workspace: Workspace, matches: list[PathMatch], scanned: int, complete: bool) -> None:
        if not self.active or revision != self.revision or workspace is not self.workspace:
            return
        ui = self.terminal
        selected = ui._menu_options[ui._menu_index][0] if ui._menu_options else None
        ui._menu_options = [(match.path, match.path) for match in matches]
        ui._choice_panel.option_highlights = {match.path: match.positions for match in matches}
        ui._choice_panel.option_links = {
            match.path: [('', '  '),
                         *CallbackLink('View', ('file-view', match.path), ui._link_hover(),
                                       partial(self.select, match.path, 'view')).fragments(),
                         ('', ' / '),
                         *CallbackLink('Edit', ('file-edit', match.path), ui._link_hover(),
                                       partial(self.select, match.path, 'edit')).fragments()]
            for match in matches
        }
        paths = [match.path for match in matches]
        ui._menu_index = paths.index(selected) if selected in paths else 0
        self.status = f'{scanned} files, {len(matches)} matches' + ('' if complete else '…')
        ui._choice_panel.reveal()
        ui.app.invalidate()

    async def refresh(self, future: asyncio.Future[str | None]) -> None:
        loop = asyncio.get_running_loop()
        while self.active and self.terminal._menu_future is future:
            await self.changed.wait()
            self.changed.clear()
            if not self.active or self.terminal._menu_future is not future:
                return
            query = self.field.text
            if len(query) > MAX_QUERY:
                self.status = 'Query limit: 256 characters'
                self.terminal.app.invalidate()
                continue
            revision, workspace = self.revision, self.workspace
            assert workspace is not None
            stop = threading.Event()
            self.stop = stop

            def publish(matches: list[PathMatch], scanned: int, complete: bool) -> None:
                loop.call_soon_threadsafe(self.publish, revision, workspace, matches, scanned, complete)

            try:
                await run_blocking(scan_paths, workspace, query, stop, publish, on_cancel=stop.set)
            except (OSError, ValueError, RuntimeError) as exc:
                if self.active and revision == self.revision:
                    self.status = 'Could not scan files'
                    self.terminal.notify(str(exc))
                    self.terminal.app.invalidate()

    def close(self) -> None:
        self.active = False
        self.revision += 1
        if self.stop is not None:
            self.stop.set()
        self.changed.set()

    async def choose(self, workspace: Workspace, query: str = '') -> str | None:
        ui = self.terminal
        if ui._menu_future is not None:
            ui.notify('Close the current menu before finding a file.')
            return None
        self.workspace = workspace
        self.action = 'view'
        self.field.buffer.set_document(Document(query))
        self.active = True
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        ui._menu_future = future
        ui._menu_options = []
        ui._menu_title = 'Find File'
        ui._menu_index = 0
        ui._menu_window.seek()
        self.query_changed(self.field.buffer)
        task = asyncio.create_task(self.refresh(future))
        self.task = task

        def cleanup() -> None:
            self.close()
            ui._menu_future = None
            ui._menu_options = []
            ui._choice_panel.option_highlights = {}
            ui._choice_panel.option_links = {}

        try:
            with MenuLifecycle(ui).opened(self.field, cleanup):
                return await future
        finally:
            await finish_cleanup(task)
            if self.task is task:
                self.task = None
