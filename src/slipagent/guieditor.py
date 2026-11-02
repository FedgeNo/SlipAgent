"""Browser adapter for the existing file editor and shared edit-boundary owner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .application import application, settle
from .editor import FileEditor
from .events import AgentEvent
from .lifecycle import finish_cleanup
from .sessionstate import Session
from .tools.blocking import run_blocking
from .useredits import UserEdits


class GuiEditor:
    """Lease one paged editor to a browser tab and release it on disconnect."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.file: FileEditor | None = None
        self.client_id = ''
        self.session_id = ''
        self.message = ''
        self.error = ''
        self.request_task: asyncio.Task[None] | None = None
        self.closing: asyncio.Task[None] | None = None
        self.lock = asyncio.Lock()
        self.opened = asyncio.Event()
        self.opened.set()
        coordinator = session.registry.resources.user_edits
        if coordinator is None:
            coordinator = UserEdits(session, self)
            session.registry.resources.user_edits = coordinator
            session.agent.on_boundary = coordinator.checkpoint
        self.coordinator = coordinator

    @property
    def active(self) -> bool:
        return self.file is not None

    async def begin(self, path: Path, client_id: str) -> None:
        async with self.lock:
            if self.client_id or self.coordinator.editor.active or self.coordinator.pending is not None:
                raise RuntimeError('Close the current editor before opening another one.')
            self.client_id = client_id
            self.session_id = self.session.agent.session_id
            self.message = 'Waiting for the current turn to finish.'
            self.error = ''
            self.closing = None
            self.request_task = asyncio.create_task(self.coordinator.request(path, editor=self))
            self.request_task.add_done_callback(self.finished)

    def finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            self.error = str(error)

    async def open(self, path: Path, line: int) -> None:
        self.opened.clear()
        worker = asyncio.create_task(asyncio.to_thread(FileEditor, path))
        try:
            try:
                self.file = await asyncio.shield(worker)
            except asyncio.CancelledError:
                loaded = await finish_cleanup(worker)
                await run_blocking(loaded.close)
                raise
        finally:
            self.opened.set()
        self.message = 'Agent held while this editor is open.'
        application(self.session).changed()

    async def page(self, client_id: str, page: int) -> dict[str, Any]:
        async with self.lock:
            if self.client_id != client_id:
                return {'state': 'closed', 'error': 'This editor is no longer owned by this tab.'}
            if self.error:
                return {'state': 'error', 'error': self.error}
            if self.file is None:
                return {'state': 'waiting', 'message': self.message}
            if not 0 <= page < self.file.page_count:
                raise ValueError('Invalid editor page.')
            return {'state': 'open', 'path': str(self.file.path), 'page': page,
                    'pages': self.file.page_count, 'text': await run_blocking(self.file.page_text, page),
                    'readOnly': self.file.read_only, 'message': self.message}

    async def update(self, client_id: str, page: int, text: str, *, save: bool) -> None:
        # Save receipts must finish even if the browser's HTTP waiter disappears.
        task = asyncio.create_task(self.write(client_id, page, text, save=save))
        await finish_cleanup(task)

    async def write(self, client_id: str, page: int, text: str, *, save: bool) -> None:
        async with self.lock:
            if self.client_id != client_id or self.file is None:
                raise RuntimeError('The editor is no longer open in this tab.')
            if self.session_id != self.session.agent.session_id:
                raise RuntimeError('The session changed; close and reopen this editor.')
            file = self.file
            if not 0 <= page < file.page_count:
                raise ValueError('Invalid editor page.')
            def write() -> str:
                file.set_page(page, text)
                if save:
                    file.save()
                    return file.diff()
                return ''
            diff = await run_blocking(write)
            if save:
                self.message = 'File saved. Close the editor to let the agent continue.'
                self.coordinator.record(file.path, diff, file.edit_id)

    async def close_for_switch(self) -> bool:
        self.notify('Close the browser editor before switching files.')
        return False

    def notify(self, text: str) -> None:
        self.message = text

    def clear_pending_notice(self) -> None:
        self.message = ''

    def recorded(self, path: Path, artifact: dict[str, Any] | None, error: str) -> None:
        self.session.agent._emit(AgentEvent(kind='notice', text=f'User edited {path}'))
        if error:
            self.message = f'File saved; diff archive failed: {error}'

    async def shutdown(self) -> None:
        async with self.lock:
            if self.file is not None:
                await run_blocking(self.file.close)
                self.file = None
            self.client_id = ''

    async def close(self) -> None:
        task = self.request_task
        if task is not None and not self.active:
            task.cancel()
            await settle(task)
        await self.opened.wait()
        await self.shutdown()
        if self.coordinator.editor is self:
            self.coordinator.close()
        if task is not None:
            await settle(task)
        self.request_task = None
        application(self.session).changed()

    async def aclose(self) -> None:
        if self.closing is None:
            self.closing = asyncio.create_task(self.close())
        await finish_cleanup(self.closing)
