"""Interface-independent ownership and operations for one running session."""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeVar, TYPE_CHECKING

from .inputqueue import input_queue
from .events import AgentEvent, EventHandler
from .lifecycle import finish_cleanup
from .submissions import cancelled
from .task import TaskMemory
from .tools.blocking import run_blocking

if TYPE_CHECKING:
    from .sessionstate import Session
    from .sessions import SessionJournal

T = TypeVar('T')


class EventPublisher(Generic[T]):
    """Fan out ephemeral notifications without retaining history or payload copies."""

    def __init__(self) -> None:
        self.listeners: dict[object, Callable[[T], None]] = {}

    def subscribe(self, listener: Callable[[T], None]) -> Callable[[], None]:
        key = object()
        self.listeners[key] = listener
        def unsubscribe() -> None:
            self.listeners.pop(key, None)
        return unsubscribe

    def emit(self, event: T) -> None:
        for key, listener in tuple(self.listeners.items()):
            try:
                listener(event)
            except Exception:
                self.listeners.pop(key, None)
                logging.getLogger(__name__).exception('Detached a failing application event listener')


@dataclass(frozen=True)
class ApplicationState:
    session_id: str
    workspace: str
    model: str
    running: bool
    stopped: bool
    queued_inputs: int
    changing: bool
    closed: bool


@dataclass(frozen=True)
class RunContinuation:
    continue_run: bool


async def settle(*tasks: asyncio.Task[Any]) -> None:
    for task in tasks:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


class SessionApplication:
    """Borrow the session's engine and resources; never create a second backend."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.changing = False
        self.cancel_deferred_input: Callable[[list[str]], None] | None = None
        self.events: EventPublisher[AgentEvent] = EventPublisher()
        self.states: EventPublisher[ApplicationState] = EventPublisher()
        if session.agent.on_event is not None:
            self.events.subscribe(session.agent.on_event)
        session.agent.on_event = self.events.emit

    def subscribe(self, listener: EventHandler) -> Callable[[], None]:
        return self.events.subscribe(listener)

    def state(self) -> ApplicationState:
        return ApplicationState(self.session.agent.session_id, str(self.session.workspace.root),
                                self.session.agent.model, self.busy, self.session.agent.stopped,
                                len(self.session.agent.pending) + len(self.session.agent.queued_messages),
                                self.changing, bool(self.session.extensions.get('application_closed')))

    def changed(self) -> None:
        if self.states.listeners:
            self.states.emit(self.state())

    @property
    def busy(self) -> bool:
        task = self.session.work.agent_task
        return self.session.agent.running or task is not None and not task.done()

    def require_idle(self) -> None:
        command = self.session.work.command_task
        if self.busy or self.changing or command is not None and not command.done() and command is not asyncio.current_task():
            raise RuntimeError('Wait for the current agent work to settle before changing the session.')
        if self.session.extensions.get('application_closed'):
            raise RuntimeError('The application session is closed.')

    @contextmanager
    def change(self, *, allow_running: bool = False) -> Iterator[None]:
        if allow_running:
            if self.changing or self.session.extensions.get('application_closed'):
                raise RuntimeError('The application session is changing or closed.')
        else:
            self.require_idle()
            edits = self.session.registry.resources.user_edits
            if edits is not None and not edits.released.is_set():
                raise RuntimeError('Close the current editor before changing the session.')
        self.changing = True
        frame = self.session.reloader
        if frame is not None:
            frame.busy += 1
        self.changed()
        try:
            yield
            self.session.agent._persist()
        finally:
            self.changing = False
            if frame is not None:
                frame.busy -= 1
            self.changed()

    def start(self, prompt: str, *, continue_run: bool = False) -> asyncio.Task[str]:
        """Return the owned run task; cancelling a UI waiter need not cancel it."""
        self.require_idle()
        work = self.session.agent.run(prompt, continue_run=True) if continue_run else self.session.agent.run(prompt)
        task = asyncio.create_task(work)
        self.session.work.agent_task = task
        task.add_done_callback(self._finished)
        self.changed()
        return task

    def _finished(self, task: asyncio.Task[str]) -> None:
        # Retrieve failures even when every interface has detached.
        if not task.cancelled():
            task.exception()
        if self.session.work.agent_task is task:
            self.session.work.agent_task = None
        self.changed()

    async def run(self, prompt: str, *, continue_run: bool = False) -> str:
        return await self.start(prompt, continue_run=continue_run)

    async def wait(self) -> str | None:
        task = self.session.work.agent_task
        return await asyncio.shield(task) if task is not None else None

    def next_run(self, answer: str, *, resume: bool = False, exiting: bool = False) -> RunContinuation | None:
        from .agent import STEP_LIMIT_NOTICE, STOP_NOTICE
        if (exiting or answer == STEP_LIMIT_NOTICE or self.session.extensions.get('pending_review') is not None
                or self.session.registry.services.get('active_review') is not None):
            return None
        if resume and (answer == STOP_NOTICE or self.session.agent.has_queued_input):
            return RunContinuation(continue_run=True)
        if self.session.agent.has_queued_input and not self.session.agent.stopped:
            return RunContinuation(continue_run=False)
        return None

    def submit(self, text: str) -> bool:
        if cancelled(text):
            return False
        if self.session.extensions.get('application_closed'):
            raise RuntimeError('The application session is closed.')
        self.session.agent.enqueue(text)
        if self.session.agent.stop_requested:
            self.session.work.resume_after_commands = True
        self.input_ready.set()
        self.changed()
        return True

    @property
    def input_ready(self) -> asyncio.Event:
        """Wake a frontend driver when another interface has accepted new input."""
        event: asyncio.Event = self.session.extensions.setdefault('input_ready', asyncio.Event())
        return event

    def queued_input(self) -> list[tuple[str, str]]:
        return input_queue(self.session.agent).snapshot()

    def discard_input(self, identifier: str) -> bool:
        removed = input_queue(self.session.agent).discard(identifier)
        self.changed()
        return removed

    def stop(self) -> bool:
        self.session.work.resume_after_commands = False
        stopping = self.session.agent.request_stop()
        self.changed()
        return stopping

    def interrupt(self, *, preserve_deferred_input: Callable[[list[str]], None] | None = None,
                  on_finished: Callable[[], None] | None = None) -> bool:
        """Cancel owned work now; cleanup remains owned after any interface detaches."""
        session = self.session
        if session.extensions.get('interrupt_task') is not None:
            return False
        session.extensions['interrupt_requested'] = True
        session.extensions['interrupt_sequence'] = session.extensions.get('interrupt_sequence', 0) + 1
        session.registry.services['interrupt_requested'] = True
        discarded = session.extensions.pop('deferred_commands', [])
        handler = preserve_deferred_input
        if handler is None:
            handler = self.cancel_deferred_input
        if handler is not None:
            handler(discarded)
        session.work.resume_after_commands = False
        session.agent.request_stop()
        session.agent.stopped = True
        tasks = [task for task in [session.work.agent_task, session.work.command_task,
                                   *(session.work.command_tasks or ())] if task is not None]
        for task in tasks:
            task.cancel()
        compactor = session.registry.resources.compactor
        if compactor is not None:
            compactor.reset()

        async def drain() -> None:
            jobs = session.registry.resources.jobs
            job_cleanup = asyncio.create_task(jobs.stop_all()) if jobs is not None else None
            await settle(*tasks, *([job_cleanup] if job_cleanup is not None else []))
            if compactor is not None:
                await compactor.wait()
            session.registry.services.pop('interrupt_requested', None)
            if on_finished is not None:
                try:
                    on_finished()
                except Exception:
                    logging.getLogger(__name__).exception('Interrupt completion display failed')
            self.changed()

        session.extensions['interrupt_task'] = asyncio.create_task(drain())
        self.changed()
        return True

    async def wait_for_interrupt(self) -> None:
        task = self.session.extensions.get('interrupt_task')
        if task is not None:
            await finish_cleanup(task)
            if self.session.extensions.get('interrupt_task') is task:
                self.session.extensions.pop('interrupt_task', None)
                self.session.extensions.pop('interrupt_requested', None)

    def background(self, work: Awaitable[Any]) -> None:
        tasks: set[asyncio.Task[Any]] = self.session.extensions.setdefault('background_tasks', set())
        task = asyncio.ensure_future(work)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def reset(self) -> None:
        with self.change():
            jobs = self.session.registry.resources.jobs
            if jobs is not None:
                await jobs.stop_all()
            self.session.agent.reset()
            self.session.extensions.pop('title', None)

    def new_task(self) -> None:
        with self.change():
            if self.session.agent.has_queued_input:
                raise ValueError('Queued input is still pending; resume or discard it before starting another task.')
            self.session.agent.history.task = TaskMemory(start_message=len(self.session.agent.messages))

    async def rename(self, title: str) -> str:
        from .sessions import session_title
        with self.change(allow_running=True):
            formatted = session_title(title)
            journal = self.session.registry.resources.journal
            if journal is not None:
                await run_blocking(journal.rename, title)
            else:
                self.session.extensions['title'] = formatted
            return formatted

    async def saved_sessions(self) -> list[dict[str, str]]:
        return await run_blocking(self._journal().listing)

    def _journal(self) -> SessionJournal:
        journal = self.session.registry.resources.journal
        if journal is None:
            raise ValueError('Session persistence is disabled.')
        return journal

    async def resume(self, identifier: str) -> None:
        with self.change():
            journal = self._journal()
            await self.session.agent.wait_for_compaction()
            data = await run_blocking(journal.load, identifier)
            jobs = self.session.registry.resources.jobs
            if jobs is not None:
                await jobs.stop_all()
            journal.restore(self.session.agent, data)

    async def fork(self) -> str:
        with self.change():
            journal = self._journal()
            await self.session.agent.wait_for_compaction()
            jobs = self.session.registry.resources.jobs
            if jobs is not None:
                await jobs.stop_all()
            self.session.agent._persist()
            data = await run_blocking(journal.load, journal.session_id)
            journal.restore(self.session.agent, data, fork=True)
            return journal.session_id

    async def delete_current(self) -> None:
        with self.change():
            await self.session.agent.wait_for_compaction()
            jobs = self.session.registry.resources.jobs
            if jobs is not None:
                await jobs.stop_all()
            self.session.agent.reset(new_session=False)
            await run_blocking(self._journal().delete_current)
            self.session.extensions.pop('title', None)

    async def delete_saved(self, identifier: str) -> None:
        with self.change():
            await run_blocking(self._journal().delete_saved, identifier)

    async def rename_saved(self, identifier: str, title: str) -> None:
        with self.change():
            await run_blocking(self._journal().rename_saved, identifier, title)

    async def rewind(self, identifier: str) -> list[str]:
        from .prompts import load_prompt
        with self.change():
            checkpoints = self.session.agent._checkpoints()
            if checkpoints is None:
                raise ValueError('No file-edit checkpoints in this session.')
            jobs = self.session.registry.resources.jobs
            if jobs is not None:
                await jobs.stop_all()
            restored = await run_blocking(checkpoints.rewind, identifier)
            notice = 'Files restored by /rewind: ' + ', '.join(restored)
            self.session.registry.context_notes['file_rewind'] = load_prompt('file-rewind.md', notice=notice)
            return restored

    def set_planning(self, enabled: bool) -> None:
        with self.change():
            self.session.agent.set_planning(enabled)

    def set_overthinking(self, enabled: bool) -> None:
        with self.change():
            self.session.agent.overthinking = enabled

    def set_danger(self, enabled: bool) -> None:
        with self.change():
            self.session.workspace.access.danger = enabled

    def set_hashline(self, enabled: bool) -> None:
        from .tools.hashline import set_hashline
        with self.change():
            set_hashline(self.session.registry, enabled)
            frame = self.session.reloader
            if frame is not None:
                frame._builtin_tools = tuple(tool for tool in self.session.registry.tools
                                             if tool.name in self.session.registry.builtin_names)

    async def select_skill(self, selector: str | None) -> None:
        from .skills import select
        with self.change():
            if selector is None:
                self.session.registry.services.pop('selected_skill', None)
            else:
                self.session.registry.services['selected_skill'] = await run_blocking(select, self.session.workspace, selector)

    async def select_model(self, selector: str) -> None:
        from .config import ConfigError, save_model_choice
        with self.change():
            model = next((model for model in await self.session.catalog() if model.selector == selector), None)
            if model is None:
                raise ValueError('Choose an available provider and model.')
            await self.session.select_model(model)
            try:
                await run_blocking(save_model_choice, model.id, model.provider)
            except (OSError, ConfigError) as exc:
                self.session.notice(f'Model selected, but its preference could not be saved: {exc}')

    async def enable_optimal(self) -> None:
        from .config import ConfigError, save_optimal_choice
        with self.change():
            await self.session.model_routing().enable()
            try:
                await run_blocking(save_optimal_choice, True)
            except (OSError, ConfigError) as exc:
                self.session.notice(f'Optimal selection enabled, but its preference could not be saved: {exc}')

    async def close(self) -> None:
        """Drain active operations before closing any services they borrow."""
        if self.session.extensions.get('application_close_task') is None:
            self.session.extensions['application_close_task'] = asyncio.create_task(self._close())
        await finish_cleanup(self.session.extensions['application_close_task'])

    async def _close(self) -> None:
        session = self.session
        session.extensions['application_closed'] = True
        if self.busy or session.work.command_task is not None or session.work.command_tasks:
            self.interrupt()
        await self.wait_for_interrupt()
        tasks = list(session.work.command_tasks or ()) + list(session.extensions.get('background_tasks', ()))
        for task in tasks:
            task.cancel()
        await settle(*tasks)
        try:
            if session.mcp is not None:
                await session.mcp.aclose()
        finally:
            try:
                await session.registry.aclose()
            finally:
                try:
                    await session.client.aclose()
                finally:
                    if session.reloader is not None:
                        session.reloader.close()
                    self.changed()
                    self.events.listeners.clear()
                    self.states.listeners.clear()


def application(session: Session) -> SessionApplication:
    owner = session.extensions.get('application')
    if owner is None:
        owner = SessionApplication(session)
        session.extensions['application'] = owner
    return owner
