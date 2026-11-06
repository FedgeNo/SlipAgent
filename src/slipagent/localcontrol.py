"""Versioned local control borrowing the application's single execution owner."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import OrderedDict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from .application import application, ApplicationState
from .api import APIError
from .events import AgentEvent
from .lifecycle import Lifetime, finish_cleanup
from . import localipc
from .sessionstate import Session
from .tools.blocking import run_blocking

MAX_CLIENTS = 16
QUEUE_SIZE = 64
CACHE_SIZE = 128


class Params(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Request(Params):
    jsonrpc: Literal['2.0']
    id: str = Field(min_length=1, max_length=128)
    method: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = Field(default=None, alias='sessionId')


class TextParams(Params):
    text: str = Field(min_length=1, max_length=65536)


class IdentifierParams(Params):
    identifier: str = Field(min_length=1, max_length=4096)


class SettingParams(Params):
    setting: Literal['danger', 'planning', 'overthinking', 'hashline']
    enabled: StrictBool


class PageParams(Params):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=16384, ge=1, le=16384)


class HistoryParams(PageParams):
    step_id: int = Field(alias='stepId', ge=1)


class SessionsParams(Params):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=100)


def encode(value: Any) -> bytes:
    result = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(result) > localipc.LIMIT:
        raise ValueError('Control message exceeds 128 KiB; use smaller pages.')
    return result


class Connection:
    def __init__(self, host: ControlHost, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.host, self.reader, self.writer = host, reader, writer
        self.peer: str | None = None
        self.subscribed = False
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(QUEUE_SIZE)

    def close(self) -> None:
        self.writer.close()
        self.writer.transport.abort()

    def send(self, value: Any) -> None:
        if self.queue.full():
            self.close()
        elif not self.writer.is_closing():
            self.queue.put_nowait(encode(value))

    def allowed(self) -> bool:
        return self.host.danger or self.peer == self.host.owner

    async def output(self) -> None:
        try:
            while True:
                self.writer.write(await self.queue.get())
                await asyncio.wait_for(self.writer.drain(), 5)
        finally:
            self.close()

    async def run(self) -> None:
        sender = asyncio.create_task(self.output())
        try:
            try:
                self.peer = localipc.peer_identity(self.writer)
            except OSError:
                if not self.host.danger:
                    raise
                self.peer = ''  # Protected Windows clients are still allowed in public mode.
            if not self.allowed():
                return
            while not self.writer.is_closing():
                line = await self.reader.readline()
                if not line:
                    break
                if len(line) > localipc.LIMIT or not line.endswith(b'\n'):
                    break
                if not self.allowed():
                    break
                try:
                    request = Request.model_validate(json.loads(line))
                except (ValueError, ValidationError):
                    self.send({'jsonrpc': '2.0', 'id': None,
                               'error': {'code': -32600, 'message': 'Invalid control request.'}})
                    continue
                if request.method in ('subscribe', 'unsubscribe'):
                    try:
                        Params.model_validate(request.params)
                        self.subscribed = request.method == 'subscribe'
                        self.send({'jsonrpc': '2.0', 'id': request.id,
                                   'result': self.host.status() if self.subscribed else True})
                    except ValueError:
                        self.send({'jsonrpc': '2.0', 'id': request.id,
                                   'error': {'code': -32602, 'message': 'Invalid subscription parameters.'}})
                else:
                    self.send(await self.host.request(request))
        except (OSError, ValueError, asyncio.TimeoutError):
            pass
        finally:
            self.close()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            try:
                await self.writer.wait_closed()
            except OSError:
                pass
            self.host.connections.discard(self)


class ControlHost:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.controller = application(session)
        self.instance = uuid.uuid4().hex
        self.endpoint = localipc.endpoint_name(self.instance)
        self.owner = localipc.owner_identity()
        self.danger = session.workspace.access.danger
        self.listener = localipc.listener(self.endpoint)
        self.lifetime = Lifetime('local control')
        self.connections: set[Connection] = set()
        self.requests: OrderedDict[str, tuple[str, asyncio.Task[dict[str, Any]]]] = OrderedDict()
        self.sequence = 0
        self.closing: asyncio.Task[None] | None = None
        self.unsubscribe = self.controller.subscribe(self.event)
        self.unsubscribe_state = self.controller.states.subscribe(self.state)

    async def start(self) -> None:
        await self.listener.start(self.connected, self.danger)

    def connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.closing is not None or not self.lifetime.active or len(self.connections) >= MAX_CLIENTS:
            writer.close()
            return
        connection = Connection(self, reader, writer)
        self.connections.add(connection)
        self.lifetime.spawn(connection.run())

    def status(self) -> dict[str, Any]:
        journal = self.session.registry.resources.journal
        state = self.controller.state()
        return {'protocolVersion': 1, 'instanceId': self.instance, 'endpoint': self.endpoint,
                'pid': os.getpid(), 'title': journal.title if journal is not None else self.session.extensions.get('title', ''),
                'danger': self.danger, 'steps': len(self.session.agent.history.steps),
                'sessionId': state.session_id, 'workspace': state.workspace, 'model': state.model,
                'provider': getattr(self.session.client, 'provider', ''),
                'running': state.running, 'stopped': state.stopped, 'queuedInputs': state.queued_inputs,
                'changing': state.changing, 'closed': state.closed,
                'activity': self.session.registry.services.get('activity_phase', '') if state.running else ''}

    def publish(self, method: str, params: Any) -> None:
        self.sequence += 1
        for connection in tuple(self.connections):
            if connection.subscribed and connection.allowed():
                connection.send({'jsonrpc': '2.0', 'method': method,
                                 'params': {'instanceId': self.instance, 'sequence': self.sequence, **params}})

    def event(self, event: AgentEvent) -> None:
        if event.kind == 'context':
            return
        self.publish('event', {'kind': event.kind, 'step': event.step, 'text': event.text[:2048],
            'truncated': len(event.text) > 2048,
            'toolCall': {'id': event.tool_call.id, 'name': event.tool_call.name} if event.tool_call else None})

    def state(self, state: ApplicationState) -> None:
        self.publish('state', self.status())

    def set_access(self, danger: bool) -> None:
        try:
            self.listener.set_access(danger)
        except OSError:
            # Permission changes must never leave a public listener accidentally enabled.
            self.controller.background(self.aclose())
            raise
        self.danger = danger
        if not danger:
            for connection in tuple(self.connections):
                if connection.peer != self.owner:
                    connection.close()

    async def request(self, request: Request) -> dict[str, Any]:
        signature = json.dumps([request.method, request.params, request.session_id], sort_keys=True)
        cached = self.requests.get(request.id)
        if cached is not None:
            if cached[0] != signature:
                return {'jsonrpc': '2.0', 'id': request.id, 'error': {
                    'code': -32600, 'message': 'Request ID was already used for another operation.'}}
            self.requests.move_to_end(request.id)
            return await asyncio.shield(cached[1])
        while len(self.requests) >= CACHE_SIZE:
            completed = next((key for key, (_, task) in self.requests.items() if task.done()), None)
            if completed is None:
                return {'jsonrpc': '2.0', 'id': request.id, 'error': {'code': -32000, 'message': 'Control request limit reached.'}}
            self.requests.pop(completed)
        task = self.lifetime.spawn(self.dispatch(request))
        self.requests[request.id] = signature, task
        return await asyncio.shield(task)

    async def dispatch(self, request: Request) -> dict[str, Any]:
        try:
            if request.session_id is not None and request.session_id != self.session.agent.session_id:
                raise RuntimeError('The session changed; refresh status before applying the action.')
            result = await self.execute(request.method, request.params)
            if (request.method in ('history', 'plan') and request.session_id is not None
                    and request.session_id != self.session.agent.session_id):
                raise RuntimeError('The session changed while reading; refresh status before continuing.')
            response = {'jsonrpc': '2.0', 'id': request.id, 'result': result}
            encode(response)
            return response
        except (ValueError, RuntimeError, OSError, APIError) as exc:
            return {'jsonrpc': '2.0', 'id': request.id, 'error': {'code': -32000, 'message': str(exc)[:2048]}}

    async def execute(self, method: str, params: dict[str, Any]) -> Any:
        app = self.controller
        if method in ('submit', 'rename'):
            value = TextParams.model_validate(params).text
            if not value.strip():
                raise ValueError('Enter nonempty text.')
            return {'accepted': app.submit(value)} if method == 'submit' else await app.rename(value)
        if method in ('discard', 'resume', 'model'):
            identifier = IdentifierParams.model_validate(params).identifier
            if method == 'discard':
                return app.discard_input(identifier)
            if method == 'resume':
                await app.resume(identifier)
            else:
                await app.select_model(identifier)
            return True
        if method == 'setting':
            setting = SettingParams.model_validate(params)
            setters = {'danger': app.set_danger, 'planning': app.set_planning,
                       'overthinking': app.set_overthinking, 'hashline': app.set_hashline}
            setters[setting.setting](setting.enabled)
            return True
        if method == 'history':
            page = HistoryParams.model_validate(params)
            return await run_blocking(self.history_page, page)
        if method == 'plan':
            paging = PageParams.model_validate(params)
            return await run_blocking(lambda: self.text_page(app.plan(), paging))
        if method == 'sessions':
            listing = SessionsParams.model_validate(params)
            sessions = await app.saved_sessions()
            end = listing.offset + listing.limit
            return {'items': sessions[listing.offset:end],
                    'nextOffset': end if end < len(sessions) else None}
        Params.model_validate(params)
        if method == 'status':
            return self.status()
        if method == 'stop':
            return app.stop()
        if method == 'interrupt':
            return app.interrupt()
        if method == 'queue':
            return [{'identifier': key, 'text': text[:160]} for key, text in app.queued_input()[:100]]
        if method == 'reset':
            await app.reset()
            return True
        if method == 'fork':
            return await app.fork()
        if method == 'new_task':
            app.new_task()
            return True
        raise ValueError(f'Unknown control method: {method}')

    def history_page(self, page: HistoryParams) -> dict[str, Any]:
        from .context import CompletedStep
        steps = self.session.agent.history.steps
        if page.step_id > len(steps):
            raise ValueError('No such history step.')
        step = steps[page.step_id - 1]
        value = step.parts(stream=True) if isinstance(step, CompletedStep) else step.record()
        return {'stepId': page.step_id, **self.text_page(value, page)}

    def text_page(self, value: Any, page: PageParams) -> dict[str, Any]:
        from .sources import json_chunks
        chunks, position, end = [], 0, page.offset + page.limit + 1
        for chunk in json_chunks(value):
            if position + len(chunk) > page.offset:
                chunks.append(chunk[max(0, page.offset - position):max(0, end - position)])
            position += len(chunk)
            if position >= end:
                break
        text = ''.join(chunks)
        return {'offset': page.offset, 'text': text[:page.limit],
                'nextOffset': page.offset + page.limit if len(text) > page.limit else None}

    async def aclose(self) -> None:
        if self.closing is None:
            self.closing = asyncio.create_task(self.close())
        await finish_cleanup(self.closing)

    async def close(self) -> None:
        self.unsubscribe()
        self.unsubscribe_state()
        # Newer asyncio servers wait for clients during wait_closed(). Close
        # clients first, and reject pending accepts once closing has started.
        for connection in tuple(self.connections):
            connection.close()
        try:
            await self.listener.aclose()
        finally:
            for connection in tuple(self.connections):
                connection.close()
            await self.lifetime.aclose()
            self.requests.clear()


async def start_control(session: Session) -> ControlHost:
    lock = session.extensions.setdefault('local_control_lock', asyncio.Lock())
    async with lock:
        if session.extensions.get('application_closed'):
            raise RuntimeError('The application session is closed.')
        existing = session.extensions.get('local_control')
        if isinstance(existing, ControlHost) and existing.closing is None:
            return existing
        host = ControlHost(session)
        try:
            await host.start()
            if session.extensions.get('application_closed'):
                raise RuntimeError('The application session closed during control startup.')
        except BaseException:
            await host.aclose()
            raise
        session.extensions['local_control'] = host
        return host
