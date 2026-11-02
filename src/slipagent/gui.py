"""Loopback browser host borrowing the running application's engine and lifetime."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import secrets
import socket
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.responses import FileResponse, JSONResponse, Response
import uvicorn

from .application import application, settle
from .guitransport import GuiTransport, MAX_CLIENTS
from .guiapi import install_routes
from .guieditor import GuiEditor
from .lifecycle import finish_cleanup
from .sessionstate import Session

AUTH_TIMEOUT = 5.0
SEND_TIMEOUT = 5.0
ASSETS = Path(__file__).with_name('web')


class EmbeddedServer(uvicorn.Server):
    """Leave process signal ownership with the CLI and its asyncio runner."""

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class GuiHost:
    """Own the listener and subscribers, never the borrowed agent session."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.transport = GuiTransport(application(session))
        self.editor = GuiEditor(session)
        self.client_connections: dict[str, int] = {}
        self.token = secrets.token_urlsafe(32)
        self.port = 0
        self.connections = 0
        self.socket: socket.socket | None = None
        self.server: EmbeddedServer | None = None
        self.task: asyncio.Task[None] | None = None
        self.closing: asyncio.Task[None] | None = None
        self.app = self.build_app()

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.port}/#{self.token}'

    @property
    def connection_message(self) -> str:
        return (f'  Browser: {self.url}\n'
                '  For SSH access, run this on your own computer:\n'
                f'    ssh -N -L {self.port}:127.0.0.1:{self.port} YOUR_SSH_HOST\n'
                '  Replace YOUR_SSH_HOST with the SSH destination you used to connect.\n'
                '  Then open the same Browser link above on your computer. Keep the CLI and tunnel open.')

    def allowed_host(self, host: str) -> bool:
        return re.fullmatch(r'(?:127\.0\.0\.1|localhost)(?::[0-9]{1,5})?', host) is not None

    def authorized(self, token: str) -> bool:
        return hmac.compare_digest(token.encode(), self.token.encode())

    def build_app(self) -> FastAPI:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        app.state.gui_clients = self.client_connections

        @app.middleware('http')
        async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            host = request.headers.get('host', '')
            origin = request.headers.get('origin')
            if not self.allowed_host(host) or origin is not None and origin != f'http://{host}':
                return Response(status_code=403)
            if request.url.path.startswith('/api/') and not self.authorized(
                    request.headers.get('authorization', '').removeprefix('Bearer ')):
                return Response(status_code=401)
            if request.method not in ('GET', 'HEAD') and origin != f'http://{host}':
                return Response(status_code=403)
            response = await call_next(request)
            response.headers.update({'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
                                     'X-Content-Type-Options': 'nosniff',
                                     'Content-Security-Policy': "default-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})
            return response

        @app.get('/')
        async def index() -> FileResponse:
            return FileResponse(ASSETS / 'index.html')

        @app.get('/app.js')
        async def script() -> FileResponse:
            return FileResponse(ASSETS / 'app.js', media_type='text/javascript')

        @app.get('/app.css')
        async def stylesheet() -> FileResponse:
            return FileResponse(ASSETS / 'app.css', media_type='text/css')

        @app.get('/third-party-notices.txt')
        async def notices() -> FileResponse:
            return FileResponse(ASSETS / 'third-party-notices.md', media_type='text/plain')

        @app.get('/api/snapshot')
        async def snapshot(request: Request) -> Response:
            if not self.authorized(request.headers.get('authorization', '').removeprefix('Bearer ')):
                return Response(status_code=401)
            return JSONResponse(self.transport.snapshot())

        @app.websocket('/api/events')
        async def events(ws: WebSocket) -> None:
            host = ws.headers.get('host', '')
            if (not self.allowed_host(host) or ws.headers.get('origin') != f'http://{host}'
                    or self.connections >= MAX_CLIENTS or self.transport.closed):
                await ws.close(code=1008)
                return
            self.connections += 1
            queue: asyncio.Queue[dict[str, Any] | None] | None = None
            sender: asyncio.Task[None] | None = None
            receiver: asyncio.Task[None] | None = None
            client_id = ''
            try:
                await ws.accept()
                async with asyncio.timeout(AUTH_TIMEOUT):
                    raw = await ws.receive_text()
                auth = json.loads(raw)
                if (not isinstance(auth, dict) or auth.get('type') != 'authenticate'
                        or not isinstance(auth.get('token'), str) or not self.authorized(auth['token'])):
                    await ws.close(code=1008)
                    return
                client_id = auth.get('clientId', '')
                if not isinstance(client_id, str) or len(client_id) > 128:
                    client_id = ''
                    await ws.close(code=1008)
                    return
                if client_id:
                    self.client_connections[client_id] = self.client_connections.get(client_id, 0) + 1
                queue = self.transport.attach()

                async def send() -> None:
                    assert queue is not None
                    while True:
                        record = await queue.get()
                        async with asyncio.timeout(SEND_TIMEOUT):
                            if record is None:
                                await ws.close(code=1013)
                                return
                            await ws.send_json(record)

                async def receive() -> None:
                    # This stage is an observer. No browser message can execute actions.
                    await ws.receive_text()
                    await ws.close(code=1008)

                sender = asyncio.create_task(send())
                receiver = asyncio.create_task(receive())
                await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            except (WebSocketDisconnect, TimeoutError, ValueError, RuntimeError, KeyError):
                try:
                    await asyncio.wait_for(ws.close(code=1008), SEND_TIMEOUT)
                except (WebSocketDisconnect, RuntimeError, TimeoutError):
                    pass
            finally:
                if queue is not None:
                    self.transport.detach(queue)
                for task in (sender, receiver):
                    if task is not None:
                        task.cancel()
                await settle(*(task for task in (sender, receiver) if task is not None))
                if client_id:
                    remaining = self.client_connections.get(client_id, 1) - 1
                    if remaining:
                        self.client_connections[client_id] = remaining
                    else:
                        self.client_connections.pop(client_id, None)
                        if self.editor.client_id == client_id:
                            await self.editor.aclose()
                self.connections -= 1

        install_routes(app, self.session, self.editor)
        return app

    async def start(self, port: int = 0) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket = listener
        listener.bind(('127.0.0.1', port))
        listener.listen(32)
        listener.setblocking(False)
        self.port = listener.getsockname()[1]
        config = uvicorn.Config(self.app, log_config=None, log_level='warning', access_log=False,
                                lifespan='off', proxy_headers=False, server_header=False,
                                ws='websockets-sansio', ws_max_size=4096,
                                ws_per_message_deflate=False, limit_concurrency=32,
                                timeout_graceful_shutdown=2)
        self.server = EmbeddedServer(config)
        self.task = asyncio.create_task(self.server.serve(sockets=[listener]), name='GUI host')
        self.task.add_done_callback(self.finished)
        async with asyncio.timeout(5):
            while not self.server.started:
                if self.task.done():
                    await self.task
                    raise RuntimeError('GUI host stopped before startup.')
                await asyncio.sleep(.01)

    def finished(self, task: asyncio.Task[None]) -> None:
        self.transport.close()
        if not task.cancelled() and task.exception() is not None:
            logging.getLogger(__name__).error('GUI host stopped', exc_info=task.exception())

    async def aclose(self) -> None:
        if self.closing is None:
            self.closing = asyncio.create_task(self.close())
        await finish_cleanup(self.closing)

    async def close(self) -> None:
        self.transport.close()
        try:
            await self.editor.aclose()
        finally:
            try:
                if self.server is not None:
                    self.server.should_exit = True
                needs_shutdown = False
                if self.task is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(self.task), 5)
                    except asyncio.CancelledError:
                        if not self.task.cancelled():
                            raise
                        needs_shutdown = True
                    except Exception:
                        self.task.cancel()
                        await settle(self.task)
                        needs_shutdown = True
                if needs_shutdown and self.server is not None and self.server.started:
                    # A failed serve task can bypass Uvicorn's normal connection drain.
                    await self.server.shutdown(sockets=[self.socket] if self.socket is not None else None)
            finally:
                if self.socket is not None:
                    self.socket.close()


async def start_gui(session: Session, *, port: int = 0) -> GuiHost:
    """Reuse the session host; registry shutdown closes it without closing the agent."""
    lock = session.extensions.setdefault('gui_lock', asyncio.Lock())
    async with lock:
        if session.extensions.get('application_closed'):
            raise RuntimeError('The application session is closed.')
        existing: GuiHost | None = session.registry.services.get('gui_host')
        if existing is not None and not existing.transport.closed:
            return existing
        if existing is not None:
            await existing.aclose()
        if not 0 <= port <= 65535:
            raise ValueError('GUI port must be between 0 and 65535.')
        host = GuiHost(session)
        try:
            await host.start(port)
        except BaseException:
            await host.aclose()
            raise
        session.registry.services['gui_host'] = host
        return host
