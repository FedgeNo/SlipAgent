"""Typed browser actions and paged original data, served by the loopback host."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import JSONResponse

from .application import application
from .api import APIError
from .guibrowser import GuiBrowser, text_page
from .guieditor import GuiEditor
from .historyquestions import history_question, history_reference, display_question
from .sessionstate import Session
from .tools.blocking import run_blocking
from .workspace import WorkspaceError


class HistoryTarget(BaseModel):
    model_config = ConfigDict(extra='forbid')
    step_id: int = Field(alias='stepId', ge=1)
    section: str = Field(default='all', max_length=64)
    call_id: str | None = Field(default=None, alias='callId', max_length=4096)
    field_path: list[str | int] = Field(default_factory=list, alias='fieldPath')


class BrowserAction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(alias='sessionId')
    action: Literal['submit', 'stop', 'interrupt', 'discard', 'reset', 'new_task', 'rename',
                    'resume', 'fork', 'delete_saved', 'delete_current', 'rename_saved', 'setting', 'model', 'optimal', 'skill', 'rewind']
    text: str = Field(default='', max_length=65536)
    identifier: str = Field(default='', max_length=4096)
    setting: Literal['planning', 'overthinking', 'danger', 'hashline'] | None = None
    enabled: bool | None = None
    history_reference: HistoryTarget | None = Field(default=None, alias='historyReference')


class EditorAction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['open', 'stage', 'save', 'close']
    client_id: str = Field(alias='clientId', min_length=1, max_length=128)
    session_id: str = Field(alias='sessionId')
    node: str = Field(default='', max_length=4096)
    page: int = Field(default=0, ge=0)
    text: str = Field(default='', max_length=262144)


async def read_action(request: Request, limit: int) -> Any:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise HTTPException(413, 'Action exceeds its size limit.')
        body.extend(chunk)
    try:
        return json.loads(body)
    except ValueError as exc:
        raise HTTPException(422, 'Invalid JSON action.') from exc


def install_routes(app: FastAPI, session: Session, editor: GuiEditor) -> None:
    browser = GuiBrowser(session)
    controller = application(session)

    @app.exception_handler(WorkspaceError)
    @app.exception_handler(ValueError)
    @app.exception_handler(APIError)
    @app.exception_handler(RuntimeError)
    @app.exception_handler(OSError)
    async def operation_error(request: Request, exc: Exception) -> JSONResponse:
        status = 400
        if isinstance(exc, WorkspaceError):
            status = 403
        elif isinstance(exc, RuntimeError):
            status = 409
        return JSONResponse({'detail': str(exc)}, status_code=status)

    @app.get('/api/children')
    async def children(node: str = Query(default='/', max_length=4096),
                       cursor: str = Query(default='', max_length=4096),
                       limit: int = Query(default=100, ge=1, le=100)) -> dict[str, Any]:
        try:
            if node.startswith('/models'):
                await session.catalog()
            return await run_blocking(lambda: {**browser.children(node, cursor, limit),
                'historyReference': history_reference(session.agent, browser.session_node(node))})
        except (KeyError, IndexError, FileNotFoundError) as exc:
            raise HTTPException(404, 'This collection is no longer available.') from exc

    @app.get('/api/search')
    async def search(query: str = Query(min_length=1, max_length=256),
                     cursor: str = Query(default='0:0', max_length=100),
                     limit: int = Query(default=100, ge=1, le=100)) -> dict[str, Any]:
        return await run_blocking(browser.search, query, cursor, limit)

    @app.get('/api/detail')
    async def detail(node: str = Query(max_length=4096),
                     offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
        try:
            return await run_blocking(lambda: {**browser.detail(node, offset),
                'historyReference': history_reference(session.agent, browser.session_node(node))})
        except (KeyError, IndexError, FileNotFoundError) as exc:
            raise HTTPException(404, 'This object is no longer available.') from exc

    @app.get('/api/feed')
    async def feed(offset: int = Query(default=-1, ge=-1),
                   limit: int = Query(default=40, ge=1, le=100)) -> dict[str, Any]:
        return await run_blocking(browser.feed, offset, limit)

    @app.get('/api/controls')
    async def controls() -> dict[str, Any]:
        queue = controller.queued_input()
        return {'settings': browser.settings(), 'queue': [(key, text_page(display_question(text), limit=160)['text']) for key, text in queue[:5]],
                'queuedTotal': len(queue),
                'editing': bool(editor.client_id)}

    @app.get('/api/plan')
    async def plan() -> Any:
        from .sources import materialize
        return await run_blocking(lambda: materialize(browser.plan()))

    @app.get('/api/editor')
    async def editor_page(client_id: str = Query(alias='clientId', min_length=1, max_length=128),
                          page: int = Query(default=0, ge=0)) -> dict[str, Any]:
        return await editor.page(client_id, page)

    @app.post('/api/editor')
    async def editor_action(request: Request) -> dict[str, Any]:
        try:
            command = EditorAction.model_validate(await read_action(request, 2 * 1024 * 1024))
        except ValidationError as exc:
            raise HTTPException(422, 'Invalid editor action.') from exc
        if command.session_id != session.agent.session_id:
            raise HTTPException(409, 'The session changed. Reopen the editor.')
        if command.client_id not in app.state.gui_clients:
            raise HTTPException(409, 'Connect the browser before opening an editor.')
        match command.action:
            case 'open':
                if not command.node.startswith('/files/'):
                    raise HTTPException(422, 'Choose a workspace file.')
                path = await run_blocking(browser.path, command.node)
                await editor.begin(path, command.client_id)
            case 'close':
                if editor.client_id != command.client_id:
                    raise HTTPException(409, 'This tab does not own the editor.')
                await editor.aclose()
            case 'stage' | 'save':
                await editor.update(command.client_id, command.page, command.text, save=command.action == 'save')
        return {'ok': True}

    @app.post('/api/actions')
    async def action(request: Request) -> dict[str, Any]:
        try:
            command = BrowserAction.model_validate(await read_action(request, 131072))
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422, 'Invalid browser action.') from exc
        if command.session_id != session.agent.session_id:
            raise HTTPException(409, 'The session changed. Refresh before applying this action.')
        if command.history_reference is not None and command.action != 'submit':
            raise HTTPException(422, 'History references belong to prompts.')
        if editor.client_id and command.action not in ('submit', 'stop', 'interrupt', 'discard', 'rename'):
            raise HTTPException(409, 'Close the editor before changing the session.')
        result: Any = None
        match command.action:
            case 'submit':
                if not command.text.strip():
                    raise HTTPException(422, 'Enter a prompt.')
                text = command.text
                if command.history_reference is not None:
                    text = await run_blocking(history_question, session.agent, text, **command.history_reference.model_dump())
                    if command.session_id != session.agent.session_id:
                        raise HTTPException(409, 'The session changed. Select the history object again.')
                result = controller.submit(text)
            case 'stop':
                result = controller.stop()
            case 'interrupt':
                result = controller.interrupt()
                await controller.wait_for_interrupt()
            case 'discard':
                result = controller.discard_input(command.identifier)
            case 'reset':
                await controller.reset()
            case 'new_task':
                controller.new_task()
            case 'rename':
                result = await controller.rename(command.text)
            case 'resume':
                await controller.resume(command.identifier)
            case 'fork':
                result = await controller.fork()
            case 'delete_saved':
                await controller.delete_saved(command.identifier)
            case 'delete_current':
                await controller.delete_current()
            case 'rename_saved':
                await controller.rename_saved(command.identifier, command.text)
            case 'setting':
                if command.enabled is None or command.setting is None:
                    raise HTTPException(422, 'Choose a setting and its enabled state.')
                setters = {'planning': controller.set_planning, 'overthinking': controller.set_overthinking,
                           'danger': controller.set_danger, 'hashline': controller.set_hashline}
                setters[command.setting](command.enabled)
            case 'model':
                await controller.select_model(command.identifier)
            case 'optimal':
                await controller.enable_optimal()
            case 'skill':
                await controller.select_skill(command.identifier or None)
            case 'rewind':
                result = await controller.rewind(command.identifier)
        return {'result': result, 'sessionId': session.agent.session_id}
