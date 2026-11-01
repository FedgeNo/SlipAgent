"""Construct the shared session engine without importing a user interface."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from collections.abc import Callable

from .agent import Agent, build_system_prompt
from .config import Config, ConfigError
from .events import EventHandler
from .instructions import load_project_instructions
from .mcp import MCPManager
from .providers import create_client
from .sessionstate import Session
from .tools import build_default_registry
from .workspace import Workspace, WorkspaceError


@dataclass(frozen=True, kw_only=True)
class SessionOptions:
    danger: bool = False
    python: str | None = None
    planning: bool = True
    hashline: bool = True
    context_tokens: int | None = None
    reasoning_history_steps: int = 25
    filtered_thoughts: bool = True
    exposed_tools: tuple[str, ...] | None = None
    mcp: bool = False


async def create_session(config: Config, options: SessionOptions, *,
                         session_factory: Callable[..., Session] = Session,
                         on_event: EventHandler | None = None) -> Session:
    try:
        workspace = Workspace(config.workspace, danger=options.danger)
        load_project_instructions(workspace)
    except WorkspaceError as exc:
        raise ConfigError(str(exc)) from exc
    client = create_client(config.provider, api_key=config.api_key, base_url=config.base_url,
                           http_referer=config.http_referer, app_title=config.app_title)
    async with AsyncExitStack() as resources:
        resources.push_async_callback(client.aclose)
        registry = build_default_registry(workspace)
        resources.push_async_callback(registry.aclose)
        if not options.hashline:
            from .tools.hashline import set_hashline
            set_hashline(registry, False)
        registry.services['editable_prompts'] = True
        registry.services['project_environment'].python_override = options.python
        agent = Agent(client=client, registry=registry, model=config.model,
                      max_steps=config.max_steps, temperature=config.temperature, max_tokens=config.max_tokens,
                      context_steps=config.context_steps, overthinking=config.overthinking,
                      planning=options.planning, context_tokens=options.context_tokens,
                      reasoning_history_steps=options.reasoning_history_steps,
                      filtered_thoughts=options.filtered_thoughts, exposed_tools=options.exposed_tools,
                      system_prompt=build_system_prompt(str(workspace.root)), on_event=on_event)
        manager = MCPManager(workspace.root, registry=registry) if options.mcp else None
        if manager is not None:
            resources.push_async_callback(manager.aclose)
        session = session_factory(agent=agent, registry=registry, client=client, workspace=workspace,
                                  api_key=config.api_key, base_url=config.base_url,
                                  http_referer=config.http_referer, app_title=config.app_title,
                                  mcp=manager)
        client.on_retry = session.retry_handler
        async def close_selected_client() -> None:
            if session.client is not client:
                await session.client.aclose()
        resources.push_async_callback(close_selected_client)
        session.configure_compaction()
        resources.pop_all()
        return session
