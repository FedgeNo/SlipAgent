"""Typed active work and cancellation-safe holds at agent boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from .resources import ResourceOwner, ResourceRef

if TYPE_CHECKING:
    from .runtime import RuntimeFrame


class SessionWork(ResourceOwner):
    agent_task = ResourceRef[asyncio.Task[str]]("active_agent_task")
    command_task = ResourceRef[asyncio.Task[bool]]("active_command_task")
    command_tasks = ResourceRef[set[asyncio.Task[None]]]("command_tasks")
    deferred_commands = ResourceRef[list[str]]("deferred_commands")
    resume_after_commands = ResourceRef[bool]("resume_after_commands")


class WorkCoordinator:
    """Hold new agent batches and reload commits without changing UI focus."""

    def __init__(self, frame: Callable[[], RuntimeFrame | None]) -> None:
        self.frame = frame
        self.held_frame: RuntimeFrame | None = None
        self.released = asyncio.Event()
        self.released.set()

    def hold(self) -> None:
        self.released.clear()
        frame = self.frame()
        if frame is not None and self.held_frame is None:
            frame.busy += 1
            self.held_frame = frame

    def release(self) -> None:
        if self.held_frame is not None:
            self.held_frame.busy -= 1
            self.held_frame = None
        self.released.set()

    async def wait(self) -> None:
        await self.released.wait()

    async def boundary(self) -> None:
        await self.wait()

    async def checkpoint(self) -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            return
        await self.boundary()
        frame = self.frame()
        if frame is not None:
            await frame.checkpoint(boundary=True)
