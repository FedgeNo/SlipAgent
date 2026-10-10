"""Session-owned coordination between editable files and agent boundaries."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class UserEdits:
    def __init__(self, session: Any, terminal: Any) -> None:
        self.session = session
        self.terminal = terminal
        self.pending: tuple[Path, int] | None = None
        self.released = asyncio.Event()
        self.released.set()
        self.opening = False
        self.frame_held = False

    async def request(self, path: Path, line: int = 1) -> None:
        if self.pending is not None or self.opening or self.terminal.__dict__.get("_editor_active"):
            self.terminal.notify("Close the current editor first.")
            return
        self.pending = path, line
        self.released.clear()
        if self.session.reloader is not None:
            self.session.reloader.busy += 1
            self.frame_held = True
        if self.session.agent.running:
            self.terminal.notify("Editor queued until this turn finishes.")
        else:
            await self.boundary()

    async def boundary(self) -> None:
        if self.pending is not None and not self.opening:
            self.opening = True
            path, line = self.pending
            self.pending = None
            try:
                jobs = self.session.registry.services.get("command_jobs")
                if jobs is not None:
                    await jobs.stop_all()
                await self.terminal._open_editor_now(path, line=line)
            except BaseException:
                self.close()
                raise
            finally:
                self.opening = False
            if not self.terminal.__dict__.get("_editor_active"):
                self.close()
        await self.released.wait()

    def record(self, path: Path, diff: str, edit_id: str = "") -> None:
        edits = self.session.registry.context_notes.setdefault("user_edits", {})
        key = edit_id or str(path)
        if not diff:
            edits.pop(key, None)
            self.session.agent._persist()
            return
        entry: dict[str, Any] = {"path": str(path), "diff": diff, "source": "user",
                                 "saved_before_step": len(self.session.agent.history.steps) + 1}
        store = self.session.registry.services.get("artifact_store")
        if store is not None:
            try:
                record = store.save(f"User edited {path}", diff, kind="diff", call_id="user-edit", step_id=len(self.session.agent.history.steps) + 1)
                entry["artifact"] = record
                self.terminal.write_artifact_reference(record)
            except (OSError, ValueError) as exc:
                self.terminal.notify("Edit saved; diff archive failed.")
                self.terminal.write(f"Could not archive user edit: {exc}")
        edits[key] = entry
        self.terminal._touched_files[str(path)] = "user edited"
        self.session.agent._persist()

    def close(self) -> None:
        if self.frame_held:
            self.session.reloader.busy -= 1
            self.frame_held = False
        self.pending = None
        self.released.set()

    async def checkpoint(self) -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            return
        await self.boundary()
        if self.session.reloader is not None:
            await self.session.reloader.checkpoint(boundary=True)

    async def aclose(self) -> None:
        try:
            panel = self.terminal.__dict__.get("_editor_panel")
            if panel is not None:
                await panel.shutdown()
        finally:
            self.close()
