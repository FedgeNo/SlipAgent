"""Session-owned coordination between editable files and agent boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .work import WorkCoordinator

if TYPE_CHECKING:
    from .sessionstate import Session
    from .terminal import TerminalUI


class UserEdits(WorkCoordinator):
    def __init__(self, session: Session, terminal: TerminalUI) -> None:
        super().__init__(lambda: session.reloader)
        self.session = session
        self.terminal = terminal
        self.pending: tuple[Path, int] | None = None
        self.opening = False

    async def request(self, path: Path, line: int = 1) -> None:
        if self.pending is not None or self.opening:
            self.terminal.notify("An editor request is already pending.")
            return
        switching = bool(self.terminal.__dict__.get("_editor_active"))
        if switching:
            self.opening = True
            try:
                if not await self.terminal._editor_panel.close(release=False):
                    return
            finally:
                self.opening = False
        self.pending = path, line
        self.hold()
        if self.session.agent.running and not switching:
            self.terminal.notify("Editor queued until this turn finishes.")
        else:
            await self.boundary()

    async def boundary(self) -> None:
        if self.pending is not None and not self.opening:
            self.opening = True
            path, line = self.pending
            self.pending = None
            try:
                jobs = self.session.registry.resources.jobs
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
        store = self.session.registry.resources.artifacts
        if store is not None:
            try:
                record = store.save(f"User edited {path}", diff, kind="diff", call_id="user-edit", step_id=len(self.session.agent.history.steps) + 1)
                entry["artifact"] = record
                self.terminal.write_artifact_reference(record)
            except (OSError, ValueError) as exc:
                self.terminal.notify("Edit saved; diff archive failed.")
                self.terminal.write(f"Could not archive user edit: {exc}")
        edits[key] = entry
        self.terminal._touched_items[str(path)] = "user edited"
        self.session.agent._persist()

    def close(self) -> None:
        self.pending = None
        self.release()

    async def aclose(self) -> None:
        try:
            panel = self.terminal.__dict__.get("_editor_panel")
            if panel is not None:
                await panel.shutdown()
        finally:
            self.close()
