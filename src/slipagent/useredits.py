"""Session-owned coordination between editable files and agent boundaries."""

from __future__ import annotations

from pathlib import Path
import hashlib
import uuid
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
        if not self.session.agent.running or switching:
            await self.boundary()
        else:
            self.terminal.notify("Opening editor after this turn")

    async def boundary(self) -> None:
        if self.pending is not None and not self.opening:
            self.opening = True
            path, line = self.pending
            self.pending = None
            self._clear_opening_notification()
            try:
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
        previous = edits.get(key)
        if not diff and previous is None:
            self.session.agent._persist()
            return
        digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        store = self.session.registry.resources.artifacts
        if (previous is not None and previous["path"] == str(path)
                and previous.get("diff_digest", hashlib.sha256(previous.get("diff", "").encode("utf-8")).hexdigest()) == digest
                and (store is None or "artifact" in previous)):
            return
        from .editdelivery import EDIT_PREVIEW_CHARS
        entry: dict[str, Any] = {"path": str(path), "diff_preview": diff[:EDIT_PREVIEW_CHARS],
                                 "diff_characters": len(diff), "diff_digest": digest,
                                 "revision": uuid.uuid4().hex, "source": "user", "delivered": False,
                                 "saved_before_step": len(self.session.agent.history.steps) + 1}
        if not diff:
            entry["reverted_to_editor_baseline"] = True
        if store is not None:
            try:
                record = store.save(f"User edited {path}", diff, kind="diff", call_id="user-edit", step_id=len(self.session.agent.history.steps) + 1)
                entry["artifact"] = record
                self.terminal.write_artifact_reference(record)
            except (OSError, ValueError) as exc:
                self.terminal.notify("Edit saved; diff archive failed.")
                self.terminal.write(f"Could not archive user edit: {exc}")
        if "artifact" not in entry:
            entry["diff"] = diff  # Preserve full data until archival can succeed.
        edits[key] = entry
        self.terminal._touched_items[str(path)] = "user edited"
        self.session.agent._persist()

    def close(self) -> None:
        self.pending = None
        self._clear_opening_notification()
        self.release()

    def _clear_opening_notification(self) -> None:
        if getattr(self.terminal, "_notification", "") == "Opening editor after this turn":
            self.terminal.notify("")

    async def aclose(self) -> None:
        try:
            panel = self.terminal.__dict__.get("_editor_panel")
            if panel is not None:
                await panel.shutdown()
        finally:
            self.close()
