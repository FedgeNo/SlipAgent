"""Private append-only session journals; resume never replays tools.

Records reach the operating system when written. Message deltas and tool-start
markers are also synced to disk before execution, off the event loop so a slow
disk delays the agent rather than the interface. A missing result means
uncertainty, not permission to retry. Only a torn final line can
be ignored during recovery; malformed complete records reject the journal.
"""

from __future__ import annotations

from .prompts import load_prompt

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .context import ConversationHistory, CompletedStep
from .config import MIN_CONTEXT_TOKENS, data_directory
from .api import APIError
from .task import TaskMemory
from .tools.blocking import run_blocking
from .tools.output import CommandLog, OutputStream
from .types import Message, Usage

if TYPE_CHECKING:
    from .agent import Agent


class SessionError(APIError):
    """Session storage cannot safely save or restore its records."""


def _validate_title(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or any(
        ord(char) < 32 or 127 <= ord(char) <= 159 for char in value
    ):
        raise SessionError("Session names must be nonempty and contain no control characters.")
    return value


def session_title(name: str) -> str:
    """Format the shared saved-conversation and terminal title."""
    return f"{_validate_title(name)} | SlipAgent"


def _private_directory(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)


class SessionJournal:
    def __init__(self, project: str, directory: Path | None = None) -> None:
        # Keep lexical path spelling (including symlink names); resolve only ~
        # and relative cwd. The full spelling is retained to detect a collision.
        expanded = os.path.expanduser(project)
        self.project = expanded
        if not os.path.isabs(expanded):
            self.project = os.getcwd() if expanded == "." else os.path.join(os.getcwd(), expanded)
        raw = os.fsencode(self.project)
        identity = hashlib.md5(raw, usedforsecurity=False).hexdigest() + "-" + hashlib.sha1(raw, usedforsecurity=False).hexdigest()
        root = Path(os.path.abspath(os.path.expanduser(str(directory)))) if directory is not None else data_directory()
        # Storage reads open each component without following links, so the
        # store itself lives at the resolved location. The project identity
        # above keeps the lexical spelling.
        self.directory = root.resolve() / identity
        self.session_id = ""
        self.path: Path | None = None
        self.cursor = 0
        self.last_message: dict[str, Any] | None = None
        self.system: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None
        self.step_count = 0
        self.failed = False
        # Records written since the last sync to disk.
        self._unsynced = False
        try:
            _private_directory(self.directory)
            metadata = self.directory / "project.json"
            if metadata.exists():
                if json.loads(metadata.read_text())["project"] != self.project:
                    raise SessionError("Project session path collision; existing storage was preserved.")
            else:
                fd = os.open(metadata, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as target:
                    json.dump({"project": self.project}, target, ensure_ascii=False)
        except (OSError, ValueError, KeyError) as exc:
            raise SessionError(f"Cannot open session storage {self.directory}: {exc}") from exc

    def _append(self, kind: str, **data: Any) -> None:
        if self.path is None or self.failed:
            raise SessionError("Session journal is unavailable; no further actions can be recorded.")
        from .sources import json_chunks
        try:
            # Validate and serialize before appending; partial serialization must
            # not corrupt the append-only journal.
            with tempfile.SpooledTemporaryFile(max_size=65536, mode="w+t", encoding="utf-8", dir=str(self.directory)) as record:
                for chunk in json_chunks({"type": kind, **data}, indent=None, allow_nan=False):
                    record.write(chunk)
                record.write("\n")
                record.seek(0)
                with self.path.open("a", encoding="utf-8") as target:
                    while chunk := record.read(65536):
                        target.write(chunk)
                    target.flush()
            self._unsynced = True
        except OSError as exc:
            self.failed = True
            raise SessionError(f"Could not save session {self.session_id}: {exc}. Inspect its journal before resuming.") from exc

    def _sync(self) -> None:
        # Syncing any descriptor of the file covers every earlier write to it.
        try:
            descriptor = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            self.failed = True
            raise SessionError(f"Could not save session {self.session_id}: {exc}. Inspect its journal before resuming.") from exc

    async def flush(self) -> None:
        """Wait until every record written so far is on disk, without blocking the event loop."""
        if self.path is None or self.failed:
            raise SessionError("Session journal is unavailable; no further actions can be recorded.")
        if self._unsynced:
            self._unsynced = False
            try:
                await run_blocking(self._sync)
            except BaseException:
                self._unsynced = True
                raise

    @property
    def title(self) -> str:
        # Older live journals predate titles; their project path is still known.
        return getattr(self, "_title", session_title(self.project))

    @property
    def named(self) -> bool:
        """Whether a title other than the project default was given."""
        return self.title != session_title(self.project)

    def rename(self, name: str) -> None:
        """Atomically save mutable title metadata without rewriting the journal."""
        title = session_title(name)
        if self.failed:
            raise SessionError("Session journal is unavailable; the title was not changed.")
        if self.path is None:
            self._title = title
            return
        temporary: Path | None = None
        try:
            fd, filename = tempfile.mkstemp(prefix=".title-", dir=self.directory)
            temporary = Path(filename)
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump({"title": title}, target, ensure_ascii=False)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(self.directory / (self.session_id + ".title.json"))
        except OSError as exc:
            raise SessionError(f"Could not save the session title: {exc}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._title = title

    def _saved_title(self, session_id: str, header: dict[str, Any]) -> str:
        title = header.get("title", session_title(self.project))
        try:
            with (self.directory / (session_id + ".title.json")).open(encoding="utf-8") as source:
                title = json.load(source)["title"]
        except FileNotFoundError:
            pass  # Unrenamed and older sessions use their header/project title.
        return _validate_title(title)

    def begin(self, agent: Agent, parent: str | None = None, *, title: str | None = None) -> None:
        title = session_title(self.project) if title is None else _validate_title(title)
        self.session_id = uuid.uuid4().hex
        self.path = self.directory / (self.session_id + ".jsonl")
        self.cursor, self.step_count = 0, 0
        self.last_message, self.system, self.state = None, None, None
        self.failed = False
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        except OSError as exc:
            raise SessionError(f"Cannot create session journal: {exc}") from exc
        self._append("header", version=1, project=self.project, session_id=self.session_id,
                     parent=parent, created=datetime.now(timezone.utc).isoformat(), title=title)
        self._title = title
        agent.session_id = self.session_id
        self._artifact_directory(agent, parent=parent)
        self._checkpoint_directory(agent, parent=parent)
        diagnostics = agent.registry.services.get("request_diagnostics")
        if diagnostics is not None:
            diagnostics.use_directory(self.directory / (self.session_id + "-requests"))
        archive = agent.registry.services.get("command_archive")
        if archive is not None:
            directory = self.directory / (self.session_id + "-logs")
            _private_directory(directory)
            archive.use_directory(directory, self.log)
        self.record(agent)

    def _artifact_directory(self, agent: Agent, *, parent: str | None = None) -> None:
        store = agent.registry.services.get("artifact_store")
        if store is not None:
            try:
                store.use_directory(self.directory / (self.session_id + "-artifacts"),
                                    source=self.directory / (parent + "-artifacts") if parent else None)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SessionError(f"Cannot restore inspection records: {exc}") from exc

    def _checkpoint_directory(self, agent: Agent, *, parent: str | None = None) -> None:
        checkpoints = agent._checkpoints()
        if checkpoints is not None:
            try:
                checkpoints.use_directory(
                    self.directory / (self.session_id + "-checkpoints"),
                    source=self.directory / (parent + "-checkpoints") if parent is not None else None,
                )
                if any(batch["rewound"] for batch in checkpoints.batches):
                    agent.registry.context_notes["file_rewind"] = load_prompt('file-rewind.md', notice=load_prompt('session-rewind.md').strip())
            except OSError as exc:
                raise SessionError(f"Cannot restore file checkpoints: {exc}") from exc

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        # Preserve reasoning independently of working-context retention and
        # compaction. Keep its originating model as well.
        stored = getattr(message, "_stored", None)
        if stored is not None:
            return {**stored, "reasoning": stored.get("reasoning"), "reasoning_model": stored.get("reasoning_model")}
        return {**message.to_record(), "reasoning": message.reasoning, "reasoning_model": message.reasoning_model}

    def record(self, agent: Agent) -> None:
        if self.path is None:
            # Startup commands and previews do not create empty conversations.
            if not agent.pending and not agent.registry.context_notes.get("user_edits") and not any(message.role == "user" for message in agent.messages):
                return
            self.begin(agent, title=self.title)
            return
        if self.cursor > len(agent.messages):
            raise SessionError("Conversation changed without starting a new session journal.")
        if self.cursor and self.last_message != self._message(agent.messages[self.cursor - 1]):
            self._append("replace", index=self.cursor - 1, message=self._message(agent.messages[self.cursor - 1]))
        if agent.messages and agent.messages[0].role == "system":
            system = self._message(agent.messages[0])
            if self.cursor and system != self.system:
                self._append("replace", index=0, message=system)
            self.system = copy.deepcopy(system)
        # A failed turn's record shares its message's line, so a crash between
        # lines cannot restore a partial reply as an accepted one.
        failed = {id(step.messages[-1]): step for step in agent.history.steps[self.step_count:]
                  if getattr(step, "request_failure", None) is not None}
        for message in agent.messages[self.cursor:]:
            step = failed.pop(id(message), None)
            if step is None:
                self._append("message", message=self._message(message))
            else:
                self._append("message", message=self._message(message),
                             request_failure={"id": step.id, "failure": step.request_failure})
            self.cursor += 1
        self.last_message = copy.deepcopy(self._message(agent.messages[-1])) if agent.messages else None
        state = {"task_start": agent.history.task.start_message,
                 "usage": asdict(agent.usage), "pending": list(agent.pending),
                 "queued_message_indices": list(agent.queued_messages),
                 "model": agent.model, "temperature": agent.temperature,
                 "context_tokens": agent.context_tokens,
                 "user_edits": agent.registry.context_notes.get("user_edits", {})}
        if state != self.state:
            self._append("state", **state)
            self.state = copy.deepcopy(state)
        for step in agent.history.steps[self.step_count:]:
            self.step(step)
        for step in failed.values():
            self._append("request_failure", id=step.id, failure=step.request_failure)
        self.step_count = len(agent.history.steps)

    def discard_input(self, agent: Agent, index: int | None, pending_index: int | None) -> None:
        from .inputqueue import discarded_state

        self.record(agent)
        assert self.state is not None
        state = discarded_state(self.state, index, pending_index)
        self._append("discard_input", index=index, pending_index=pending_index)
        self.state = state
        if index is not None:
            self.cursor -= 1
            messages = agent.messages[:index] + agent.messages[index + 1:]
            self.last_message = copy.deepcopy(self._message(messages[-1])) if messages else None

    def step(self, step: Any) -> None:
        self._append("step", id=step.id, summary=step.summary,
                     reasoning_summary=getattr(step, 'reasoning_summary', None),
                     compaction_status=getattr(step, "compaction_status", "pending"),
                     compaction_error=getattr(step, "compaction_error", None),
                     user_edits=getattr(step, "user_edits", None))
        if (getattr(step, "historical_summary", None) is not None or getattr(step, "history_merge_attempt", 0) > 0
                or getattr(step, "era_summary", None) is not None or getattr(step, "era_attempt", 0) > 0):
            self._append("history_merge", id=step.id,
                         summary=getattr(step, "historical_summary", None),
                         attempted_at=getattr(step, "history_merge_attempt", 0),
                         era=getattr(step, "era_summary", None),
                         era_attempted_at=getattr(step, "era_attempt", 0))

    def tool_started(self, step_id: int, call_id: str) -> None:
        self._append("tool_start", step=step_id, call_id=call_id)

    def edits_delivered(self, step_id: int, edits: list[dict[str, Any]]) -> None:
        """Persist accepted edit observations before their step's tools start."""
        self._append("user_edits", id=step_id, edits=edits)

    def log(self, log: CommandLog) -> None:
        self._append("log", metadata={**log.metadata(), "command": log.command})

    def replace_message(self, index: int, message: Message) -> None:
        """Persist late observations without replaying or rewriting their turn."""
        self._append("replace", index=index, message=self._message(message))

    def listing(self) -> list[dict[str, str]]:
        result = []
        try:
            for path in sorted(self.directory.glob("*.jsonl"), key=lambda item: item.stat().st_mtime_ns, reverse=True):
                with path.open(encoding="utf-8") as source:
                    header = json.loads(source.readline())
                if header.get("type") != "header" or header.get("project") != self.project:
                    raise SessionError(f"Invalid session header: {path}")
                result.append({"id": path.stem, "created": header["created"], "current": str(path == self.path),
                               "title": self._saved_title(path.stem, header)})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise SessionError(f"Cannot list saved sessions: {exc}") from exc
        return result

    def delete_current(self) -> None:
        """Remove the current journal and its sidecars after owners are drained."""
        if self.path is None:
            raise SessionError("No current saved session to delete.")
        try:
            for suffix in ("-logs", "-requests", "-checkpoints", "-artifacts", "-turns"):
                directory = self.directory / (self.session_id + suffix)
                if directory.exists():
                    shutil.rmtree(directory)
            (self.directory / (self.session_id + ".title.json")).unlink(missing_ok=True)
            self.path.unlink()
        except OSError as exc:
            self.failed = True
            raise SessionError(f"Could not delete session {self.session_id}: {exc}. Some session files may have been removed.") from exc
        self.path = None
        self.session_id = ""
        self.cursor, self.step_count = 0, 0
        self.last_message, self.system, self.state = None, None, None
        self.failed = False
        self._title = session_title(self.project)

    def load(self, selected: str) -> dict[str, Any]:
        if selected == "latest":
            choices = [item for item in self.listing() if item["id"] != self.session_id]
            if not choices:
                raise SessionError("No earlier sessions are saved for this project.")
            selected = choices[0]["id"]
        if not re.fullmatch(r"[0-9a-f]{32}", selected):
            raise SessionError("Session ID must be a complete ID from /sessions, or latest.")
        path = self.directory / (selected + ".jsonl")
        data: dict[str, Any] = {"messages": [], "state": {}, "steps": {}, "started": set(), "logs": {}, "id": selected, "torn": False}
        try:
            with path.open("rb") as source:
                for index, raw in enumerate(source):
                    if not raw.endswith(b"\n"):
                        data["torn"] = True
                        break
                    event = json.loads(raw)
                    # Read journals written before history posts were named steps.
                    if event.get("type") == "post":
                        event["type"] = "step"
                    if event.get("type") == "tool_start" and "post" in event:
                        event["step"] = event.pop("post")
                    if event.get("type") == "log":
                        metadata = event["metadata"]
                        if "post_id" in metadata:
                            metadata["step_id"] = metadata.pop("post_id")
                    kind = event["type"]
                    if index == 0:
                        if kind != "header" or event.get("version") != 1 or event.get("project") != self.project or event.get("session_id") != selected:
                            raise ValueError("unsupported or mismatched session header")
                        data["title"] = self._saved_title(selected, event)
                    elif kind == "message":
                        data["messages"].append(event["message"])
                        if "request_failure" in event:
                            marker = event["request_failure"]
                            data.setdefault("request_failures", {})[marker["id"]] = marker["failure"]
                    elif kind == "replace":
                        if type(event["index"]) is not int or not 0 <= event["index"] < len(data["messages"]):
                            raise ValueError("invalid message replacement")
                        data["messages"][event["index"]] = event["message"]
                    elif kind == "state":
                        data["state"] = event
                    elif kind == "discard_input":
                        from .inputqueue import discarded_state

                        removed = event["index"]
                        data["state"] = discarded_state(data["state"], removed, event["pending_index"])
                        if removed is not None:
                            if not 0 <= removed < len(data["messages"]) or data["messages"][removed]["role"] != "user":
                                raise ValueError("invalid unsent message discard")
                            del data["messages"][removed]
                    elif kind == "step":
                        data["steps"][event["id"]] = event
                    elif kind == "history_merge":
                        data.setdefault("history_merges", {})[event["id"]] = event
                    elif kind == "request_failure":
                        data.setdefault("request_failures", {})[event["id"]] = event["failure"]
                    elif kind == "user_edits":
                        data.setdefault("delivered_user_edits", {})[event["id"]] = event["edits"]
                    elif kind == "tool_start":
                        data["started"].add((event["step"], event["call_id"]))
                    elif kind == "log":
                        data["logs"][event["metadata"]["log_id"]] = event["metadata"]
                    else:
                        raise ValueError(f"unknown event {kind!r}")
            if not data["state"]:
                raise ValueError("no complete saved session state")
            data["message_count"] = len(data["messages"])
            data["messages"] = self._restore_messages(data)
            state = data["state"]
            if type(state.get("task_start")) is not int or not 0 <= state["task_start"] <= len(data["messages"]):
                raise ValueError("invalid task start")
            if not isinstance(state.get("pending"), list) or any(not isinstance(text, str) for text in state["pending"]):
                raise ValueError("invalid queued input")
            queued = state.get("queued_message_indices", [])
            if (not isinstance(queued, list) or any(
                    type(index) is not int or not 0 <= index < len(data["messages"])
                    or data["messages"][index].role != "user" for index in queued)
                    or queued != sorted(set(queued))):
                raise ValueError("invalid queued message indices")
            cap = state.get("context_tokens")
            if cap is not None and (type(cap) is not int or cap < MIN_CONTEXT_TOKENS):
                raise ValueError("invalid context cap")
            usage = state["usage"]
            if not isinstance(usage, dict) or any(type(usage.get(name)) is not int or usage[name] < 0 for name in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError("invalid token accounting")
            cost = usage.get("cost")
            if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                raise ValueError("invalid cost accounting")
            for log_id, metadata in data["logs"].items():
                if not isinstance(log_id, str) or not re.fullmatch(r"[0-9a-f]{32}", log_id):
                    raise ValueError("invalid command log ID")
                if not isinstance(metadata["command"], str) or type(metadata["finished"]) is not bool or type(metadata["timed_out"]) is not bool:
                    raise ValueError("invalid command log metadata")
                if metadata["returncode"] is not None and type(metadata["returncode"]) is not int:
                    raise ValueError("invalid command exit code")
                if metadata["step_id"] is not None and (type(metadata["step_id"]) is not int or metadata["step_id"] < 1):
                    raise ValueError("invalid command step ID")
                if metadata["call_id"] is not None and not isinstance(metadata["call_id"], str):
                    raise ValueError("invalid command call ID")
                for name in ("stdout", "stderr"):
                    stream = metadata["streams"][name]
                    if any(type(stream[field]) is not int or stream[field] < 0 for field in ("retained_bytes", "lost_bytes")) or not isinstance(stream["retention_error"], str):
                        raise ValueError("invalid command stream metadata")
                    if "payload" in stream and (not isinstance(stream["payload"], str)
                            or stream["payload"] != "" and re.fullmatch(r"[0-9a-f]{64}", stream["payload"]) is None):
                        raise ValueError("invalid shared command payload ID")
            task = TaskMemory(start_message=data["state"]["task_start"])
            history = ConversationHistory(task=task)
            history.sync(data["messages"])
            for step_id, saved in data["steps"].items():
                if type(step_id) is not int or not 1 <= step_id <= len(history.steps):
                    raise ValueError("invalid saved step ID")
                if saved.get("summary") is not None and (not isinstance(saved["summary"], str) or not saved["summary"].strip()):
                    raise ValueError("invalid saved summary")
                if saved.get('reasoning_summary') is not None and not isinstance(saved['reasoning_summary'], str):
                    raise ValueError('invalid saved reasoning summary')
            for step in history.steps:
                saved = data["steps"].get(step.id, {})
                step.summary = saved.get("summary")
                step.results_summarized = step.summary is not None
                if isinstance(step, CompletedStep):
                    step.reasoning_summary = saved.get('reasoning_summary')
                    step.compaction_status = saved.get("compaction_status", "interrupted")
                    if step.compaction_status == "pending":
                        step.compaction_status = "interrupted"
                    step.compaction_error = saved.get("compaction_error")
                    step.observed = step.id < len(history.steps) or not step.has_results
            from .editdelivery import acknowledge, validate_notifications
            delivered_edits = dict(data.get("delivered_user_edits", {}))
            for step_id, saved in data["steps"].items():
                if saved.get("user_edits") is not None:
                    delivered_edits[step_id] = saved["user_edits"]
            for step_id, edits in delivered_edits.items():
                if type(step_id) is not int or not 1 <= step_id <= len(history.steps):
                    raise ValueError("Invalid saved-edit delivery turn")
                validate_notifications(edits)
                setattr(history.steps[step_id - 1], "user_edits", edits)
                acknowledge(data["state"].get("user_edits", {}), edits)
            from .requestfailures import retry_state, validate_failure
            for step_id, failure in data.get("request_failures", {}).items():
                if type(step_id) is not int or not 1 <= step_id <= len(history.steps):
                    raise ValueError("invalid failed request turn number")
                validate_failure(failure)
                step = history.steps[step_id - 1]
                if step.has_results or any(message.tool_calls for message in step.messages):
                    raise ValueError("a failed request cannot contain executed tools")
                setattr(step, "request_failure", failure)
                if isinstance(step, CompletedStep):
                    step.observed = step.id < len(history.steps)
            recovery = retry_state(history.steps)
            if recovery:
                previous = len(history.steps) - recovery["consecutive_failures"] - 1
                if previous >= 0:
                    batch = history.steps[previous]
                    if isinstance(batch, CompletedStep) and batch.has_results:
                        batch.observed = False
            from .historymerge import MERGE_INTERVAL, servable, validate_era_record, validate_historical_record
            for step_id, saved in data.get("history_merges", {}).items():
                if type(step_id) is not int or not 1 <= step_id <= len(history.steps):
                    raise ValueError("invalid historical summary anchor")
                step = history.steps[step_id - 1]
                for attempt_key, attribute in (("attempted_at", "history_merge_attempt"), ("era_attempted_at", "era_attempt")):
                    attempted = saved.get(attempt_key, 0)
                    if type(attempted) is not int or (attempted and not step_id <= attempted <= len(history.steps)):
                        raise ValueError("invalid historical merge attempt")
                    setattr(step, attribute, attempted)
                record = saved.get("summary")
                # Narratives from before fixed 20-turn blocks are not served;
                # their blocks are summarized again in the background.
                if record is not None and servable(record, MERGE_INTERVAL) and record.get("step_id") == step_id:
                    validate_historical_record(record, step_id, len(history.steps))
                    setattr(step, "historical_summary", record)
                era = saved.get("era")
                if era is not None:
                    validate_era_record(era, step_id, len(history.steps))
                    setattr(step, "era_summary", era)
            data["history"] = history
            data["usage"] = Usage.from_api(data["state"]["usage"])
            return data
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise SessionError(f"Cannot resume {selected}: {exc}. The saved journal was not modified.") from exc

    def _restore_messages(self, data: dict[str, Any]) -> list[Message]:
        messages = []
        for raw in data["messages"]:
            if not isinstance(raw, dict) or raw.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("invalid message")
            allowed = (str, dict, list) if raw["role"] == "tool" else (str,)
            if raw.get("content") is not None and not isinstance(raw["content"], allowed):
                raise ValueError("invalid message content")
            message = Message.from_api(raw)
            message.reasoning_model = raw.get("reasoning_model")
            messages.append(message)
        # Only the final batch may be incomplete. Fill missing observations
        # explicitly so providers get a valid call/result sequence on resume.
        index, step = 0, 0
        while index < len(messages):
            message = messages[index]
            index += 1
            if message.role == "assistant":
                step += 1
                for call in message.tool_calls or []:
                    if index < len(messages):
                        if messages[index].role != "tool" or messages[index].tool_call_id != call.id:
                            raise ValueError("broken tool-call/result sequence")
                    else:
                        detail = "Tool interrupted; outcome unknown and effects may be partial. Inspect before retrying." if (step, call.id) in data["started"] else "Tool was not started before interruption. No automatic replay was performed."
                        messages.append(Message.tool_result(call.id, {"tool": call.name, "call_id": call.id, "status": "error", "content": detail}))
                    index += 1
            elif message.role == "tool":
                raise ValueError("orphan tool result")
        return messages

    def restore(self, agent: Agent, data: dict[str, Any], *, fork: bool = False) -> None:
        if agent.running:
            raise SessionError("Resume is available only while the agent is idle.")
        agent.reset(new_session=False)
        # Refresh the operating/project instructions from this launch. Archived
        # instructions remain preserved in the parent journal for inspection.
        prefix = [message for message in agent.messages if message.role == "system"]
        messages = data["messages"]
        if prefix and messages and messages[0].role == "system":
            messages[0] = prefix[0]
        agent.messages = messages
        loaded = data["history"]
        agent.history.steps = loaded.steps
        agent.history.cursor = loaded.cursor
        agent.history._request = loaded._request
        agent.history.task = loaded.task
        agent.pending = list(data["state"]["pending"])
        # Journals from before the cap existed keep this launch's setting.
        if "context_tokens" in data["state"]:
            agent.context_tokens = data["state"]["context_tokens"]
        if data["state"].get("user_edits"):
            agent.registry.context_notes["user_edits"] = data["state"]["user_edits"]
        agent.queued_messages.update({index: messages[index].content or ""
                                      for index in data["state"].get("queued_message_indices", [])})
        agent.usage = data["usage"]
        if fork:
            self.begin(agent, parent=data["id"], title=data["title"])
        else:
            self.session_id = data["id"]
            self.path = self.directory / (self.session_id + ".jsonl")
            if data["torn"]:
                # Remove only the incomplete final event before appending.
                with self.path.open("r+b") as journal_file:
                    payload = journal_file.read()
                    journal_file.truncate(payload.rfind(b"\n") + 1)
            self.cursor = data["message_count"]
            self.step_count = len(data["steps"])
            self.last_message = None
            self.system = None
            self.state = None
            self.failed = False
            self._title = data["title"]
            agent.session_id = self.session_id
            self._artifact_directory(agent)
            self._checkpoint_directory(agent)
            diagnostics = agent.registry.services.get("request_diagnostics")
            if diagnostics is not None:
                diagnostics.use_directory(self.directory / (self.session_id + "-requests"))
            archive = agent.registry.services.get("command_archive")
            if archive is not None:
                directory = self.directory / (self.session_id + "-logs")
                _private_directory(directory)
                archive.use_directory(directory, self.log)
            self.record(agent)
        agent._archive_history()
        archive = agent.registry.services.get("command_archive")
        if archive is not None:
            for log_id, metadata in data["logs"].items():
                if not re.fullmatch(r"[0-9a-f]{32}", log_id):
                    raise SessionError("Invalid archived command ID")
                log = CommandLog(archive, metadata["command"], metadata["step_id"], metadata["call_id"])
                log.id = log_id
                if metadata.get("artifacts"):
                    log.artifact_record = metadata["artifacts"][0]
                log.returncode, log.timed_out = metadata["returncode"], metadata["timed_out"]
                log.finished = True
                for name in ("stdout", "stderr"):
                    target = OutputStream()
                    details = metadata["streams"][name]
                    source = self.directory / (data["id"] + "-logs") / (log_id + "-" + name)
                    target.lost_bytes, target.error = details["lost_bytes"], details["retention_error"]
                    try:
                        payload = details.get("payload")
                        store = archive.artifacts
                        saved = store.records.get(log.artifact_record["record_id"]) if store is not None and hasattr(log, "artifact_record") else None
                        if payload is None and saved is not None and saved.get("format_version") == 2:
                            payload = saved["payloads"]["output" if name == "stdout" else "error"]["sha256"]
                        if payload is not None:
                            if store is None:
                                raise OSError("Shared output store is unavailable")
                            from .payloads import payload_path
                            source = payload_path(store.directory, payload) if payload else None
                            target.__dict__["payload"] = payload
                        if source is not None and source.exists():
                            destination = source if payload is not None else archive.directory / source.name
                            if source != destination:
                                shutil.copyfile(source, destination)
                            destination.chmod(0o600)
                            target.path, target.retained_bytes = destination, destination.stat().st_size
                            archive.used_bytes += target.retained_bytes
                            missing = max(0, details["retained_bytes"] - target.retained_bytes)
                            if missing:
                                target.lost_bytes += missing
                                target.error += f" Saved output is shorter than its journal by {missing} bytes."
                        elif details["retained_bytes"]:
                            raise OSError("retained output file is missing")
                    except OSError as exc:
                        target.error = f"Could not restore output: {exc}"
                        target.lost_bytes += details["retained_bytes"]
                    if not metadata["finished"]:
                        target.error += " Command interrupted; complete output and exit status are unknown."
                    log.streams[name] = target
                archive.logs[log.id] = log
                log.consolidate()
                # A fork's new journal needs every record; a resume appends one
                # only when restoration or consolidation changed the metadata.
                if fork or {**log.metadata(), "command": log.command} != metadata:
                    self.log(log)
                log.remove_legacy_output()
        agent.registry.context_notes["resume"] = (
            load_prompt('session-resume.md', session_id=data['id'], torn_notice=load_prompt('session-torn-entry.md') if data['torn'] else '')
        )
