"""Private append-only session journals; resume forks and never replays tools.

Message deltas and tool-start markers are flushed before execution. A missing
result means uncertainty, not permission to retry. Only a torn final line can
be ignored during recovery; malformed complete records reject the journal.
"""

from __future__ import annotations

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

from .context import ConversationHistory, TurnPost
from .openrouter import OpenRouterError
from .task import TaskMemory, validate_task
from .tools.output import CommandLog, OutputStream
from .types import Message, Usage

if TYPE_CHECKING:
    from .agent import Agent


class SessionError(OpenRouterError):
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
        root = Path(os.path.abspath(os.path.expanduser(str(directory or os.environ.get("SLIPAGENT_STATE_DIR") or Path.home() / ".SlipAgent"))))
        self.directory = root / identity
        self.session_id = ""
        self.path: Path | None = None
        self.cursor = 0
        self.last_message: dict[str, Any] | None = None
        self.system: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None
        self.post_count = 0
        self.failed = False
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
        record = json.dumps({"type": kind, **data}, ensure_ascii=False, allow_nan=False) + "\n"
        try:
            # Each event has one writer. Resumes always create a new journal.
            with self.path.open("a", encoding="utf-8") as target:
                target.write(record)
                target.flush()
                os.fsync(target.fileno())
        except OSError as exc:
            self.failed = True
            raise SessionError(f"Could not save session {self.session_id}: {exc}. Inspect its journal before resuming.") from exc

    @property
    def title(self) -> str:
        # Older live journals predate titles; their project path is still known.
        return getattr(self, "_title", session_title(self.project))

    def rename(self, name: str) -> None:
        """Atomically save mutable title metadata without rewriting the journal."""
        title = session_title(name)
        if self.path is None or self.failed:
            raise SessionError("Session journal is unavailable; the title was not changed.")
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
        self.cursor, self.post_count = 0, 0
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
        diagnostics = agent.registry.services.get("request_diagnostics")
        if diagnostics is not None:
            diagnostics.use_directory(self.directory / (self.session_id + "-requests"))
        archive = agent.registry.services.get("command_archive")
        if archive is not None:
            directory = self.directory / (self.session_id + "-logs")
            _private_directory(directory)
            archive.use_directory(directory, self.log)
        self.record(agent)

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        # The durable record contains reasoning, although working requests and
        # compaction deliberately omit it. Keep its originating model as well.
        return {**message.to_api(), "reasoning": message.reasoning, "reasoning_model": message.reasoning_model}

    def record(self, agent: Agent) -> None:
        if self.cursor > len(agent.messages):
            raise SessionError("Conversation changed without starting a new session journal.")
        if self.cursor and self.last_message != self._message(agent.messages[self.cursor - 1]):
            self._append("replace", index=self.cursor - 1, message=self._message(agent.messages[self.cursor - 1]))
        if agent.messages and agent.messages[0].role == "system":
            system = self._message(agent.messages[0])
            if self.cursor and system != self.system:
                self._append("replace", index=0, message=system)
            self.system = copy.deepcopy(system)
        for message in agent.messages[self.cursor:]:
            self._append("message", message=self._message(message))
            self.cursor += 1
        self.last_message = copy.deepcopy(self._message(agent.messages[-1])) if agent.messages else None
        state = {"task_start": agent.history.task.start_message, "task": agent.history.task.record,
                 "usage": asdict(agent.usage), "pending": list(agent.pending),
                 "model": agent.model, "temperature": agent.temperature}
        if state != self.state:
            self._append("state", **state)
            self.state = copy.deepcopy(state)
        for post in agent.history.posts[self.post_count:]:
            self.post(post)
        self.post_count = len(agent.history.posts)

    def post(self, post: Any) -> None:
        self._append("post", id=post.id, summary=post.summary,
                     task=getattr(post, "task_record", None),
                     compaction_status=getattr(post, "compaction_status", "pending"),
                     compaction_error=getattr(post, "compaction_error", None))

    def tool_started(self, post_id: int, call_id: str) -> None:
        self._append("tool_start", post=post_id, call_id=call_id)

    def log(self, log: CommandLog) -> None:
        self._append("log", metadata={**log.metadata(), "command": log.command})

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

    def load(self, selected: str) -> dict[str, Any]:
        if selected == "latest":
            choices = [item for item in self.listing() if item["id"] != self.session_id]
            if not choices:
                raise SessionError("No earlier sessions are saved for this project.")
            selected = choices[0]["id"]
        if not re.fullmatch(r"[0-9a-f]{32}", selected):
            raise SessionError("Session ID must be a complete ID from /sessions, or latest.")
        path = self.directory / (selected + ".jsonl")
        data: dict[str, Any] = {"messages": [], "state": {}, "posts": {}, "started": set(), "logs": {}, "id": selected, "torn": False}
        try:
            with path.open("rb") as source:
                for index, raw in enumerate(source):
                    if not raw.endswith(b"\n"):
                        data["torn"] = True
                        break
                    event = json.loads(raw)
                    kind = event["type"]
                    if index == 0:
                        if kind != "header" or event.get("version") != 1 or event.get("project") != self.project or event.get("session_id") != selected:
                            raise ValueError("unsupported or mismatched session header")
                        data["title"] = self._saved_title(selected, event)
                    elif kind == "message":
                        data["messages"].append(event["message"])
                    elif kind == "replace":
                        if type(event["index"]) is not int or not 0 <= event["index"] < len(data["messages"]):
                            raise ValueError("invalid message replacement")
                        data["messages"][event["index"]] = event["message"]
                    elif kind == "state":
                        data["state"] = event
                    elif kind == "post":
                        data["posts"][event["id"]] = event
                    elif kind == "tool_start":
                        data["started"].add((event["post"], event["call_id"]))
                    elif kind == "log":
                        data["logs"][event["metadata"]["log_id"]] = event["metadata"]
                    else:
                        raise ValueError(f"unknown event {kind!r}")
            if not data["state"]:
                raise ValueError("no complete saved session state")
            data["messages"] = self._restore_messages(data)
            state = data["state"]
            if type(state.get("task_start")) is not int or not 0 <= state["task_start"] <= len(data["messages"]):
                raise ValueError("invalid task start")
            if state.get("task") is not None and validate_task(state["task"]):
                raise ValueError("invalid task record")
            if not isinstance(state.get("pending"), list) or any(not isinstance(text, str) for text in state["pending"]):
                raise ValueError("invalid queued input")
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
                if metadata["post_id"] is not None and (type(metadata["post_id"]) is not int or metadata["post_id"] < 1):
                    raise ValueError("invalid command post ID")
                if metadata["call_id"] is not None and not isinstance(metadata["call_id"], str):
                    raise ValueError("invalid command call ID")
                for name in ("stdout", "stderr"):
                    stream = metadata["streams"][name]
                    if any(type(stream[field]) is not int or stream[field] < 0 for field in ("retained_bytes", "lost_bytes")) or not isinstance(stream["retention_error"], str):
                        raise ValueError("invalid command stream metadata")
            task = TaskMemory(start_message=data["state"]["task_start"])
            history = ConversationHistory(task=task)
            history.sync(data["messages"])
            for post_id, saved in data["posts"].items():
                if type(post_id) is not int or not 1 <= post_id <= len(history.posts):
                    raise ValueError("invalid saved post ID")
                if saved.get("summary") is not None and (not isinstance(saved["summary"], str) or not saved["summary"].strip()):
                    raise ValueError("invalid saved summary")
                if saved.get("task") is not None and validate_task(saved["task"]):
                    raise ValueError("invalid saved task snapshot")
            task.record = data["state"]["task"]
            for post in history.posts:
                saved = data["posts"].get(post.id, {})
                post.summary = saved.get("summary")
                post.results_summarized = post.summary is not None
                if isinstance(post, TurnPost):
                    post.task_record = saved.get("task")
                    post.compaction_status = saved.get("compaction_status", "interrupted")
                    if post.compaction_status == "pending":
                        post.compaction_status = "interrupted"
                    post.compaction_error = saved.get("compaction_error")
                    post.observed = post.id < len(history.posts) or not post.has_results
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
            if raw.get("content") is not None and not isinstance(raw["content"], str):
                raise ValueError("invalid message content")
            message = Message.from_api(raw)
            message.reasoning_model = raw.get("reasoning_model")
            messages.append(message)
        # Only the final batch may be incomplete. Fill missing observations
        # explicitly so providers get a valid call/result sequence on resume.
        index, post = 0, 0
        while index < len(messages):
            message = messages[index]
            index += 1
            if message.role == "assistant":
                post += 1
                for call in message.tool_calls or []:
                    if index < len(messages):
                        if messages[index].role != "tool" or messages[index].tool_call_id != call.id:
                            raise ValueError("broken tool-call/result sequence")
                    else:
                        detail = "Tool interrupted; outcome unknown and effects may be partial. Inspect before retrying." if (post, call.id) in data["started"] else "Tool was not started before interruption. No automatic replay was performed."
                        messages.append(Message.tool_result(call.id, json.dumps({"tool": call.name, "call_id": call.id, "status": "error", "content": detail})))
                    index += 1
            elif message.role == "tool":
                raise ValueError("orphan tool result")
        return messages

    def restore(self, agent: Agent, data: dict[str, Any]) -> None:
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
        agent.history.posts = loaded.posts
        agent.history.cursor = loaded.cursor
        agent.history._request = loaded._request
        agent.history.task = loaded.task
        agent.pending = list(data["state"]["pending"])
        agent.usage = data["usage"]
        self.begin(agent, parent=data["id"], title=data["title"])
        archive = agent.registry.services.get("command_archive")
        if archive is not None:
            for log_id, metadata in data["logs"].items():
                if not re.fullmatch(r"[0-9a-f]{32}", log_id):
                    raise SessionError("Invalid archived command ID")
                log = CommandLog(archive, metadata["command"], metadata["post_id"], metadata["call_id"])
                log.id = log_id
                log.returncode, log.timed_out = metadata["returncode"], metadata["timed_out"]
                log.finished = True
                for name in ("stdout", "stderr"):
                    target = OutputStream()
                    details = metadata["streams"][name]
                    source = self.directory / (data["id"] + "-logs") / (log_id + "-" + name)
                    target.lost_bytes, target.error = details["lost_bytes"], details["retention_error"]
                    try:
                        if source.exists():
                            destination = archive.directory / source.name
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
                self.log(log)
        agent.registry.context_notes["resume"] = (
            f"Restored session {data['id']}. No tools were replayed. Inspect any interrupted tool's effects before retrying. "
            "The model and request settings are those currently selected; saved usage includes the parent session."
            " Background jobs from the prior process are not reattached or relaunched; their outcomes may be unknown. Inspect retained logs before rerunning commands."
            + (" A torn final journal entry was ignored; its operation may be uncertain." if data["torn"] else "")
        )
