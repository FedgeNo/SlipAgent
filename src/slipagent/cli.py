"""Command-line interface.

Two modes:
  slipagent "fix the failing test"   one-shot; final answer on stdout, progress on stderr
  slipagent                         interactive REPL
"""

from __future__ import annotations

from .commands import BY_NAME, CommandContext, all_help, menu_options, banner_commands
from .prompts import PromptError, load_prompt
from .markdown import THEMES, MarkdownRenderer, literal

import argparse
import asyncio
import re
import getpass
import io
import itertools
import json
import math
import os
import shutil
import sys
import time
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from contextlib import AsyncExitStack
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TextIO, cast

from .agent import Agent, AgentEvent, STEP_LIMIT_NOTICE, STOP_NOTICE, build_system_prompt
from .automation import JsonEventStream
from .config import (Config, ConfigError, configured_provider, context_tokens_preference, dotenv_path, dotenv_values, environment_source,
                     forget_environment, hide_from_commands, remove_dotenv_value, save_dotenv_value, save_model_choice)
from .instructions import load_project_instructions
from .lifecycle import finish_cleanup
from .mcp import MCPManager, MCPError
from .api import (
    DEFAULT_TEMPERATURE,
    APIAuthError,
    APIResponseError,
    APIClient,
    APIConfigError,
    APIError,
    APILimitError,
)
from .providers import PROVIDERS, active_providers, create_client, provider_class, fetch_catalog, key_provider
from .model_catalog import coding_models
from .modelpricing import free_label, model_order, price_label
from .tools.base import ToolRegistry, ToolResult
from .tools.editing import edit_result_line
from .tools import build_default_registry
from .types import KeyInfo, Message, ModelInfo, ToolCall
from .workspace import Workspace, WorkspaceError
from .terminal import TerminalUI, SuggestedPrompt
from .submissions import Submission, cancelled, suggested as submitted_suggestion
from .inputqueue import input_queue
from .palette import ERROR_COLOR, MUTED_COLOR, THOUGHT_COLOR, USER_BACKGROUND_COLOR, USER_COLOR, USER_TEXT_COLOR, TOOL_COLOR, LINK_COLOR, WARNING_COLOR, foreground_code
from .runtime import CORE_MODULES, RuntimeFrame
from . import plugins
from .repomap import RepositoryMap
from .requestfailures import failed_steps, failure_summary
from .tools.blocking import run_blocking
from .sessionstate import Session as Session
from .sessions import SessionJournal, SessionError, session_title
from wcwidth import iter_graphemes, strip_sequences, wcswidth
from tabulate import tabulate

BANNER = """SlipAgent — coding agent with selectable API providers

  Model:     {model}

  Workspace: {workspace}

{mcp}  Commands:  {commands}

Type a task and press Enter. Follow-ups queue while the agent works.
Settings changes also queue until the current response and tool batch finish.
/stop stops after this step; Esc interrupts now. /exit, /quit, or Ctrl-D quits.
Use /help for command details. Ctrl+\\ toggles the context view.

""".replace("{commands}", banner_commands())

QUOTA_REFRESH_INTERVAL = 15 * 60
# Seconds without model output before the status line shows the deadline.
FIRST_OUTPUT_NOTICE_SECONDS = 20
HOLD_NOTICE = "Staying on this model; press Esc again to cancel the turn"
SKIP_NOTICE = "Ctrl+N: next model; Esc: stay on this model"
WAIT_NOTICE = "Esc: keep waiting without a timeout"

HELP = all_help()


def command_menu_options() -> list[tuple[str, str]]:
    return menu_options()


MENU_OPTIONS = command_menu_options()

# Divider between the working directory and the free-call readout.
_READOUT_SEP = "  │  "


# --------------------------------------------------------------------------- #
# Terminal styling
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Style:
    enabled: bool

    def _wrap(self, code: str, text: str) -> str:
        text = literal(text)
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap(foreground_code(MUTED_COLOR), text)

    def thought(self, text: str) -> str:
        return self._wrap(foreground_code(THOUGHT_COLOR), text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def red(self, text: str) -> str:
        return self._wrap(foreground_code(ERROR_COLOR), text)

    def green(self, text: str) -> str:
        return self._wrap(foreground_code(USER_COLOR), text)

    def user(self, text: str) -> str:
        background = foreground_code(USER_BACKGROUND_COLOR).replace("38;2;", "48;2;", 1)
        return self._wrap(foreground_code(USER_TEXT_COLOR) + ";" + background, text)

    def magenta(self, text: str) -> str:
        return self._wrap(foreground_code(TOOL_COLOR), text)

    def cyan(self, text: str) -> str:
        return self._wrap(foreground_code(LINK_COLOR), text)

    def warning(self, text: str) -> str:
        return self._wrap(foreground_code(WARNING_COLOR), text)


KEY_EXPIRY_WARNING_DAYS = 7


def _key_remedy(provider: str) -> str:
    return f"Replace it with /key <key> or remove it with /key delete {provider}."


def _failure_text(session: Session, exc: BaseException) -> str:
    """A run failure as shown to the user; a rejected key names its remedy."""
    text = str(exc)
    if isinstance(exc, APIAuthError):
        text += " " + _key_remedy(getattr(session.client, "provider", "<provider>"))
    return text


def _key_expiry(value: str | None) -> datetime | None:
    """Parse a provider's key expiry; None when absent or unreadable."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _use_color(stream: TextIO, force_off: bool) -> bool:
    if force_off or os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and bool(stream.isatty())


def _literal_tool_output(text: str) -> str:
    """Keep captured terminal controls inert in both live and restored output."""
    return literal(text)


COMMAND_OUTPUT_TOOLS = frozenset({
    "run_command", "read_command_output", "command_jobs", "monitor_command",
    "git_status", "git_diff", "git_log", "git_add", "git_commit",
})


# --------------------------------------------------------------------------- #
# Event rendering
# --------------------------------------------------------------------------- #


class Renderer:
    """Render agent events as terminal output."""

    def __init__(self, style: Style, stream: TextIO, verbose: bool) -> None:
        self.style = style
        self.stream = stream
        self.verbose = verbose
        # One-shot mode writes the final answer to stdout and streams previews
        # here on stderr. Buffered replies need no second rendering on stderr.
        self.show_assistant_text = True
        self._queued_settings_pause = False
        self._active_block: object | None = None
        self._model_block = object()
        self._stream_kind: str | None = None
        self._last_output_blank = False
        self._tool_group = False
        # The live prompt line: the text drawn on it, so output arriving mid-step
        # can be written around it and the prompt put back. Holding the whole
        # line is what stops a redraw from erasing a line the agent has just
        # written.
        self._prompt: tuple[str, str] | None = None
        self.terminal: TerminalUI | None = None
        self.markdown_theme = "dark"
        self.output_markdown = False
        self.stdout_color = _use_color(sys.stdout, not style.enabled)

    def display_result(self, result: ToolResult, call: ToolCall | None) -> ToolResult:
        """Keep edit recovery source in model history, outside user-facing output."""
        if result.is_error and call is not None and call.name == "edit_file":
            text = re.sub(
                r"(?m)^Nearby actual source \(line numbers are not part of the file\):\n"
                r"(?:[ ]*\d+\t[^\n]*\n)*",
                "", result.text(),
            )
            text = text.replace("Read the file and copy the exact target text including whitespace.\n", "")
            return ToolResult.error(text)
        return result

    async def ask_input(self, prompt: str, *, password: bool = False) -> str:
        if self.terminal is not None:
            return await self.terminal.ask(prompt, password=password)
        reader = getpass.getpass if password else input
        return await asyncio.to_thread(reader, prompt)

    def _write(
        self, text: str, *, continuation_indents: dict[int, int] | None = None,
        user_prompt: bool = False,
    ) -> None:
        plain = strip_sequences(text)
        self._last_output_blank = not plain.rsplit("\n", 1)[-1].strip()
        if self.terminal is not None:
            self.terminal.write(text, continuation_indents=continuation_indents, user_prompt=user_prompt)
            return
        prompt = self._prompt
        if prompt is not None:
            self._erase()
        print(text, file=self.stream)
        if prompt is not None:
            # Output landed above a live prompt, so the prompt goes back under
            # it: the user keeps a line to type into rather than a lost one.
            self._draw(*prompt)

    def _begin_block(self, block: object | None = None, *, text: str = "", tool: bool = False) -> None:
        if getattr(self, "_tool_group", False) and not tool:
            self._active_block = None
        self._tool_group = tool
        if block is None or block != self._active_block:
            first, newline, _ = strip_sequences(text).partition("\n")
            if not self._last_output_blank and not (newline and not first.strip()):
                self._write("")
            self._active_block = block

    def emit(
        self, text: str = "", *, block: object | None = None, separate: bool = True,
        continuation_indents: dict[int, int] | None = None,
        tool: bool = False,
    ) -> None:
        """Separate output units while keeping their lines together."""
        self._finish_stream()
        if (separate or (getattr(self, "_tool_group", False) and not tool)) and strip_sequences(text).strip():
            self._begin_block(block, text=text, tool=tool)
        self._write(text, continuation_indents=continuation_indents)

    def show_banner(self, *, model: str, workspace: Path, mcp: str = "") -> None:
        banner = BANNER.format(model=literal(model), workspace=literal(str(workspace)), mcp=literal(mcp))
        self.emit(
            banner,
            continuation_indents={index: prefix.end() for index, line in enumerate(banner.splitlines())
                                  if (prefix := re.match(r"  Commands: +", line)) is not None},
        )

    async def show_welcome(self, *, model: str, workspace: Path, mcp: str = "") -> None:
        """Reveal the welcome a line at a time while the terminal stays responsive."""
        banner = BANNER.format(model=literal(model), workspace=literal(str(workspace)), mcp=literal(mcp))
        lines = banner.splitlines()
        for index, line in enumerate(lines):
            prefix = re.match(r"  Commands: +", line)
            self.emit(line, separate=False,
                      continuation_indents={0: prefix.end()} if prefix else None)
            if self.terminal is not None and index < len(lines) - 1:
                await asyncio.sleep(.06)

    def _finish_stream(self) -> None:
        if getattr(self, "_stream_kind", None) is None:
            return
        if self.terminal is not None and self._stream_kind in {"assistant_delta", "reasoning_delta"}:
            self.terminal.finish_markdown()
        if self.terminal is None:
            self.stream.write("\n")
            self.stream.flush()
        self._stream_kind = None
        self._last_output_blank = False

    def _stream_text(self, kind: str, text: str) -> None:
        first = getattr(self, "_stream_kind", None) != kind
        if first:
            self._finish_stream()
            if kind != "tool_output":
                self._model_block = object()
            if kind == "reasoning_delta":
                text = ("Thinking:\n\n" if self.terminal is not None else "Thinking: ") + text
            leading = text
            if kind == "assistant_delta" and self.terminal is not None and not getattr(self.terminal, "_markdown_source", False):
                leading = ""  # Rendered Markdown discards leading blank lines.
            self._begin_block(self._model_block, text=leading, tool=kind == "tool_output")
        styled = literal(text)
        if kind == "reasoning_delta":
            styled = self.style.thought(text)
        elif kind == "tool_output":
            styled = self.style.dim(text)
        if self.terminal is not None:
            if kind in {"assistant_delta", "reasoning_delta"}:
                self.terminal.write_markdown(text, first=first, final=False, reasoning=kind == "reasoning_delta")
            else:
                self.terminal.write_chunk(styled, first=first)
        else:
            self.stream.write(styled)
            self.stream.flush()
        self._stream_kind = kind
        self._last_output_blank = False

    def user_prompt(self, text: str, *, queued: bool = False) -> None:
        """Separate each submitted prompt from the surrounding transcript."""
        if queued and self.terminal is not None:
            self.terminal.queue_prompt(text)
            return
        self._finish_stream()
        self.__dict__.pop("_suggestion_step", None)
        self._begin_block()
        block = self.style.user(f"> {text}")
        self._write(f"{block}\n", user_prompt=True)

    def clear_prompt(self) -> None:
        """Erase the live prompt row, leaving everything above it alone."""
        if self._prompt is None:
            return
        self._erase()
        self._prompt = None

    async def restore_transcript(self, messages: list[Message], pending: list[str], *,
                                 queued_messages: dict[int, str] | None = None,
                                 failures: dict[int, dict[str, Any]] | None = None) -> None:
        """Display archived originals only; never dispatch their tools or requests."""
        self._finish_stream()
        if self.terminal is not None:
            self.terminal.clear_transcript()
            self.terminal.hold_layout("Loading conversation, this can take some time…")
        self._active_block = None
        self._model_block = object()
        self._last_output_blank = True
        self._tool_group = False
        try:
            await self._restore_records(messages, queued_messages or {}, failures or {})
            for text in pending:
                self.user_prompt(text, queued=True)
        finally:
            if self.terminal is not None:
                # Show the newest output now; earlier blocks fill in above it, newest first.
                self.terminal.show_tail_first()
                self.terminal.scroll_to_end()

    async def _restore_records(self, messages: list[Message], queued_messages: dict[int, str],
                               failures: dict[int, dict[str, Any]]) -> None:
        restored_step = 0
        restored_calls: dict[str, ToolCall] = {}
        yielded_at = time.monotonic()
        for index, message in enumerate(messages):
            if message.role == "user":
                self.user_prompt(message.content or "", queued=index in queued_messages)
            elif message.role == "assistant":
                restored_step += 1
                if message.reasoning:
                    self._stream_text("reasoning_delta", message.reasoning)
                    self._finish_stream()
                if message.content:
                    self.handle(AgentEvent(kind="assistant_text", text=message.content))
                if restored_step in failures:
                    # Shown after any partial reply, so it never reads as an accepted answer.
                    self.handle(AgentEvent(kind="warning", text=failure_summary(failures[restored_step])))
                for call in message.tool_calls or []:
                    restored_calls[call.id] = call
                    self.handle(AgentEvent(kind="tool_start", tool_call=call, step=restored_step))
            elif message.role == "tool":
                result = ToolResult.ok(message.content or "")
                saved = result.content
                if (isinstance(saved, dict) and saved.get("status") in ("success", "error")
                        and "content" in saved):
                    result = ToolResult(saved["content"], saved["status"] == "error")
                restored_call = restored_calls.get(message.tool_call_id or "")
                result = self.display_result(result, restored_call)
                saved_diff = (self.terminal is not None and restored_call is not None
                              and restored_call.name in {"write_file", "edit_file", "edit_hashlines"} and not result.is_error
                              and isinstance(saved, dict)
                              and any(reference.get("kind") == "diff" for reference in saved.get("artifacts", [])))
                if self.terminal is not None and restored_call is not None:
                    self.terminal.finish_call(restored_call, restored_step, result.content, result.is_error)
                if isinstance(saved, dict) and saved.get("tool") == "answer" and not result.is_error:
                    self.handle(AgentEvent(kind="assistant_text", text=result.content["text"]))
                    continue
                if result.content:
                    body = result.text()
                    if saved_diff:
                        body = body.splitlines()[0]
                    if self.terminal is not None and restored_call is not None and restored_call.name in COMMAND_OUTPUT_TOOLS:
                        body = "Command failed; see Calls." if result.is_error else "Command completed."
                    body = _literal_tool_output(body)
                    if result.is_error:
                        self.emit(self.style.red(f"  ✗ {body}"), block=self._model_block, tool=True)
                    else:
                        self.emit(self.style.dim(body), block=self._model_block, tool=True)
                if restored_call is not None:
                    self._result_references(restored_call, result)
                if isinstance(saved, dict):
                    for reference in saved.get("artifacts", []):
                        self.handle(AgentEvent(kind="artifact", result=ToolResult.ok(reference)))
            if time.monotonic() - yielded_at >= 0.02:
                # Keep keys and redraws responsive while a long history is written.
                await asyncio.sleep(0)
                yielded_at = time.monotonic()

    def _erase(self) -> None:
        if self.terminal is not None:
            return
        if not self.stream.isatty():
            # Nothing to erase without a cursor to move; the row is simply left
            # in the captured output.
            return
        print("\r\033[2K", end="", file=self.stream, flush=True)

    def _draw(self, status: str, marker: str) -> None:
        if self.terminal is not None:
            self.terminal.app.invalidate()
            return
        print(f"{marker}\n\n{status}", file=self.stream, flush=True)

    def draw_prompt(self, status: str, marker: str) -> None:
        """Refresh the input area and its separate bottom readout."""
        self.clear_prompt()
        self._prompt = (status, marker)
        self._draw(status, marker)

    def end_prompt(self) -> None:
        """Hand the drawn prompt over to history.

        Called once the line has been read: the terminal has already moved past
        it, so output arriving later must not reach back and erase it.
        """
        self._prompt = None

    @property
    def prompt_is_live(self) -> bool:
        """True while a prompt drawn by this renderer is still the last thing on
        screen, meaning output must be written around it rather than over it."""
        return self._prompt is not None

    def request_phase(self, text: str) -> None:
        """Show what the running request is doing in the status line; empty text clears it."""
        if self.terminal is not None:
            self.terminal.set_phase(text)

    def first_output_wait(self, deadline: Any, *, skippable: bool) -> None:
        """Count down a silent request's deadline; Ctrl+N skips to the next model and Esc stays."""
        ticker = self.__dict__.pop("_first_output_ticker", None)
        if ticker is not None:
            ticker.cancel()
        terminal = self.terminal
        if terminal is None:
            return
        terminal.set_skip_handler(deadline.skip if deadline is not None and skippable else None)
        terminal.set_hold_handler(None)
        if deadline is None:
            if ticker is not None:
                terminal.set_retry_status("")
            if terminal.status_line.notification in (SKIP_NOTICE, WAIT_NOTICE):
                terminal.notify("")
            return

        def hold() -> None:
            deadline.hold()
            if deadline.held:
                terminal.set_retry_status("")
                terminal.notify(HOLD_NOTICE)

        async def tick() -> None:
            started = time.monotonic()
            while True:
                await asyncio.sleep(1)
                remaining = deadline.remaining()
                # Ordinary responses start within seconds; only a long silence is shown.
                if remaining is not None and time.monotonic() - started >= FIRST_OUTPUT_NOTICE_SECONDS:
                    action = "next model" if skippable else "timeout"
                    terminal.set_retry_status(f"No output yet; {math.ceil(remaining)}s until {action}",
                                              spinning=True)
                    # Esc keeps this model only while the countdown offers it.
                    if terminal.__dict__.get("_hold_callback") is None and not getattr(deadline, "held", False):
                        terminal.set_hold_handler(hold)
                        terminal.notify(SKIP_NOTICE if skippable else WAIT_NOTICE)
        self.__dict__["_first_output_ticker"] = asyncio.ensure_future(tick())

    def retry_countdown(self, error: str, seconds: int) -> None:
        """Keep countdown ticks out of scrollback while retaining the error."""
        previous = self.__dict__.get("_retry_error", "")
        self.__dict__["_retry_error"] = error
        if error and error != previous:
            self.emit(self.style.red(f"  ✗ {error}"))
        text = f"Retrying in {seconds}s (Esc to interrupt)" if error else ""
        if self.terminal is not None:
            self.terminal.set_retry_status(text)
        elif error and (not previous or seconds % 10 == 0):
            self.emit(self.style.dim(f"  Retrying in {seconds}s."), separate=False)

    def _tool_heading(self, call: ToolCall) -> None:
        from .calllabels import call_label, raw_call
        label = call_label(call)
        if label:
            # The label is for everyone; the exact call stays visible beneath it.
            self.emit(self.style.magenta(f"  ⚙ {label}") + "\n" + self.style.thought(f"    {raw_call(call)}"),
                      block=self._model_block, tool=True)
        else:
            self.emit(self.style.magenta(f"  ⚙ {raw_call(call)}"), block=self._model_block, tool=True)

    def _result_references(self, call: ToolCall, result: ToolResult) -> None:
        """Link the paths a successful call touched; refused paths never become links."""
        if self.terminal is None or result.is_error:
            return
        if call.name == "edit_file":
            self._edit_file_reference(call, result)
            return
        path = call.arguments.get("path")
        if path is None and call.name == "list_dir":
            path = "."
        if isinstance(path, str):
            self.terminal.write_directory_reference(path)
            offset = call.arguments.get("offset", 1)
            self.terminal.write_file_reference(path, line=offset if isinstance(offset, int) and offset > 0 else 1)
        if call.name in {"copy_path", "move_path"}:
            destination = call.arguments.get("destination")
            if isinstance(destination, str):
                self.terminal.write_file_reference(destination)

    def _edit_file_reference(self, call: ToolCall, result: ToolResult) -> None:
        if self.terminal is None or result.is_error:
            return
        path = call.arguments.get("path")
        if not isinstance(path, str):
            return
        text = result.text()
        line = edit_result_line(text)
        occurrences = re.match(r"Replaced (\d+) occurrence\(s\) in ", text)
        edits = call.arguments.get("edits")
        multiple = (occurrences is not None and int(occurrences[1]) > 1
                    or isinstance(edits, list) and len(edits) > 1)
        self.terminal.write_file_reference(path, line=line or 1, show_line=line is not None and not multiple)

    def handle(self, event: AgentEvent) -> None:
        if event.kind == "context":
            self.__dict__["_reply_preview"] = []
            if self.terminal is not None:
                self.terminal.set_context(event.request_body if event.request_body is not None else event.text)

        elif event.kind == "step_start":
            self._finish_stream()
            self.__dict__["_reply_preview"] = []
            if event.step == 1:
                self._model_block = object()

        elif event.kind in {"assistant_delta", "reasoning_delta"}:
            if event.kind == "assistant_delta":
                self.__dict__.setdefault("_reply_preview", []).append(event.text)
            self._stream_text(event.kind, event.text)

        elif event.kind == "stream_end":
            self._finish_stream()

        elif event.kind == "retry":
            self.__dict__["_reply_preview"] = []
            self.emit(self.style.warning(f"  ✗ {event.text}"))

        elif event.kind == "retry_wait":
            self.__dict__["_reply_preview"] = []
            self.retry_countdown(event.text, event.step)

        elif event.kind == "assistant_text":
            preview = "".join(self.__dict__.pop("_reply_preview", []))
            if self.show_assistant_text and preview and preview != event.text and self.terminal is not None:
                # The preview streamed text the validated reply omits, such as a call tag.
                self._finish_stream()
                if self.terminal.replace_markdown(event.text):
                    self._last_output_blank = False
                    return
            if self.show_assistant_text and preview != event.text:
                self._finish_stream()
                self._model_block = object()
                if self.terminal is not None:
                    self._begin_block(self._model_block, text=event.text if getattr(self.terminal, "_markdown_source", False) else "")
                    self.terminal.write_markdown(event.text)
                    self._last_output_blank = False
                else:
                    self.__dict__.setdefault("output_markdown", False)
                    text = literal(event.text) if self.output_markdown else MarkdownRenderer(color=False).plain(event.text)
                    self.emit(text, block=self._model_block)

        elif event.kind == "suggestion":
            self._finish_stream()
            if self.terminal is not None:
                if self.__dict__.get("_suggestion_step") != event.step:
                    self.emit("Suggested Next Steps", block=object())
                    self.__dict__["_suggestion_step"] = event.step
                self.terminal.write_suggestion(event.text)

        elif event.kind == "user_message":
            # Waiting input stays outside the feed until its request is sent.
            self.user_prompt(event.text, queued=True)

        elif event.kind == "user_message_sent":
            if self.terminal is not None and self.terminal.queued_prompt_sent(event.text):
                self.user_prompt(event.text)

        elif event.kind == "user_queue_reset":
            if self.terminal is not None:
                self.terminal.clear_queued_prompts()

        elif event.kind == "user_queue_changed":
            if self.terminal is not None:
                self.terminal.set_queued_prompts(json.loads(event.text))

        elif event.kind == "tool_start" and event.tool_call is not None:
            self._tool_streamed = False
            if self.terminal is not None:
                self.terminal.track_call(event.tool_call, event.step)
            if event.tool_call.name == "answer":
                return
            # Calls tracks running searches immediately. Their transcript heading
            # waits for the result so concurrent calls cannot split the file list.
            if self.terminal is None or event.tool_call.name != "grep":
                self._tool_heading(event.tool_call)

        elif event.kind == "tool_output":
            if self.terminal is not None:
                return
            self._tool_streamed = True
            # Commands may emit terminal controls, including split escape
            # sequences. Display them literally instead of executing them.
            text = _literal_tool_output(event.text)
            self._stream_text("tool_output", text)

        elif event.kind == "tool_end":
            self._finish_stream()
            result = event.result
            if result is None:
                return
            result = self.display_result(result, event.tool_call)
            if self.terminal is not None and event.tool_call is not None and event.tool_call.name == "grep":
                self._tool_heading(event.tool_call)
            if self.terminal is not None and event.tool_call is not None:
                self.terminal.finish_call(event.tool_call, event.step, result.content, result.is_error)
            if self.terminal is not None and event.tool_call is not None and event.tool_call.name in COMMAND_OUTPUT_TOOLS:
                body = "Command failed; see Calls." if result.is_error else "Command completed."
                color = self.style.red if result.is_error else self.style.green
                self.emit(color(f"  {'✗' if result.is_error else '✓'} {body}"), block=self._model_block, tool=True)
            elif result.is_error:
                # Errors are the most useful signal; always surface them.
                self.emit(self.style.red(f"  ✗ {result.text()}"), block=self._model_block, tool=True)
            elif event.tool_call is not None and event.tool_call.name == "answer":
                self.handle(AgentEvent(kind="assistant_text", step=event.step, text=result.content["text"]))
            elif self.terminal is not None and event.tool_call is not None and event.tool_call.name in {"write_file", "edit_file"} and self.terminal.has_diff(event.tool_call.id, event.step):
                self.emit(self.style.dim("    " + result.text().splitlines()[0]), block=self._model_block, tool=True)
            elif getattr(self, "_tool_streamed", False):
                self.emit(self.style.dim("  ✓ Command completed."), block=self._model_block, tool=True)
            elif self.verbose and result.text().strip():
                body = result.text().strip()
                for line in body.splitlines():
                    self.emit(self.style.dim(f"    {line}"), block=self._model_block, tool=True)
            if self.terminal is not None and event.tool_call is not None and not result.is_error:
                self._result_references(event.tool_call, result)
                if event.tool_call.name == "grep":
                    for match in re.finditer(r"^(.+?):(\d+):", result.text(), re.MULTILINE):
                        self.terminal.write_file_reference(match[1], line=int(match[2]))

        elif event.kind == "artifact" and event.result is not None:
            reference = event.result.content
            if self.terminal is not None:
                self._finish_stream()
                self._begin_block(self._model_block, tool=True)
                self.terminal.write_artifact_reference(reference)
                self._last_output_blank = False
            else:
                self.emit(self.style.dim(f"  Saved {reference['kind']} record {reference['record_id']}."), block=self._model_block, tool=True)

        elif event.kind == "warning":
            if event.text == STOP_NOTICE and getattr(self, '_queued_settings_pause', False):
                self.emit(self.style.dim("  Current step finished; applying queued commands."))
            else:
                self.emit(self.style.warning(f"  Warning: {event.text}"), block=self._model_block)
        elif event.kind == "notice":
            self.emit(self.style.dim(f"  {event.text}"))

        elif event.kind == "step_end":
            if self.verbose and event.usage is not None:
                self.emit(self.style.dim(f"    ({event.usage.summary()})"), block=self._model_block)


# --------------------------------------------------------------------------- #
# Session wiring
# --------------------------------------------------------------------------- #


class Launcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    async def build_session(self, *, defer_startup: bool = False) -> Session:
        args = self.args
        if getattr(args, "no_session", False) and getattr(args, "resume", None):
            raise ConfigError("--resume cannot be combined with --no-session.")
        # Overrides apply before any session object exists, so constructors use them too.
        plugins.disabled = getattr(args, "no_plugins", False)
        plugins.all_plugins = getattr(args, "all_plugins", False)
        if not plugins.__dict__.get("applied_to_live"):
            plugins.applied_to_live = True
            await run_blocking(plugins.apply_to_live, CORE_MODULES)
        config = Config.from_env(
            api_key=args.api_key,
            provider=getattr(args, "provider", None),
            model=args.model,
            base_url=args.base_url,
            workspace=args.workspace,
            max_steps=args.max_steps,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        try:
            workspace = Workspace(config.workspace, danger=getattr(args, "danger", False))
            load_project_instructions(workspace)
        except WorkspaceError as exc:
            raise ConfigError(str(exc)) from exc

        style = Style(_use_color(sys.stderr, args.no_color))
        renderer = Renderer(style, sys.stderr, args.verbose)
        renderer.markdown_theme = getattr(args, "theme", "dark")
        renderer.output_markdown = getattr(args, "markdown", False)
        renderer.stdout_color = _use_color(sys.stdout, args.no_color)

        client = create_client(
            config.provider, api_key=config.api_key,
            base_url=config.base_url,
            http_referer=config.http_referer,
            app_title=config.app_title,
            on_retry=renderer.retry_countdown,
        )
        async with AsyncExitStack() as resources:
            resources.push_async_callback(client.aclose)
            registry = build_default_registry(workspace)
            if not getattr(args, "hashline", True):
                from .tools.hashline import set_hashline
                set_hashline(registry, False)
            registry.services["editable_prompts"] = True
            resources.push_async_callback(registry.aclose)
            registry.services["project_environment"].python_override = args.python

            agent = Agent(
                client=client,
                registry=registry,
                model=config.model,
                max_steps=config.max_steps,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                context_steps=config.context_steps,
                overthinking=config.overthinking,
                planning=getattr(args, "planning", True),
                context_tokens=getattr(args, "context_tokens", None) or context_tokens_preference(),
                reasoning_history_steps=getattr(args, "reasoning_history_steps", 25),
                filtered_thoughts=not getattr(args, "raw_reasoning_history", False),
                exposed_tools=tuple(args.tools.split(',')) if getattr(args, "tools", None) else None,
                system_prompt=build_system_prompt(str(workspace.root)),
                on_event=lambda event: renderer.handle(event),
            )

            # Skip automatic MCP startup for one-shot runs unless --mcp is set.
            mcp: MCPManager | None = None
            if not args.no_mcp and (args.mcp or not args.prompt_flag and not args.prompt and not getattr(args, "prompt_file", None)):
                manager = MCPManager(workspace.root, registry=registry)
                mcp = manager
                resources.push_async_callback(manager.aclose)

            session = Session(
                agent=agent,
                registry=registry,
                client=client,
                renderer=renderer,
                workspace=workspace,
                api_key=config.api_key,
                base_url=config.base_url,
                http_referer=config.http_referer,
                app_title=config.app_title,
                mcp=mcp,
            )
            async def close_selected_client() -> None:
                if session.client is not client:
                    await session.client.aclose()

            resources.push_async_callback(close_selected_client)
            registry.services["first_output_wait"] = lambda deadline: renderer.first_output_wait(
                deadline, skippable=session.model_routing().enabled)
            registry.services["request_phase"] = renderer.request_phase
            session.configure_compaction()
            if defer_startup:
                session.extensions["startup"] = lambda: self.initialize(session, config, announce=True)
            else:
                await self.initialize(session, config)
            resources.pop_all()
            return session

    async def initialize(self, session: Session, config: Config, *, announce: bool = False) -> None:
        """Overlap independent startup work; build context only after its services settle."""
        args = self.args
        renderer, registry, agent = session.renderer, session.registry, session.agent
        model_error: APIError | None = None
        routing = session.model_routing()
        routing.enabled = not (args.model or getattr(args, "provider", None))
        # Interactive startup selects from saved metadata and refreshes it afterwards.
        cached_start = False

        def notice(text: str) -> None:
            if announce:
                renderer.emit(renderer.style.dim(f"  {text}"), separate=False)

        async def phase(label: str, operation: Callable[[], Awaitable[Any]]) -> None:
            notice(f"{label}…")
            action = label[0].lower() + label[1:]
            try:
                outcome = await operation()
            except BaseException:
                notice(f"Interrupted or failed while {action}.")
                raise
            notice(outcome if isinstance(outcome, str) else f"Finished {action}.")

        async def metadata() -> str | None:
            nonlocal model_error, cached_start
            try:
                if routing.enabled:
                    if announce and await session.use_cached_catalog():
                        cached_start = True
                        return "Model catalog: using the saved catalog; refreshing it in the background."
                    await routing.candidates("working")
                else:
                    if announce:
                        from . import startupcache
                        profile = await run_blocking(startupcache.cached_profile, ModelInfo(config.model, provider=config.provider),
                                                     session.provider_identity(config.provider))
                        if profile is not None:
                            session.client.cache_capabilities(config.model, profile)
                            cached_start = True
                            return "Model capabilities: using the saved profile; refreshing it in the background."
                    if await session.client.model_capabilities(config.model, refresh=True) is None:
                        raise APIConfigError("The API did not supply capabilities to verify the configured model's compatibility.")
            except APIError as exc:
                model_error = exc
                renderer.emit(literal(f"  model check failed: {exc} Select another model with /models or /model <slug>."))
                return "Model capability check: failed."
            return None

        async def mcp() -> str:
            if session.mcp is None:
                return "MCP: disabled."
            def report(state: Any) -> None:
                if state.status == "error":
                    renderer.emit(renderer.style.warning(f"  Warning: mcp: {state.line()}"), separate=False)
                else:
                    notice(f"MCP: {state.line()}")
            try:
                states = await session.mcp.connect_all(on_state=report)
                if not states:
                    return "MCP: no servers configured."
                connected = sum(state.status == "connected" for state in states)
                failed = sum(state.status == "error" for state in states)
                other = len(states) - connected - failed
                return f"MCP: {connected} connected; {failed} failed." + (f" {other} disconnected." if other else "")
            except MCPError as exc:
                renderer.emit(renderer.style.warning(f"  Warning: mcp: {exc}"), separate=False)
                return "MCP: connection failed."

        async def repository() -> None:
            environment = registry.resources.environment
            if environment is not None:
                if registry.resources.repository_map is None:
                    registry.resources.repository_map = RepositoryMap(environment.workspace)
                # Let an in-flight map finish before closing its workspace or reloading.
                worker = asyncio.create_task(registry.resources.repository_map.snapshot(""))
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    await finish_cleanup(worker)
                    raise

        async def environment() -> None:
            if registry.resources.environment is not None:
                await registry.resources.environment.snapshot()

        async def quota() -> str:
            await Quota(session).refresh()
            return "Quota: unavailable." if session.free_calls is None else f"Quota: {session.free_calls} free calls remaining."

        async def resume() -> None:
            if not getattr(args, "no_session", False):
                journal = await run_blocking(SessionJournal, str(config.workspace))
                registry.services["session_journal"] = journal
                if getattr(args, "resume", None):
                    data = await run_blocking(journal.load, args.resume)
                    journal.restore(agent, data)
                    session.extensions["restore_transcript"] = True
                    renderer.emit(f"  Resumed {journal.session_id}; no tools were replayed.")

        operations = [
            ("Fetching the model catalog and checking model capabilities", metadata),
            ("Connecting to MCP servers and loading their tools", mcp),
            ("Scanning repository files and building the repository map", repository),
            ("Finding and validating the project Python environment", environment),
            ("Preparing session storage and restoring any requested session", resume),
        ]
        if announce:
            # The quota is informational; the prompt need not wait for it.
            Repl(session).start_background_task(Quota(session).refresh())
        else:
            operations.insert(4, ("Retrieving the remaining API quota", quota))
        tasks = [asyncio.create_task(phase(label, operation)) for label, operation in operations]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await finish_cleanup(asyncio.create_task(_gather_quietly(*tasks)))
        # Optimal's selection and skip notices print after the startup sequence
        # rather than inside it; the selection itself still happens here, since
        # it preflights the model context.
        routing.defer_notices()
        try:
            if model_error is None:
                notice("Preparing model context…")
                try:
                    # Selection preflights context after MCP, environment, and restored history settle.
                    preview = asyncio.ensure_future(routing.enable(cached=cached_start) if routing.enabled else
                                                    agent._context_view(registry.specs(), 0, preview=True))
                    try:
                        await asyncio.shield(preview)
                    except asyncio.CancelledError:
                        await finish_cleanup(preview)
                        raise
                except APIError as exc:
                    model_error = exc
                    renderer.emit(literal(f"  model check failed: {exc} Select another model with /models or /model <slug>."))
            if cached_start:
                Repl(session).start_background_task(Repl(session).refresh_model_metadata())
            if not getattr(args, "no_reload", False):
                frame = await run_blocking(RuntimeFrame, session, sys.modules[__name__])
                session.reloader = frame
                edits = registry.services.get("user_edits")
                agent.on_boundary = edits.checkpoint if edits is not None else lambda: frame.checkpoint(boundary=True)
                frame.report_plugins(plugins.reports)
            else:
                for report in plugins.reports:
                    for text in report.notices():
                        renderer.emit(literal(f"  {text}"))
            if args.model and model_error is None and not getattr(args, "json_events", False):
                try:
                    await run_blocking(save_model_choice, config.model, config.provider)
                except (OSError, ConfigError) as exc:
                    renderer.emit(literal(f"  could not remember model choice: {exc}"))
            notice("Startup complete." if model_error is None else "Startup finished; model capability check failed.")
        finally:
            routing.flush_notices()

    async def run_one_shot(self, session: Session, prompt: str, events: JsonEventStream | None = None) -> int:
        """Run a single task. Progress goes to stderr so stdout stays pipeable."""
        session.renderer.show_assistant_text = False
        failure: Exception | None = None
        answer = ""
        try:
            answer = await session.agent.run(prompt)
            await session.agent.wait_for_compaction()
        except (APIError, ConfigError, WorkspaceError) as exc:
            print(literal(f"slipagent: {_failure_text(session, exc)}"), file=sys.stderr)
            failure = exc
        finally:
            await Repl(session).shutdown()

        if events is not None:
            status = "completed"
            if failure is not None:
                status = "error"
            elif answer == STEP_LIMIT_NOTICE:
                status = "step_limit"
            elif session.agent.stopped:
                status = "stopped"
            exit_code = 0 if status == "completed" else 1
            events.write("run_end", status=status, exit_code=exit_code, answer=answer,
                         error={"type": type(failure).__name__, "message": str(failure)} if failure else None,
                         session_id=session.agent.session_id, requests=session.agent._requests,
                         usage=asdict(session.agent.usage))
            return exit_code
        if failure is not None:
            return 1

        # Rendered to stdout so `slipagent -p "..." | pbcopy` behaves.
        if session.renderer.output_markdown:
            print(literal(answer))
        elif getattr(session.renderer, "stdout_color", _use_color(sys.stdout, not session.renderer.style.enabled)):
            from prompt_toolkit.formatted_text import FormattedText
            from prompt_toolkit.shortcuts import print_formatted_text
            rows = MarkdownRenderer(session.renderer.markdown_theme).render(answer, shutil.get_terminal_size().columns)
            fragments = [(style, text) for row in rows for style, text, *_ in [*row.fragments, ("", "\n")]]
            print_formatted_text(FormattedText(fragments), file=sys.stdout, end="")
        else:
            print(MarkdownRenderer(color=False).plain(answer))
        return 1 if answer == STEP_LIMIT_NOTICE or session.agent.stopped else 0

    async def list_models(self) -> int:
        args = self.args
        from .config import load_dotenv
        load_dotenv()
        selected = configured_provider(args.provider)
        provider_class(selected)
        names = active_providers(keys={selected: args.api_key} if args.api_key else None)
        if not names:
            keys = ", ".join(provider_class(name).key_env for name in PROVIDERS)
            raise ConfigError(f"No active API providers. Set one of: {keys}.")
        renderer = Renderer(Style(_use_color(sys.stderr, args.no_color)), sys.stderr, args.verbose)
        results = await asyncio.gather(*(fetch_catalog(name, base_url=(args.base_url if name == selected else None)
                                        or os.environ.get(provider_class(name).base_url_env),
                                        api_key=args.api_key if name == selected else None,
                                        on_retry=renderer.retry_countdown) for name in names),
                                       return_exceptions=True)
        models: list[ModelInfo] = []
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                if not isinstance(result, (APIError, ConfigError)):
                    raise result
                print(literal(f"slipagent: {name}: {result}"), file=sys.stderr)
            else:
                models.extend(coding_models(result))
        if not models:
            return 1

        style = Style(_use_color(sys.stdout, args.no_color))
        for model in sorted(models, key=model_order):
            context = f"  {model.context_length:,} ctx" if model.context_length else ""
            print(f"{literal(model.selector)}{style.dim(context)}")
        print(style.dim(f"\n{len(models)} models"), file=sys.stderr)
        return 0

    def notice_ignored_project_settings(self) -> None:
        args = self.args
        from .config import _dotenv_disabled, dotenv_path, ignored_project_settings

        if _dotenv_disabled(os.environ):
            return
        start = Path(getattr(args, "workspace", None) or os.environ.get("SLIPAGENT_WORKSPACE") or ".").expanduser()
        try:
            ignored = ignored_project_settings(start)
        except (OSError, ConfigError, APIConfigError):
            return
        if ignored is not None:
            path, names = ignored
            print(literal(f"slipagent: ignoring {', '.join(names)} in {path}; project .env files are not read. "
                          f"Put SlipAgent settings in {dotenv_path()} or export them."), file=sys.stderr)

    def notice_project_language_servers(self) -> None:
        args = self.args
        from .environment import PROJECT_SETTINGS_PATH
        from .lsp import language_servers_path

        start = Path(getattr(args, "workspace", None) or os.environ.get("SLIPAGENT_WORKSPACE") or ".").expanduser()
        try:
            data = json.loads((start / PROJECT_SETTINGS_PATH).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and data.get("language_servers"):
            print(literal(f"slipagent: ignoring language_servers in {start / PROJECT_SETTINGS_PATH}; "
                          f"projects cannot name commands to start. Move them to {language_servers_path()}."),
                  file=sys.stderr)

    def first_run_key(self) -> bool:
        """Ask for a first API key before an interactive launch that has none."""
        print("SlipAgent needs an API key from one of these providers; each offers free models:", file=sys.stderr)
        for _, name, url in FIRST_RUN_PROVIDERS:
            print(f"  {name:<14} {url}", file=sys.stderr)
        try:
            key = getpass.getpass("Paste an API key (hidden; Enter exits): ").strip()
            if not key:
                return False
            provider = key_provider(key)
            while provider is None:
                answer = input("Which provider is this key for? (" + ", ".join(PROVIDERS) + "; Enter exits) ").strip().lower()
                if not answer:
                    return False
                provider = answer if answer in PROVIDERS else None
            implementation = provider_class(provider)
            print(f"Using it as the {provider} key for this session.", file=sys.stderr)
            save = input(f"Save it to {dotenv_path()} for future launches? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return False
        os.environ[implementation.key_env] = key
        # The key came from the prompt, not the user's shell; commands must not inherit it.
        hide_from_commands(implementation.key_env)
        if save in {"", "y", "yes"}:
            try:
                save_dotenv_value(dotenv_path(), implementation.key_env, key)
                print(f"Saved to {dotenv_path()}.", file=sys.stderr)
            except (OSError, ConfigError) as exc:
                print(f"Could not save the key: {exc}", file=sys.stderr)
        return True

    async def dispatch(self, prompt: str | None) -> int:
        """Run everything inside a single event loop.

        The httpx client must be created and used in the same loop, so setup and
        execution cannot live in separate `asyncio.run` calls.
        """
        args = self.args
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=4, thread_name_prefix="slipagent"))
        self.notice_ignored_project_settings()
        self.notice_project_language_servers()
        if args.list_models:
            try:
                return await self.list_models()
            except (ConfigError, APIConfigError) as exc:
                print(literal(f"slipagent: {exc}"), file=sys.stderr)
                return 2

        events = JsonEventStream(sys.stdout) if getattr(args, "json_events", False) else None
        interactive = not prompt and sys.stdin.isatty() and sys.stderr.isatty() and os.environ.get("TERM") != "dumb"
        if interactive and not args.api_key:
            from .config import load_dotenv
            load_dotenv()
        if interactive and not args.api_key and not active_providers():
            if not await asyncio.to_thread(self.first_run_key):
                return 2
        try:
            session = await self.build_session(defer_startup=interactive)
        except (ConfigError, APIError) as exc:
            print(literal(f"slipagent: {exc}"), file=sys.stderr)
            if events is not None:
                events.write("run_end", status="error", exit_code=2, phase="setup",
                             error={"type": type(exc).__name__, "message": str(exc)})
            return 2

        if prompt:
            if events is not None:
                previous_handler = session.agent.on_event
                def handle(event: AgentEvent) -> None:
                    events.event(event)
                    if previous_handler is not None:
                        previous_handler(event)
                session.agent.on_event = handle
                events.write("run_start", model=session.agent.model, provider=session.client.provider,
                             workspace=str(session.workspace.root), session_id=session.agent.session_id)
            return await self.run_one_shot(session, prompt, events)
        return await Repl(session).run()


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


class Repl:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def run(self) -> int:
        session = self.session
        style = session.renderer.style
        renderer = session.renderer
        terminal_task: asyncio.Task[None] | None = None
        if sys.stdin.isatty() and renderer.stream.isatty() and os.environ.get("TERM") != "dumb":
            terminal = TerminalUI(
                lambda columns: StatusLine(session, style).text(columns=columns),
                renderer.stream, color=style.enabled,
                status_suffix=lambda: StatusLine(session, style).danger_suffix(),
                project_directory=session.workspace.root,
            )
            renderer.terminal = terminal
            terminal.__dict__["_agent_input_queue"] = lambda: input_queue(session.agent)
            from .useredits import UserEdits
            user_edits = UserEdits(session, terminal)
            registry_edits = session.registry.services.get("user_edits")
            if registry_edits is not None:
                registry_edits.close()
            session.registry.services["user_edits"] = user_edits
            terminal.__dict__["_user_edits"] = user_edits
            session.agent.on_boundary = user_edits.checkpoint
            terminal.__dict__["_artifact_store"] = session.registry.services.get("artifact_store")
            terminal.configure_markdown(theme=renderer.markdown_theme, source=renderer.output_markdown)
            terminal.set_interrupt_handler(lambda: self.request_interrupt())
            terminal_task = asyncio.create_task(terminal.run())
        quota_task: asyncio.Task[None] | None = None
        reload_task: asyncio.Task[None] | None = None
        try:
            try:
                ready = await self.start()
            except (ConfigError, APIError) as exc:
                renderer.emit(style.red(f"  Startup failed: {exc}"))
                return 2
            if not ready:
                return 0
            if renderer.terminal is not None:
                await PluginMenu(session, style, _RendererStream(renderer)).offer()
            if session.extensions.pop("restore_transcript", False):
                await renderer.restore_transcript(session.agent.messages, session.agent.pending,
                                                  queued_messages=session.agent.queued_messages,
                                                  failures=failed_steps(session.agent.history.steps))
            quota_task = asyncio.create_task(Quota(session).poll())
            reload_task = asyncio.create_task(session.reloader.watch()) if session.reloader is not None else None
            while True:
                try:
                    line = await self.read_line(style)
                except EOFError:
                    renderer.emit()
                    break
                if line is None:
                    break
                await self.wait_for_interrupt()
                if cancelled(line):
                    continue
                suggested = isinstance(line, SuggestedPrompt) or submitted_suggestion(line)
                if not suggested:
                    line = line.strip()

                if not line:
                    continue

                if not suggested and line.startswith("/"):
                    try:
                        if await CommandRunner(session).run_interactive(line):
                            break
                    except Exception as exc:
                        if session.reloader is not None:
                            session.reloader.report_error(f"Command failed: {type(exc).__name__}: {exc}")
                        else:
                            renderer.emit(style.red(f"  ✗ Command failed: {type(exc).__name__}: {exc}"))
                    continue

                # The prompt stays live while the agent works: anything typed in
                # the meantime is queued onto the step instead of being ignored.
                if isinstance(line, Submission):
                    if renderer.terminal is not None:
                        renderer.terminal.consume_submission(line)
                    session.agent.enqueue(line)
                    prompt = ""
                else:
                    renderer.user_prompt(line)
                    prompt = line
                try:
                    if await self.run_request(prompt, style):
                        break
                except KeyboardInterrupt:
                    # Preserve completed results; interrupted actions may be partial.
                    renderer.emit(style.dim("  interrupted."))
                    continue
                except Exception as exc:
                    if session.reloader is not None:
                        session.reloader.report_error(_failure_text(session, exc))
                    else:
                        renderer.emit(style.red(f"  ✗ {_failure_text(session, exc)}"))
                    continue
        finally:
            if reload_task is not None:
                reload_task.cancel()
                await _gather_quietly(reload_task)
            if quota_task is not None:
                quota_task.cancel()
                await _gather_quietly(quota_task)
            try:
                # The final redraw still imports accepted component code. Retire
                # that generation only after the terminal task has finished.
                if renderer.terminal is not None:
                    renderer.terminal.close()
                if terminal_task is not None:
                    await terminal_task
            finally:
                renderer.terminal = None
                await self.shutdown()
        return 0

    async def start(self) -> bool:
        """Keep input live during startup, retaining prompts until initialization settles."""
        session = self.session
        renderer, terminal = session.renderer, session.renderer.terminal
        startup = session.extensions.pop("startup", None)

        async def initialize() -> None:
            await renderer.show_welcome(model=session.agent.model, workspace=session.workspace.root)
            if startup is not None:
                await startup()

        if terminal is None:
            await initialize()
            return True
        terminal.set_working(True)
        self.refresh_title()
        task = asyncio.create_task(initialize())

        def interrupt() -> None:
            terminal.set_working(True, stopping=True)
            task.cancel()

        terminal.set_interrupt_handler(interrupt)
        reader = asyncio.create_task(self.read_line(renderer.style))
        queued: list[str | None] = []
        try:
            while not task.done():
                done, _ = await asyncio.wait({task, reader}, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    line = reader.result()
                    if line is None or line.strip() in {"/exit", "/quit"}:
                        task.cancel()
                        return False
                    queued.append(line)
                    reader = asyncio.create_task(self.read_line(renderer.style))
            if task.cancelled():
                return False
            await task
            return True
        finally:
            if reader.done() and not reader.cancelled():
                queued.append(reader.result())
            reader.cancel()
            if not task.done():
                task.cancel()
            await finish_cleanup(asyncio.create_task(_gather_quietly(task, reader)))
            session.extensions["startup_input_lines"] = queued
            terminal.set_interrupt_handler(lambda: self.request_interrupt())
            terminal.set_working(False)

    async def read_line(self, style: Style) -> str | None:
        """Read one line of input, drawing the prompt and readouts around it.

        Returns None at end of input, which the caller treats as EOF.
        """
        session = self.session
        if "pending_input_line" in session.extensions:
            return cast(str | None, session.extensions.pop("pending_input_line"))
        queued = session.extensions.get("startup_input_lines")
        if queued:
            return cast(str | None, queued.pop(0))
        renderer = session.renderer
        self.refresh_title()
        renderer.draw_prompt(StatusLine(session, style).text(), style.user("> "))
        if renderer.terminal is not None:
            line = await renderer.terminal.read_line()
            renderer.end_prompt()
            return line
        if session._input_future is None:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[str | None] = loop.create_future()
            session._input_future = future
            source = sys.stdin

            def read() -> None:
                try:
                    line = source.readline()
                except (EOFError, KeyboardInterrupt):
                    line = ""
                except Exception as exc:
                    try:
                        loop.call_soon_threadsafe(future.set_exception, exc)
                    except RuntimeError:
                        pass
                    return
                try:
                    loop.call_soon_threadsafe(future.set_result, line or None)
                except RuntimeError:
                    pass

            # A cancelled prompt retains its read so the next prompt consumes the
            # same line. A daemon reader does not hold up shutdown on an idle TTY.
            threading.Thread(target=read, daemon=True).start()
        try:
            line = await asyncio.shield(session._input_future)
        except (EOFError, KeyboardInterrupt):
            line = ""
        session._input_future = None
        if not renderer.prompt_is_live:
            # Cancelled mid-read (the step ended first): leave the prompt as the
            # last thing on screen rather than ending a repaint another task owns.
            return None
        renderer.end_prompt()
        # A non-tty has no cursor to park on, so restore the line ourselves.
        if not renderer.stream.isatty():
            print(file=renderer.stream)
        return line or None

    async def run_request(self, prompt: str, style: Style) -> bool:
        """Run a task while keeping the prompt usable in the background.

        A reader task stays alive for the duration: lines that arrive mid-step are
        queued onto the agent, and `/exit` or `/quit` finishes the step before exiting.
        Returns True when the user asked to exit.
        """
        session = self.session
        previous_boundary = getattr(session.agent, "on_boundary", None)

        async def boundary() -> None:
            if previous_boundary is not None:
                await previous_boundary()
            if not session.agent.has_queued_input and not session.agent.stop_requested:
                # Optimal changes affect the next request and need no task stop.
                commands = session.work.deferred_commands or []
                while commands and CommandRunner(session).at_boundary(commands[0]):
                    await CommandRunner(session).execute(commands.pop(0), at_boundary=True)
                if not commands:
                    session.work.deferred_commands = None
                    session.work.resume_after_commands = None
                    session.renderer._queued_settings_pause = False
            # Earlier prompts must reach a request before a later command applies.
            if session.work.deferred_commands and not session.agent.has_queued_input:
                session.agent.request_stop()

        session.agent.on_boundary = boundary
        reader: asyncio.Task[str | None] | None = asyncio.create_task(
            self.read_line(style)
        )
        agent_task: asyncio.Task[str] = asyncio.create_task(session.agent.run(prompt))
        session.work.agent_task = agent_task
        renderer = session.renderer
        exiting = False
        eof = False
        if renderer.terminal is not None:
            renderer.terminal.set_interrupt_handler(lambda: self.request_interrupt())
            renderer.terminal.set_working(True)

        try:
            while True:
                watched = {agent_task} | ({reader} if reader is not None else set())
                done, _ = await asyncio.wait(
                    watched, return_when=asyncio.FIRST_COMPLETED
                )
                # A line arriving with completion belongs to the idle REPL. Keep it
                # for the handoff below instead of queuing it on a stopped run.
                if reader is not None and reader in done and not agent_task.done():
                    line = reader.result()
                    if line is None:
                        # EOF lets the active run finish before shutdown.
                        reader = None
                        exiting = True
                        eof = True
                    else:
                        text = line.strip()
                        is_command = text.startswith("/") and not (isinstance(line, SuggestedPrompt) or submitted_suggestion(line))
                        if cancelled(line):
                            pass
                        elif is_command and text.split(None, 1)[0].lower() in {"/stop", "/exit", "/quit"}:
                            if await CommandRunner(session).handle(text):
                                renderer.emit(style.dim("  exiting."))
                                exiting = True
                                reader = None
                        elif (is_command and not session.work.deferred_commands
                              and CommandRunner(session).runs_in_background(text)):
                            # Commands that leave the run untouched apply at once.
                            if renderer.terminal is not None:
                                renderer.terminal.consume_submission(line)
                            CommandRunner(session).start_background(text)
                        elif text and (is_command or session.work.deferred_commands):
                            CommandRunner(session).queue(text, text.split(None, 1)[0].lstrip("/") if is_command else "input")
                        elif text:
                            if renderer.terminal is not None:
                                renderer.terminal.consume_submission(line)
                            session.agent.enqueue(text)
                            if session.agent.stop_requested:
                                session.work.resume_after_commands = True
                        if reader is not None:
                            reader = asyncio.create_task(self.read_line(style))

                if agent_task in done:
                    if session.extensions.get("interrupt_requested"):
                        await self.wait_for_interrupt()
                        if not exiting and session.agent.has_queued_input:
                            renderer.emit(style.dim("  Starting queued input."))
                            if renderer.terminal is not None:
                                renderer.terminal.set_working(True)
                            agent_task = asyncio.create_task(session.agent.run(""))
                            session.work.agent_task = agent_task
                            continue
                        break
                    answer = agent_task.result()
                    resume, commands_exit = await CommandRunner(session).apply_deferred(exiting=exiting and not eof)
                    exiting = exiting or commands_exit
                    if resume and not exiting and answer != STEP_LIMIT_NOTICE and (answer == STOP_NOTICE or session.agent.has_queued_input):
                        renderer.emit(style.dim("  Continuing with queued input."))
                        if renderer.terminal is not None:
                            renderer.terminal.set_working(True)
                        agent_task = asyncio.create_task(session.agent.run("", continue_run=True))
                        session.work.agent_task = agent_task
                        continue
                    if not exiting and session.agent.has_queued_input and answer != STEP_LIMIT_NOTICE and not session.agent.stopped:
                        if renderer.terminal is not None:
                            renderer.terminal.set_working(True)
                        agent_task = asyncio.create_task(session.agent.run(""))
                        session.work.agent_task = agent_task
                        continue
                    break
                if reader is None:
                    # With no reader, wait for the active run to settle.
                    if session.extensions.get("interrupt_requested"):
                        await self.wait_for_interrupt()
                        break
                    await agent_task
        except BaseException as exc:
            agent_task.cancel()
            await _gather_quietly(agent_task)
            # A failed model request still leaves an idle session where queued
            # settings can be applied. Cancellation/exit must not start new work.
            if not isinstance(exc, asyncio.CancelledError):
                await CommandRunner(session).apply_deferred(exiting=exiting)
            if isinstance(exc, asyncio.CancelledError) and session.extensions.get("interrupt_requested"):
                await self.wait_for_interrupt()
            else:
                raise
        finally:
            session.agent.on_boundary = previous_boundary
            session.work.agent_task = None
            if reader is not None and not reader.done():
                reader.cancel()
            if reader is not None:
                await _gather_quietly(reader)
                if not exiting and not reader.cancelled() and reader.exception() is None:
                    # Input may finish while deferred commands are being applied.
                    session.extensions["pending_input_line"] = reader.result()
            renderer.clear_prompt()
            if renderer.terminal is not None:
                renderer.terminal.set_working(False)

        if agent_task.done() and not agent_task.cancelled():
            failure = agent_task.exception()
            if failure is not None:
                raise failure
            if exiting and session.agent.has_queued_input:
                count = len(session.agent.pending) + len(session.agent.queued_messages)
                subject = "message was" if count == 1 else "messages were"
                renderer.emit(style.dim(
                    f"  {count} queued {subject} not run because the session is exiting."
                ))
            if session.agent.usage.cost is not None:
                renderer.emit(style.dim(f"    ({session.agent.usage.summary()})"), separate=False)

        return exiting

    def request_interrupt(self) -> None:
        """Signal cancellation immediately; retain cleanup outside the input loop."""
        session = self.session
        cleanup = session.extensions.get("interrupt_task")
        if cleanup is not None:
            return
        session.extensions["interrupt_requested"] = True
        session.extensions["interrupt_sequence"] = session.extensions.get("interrupt_sequence", 0) + 1
        session.registry.services["interrupt_requested"] = True
        discarded = session.extensions.pop("deferred_commands", [])
        CommandRunner(session).cancel_queued(discarded)
        session.extensions["resume_after_commands"] = False
        session.renderer._queued_settings_pause = False
        session.agent.request_stop()
        session.agent.stopped = True
        task = session.extensions.get("active_agent_task")
        command = session.extensions.get("active_command_task")
        commands = list(session.extensions.get("command_tasks", ()))
        for owned in [task, command, *commands]:
            if owned is not None:
                owned.cancel()
        compactor = session.registry.services.get("step_compactor")
        if compactor is not None:
            compactor.reset()
        session.renderer.emit(session.renderer.style.dim("  Interrupting agent work; cancelling active operations."))
        if session.renderer.terminal is not None:
            session.renderer.terminal.set_working(True, stopping=True)

        async def drain() -> None:
            jobs = session.registry.services.get("command_jobs")
            job_cleanup = asyncio.create_task(jobs.stop_all()) if jobs is not None else None
            await _gather_quietly(*[owned for owned in [task, command, *commands, job_cleanup] if owned is not None])
            if compactor is not None:
                await compactor.wait()
            session.registry.services.pop("interrupt_requested", None)
            session.renderer.emit(session.renderer.style.dim("  Interrupted. Conversation and completed results are preserved."))
            if session.renderer.terminal is not None:
                session.renderer.terminal.set_working(False)
        session.extensions["interrupt_task"] = asyncio.create_task(drain())

    async def wait_for_interrupt(self) -> None:
        session = self.session
        task = session.extensions.get("interrupt_task")
        if task is not None:
            await finish_cleanup(task)
            session.extensions.pop("interrupt_task", None)
            session.extensions.pop("interrupt_requested", None)

    def refresh_title(self) -> None:
        session = self.session
        terminal = session.renderer.terminal
        if terminal is not None:
            journal = session.registry.services.get("session_journal")
            title = journal.title if journal is not None else session.extensions.get("title")
            if title is None:
                title = session_title(str(session.workspace.root))
            terminal.set_title(title)

    def start_background_task(self, work: Awaitable[Any]) -> None:
        """Own startup work that must not delay the prompt; shutdown cancels it."""
        session = self.session
        tasks: set[asyncio.Task[Any]] = session.extensions.setdefault("background_tasks", set())
        task = asyncio.ensure_future(work)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def refresh_model_metadata(self) -> None:
        """Replace saved catalogs and profiles used at startup with current ones."""
        session = self.session
        try:
            await session.catalog(refresh=True)
            current = ModelInfo(session.agent.model, provider=session.client.provider)
            targets = [current]
            routing = session.model_routing()
            if routing.enabled:
                # Save the best candidate's profile so the next launch can select it from the cache.
                best = next(iter(await routing.candidates("working")), None)
                if best is not None and best.selector != current.selector:
                    targets.append(best)
            for model in targets:
                await session.refresh_profile(model)
        except (APIError, ConfigError, ValueError):
            pass

    async def shutdown(self) -> None:
        """Release every resource the session owns, MCP subprocesses included.

        Order matters: MCP tools hold references to their clients, so the servers
        are stopped before the registry and HTTP client are closed.
        """
        session = self.session
        await self.wait_for_interrupt()
        tasks = list(session.extensions.get("command_tasks", ())) + list(session.extensions.get("background_tasks", ()))
        for task in tasks:
            task.cancel()
        await _gather_quietly(*tasks)
        try:
            if session.mcp is not None:
                await session.mcp.aclose()
        finally:
            try:
                await session.registry.aclose()
            finally:
                try:
                    await session.client.aclose()
                finally:
                    if session.reloader is not None:
                        session.reloader.close()


class StatusLine:
    def __init__(self, session: Session, style: Style) -> None:
        self.session = session
        self.style = style

    def text(self, *, columns: int | None = None) -> str:
        """Bottom readout: workspace CWD and active model."""
        session, style = self.session, self.style
        cwd = self.cwd()
        model = session.agent.model
        if isinstance(getattr(session, "client", None), APIClient):
            model = f"{session.client.provider}::{model}"
        model = f"Model: {model}"
        if columns is None:
            columns = shutil.get_terminal_size(fallback=(0, 0)).columns
        danger = StatusLine(session, Style(False)).danger_suffix()
        available = max(0, columns - wcswidth(danger)) if columns else 0
        if columns:
            room = available - wcswidth(model + _READOUT_SEP)
            cwd = self.fit(cwd, room, keep_end=True) if room >= 4 else ""
            if not cwd:
                model = self.fit(model, available)
        parts = [part for part in (cwd, model) if part]
        return (style.dim(self.fit(_READOUT_SEP.join(parts), available) if columns else _READOUT_SEP.join(parts))
                + style.red(self.fit(danger, columns, keep_end=True) if columns else danger))

    def danger_suffix(self) -> str:
        session, style = self.session, self.style
        return style.red(" | Danger Mode") if session.workspace.access.danger else ""

    def cwd(self) -> str:
        """Show the tools' working directory, abbreviating the home directory."""
        cwd = self.session.workspace.root
        # Abbreviate the home prefix while retaining the workspace suffix.
        home = Path.home()
        if cwd == home:
            return "~"
        try:
            return "~/" + cwd.relative_to(home).as_posix()
        except ValueError:
            return str(cwd)

    def fit(self, text: str, columns: int, *, keep_end: bool = False) -> str:
        if wcswidth(text) <= columns:
            return text
        if columns <= 0:
            return ""
        graphemes = list(iter_graphemes(text))
        while graphemes and wcswidth("".join(graphemes)) + 1 > columns:
            if keep_end:
                graphemes.pop(0)
            else:
                graphemes.pop()
        text = "".join(graphemes)
        return "…" + text if keep_end else text + "…"


class Quota:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def refresh(self) -> None:
        session = self.session
        frame = session.reloader
        if frame is not None:
            frame.busy += 1
        try:
            await self.fetch()
        finally:
            if frame is not None:
                frame.busy -= 1

    async def fetch(self) -> None:
        """Fetch the authoritative free-call quota at most every fifteen minutes."""
        session = self.session
        now = time.monotonic()
        if session.quota_checked_at is not None and now - session.quota_checked_at < QUOTA_REFRESH_INTERVAL:
            return
        session.quota_checked_at = now
        client = session.client
        try:
            info = await client.key_info()
        except APIAuthError as exc:
            # An expired or revoked key shows up here before the next prompt fails.
            if session.client is client and session.extensions.get("key_rejected_warned") != client.api_key:
                session.extensions["key_rejected_warned"] = client.api_key
                self.warn(f"{client.provider} rejected its API key during the quota check: {exc} "
                          + _key_remedy(client.provider))
            return
        except (APIError, ConfigError):
            return
        # A response for the previous key must not overwrite the new key's quota.
        if session.client is client and info is not None:
            self.cache(info)

    async def poll(self) -> None:
        """Refresh independently of user input and model requests."""
        session = self.session
        while True:
            checked = session.quota_checked_at
            delay = 0.0 if checked is None else max(
                0.0, QUOTA_REFRESH_INTERVAL - (time.monotonic() - checked),
            )
            await asyncio.sleep(delay)
            await self.refresh()
            if session.renderer.terminal is not None:
                session.renderer.terminal.app.invalidate()

    def cache(self, info: KeyInfo) -> None:
        """Remember the key's remaining free calls for the status bar, and its expiry."""
        session = self.session
        session.free_calls = info.free_quota.remaining if info.free_quota is not None else None
        session.quota_checked_at = time.monotonic()
        provider = getattr(session.client, "provider", "")
        session.extensions.setdefault("key_expiry", {})[provider] = info.expires_at
        self.warn_expiry(provider, info.expires_at)

    def warn_expiry(self, provider: str, expires_at: str | None) -> None:
        """Warn once per key when it has expired or expires within a week."""
        import math
        expiry = _key_expiry(expires_at)
        if expiry is None:
            return
        remaining = expiry - datetime.now(timezone.utc)
        if remaining.days >= KEY_EXPIRY_WARNING_DAYS:
            return
        session = self.session
        marker = (provider, expires_at)
        if session.extensions.get("key_expiry_warned") == marker:
            return
        session.extensions["key_expiry_warned"] = marker
        when = expiry.date().isoformat()
        if remaining.total_seconds() <= 0:
            self.warn(f"the {provider} API key expired on {when}. " + _key_remedy(provider))
        else:
            days = max(1, math.ceil(remaining.total_seconds() / 86400))
            self.warn(f"the {provider} API key expires in {days} day{'' if days == 1 else 's'} ({when}); "
                      f"rotate it with /key <key> before then.")

    def warn(self, text: str) -> None:
        renderer = getattr(self.session, "renderer", None)
        if renderer is not None:
            renderer.emit(renderer.style.warning(f"  Warning: {text}"))


class CommandRunner:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def handle(self, line: str) -> bool:
        session = self.session
        terminal = session.renderer.terminal
        if terminal is not None and terminal.__dict__.get("_editor_active"):
            terminal.notify("Close the editor before running commands.")
            return False
        frame = session.reloader
        if frame is not None:
            await frame.checkpoint()
            frame.busy += 1
        try:
            return await self.execute(line)
        finally:
            if frame is not None:
                frame.busy -= 1
                await frame.checkpoint()

    async def execute(self, line: str, *, at_boundary: bool = False) -> bool:
        """Dispatch definitions while preserving batch boundaries and finalization."""
        session = self.session
        parts = line[1:].split(None, 1)
        name = parts[0].lower() if parts else ""
        argument = parts[1].strip() if len(parts) > 1 else ""
        command = BY_NAME.get(name)
        if cancelled(line):
            return False
        if (not at_boundary and command is not None and command.changes_session(argument)
                and (session.agent.running or session.extensions.get("deferred_commands"))):
            self.queue(line, name)
            return False
        if isinstance(line, Submission) and session.renderer.terminal is not None:
            session.renderer.terminal.consume_submission(line)
        if command is None:
            session.renderer.emit(session.renderer.style.red(f"  unknown command: /{name} (try /help)"))
        else:
            result = await command.run(CommandContext(session, session.renderer.style, _RendererStream(session.renderer)), argument)
            if result is not None:
                return result
        Repl(session).refresh_title()
        session.agent._persist()
        return False

    async def run_interactive(self, line: str) -> bool:
        """Keep commands such as model selection cancellable from the terminal."""
        session = self.session
        task = asyncio.create_task(self.handle(line))
        session.work.command_task = task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if session.extensions.get("interrupt_requested"):
                await Repl(session).wait_for_interrupt()
                return False
            task.cancel()
            await _gather_quietly(task)
            raise
        finally:
            session.work.command_task = None

    def queue(self, line: str, command: str) -> None:
        session = self.session
        commands = session.work.deferred_commands
        if commands is None:
            commands = []
            session.work.deferred_commands = commands
        # Fresh input resumes after a stop; /stop clears this for older input.
        session.work.resume_after_commands = True
        commands.append(line)
        pause = any(not self.at_boundary(queued) for queued in commands)
        session.renderer._queued_settings_pause = pause and bool(session.extensions.get("resume_after_commands"))
        if pause and not session.agent.has_queued_input:
            session.agent.request_stop()
        if session.renderer.terminal is not None:
            # A pause that resumes on its own to apply a command is not a stop.
            pausing = session.renderer._queued_settings_pause
            session.renderer.terminal.set_working(True, stopping=session.agent.stop_requested and not pausing, pausing=pausing)
        if session.renderer.terminal is None:
            session.renderer.emit(session.renderer.style.dim(
                f"  Queued /{command}; it will run after earlier input and the current tool batch."
            ))

    async def apply_deferred(self, *, exiting: bool = False) -> tuple[bool, bool]:
        """Dispatch the shared FIFO through its next prompt barrier while idle.

        Command text stays outside model history and journals, including credentials.
        The active response and every tool have settled before this is called.
        """
        session = self.session
        commands = session.work.deferred_commands or []
        if session.agent.has_queued_input and not exiting:
            return bool(session.work.resume_after_commands), False
        session.work.deferred_commands = None
        interrupt_sequence = session.extensions.get("interrupt_sequence", 0)
        resume = bool(session.work.resume_after_commands)
        session.work.resume_after_commands = None
        session.renderer._queued_settings_pause = False
        if exiting:
            self.cancel_queued(commands)
            if commands:
                session.renderer.emit(session.renderer.style.dim(
                    f"  {len(commands)} queued command(s) were not run because the session is exiting."
                ))
            return False, True
        for position, line in enumerate(commands):
            if cancelled(line):
                continue
            if not line.startswith("/") or submitted_suggestion(line) or isinstance(line, SuggestedPrompt):
                end = position
                while end < len(commands):
                    queued = commands[end]
                    if not cancelled(queued):
                        if queued.startswith("/") and not (submitted_suggestion(queued) or isinstance(queued, SuggestedPrompt)):
                            break
                        if session.renderer.terminal is not None:
                            session.renderer.terminal.consume_submission(queued)
                        session.agent.enqueue(queued)
                    end += 1
                # A prompt is a barrier: later commands wait for its request.
                session.work.deferred_commands = commands[end:] or None
                session.work.resume_after_commands = resume
                # Keep an explicit /stop authoritative even when a preceding
                # queued command has now finished applying.
                return resume, False
            parts = line[1:].split(None, 1)
            command = parts[0].lower()
            argument = parts[1].strip() if len(parts) > 1 else ""
            if BY_NAME.get(command) is not None and BY_NAME[command].replaces_task(argument):
                resume = False
            try:
                if session.renderer.terminal is not None:
                    session.renderer.terminal.set_working(False)
                    session.renderer.terminal.set_command_status(f"Applying /{command} (Esc to interrupt)")
                if await self.run_interactive(line):
                    return False, True
                if session.extensions.get("interrupt_sequence", 0) != interrupt_sequence:
                    self.cancel_queued(commands[position + 1:])
                    return False, False
            except Exception as exc:
                session.renderer.emit(session.renderer.style.red(f"  ✗ /{command} failed: {exc}"))
            finally:
                if session.renderer.terminal is not None:
                    session.renderer.terminal.set_command_status("")
        return resume, False

    def cancel_queued(self, lines: list[str]) -> None:
        """Cancel settings while retaining unsent user input for continuation."""
        session = self.session
        for line in lines:
            if cancelled(line):
                continue
            if isinstance(line, Submission) and session.renderer.terminal is not None:
                session.renderer.terminal.consume_submission(line)
            if not line.startswith("/") or submitted_suggestion(line) or isinstance(line, SuggestedPrompt):
                session.agent.enqueue(line)

    def start_background(self, text: str) -> None:
        session = self.session
        tasks = session.work.command_tasks
        if tasks is None:
            tasks = set()
            session.work.command_tasks = tasks
        async def run() -> None:
            try:
                await self.handle(text)
            except Exception as exc:
                session.renderer.emit(session.renderer.style.red(f"  ✗ Command failed: {exc}"))
        task = asyncio.create_task(run())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def at_boundary(self, line: str) -> bool:
        """Routing commands can apply between batches without ending the task."""
        return (not isinstance(line, SuggestedPrompt) and not submitted_suggestion(line)
                and line.split(None, 1)[0].lower() == "/optimal")

    def runs_in_background(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        parts = text.split(None, 1)
        name = parts[0].lower().removeprefix("/") if parts else ""
        argument = parts[1].strip().lower() if len(parts) > 1 else ""
        command = BY_NAME.get(name)
        return command.background(argument) if command is not None else False

    def changes_session(self, command: str, argument: str) -> bool:
        definition = BY_NAME.get(command)
        return definition.changes_session(argument) if definition is not None else False


async def _gather_quietly(*tasks: asyncio.Task[object]) -> None:
    for task in tasks:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


class KeyEntry:
    def __init__(self, session: Session, style: Style, out: io.TextIOBase) -> None:
        self.session = session
        self.style = style
        self.out = out

    async def command(self, argument: str) -> None:
        session, style, out = self.session, self.style, self.out
        words = argument.split(maxsplit=1)
        if words and words[0].lower() == "list":
            self.listing()
            return
        if words and words[0].lower() in {"delete", "remove"}:
            name = words[1].strip().lower() if len(words) == 2 else ""
            if name not in PROVIDERS:
                print(style.red("  usage: /key delete <provider>; providers: " + ", ".join(PROVIDERS)), file=out)
                return
            await self.delete(name)
            return
        if argument.lower() in ("show", "status", ""):
            if argument == "" and getattr(getattr(session, "renderer", None), "terminal", None) is not None:
                # Bare /key on a terminal lists the configured keys to delete, or takes a new one.
                await self.menu()
                return
            if argument == "" and sys.stdin.isatty():
                await self.replace()
                return
            print(f"  key: {style.cyan(self.mask(session.api_key))}", file=out)
            try:
                info = await session.client.key_info()
            except APIError as exc:
                print(style.red(f"  could not verify key: {exc}"), file=out)
                return
            if info is None:
                print(style.dim("  This provider has no key-status API; authentication is checked on inference."), file=out)
            else:
                Quota(session).cache(info)
                self.describe(info)
            return

        parts = argument.split(maxsplit=1)
        provider = None
        candidate = argument
        if len(parts) == 2:
            provider, candidate = parts[0].lower(), parts[1]
            if provider not in PROVIDERS:
                print(style.red("  usage: /key [provider] <key>; providers: " + ", ".join(PROVIDERS)), file=out)
                return
        await self.replace(candidate=candidate, provider=provider)

    async def replace(self, candidate: str | None = None, provider: str | None = None) -> None:
        session, style, out = self.session, self.style, self.out
        if candidate is None:
            if not sys.stdin.isatty():
                print(
                    style.dim(
                        "  /key needs an interactive terminal to read a key. "
                        f"Set {session.client.key_env}, or run `/key <key>`."
                    ),
                    file=out,
                )
                return
            print(style.dim("  Enter an API key (input hidden; provider detected automatically)."), file=out)
            try:
                candidate = (await session.renderer.ask_input("  key: ", password=True)).strip()
            except (EOFError, KeyboardInterrupt):
                print(file=out)
                return
            if not candidate:
                print(style.dim("  cancelled"), file=out)
                return

        candidate = candidate.strip()
        if not candidate or not candidate.isascii() or any(character.isspace() or ord(character) < 32 or ord(character) == 127
                                                          for character in candidate):
            print(style.red("  API keys must be a single nonblank ASCII value."), file=out)
            return
        detected = key_provider(candidate)
        if provider is not None and detected is not None and provider != detected:
            print(style.red(f"  Provider mismatch: selected {provider}, detected {detected}. Use /key {detected} <key>."), file=out)
            return
        provider = provider or detected
        if provider is None:
            if not sys.stdin.isatty():
                print(style.red("  Could not identify the provider. Use /key <provider> <key>; providers: "
                                + ", ".join(PROVIDERS)), file=out)
                return
            try:
                provider = (await session.renderer.ask_input("  provider (" + ", ".join(PROVIDERS) + "; blank cancels): ")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if not provider:
                print(style.dim("  cancelled"), file=out)
                return
            if provider not in PROVIDERS:
                print(style.red("  Choose a supported provider: " + ", ".join(PROVIDERS)), file=out)
                return
        implementation = provider_class(provider)
        print(style.dim(f"  provider: {provider}"), file=out)
        try:
            info = await session.use_api_key(candidate, provider=provider, verify=True)
        except (APIError, ConfigError) as exc:
            print(style.red(literal(f"  rejected: {str(exc).replace(candidate, '[key]')}")), file=out)
            return

        if info is None and not getattr(implementation, "catalog_requires_auth", False):
            print(style.dim(f"  {provider} key updated for this session; authentication will be checked on inference."), file=out)
        else:
            print(f"  {provider} key accepted: {style.cyan(self.mask(candidate))}", file=out)
            if info is not None:
                if provider == session.client.provider:
                    Quota(session).cache(info)
                self.describe(info)

        # Persisting replaces a credential on disk, so require explicit consent.
        if not sys.stdin.isatty():
            print(style.dim("  not saved (no interactive terminal)"), file=out)
            return
        answer = await session.renderer.ask_input(f"  save {provider} key to .env for future runs? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print(style.dim("  kept for this session only"), file=out)
            return

        path = dotenv_path()
        try:
            save_dotenv_value(path, implementation.key_env, candidate)
            print(style.dim(f"  saved to {path}"), file=out)
        except (OSError, ConfigError) as exc:
            print(style.red(f"  could not save to {path}: {exc}"), file=out)

    def configured(self) -> list[tuple[str, str, list[str]]]:
        """Every provider with a key: its name, the key, and where that key lives."""
        session = self.session
        saved = dotenv_values(dotenv_path())
        options = session.extensions.get("provider_options", {})
        active = getattr(session.client, "provider", None)
        result = []
        for provider in PROVIDERS:
            name = provider_class(provider).key_env
            key = ""
            sources: list[str] = []
            if provider == active:
                key, sources = session.client.api_key, ["active"]
            elif isinstance(options.get(provider), dict) and options[provider].get("api_key"):
                key, sources = options[provider]["api_key"], ["this session"]
            source = environment_source(name)
            if source == "shell":
                sources.append("exported by the shell")
            elif source == "session":
                sources.append("entered at startup")
            elif source == "dotenv" and name not in saved:
                sources.append("loaded from .env")
            if name in saved:
                sources.append(f"saved in {dotenv_path()}")
            key = key or os.environ.get(name, "") or saved.get(name, "")
            expires_at = session.extensions.get("key_expiry", {}).get(provider)
            expiry = _key_expiry(expires_at)
            if expiry is not None:
                when = expiry.date().isoformat()
                sources.append(f"expired {when}" if expiry <= datetime.now(timezone.utc) else f"expires {when}")
            if key.strip():
                result.append((provider, key, sources))
        return result

    def listing(self) -> None:
        style, out = self.style, self.out
        entries = self.configured()
        if not entries:
            print(style.dim("  no API keys are configured; /key <key> adds one."), file=out)
            return
        for provider, key, sources in entries:
            print(f"  {provider:<11} {style.cyan(self.mask(key))}  {style.dim(', '.join(sources))}", file=out)

    async def menu(self) -> None:
        """Pick a configured key to delete, after confirmation, or enter a new one."""
        session, style, out = self.session, self.style, self.out
        terminal = session.renderer.terminal
        if terminal is None:
            self.listing()
            return
        entries = self.configured()
        options = [("new", "Enter a new key")]
        options += [(provider, f"{provider}  {self.mask(key)}  — {', '.join(sources)}") for provider, key, sources in entries]
        choice = await terminal.choose("API Keys", options)
        if choice is None:
            return
        if choice == "new":
            await self.replace()
            return
        provider, key, _ = next(entry for entry in entries if entry[0] == choice)
        confirmation = await terminal.choose(f"Are you sure you want to delete this key? {provider} {self.mask(key)}",
                                             [("delete", "Delete this key"), ("cancel", "Cancel")])
        if confirmation == "delete":
            await self.delete(provider)
        else:
            print(style.dim("  kept"), file=out)

    async def delete(self, provider: str) -> None:
        """Remove a provider's key from the saved .env, this process, and this session."""
        session, style, out = self.session, self.style, self.out
        implementation = provider_class(provider)
        if provider == getattr(session.client, "provider", None):
            print(style.red(f"  the {provider} key is in use by the current model; switch with /model before deleting it."), file=out)
            return
        removed = []
        path = dotenv_path()
        try:
            if remove_dotenv_value(path, implementation.key_env):
                removed.append(f"removed from {path}")
        except (OSError, ConfigError) as exc:
            print(style.red(f"  could not update {path}: {exc}"), file=out)
            return
        source = forget_environment(implementation.key_env)
        if source == "shell":
            removed.append("cleared for this process; your shell exports it, so it returns at the next launch unless you unset it there")
        elif source is not None:
            removed.append("cleared for this process")
        if session.extensions.get("provider_options", {}).pop(provider, None) is not None:
            removed.append("forgotten for this session")
        if not removed:
            print(style.dim(f"  no {provider} key is configured."), file=out)
            return
        session._catalog = None
        session.model_routing().reset(provider)
        print(f"  {provider} key deleted: " + "; ".join(removed) + ".", file=out)

    def describe(self, info: KeyInfo) -> None:
        """Print a key's limits so the user knows what it can actually reach."""
        style, out = self.style, self.out
        if info.label:
            print(literal(f"    label      {info.label}"), file=out)
        if info.limit is not None:
            remaining = info.limit_remaining
            detail = f"${info.limit:.2f} limit"
            if remaining is not None:
                detail += f", ${remaining:.2f} remaining"
            print(f"    spend      {detail}", file=out)
        print(f"    usage      ${info.usage:.2f} to date", file=out)
        if info.expires_at:
            expiry = _key_expiry(info.expires_at)
            expired = expiry is not None and expiry <= datetime.now(timezone.utc)
            print(f"    expires    {info.expires_at}" + (" (expired)" if expired else ""), file=out)
        if info.free_quota is not None:
            quota = info.free_quota
            print(
                f"    free quota {quota.remaining}/{quota.limit} "
                f"free-model requests left today",
                file=out,
            )
        if info.cannot_reach_paid_models:
            print(
                style.dim(
                    "    note       a $0 spend limit means only zero-cost models "
                    "will respond"
                ),
                file=out,
            )

    def mask(self, key: str) -> str:
        """Show the vendor prefix, or four characters, plus the last four."""
        if len(key) <= 12:
            return "*" * len(key)
        from .providers import KEY_PREFIXES
        prefix = next((value for values in KEY_PREFIXES.values() for value in values if key.startswith(value)), key[:4])
        return f"{prefix}…{key[-4:]}"


def _price_cell(pricing: dict[str, Any]) -> str:
    """Render prompt/completion pricing as $/Mtok, or 'free'."""
    return price_label(pricing)


# `/model` subcommands, shown as `[name] description` in its menu.
MODEL_MENU = (
    ("model", "Enter a model name or filter words; blank lists every model"),
    ("provider", "Choose a provider and list its models, free ones first"),
    ("free", "Choose from free models"),
    ("optimal", "Select the best ranked free model automatically"),
    ("next", "Skip the current model for the next ranked free model"),
    ("reset", "Retry from the best ranked free model, clearing failures"),
)


class ModelPicker:
    def __init__(self, session: Session, style: Style, out: io.TextIOBase) -> None:
        self.session = session
        self.style = style
        self.out = out

    async def menu(self, argument: str, *, require_free: bool = False) -> None:
        """Dispatch `/model`: a menu, a subcommand, a provider, an exact model, or filter words.

        Subcommands and provider names apply only as a single word; several words search.
        """
        session, style, out = self.session, self.style, self.out
        words = argument.strip()
        action = words.lower() if len(words.split()) == 1 else ""
        terminal = session.renderer.terminal
        if not argument.strip():
            routing = session.model_routing()
            current = f"{session.client.provider}::{session.agent.model}"
            state = f"Optimal {'on' if routing.enabled else 'off'}"
            if terminal is None:
                print(f"  current model: {style.cyan(current)} ({state})", file=out)
                for name, description in MODEL_MENU:
                    print(f"  /model {name:<9} {description}", file=out)
                print(style.dim("  /model <provider>, <model> or <words> also select directly."), file=out)
                return
            action = await terminal.choose(f"Model: {current} ({state})",
                                           [(name, f"[{name}] {description}") for name, description in MODEL_MENU]) or ""
            if not action:
                return
            if action == "model":
                try:
                    words = (await session.renderer.ask_input("  model or filter words (blank lists every model): ")).strip()
                except (EOFError, KeyboardInterrupt):
                    return
                await self.lookup(words, require_free=require_free)
                return
        if action == "provider":
            await self.provider_models("")
        elif action in PROVIDERS:
            await self.provider_models(action)
        elif action in {"optimal", "next", "reset"}:
            await self.routing_action(action)
        elif action == "free":
            await self.browse("free")
        else:
            await self.lookup(words, require_free=require_free)

    async def routing_action(self, action: str) -> None:
        """Enable optimal selection, skip to the next ranked model, or retry from the top."""
        session, style, out = self.session, self.style, self.out
        routing = session.model_routing()
        try:
            await (routing.skip() if action == "next" else routing.enable())
        except APIError as exc:
            print(style.red(f"  {exc}"), file=out)

    async def provider_models(self, provider: str) -> None:
        """List a provider's models, free ones first, choosing the provider when none is named."""
        session, style, out = self.session, self.style, self.out
        if not provider:
            terminal = session.renderer.terminal
            if terminal is None:
                print("  providers: " + ", ".join(PROVIDERS) + "  (/model <provider> lists its models)", file=out)
                return
            provider = await terminal.choose("Select Provider", [(name, name) for name in PROVIDERS]) or ""
            if not provider:
                return
        await self.browse("", provider=provider)

    async def provider_default(self, provider: str) -> None:
        """Switch to a provider's free default model."""
        session, style, out = self.session, self.style, self.out
        if provider not in PROVIDERS:
            print(style.red("  usage: /provider " + "|".join(PROVIDERS)), file=out)
            return
        await self.select(f"{provider}::{provider_class(provider).default_model}", require_free=True)

    async def lookup(self, words: str, *, require_free: bool = False) -> None:
        """Select an exact model, or list the models matching every word."""
        session, style, out = self.session, self.style, self.out
        try:
            catalog = await session.catalog(refresh=True)
        except (APIError, ConfigError) as exc:
            print(style.red(f"  could not verify against the catalog: {exc}"), file=out)
            return
        if words and any(words in (model.id, model.selector) for model in catalog):
            await self.select(words, require_free=require_free, catalog=catalog)
        else:
            await self.browse(words)

    async def select(self, argument: str, *, require_free: bool = False,
                     catalog: list[ModelInfo] | None = None) -> None:
        """Switch to an exact model, verified against a freshly fetched catalog."""
        session, style, out = self.session, self.style, self.out
        if catalog is None:
            try:
                catalog = await session.catalog(refresh=True)
            except (APIError, ConfigError) as exc:
                print(style.red(f"  could not verify against the catalog: {exc}"), file=out)
                return

        matches = [model for model in catalog if argument in (model.id, model.selector)]
        if not matches:
            print(style.red(f"  unknown model: {argument}"), file=out)
            return
        if len(matches) > 1:
            print(style.dim("  Select a provider explicitly with /model <provider>::<model>:"), file=out)
            print(self.table(matches), file=out)
            return
        entry = matches[0]
        if require_free:
            try:
                free = all(float(entry.pricing[name]) == 0 for name in ("prompt", "completion"))
                free = free and all(float(entry.pricing.get(name, 0)) == 0
                                    for name in ("request", "internal_reasoning"))
            except (KeyError, TypeError, ValueError):
                free = False
            if not free:
                print(style.red(f"  {entry.selector} is not advertised as free; provider unchanged."), file=out)
                return
        try:
            await session.select_model(entry)
        except (APIError, ConfigError) as exc:
            print(style.red(f"  could not load model properties: {exc}"), file=out)
            return
        try:
            await asyncio.to_thread(save_model_choice, entry.id, entry.provider)
        except (OSError, ConfigError) as exc:
            print(style.red(f"  could not remember model choice: {exc}"), file=out)
        price = _price_cell(entry.pricing)
        print(f"  switched to {style.cyan(entry.selector)}  {price}{'/Mtok' if price.startswith('$') else ''}",
              file=out)
        if entry.provider == "openrouter" and _price_cell(entry.pricing) == "free":
            print(style.dim("  note: free models still work with a $0-limit key"), file=out)

    async def browse(self, argument: str, *, provider: str | None = None) -> None:
        session, style, out = self.session, self.style, self.out
        try:
            catalog = await session.catalog()
        except (APIError, ConfigError) as exc:
            print(style.red(f"  could not load the catalog: {exc}"), file=out)
            return

        for error in getattr(session, "extensions", {}).get("catalog_errors", []):
            print(style.red(f"  catalog unavailable: {error}"), file=out)

        current_model = getattr(getattr(session, "agent", None), "model", None)
        current_provider = getattr(getattr(session, "client", None), "provider", "openrouter")
        labels = {m.selector: self.label(m) + ("  (current)" if m.id == current_model
                                                and m.provider == current_provider else "") for m in catalog}
        needle = argument.strip().lower()
        if needle:
            words = ("free" if needle == "0" else needle).split()
            matches = [m for m in catalog
                       if all(word in labels[m.selector].lower() for word in words)]
        else:
            matches = list(catalog)
        if provider is not None:
            matches = [model for model in matches if model.provider == provider]

        if not matches:
            print(f"  no {provider + ' ' if provider else ''}models matching {argument!r}" if argument
                  else f"  no {provider} models are available; check its API key", file=out)
            return

        terminal = session.renderer.terminal
        if terminal is not None:
            options = [
                (model.selector, labels[model.selector])
                for model in sorted(matches, key=model_order)
            ]
            selected = await terminal.choose("Select Model", options,
                                            styles={m.selector: "class:menu-free" for m in matches
                                                    if free_label(m.pricing) is not None},
                                            legend=[("class:menu-free", "   "), ("", " = free")])
            if selected is not None:
                await self.select(selected)
            return

        free_count = sum(1 for m in catalog if free_label(m.pricing) is not None)
        print(style.dim(
            f"  {len(matches)} shown of {len(catalog)} "
            f"({free_count} free — /models free)\n"
        ), file=out)

        if needle in ("free", "0"):
            # Only reachable models matter to someone on a limited key.
            try:
                info = await session.client.key_info()
            except APIError:
                info = None
            if info is not None and info.cannot_reach_paid_models:
                print(style.dim(
                    "  your key has a $0 spend limit, so only the free models "
                    "below will respond\n"
                ), file=out)

        print(self.table(sorted(matches, key=model_order)), file=out)

    def table(self, models: list[ModelInfo]) -> str:
        session = self.session
        terminal = session.renderer.terminal
        columns = terminal.output.get_size().columns if terminal is not None else shutil.get_terminal_size((80, 24)).columns
        rows = [[f"{model.provider}::\n{model.id}" + (f"\nEst. coding score: {model.score:.1f}/100" if model.score is not None else ""),
                 f"{model.context_length:,}" if model.context_length else "-", _price_cell(model.pricing)] for model in models]
        headers = ["Model", "Context", "In/Out $/Mtok"]
        context_width = max(len(headers[1]), *(len(row[1]) for row in rows))
        price_width = max(len(headers[2]), *(len(row[2]) for row in rows))
        # Grid borders and cell padding occupy ten columns; output is indented by two.
        model_width = columns - context_width - price_width - 12
        while model_width >= len(headers[0]):
            table = tabulate(rows, headers=headers, tablefmt="outline", disable_numparse=True,
                             colalign=("left", "right", "right"), maxcolwidths=[model_width, None, None])
            # Tabulate also reserves header alignment space; measure the final grid.
            excess = max(len(line) for line in table.splitlines()) + 2 - columns
            if excess <= 0:
                break
            model_width -= excess
        else:
            # Stack fields inside each row when three columns would exceed the screen.
            cells = [[f"Model: {row[0]}\nContext: {row[1]}\nIn/Out $/Mtok: {row[2]}"] for row in rows]
            table = tabulate(cells, tablefmt="outline", disable_numparse=True,
                             maxcolwidths=[max(1, columns - 6)])
        return literal("\n".join("  " + line for line in table.splitlines()))

    def label(self, model: ModelInfo) -> str:
        name = f"  {model.name}" if model.name and model.name != model.id else ""
        price = _price_cell(model.pricing)
        if price.startswith("$"):
            price += "/Mtok"
        context = f"{model.context_length:,}" if model.context_length else "unknown"
        score = f"  estimated coding score: {model.score:.1f}/100" if model.score is not None else ""
        return f"{model.selector}{name}  {price}  context: {context}{score}"

    def describe(self, model: ModelInfo) -> str:
        context = f"{model.context_length // 1000}k" if model.context_length else "-"
        price = _price_cell(model.pricing)
        return f"  {model.id:<44} {context:>8}  {price:>15}"


class PluginMenu:
    """Approve plugin code and overrides, and resolve competing replacements.

    Decisions are saved in plugins/.approvals.json and apply through a reload,
    or directly to the live classes when reload is off.
    """

    def __init__(self, session: Session, style: Style, out: io.TextIOBase) -> None:
        self.session = session
        self.style = style
        self.out = out

    async def offer(self) -> None:
        """Ask about file versions, overrides, and conflicts without saved decisions."""
        if not plugins.all_plugins and plugins.pending(plugins.reports):
            await self.menu(only_new=True)

    async def menu(self, *, only_new: bool = False) -> None:
        style, out = self.style, self.out
        terminal = self.session.renderer.terminal
        reports = [report for report in plugins.reports if not report.error
                   and (report.overrides or report.code_state != "on")]
        if not only_new:
            self.listing()
        if not reports:
            return
        if terminal is None:
            print(style.dim("  Plugin approvals need an interactive terminal."), file=out)
            return
        approvals = await run_blocking(plugins.load_approvals)
        changed = False
        if only_new:
            for report in reports:
                if report.code_state == "rejected":
                    continue
                current = await self.approve_code(terminal, approvals, report)
                if current is None:
                    continue
                report = current
                new = [override for override in report.overrides
                       if approvals.decision(report.name, override.member_key) is None]
                if new:
                    changed |= await self.approve(terminal, approvals, report, new)
        else:
            while True:
                reports = [report for report in plugins.reports if not report.error
                           and (report.overrides or report.code_state != "on")]
                options = [(report.name, f"{report.name}: {self.summary(report, approvals)}") for report in reports]
                name = await terminal.choose(f"Plugins in {plugins.plugin_directory()}", options)
                if name is None:
                    break
                report = next(report for report in reports if report.name == name)
                current = await self.approve_code(terminal, approvals, report)
                if current is not None:
                    changed |= await self.approve(terminal, approvals, current, current.overrides)
        changed |= await self.resolve(terminal, approvals, plugins.reports, only_new=only_new)
        if changed:
            await run_blocking(plugins.save_approvals, approvals)
            await self.apply()

    async def approve_code(self, terminal: TerminalUI, approvals: plugins.Approvals,
                           report: plugins.PluginReport) -> plugins.PluginReport | None:
        """Approve the exact file snapshot before importing it to discover overrides."""
        if report.code_state == "on":
            return report
        print(f"  Review {report.path}. Approving this version allows all its Python code to execute, "
              "including imports and top-level statements.", file=self.out)
        answer = await terminal.choose(f"Python Code: {report.name}", [
            ("approve", "Approve This Version"), ("reject", "Keep This Version Disabled"),
        ])
        if answer is None:
            return None
        approvals.decide_code(report.name, report.digest, answer == "approve")
        await run_blocking(plugins.save_approvals, approvals)
        await self.apply()
        return next((current for current in plugins.reports if current.name == report.name
                     and current.code_state == "on" and not current.error), None)

    async def apply(self) -> None:
        """Apply changed plugin files or decisions now: through a reload, or directly without one."""
        style, out = self.style, self.out
        if plugins.all_plugins:
            print(style.dim("  Plugin choices saved; --all-plugins applies every override for this run."), file=out)
        elif self.session.reloader is not None:
            self.session.reloader.request()
            await self.session.reloader.checkpoint()
        else:
            for report in await run_blocking(plugins.apply_to_live, CORE_MODULES):
                for text in report.notices():
                    print(f"  {text}", file=out)
            print(style.dim("  Plugin choices applied."), file=out)

    STATES = {"on": "approved", "pending": "awaiting approval", "rejected": "rejected",
              "conflict": "conflict unresolved", "dropped": "another plugin chosen"}
    STATUSES = {"changed": "changed upstream", "unstamped": "unstamped", "added": "new member"}

    def listing(self) -> None:
        style, out = self.style, self.out
        print(f"  Plugins in {plugins.plugin_directory()}", file=out)
        if not plugins.reports:
            print(style.dim("  none; /plugin copy <module.Class>[.method] starts one."), file=out)
        for report in plugins.reports:
            if report.error:
                print(style.red(f"  {report.name}: disabled: {report.error}"), file=out)
                continue
            if report.code_state != "on":
                state = "Python code awaiting approval" if report.code_state == "pending" else "Python code rejected"
                print(style.dim(f"  {report.name}: {state}"), file=out)
                continue
            print(f"  {report.name} → {report.target}", file=out)
            for override in report.overrides:
                notes = [self.STATES[override.state]]
                if override.status in self.STATUSES:
                    notes.append(self.STATUSES[override.status])
                text = f"    {override.member_key}: {', '.join(notes)}"
                print(style.warning(text) if override.status == "changed" and override.state == "on"
                      else text if override.state == "on" else style.dim(text), file=out)

    async def command(self, argument: str) -> None:
        """`/plugin copy|diff|restamp ...`."""
        from .plugintools import PluginToolError, PluginTools

        style, out = self.style, self.out
        action, _, rest = argument.strip().partition(" ")
        words = rest.split()
        tools = PluginTools()
        try:
            if action == "copy" and len(words) <= 2:
                target = words[0] if words else await self.pick_target(tools)
                if not target:
                    return
                path, members = await run_blocking(tools.copy, target, words[1] if len(words) > 1 else "")
                approvals = await run_blocking(plugins.load_approvals)
                for member in members:
                    approvals.decide(path.name, member, True)
                if tools.copied_digest is not None:
                    approvals.decide_code(path.name, tools.copied_digest, True)
                await run_blocking(plugins.save_approvals, approvals)
                print(f"  Wrote {', '.join(members)} to {path}; overrides approved. "
                      "Use /plugins to approve later code changes.",
                      file=out)
                await self.apply()
            elif action == "diff" and len(words) <= 1:
                lines = await run_blocking(tools.diff, words[0] if words else "")
                for line in lines or ["No override has changed upstream."]:
                    print("  " + (style.green(line) if line.startswith("+") and not line.startswith("+++")
                                  else style.red(line) if line.startswith("-") and not line.startswith("---")
                                  else line), file=out)
            elif action == "restamp" and 1 <= len(words) <= 2:
                members = await run_blocking(tools.restamp, words[0], words[1] if len(words) > 1 else "")
                print(f"  Restamped {', '.join(members)}." if members else "  Every override is already current.",
                      file=out)
                if members:
                    await self.apply()
            else:
                print("  usage: /plugin copy [module.Class[.method] [file]] | /plugin diff [file] | "
                      "/plugin restamp <file> [method]", file=out)
        except (PluginToolError, OSError) as exc:
            print(style.red(f"  {exc}"), file=out)

    async def pick_target(self, tools: Any) -> str:
        """Choose a class, then the whole class or one method, from the application."""
        terminal = self.session.renderer.terminal
        if terminal is None:
            print("  usage: /plugin copy <module.Class>[.method] [file]", file=self.out)
            return ""
        targets = await run_blocking(tools.targets)
        target = await terminal.choose("Copy into a plugin", [(target, target) for target in targets])
        if target is None:
            return ""
        methods = await run_blocking(tools.methods, target)
        method = await terminal.choose(f"Copy from {target}", [("", "Whole class"), *((name, name) for name in methods)])
        if method is None:
            return ""
        return f"{target}.{method}" if method else target

    def summary(self, report: plugins.PluginReport, approvals: plugins.Approvals) -> str:
        if report.code_state != "on":
            return "Python code awaiting approval" if report.code_state == "pending" else "Python code rejected"
        decisions = [approvals.decision(report.name, override.member_key) for override in report.overrides]
        parts = [f"{decisions.count(True)} of {len(decisions)} allowed"]
        if None in decisions:
            parts.append(f"{decisions.count(None)} undecided")
        return ", ".join(parts)

    async def approve(self, terminal: TerminalUI, approvals: plugins.Approvals, report: plugins.PluginReport,
                      overrides: list[plugins.Override]) -> bool:
        listed = ", ".join(override.member_key for override in overrides)
        action = await terminal.choose(f"Plugin {report.name} → {listed}", [
            ("all", "Allow all"), ("choose", "Choose"), ("none", "None"),
        ])
        if action is None:
            return False
        if action != "choose":
            for override in overrides:
                approvals.decide(report.name, override.member_key, action == "all")
            if action == "none" and not any(approvals.decision(report.name, override.member_key)
                                             for override in report.overrides):
                approvals.decide_code(report.name, report.digest, False)
            return True
        changed = False
        for override in overrides:
            current = approvals.decision(report.name, override.member_key)
            answer = await terminal.choose(f"{report.name} → {override.member_key}", [
                ("allow", "Allow" + (" (current)" if current is True else "")),
                ("reject", "Reject" + (" (current)" if current is False else "")),
            ])
            if answer is None:
                break
            approvals.decide(report.name, override.member_key, answer == "allow")
            changed = True
        return changed

    async def resolve(self, terminal: TerminalUI, approvals: plugins.Approvals,
                      reports: list[plugins.PluginReport], *, only_new: bool) -> bool:
        """Choose one plugin, or a stacking order, for each method several approved overrides target."""
        groups: dict[str, list[tuple[str, plugins.Override]]] = {}
        for report in reports:
            for override in report.overrides:
                if approvals.decision(report.name, override.member_key):
                    groups.setdefault(override.label, []).append((report.name, override))
        changed = False
        for label, entries in groups.items():
            names = [name for name, _ in entries]
            current = approvals.order(label, names)
            if len(names) < 2 or only_new and current is not None:
                continue
            calls = {name: override.calls_original for name, override in entries}
            options = [(name, f"Only {name}" + (" (current)" if current == [name] else "")) for name in names]
            # Every override but the last must call original(), or the ones after it never run.
            orders = itertools.permutations(names) if len(names) <= 3 else iter((names, names[::-1]))
            for order in map(list, orders):
                if all(calls[name] for name in order[:-1]):
                    options.append(("/".join(order), "Stacked: " + ", then ".join(order)
                                    + (" (current)" if current == order else "")))
            choice = await terminal.choose(f"{label} is overridden by {', '.join(names)}", options)
            if choice is not None:
                approvals.choose(label, names, choice.split("/"))
                changed = True
        return changed


class _RendererStream(io.TextIOBase):
    """A write-only stream that routes whole lines through the renderer.

    Slash-command handlers print with `file=`, so this lets them keep that shape
    while every line they emit still steps around the live prompt.
    """

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer
        self._pending = ""
        self._block = object()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        # print() splits a line into the text and its newline, so only whole
        # lines are emitted; the tail waits for the rest of the line to arrive.
        self._pending += text
        *lines, self._pending = self._pending.split("\n")
        for line in lines:
            self._renderer.emit(line, block=self._block)
        return len(text)

    def flush(self) -> None:
        if self._pending:
            self._renderer.emit(self._pending, block=self._block)
            self._pending = ""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slipagent",
        usage="slipagent [options] [prompt ...]",
        description="An agentic coding harness with selectable API providers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  slipagent                       start the interactive REPL\n"
            "  slipagent 'add tests for foo'   run a single task and exit\n"
            "  slipagent -p 'explain this repo' --model openai/gpt-5\n"
            "  slipagent --list-models         browse configured providers' catalogs\n"
        ),
    )
    parser.add_argument("prompt", nargs="*", help="Task to run, then exit.")
    parser.add_argument("-p", "--prompt", dest="prompt_flag",
                        help="Task to run, then exit.")
    parser.add_argument("-w", "--workspace", default=None,
                        help="Project root and default path boundary (lifted by --danger). Default: cwd.")
    parser.add_argument("-m", "--model", help="Select and remember a model in manual mode (default: optimal free-model selection).")
    parser.add_argument("--provider", choices=list(PROVIDERS), help="Select an API provider in manual mode (or set SLIPAGENT_PROVIDER for the initial provider).")
    parser.add_argument("--resume", nargs="?", const="latest", help="Restore a project session by full ID, or the latest saved session.")
    parser.add_argument("--danger", action="store_true",
                        help="Disable built-in workspace path confinement without a confirmation prompt.")
    parser.add_argument("--update", action="store_true",
                        help="Download the latest SlipAgent from GitHub and run its installer, which checks for local edits first.")
    parser.add_argument("--list-models", action="store_true",
                        help="List catalogs from providers with configured API keys and exit.")

    display = parser.add_argument_group("display and sessions")
    display.add_argument("--theme", choices=list(THEMES), default="dark", help="Markdown theme (default: dark).")
    display.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")
    display.add_argument("--markdown", action="store_true", help="Preserve Markdown source instead of readable plain output.")
    display.add_argument("-v", "--verbose", action="store_true",
                         help="Show full tool output and per-step token usage.")
    display.add_argument("--no-session", action="store_true", help="Keep conversation and command logs only until reset or exit.")

    automation = parser.add_argument_group("automation")
    automation.add_argument("--prompt-file", metavar="PATH", help="Read a UTF-8 one-shot prompt verbatim; use - for stdin.")
    automation.add_argument("--json-events", action="store_true",
                            help="One-shot automation: stream JSON Lines events and final status on stdout.")
    automation.add_argument("--max-steps", type=int, default=None,
                            help="Cap on model requests per run, including response retries. Default: 200.")
    automation.add_argument("--tools", help="Comma-separated tool allowlist; recall_history and enabled planning remain available.")
    automation.add_argument("--mcp", action="store_true",
                            help="Force connecting to MCP servers even in one-shot mode.")
    automation.add_argument("--no-mcp", action="store_true",
                            help="Do not connect to globally configured MCP servers.")

    advanced = parser.add_argument_group("advanced", "Model tuning and connection overrides; the defaults suit most use.")
    advanced.add_argument("--api-key", help="Selected provider API key (or set its API_KEY environment variable).")
    advanced.add_argument("--base-url", help="Override the API base URL.")
    advanced.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                          help="Working-model temperature override when supported; omitted by default. Compression always omits it.")
    advanced.add_argument("--max-tokens", type=int, default=None,
                          help="Cap on completion tokens per response.")
    advanced.add_argument("--context-tokens", type=int, help="Working context cap including output reserve (at least 4096; never expands the endpoint limit).")
    advanced.add_argument("--reasoning-history-steps", type=int, choices=range(26), default=25,
                          metavar="0..25", help="Recent steps whose archived thoughts may be supplied; 0 keeps outcomes only (default 25).")
    advanced.add_argument("--raw-reasoning-history", action="store_true", help="Compare original thoughts instead of filtered thoughts in Overthinking history.")
    advanced.add_argument("--planning", action=argparse.BooleanOptionalAction, default=True,
                          help="Expose a persistent working-plan tool for multi-step tasks (default: enabled).")
    advanced.add_argument("--hashline", action=argparse.BooleanOptionalAction, default=True,
                          help="Offer optional hash-anchored read/edit tools (default: enabled; --no-hashline removes their guidance).")
    advanced.add_argument("--python", default=None,
                          help="Project Python interpreter path, overriding .slipagent/project.json and venv discovery.")
    advanced.add_argument("--no-reload", action="store_true", help="Disable automatic component reloads.")
    advanced.add_argument("--no-plugins", action="store_true",
                          help="Skip every plugin in the plugins folder for this run (also SLIPAGENT_NO_PLUGINS=1).")
    advanced.add_argument("--all-plugins", action="store_true",
                          help="Apply every plugin override for this run without asking, stacking in file-name order.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.update:
        from .updater import update
        return update()
    try:
        Path.cwd()
    except OSError as exc:
        print(literal(f"slipagent: current directory is unavailable: {exc}. "
                      "Run cd to an existing directory, then start SlipAgent again."), file=sys.stderr)
        return 2

    # argparse allows both a positional prompt and -p; prefer the explicit flag.
    prompt = args.prompt_flag or " ".join(args.prompt).strip() or None
    if args.prompt_file is not None:
        if args.prompt_flag is not None or args.prompt:
            parser.error("--prompt-file cannot be combined with -p or a positional prompt.")
        try:
            if args.prompt_file == "-":
                prompt = sys.stdin.buffer.read().decode("utf-8")
            else:
                with Path(args.prompt_file).open(encoding="utf-8", newline="") as source:
                    prompt = source.read()
        except (OSError, UnicodeError) as exc:
            parser.error(f"Cannot read prompt: {exc}")
        if not prompt.strip():
            parser.error("--prompt-file requires a nonempty prompt.")
    if args.json_events and (not prompt or args.list_models):
        parser.error("--json-events requires a one-shot prompt and cannot be combined with --list-models.")

    try:
        return asyncio.run(Launcher(args).dispatch(prompt))
    except PromptError as exc:
        print(literal(f"slipagent: {exc}"), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


FIRST_RUN_PROVIDERS = (
    ("openrouter", "OpenRouter", "https://openrouter.ai/keys"),
    ("nvidia", "NVIDIA", "https://build.nvidia.com"),
    ("google", "Google Gemini", "https://aistudio.google.com/apikey"),
    ("mistral", "Mistral", "https://console.mistral.ai/api-keys"),
    ("groq", "Groq", "https://console.groq.com/keys"),
    ("kilo", "Kilo", "https://kilo.ai"),
)


if __name__ == "__main__":
    raise SystemExit(main())
