"""Command-line interface.

Two modes:
  slipagent "fix the failing test"   one-shot; final answer on stdout, progress on stderr
  slipagent                         interactive REPL
"""

from __future__ import annotations

from .commands import BY_NAME, CommandContext, all_help, command_help, menu_options, banner_commands
from .prompts import PromptError, load_prompt
from .markdown import THEMES, MarkdownRenderer, literal

import argparse
import asyncio
import re
import getpass
import io
import json
import math
import os
import shutil
import sys
import time
import threading
import shlex
from dataclasses import dataclass
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, TextIO, cast

from .agent import Agent, AgentEvent, STEP_LIMIT_NOTICE, STOP_NOTICE, build_system_prompt
from .config import Config, ConfigError, configured_provider, dotenv_path, save_dotenv_value, save_model_choice
from .instructions import load_project_instructions, ProjectInstructions
from .lifecycle import finish_cleanup
from .mcp import MCPManager, MCPError, ServerSpec, config_path, load_servers
from .api import (
    DEFAULT_TEMPERATURE,
    APIResponseError,
    APIClient,
    APIConfigError,
    APIError,
    APILimitError,
)
from .providers import PROVIDERS, active_providers, create_client, provider_class, fetch_catalog
from .model_catalog import coding_models
from .tools.base import ToolRegistry, ToolResult
from .tools import build_default_registry
from .types import KeyInfo, Message, ModelInfo, ToolCall
from .workspace import Workspace, WorkspaceError
from .terminal import TerminalUI
from .palette import ERROR_COLOR, MUTED_COLOR, THOUGHT_COLOR, USER_BACKGROUND_COLOR, USER_COLOR, USER_TEXT_COLOR, foreground_code
from .runtime import RuntimeFrame
from .sessionstate import Session as Session
from .sessions import SessionJournal, SessionError, session_title
from wcwidth import iter_graphemes, strip_sequences, wcswidth
from tabulate import tabulate

BANNER = """SlipAgent — coding agent with selectable API providers

  Model:     {model}

  Workspace: {workspace}

  Tools:     {tools}

{mcp}  Commands:  {commands}

Type a task and press Enter. Follow-ups queue while the agent works.
Settings changes also queue until the current response and tool batch finish.
/stop stops after this step; Esc interrupts now. /exit, /quit, or Ctrl-D quits.
Use /help for command details. Ctrl+\\ toggles the context view.""".replace("{commands}", banner_commands())

QUOTA_REFRESH_INTERVAL = 15 * 60

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
        return self._wrap("38;2;255;0;255", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


def _use_color(stream: TextIO, force_off: bool) -> bool:
    if force_off or os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and bool(stream.isatty())


def _literal_tool_output(text: str) -> str:
    """Keep captured terminal controls inert in both live and restored output."""
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", lambda match: f"\\x{ord(match[0]):02x}", text)


COMMAND_OUTPUT_TOOLS = frozenset({
    "run_command", "read_command_output", "command_jobs",
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
        # The live prompt line: the text drawn on it, so output arriving mid-step
        # can be written around it and the prompt put back. Holding the whole
        # line is what stops a redraw from erasing a line the agent has just
        # written.
        self._prompt: tuple[str, str] | None = None
        self.terminal: TerminalUI | None = None
        self.markdown_theme = "dark"
        self.output_markdown = False
        self.stdout_color = _use_color(sys.stdout, not style.enabled)

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

    def _begin_block(self, block: object | None = None) -> None:
        if block is None or block != self._active_block:
            if not self._last_output_blank:
                self._write("")
            self._active_block = block

    def emit(
        self, text: str = "", *, block: object | None = None, separate: bool = True,
        continuation_indents: dict[int, int] | None = None,
    ) -> None:
        """Separate output units while keeping their lines together."""
        self._finish_stream()
        if separate and strip_sequences(text).strip():
            self._begin_block(block)
        self._write(text, continuation_indents=continuation_indents)

    def show_banner(self, *, model: str, workspace: Path, tools: list[str], mcp: str = "") -> None:
        before, _, after = BANNER.partition("{tools}")
        before = before.format(model=model, workspace=workspace, mcp=mcp)
        after = after.format(model=model, workspace=workspace, mcp=mcp)
        # Keep both lists unbroken so resize can reflow them at full width.
        banner = before + ", ".join(tools) + after
        self.emit(
            banner,
            continuation_indents={index: 0 for index, line in enumerate(banner.splitlines())
                                  if line.startswith(("  Tools:", "  Commands:"))},
        )

    def _finish_stream(self) -> None:
        if getattr(self, "_stream_kind", None) is None:
            return
        if self.terminal is not None and self._stream_kind == "assistant_delta":
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
            self._begin_block(self._model_block)
            if kind == "reasoning_delta":
                text = "Thinking: " + text
        styled = literal(text)
        if kind == "reasoning_delta":
            styled = self.style.thought(text)
        elif kind == "tool_output":
            styled = self.style.dim(text)
        if self.terminal is not None:
            if kind == "assistant_delta":
                self.terminal.write_markdown(text, first=first, final=False)
            else:
                self.terminal.write_chunk(styled, first=first)
        else:
            self.stream.write(styled)
            self.stream.flush()
        self._stream_kind = kind
        self._last_output_blank = False

    def user_prompt(self, text: str, *, queued: bool = False) -> None:
        """Separate each submitted prompt from the surrounding transcript."""
        self._finish_stream()
        self._begin_block()
        block = self.style.user(f"> {text}")
        if queued and self.terminal is not None:
            self._write(block + "\n", user_prompt=True)
            self.terminal.write_queued_notice(self.style.dim("  (queued — the agent is still working)"), text)
            return
        if queued:
            block += "\n" + self.style.dim("  (queued — the agent is still working)")
        self._write(f"{block}\n", user_prompt=True)

    def clear_prompt(self) -> None:
        """Erase the live prompt row, leaving everything above it alone."""
        if self._prompt is None:
            return
        self._erase()
        self._prompt = None

    async def restore_transcript(self, messages: list[Message], pending: list[str], *,
                                 queued_messages: dict[int, str] | None = None) -> None:
        """Display archived originals only; never dispatch their tools or requests."""
        self._finish_stream()
        if self.terminal is not None:
            self.terminal.clear_transcript()
        self._active_block = None
        self._model_block = object()
        self._last_output_blank = True
        restored_step = 0
        restored_calls: dict[str, ToolCall] = {}
        for index, message in enumerate(messages):
            if message.role == "user":
                self.user_prompt(message.content or "", queued=index in (queued_messages or {}))
            elif message.role == "assistant":
                restored_step += 1
                self._model_block = object()
                if message.reasoning:
                    self._stream_text("reasoning_delta", message.reasoning)
                    self._finish_stream()
                if message.content:
                    self.handle(AgentEvent(kind="assistant_text", text=message.content))
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
                if self.terminal is not None and restored_call is not None:
                    self.terminal.finish_call(restored_call, restored_step, result.content, result.is_error)
                if isinstance(saved, dict) and saved.get("tool") == "answer" and not result.is_error:
                    self.handle(AgentEvent(kind="assistant_text", text=result.content["text"]))
                    continue
                if result.content:
                    body = result.text()
                    if self.terminal is not None and restored_call is not None and restored_call.name in COMMAND_OUTPUT_TOOLS:
                        body = "Command failed; see Calls." if result.is_error else "Command completed."
                    body = _literal_tool_output(body)
                    if result.is_error:
                        self.emit(self.style.red(f"  ✗ {body}"), block=self._model_block)
                    else:
                        self.emit(self.style.dim(body), block=self._model_block)
                if isinstance(saved, dict):
                    for reference in saved.get("artifacts", []):
                        self.handle(AgentEvent(kind="artifact", result=ToolResult.ok(reference)))
            if index % 100 == 99:
                await asyncio.sleep(0)
        for text in pending:
            self.user_prompt(text, queued=True)
        if self.terminal is not None:
            self.terminal.scroll_to_end()

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

    def handle(self, event: AgentEvent) -> None:
        if event.kind == "context":
            if self.terminal is not None:
                self.terminal.set_context(event.text)

        elif event.kind == "step_start":
            self._finish_stream()
            if event.step == 1:
                self._model_block = object()

        elif event.kind in {"assistant_delta", "reasoning_delta"}:
            self._stream_text(event.kind, event.text)

        elif event.kind == "stream_end":
            self._finish_stream()

        elif event.kind == "retry":
            self.emit(self.style.red(f"  ✗ {event.text}"))

        elif event.kind == "retry_wait":
            self.retry_countdown(event.text, event.step)

        elif event.kind == "assistant_text":
            if self.show_assistant_text:
                self._finish_stream()
                self._model_block = object()
                if self.terminal is not None:
                    self._begin_block(self._model_block)
                    self.terminal.write_markdown(event.text)
                    self._last_output_blank = False
                else:
                    self.__dict__.setdefault("output_markdown", False)
                    text = literal(event.text) if self.output_markdown else MarkdownRenderer(color=False).plain(event.text)
                    self.emit(text, block=self._model_block)

        elif event.kind == "user_message":
            # Mid-step input the user typed while the agent was working. It is
            # echoed rather than the live prompt, because the prompt they are
            # typing into is still on screen holding the text they just sent.
            self.user_prompt(event.text, queued=True)

        elif event.kind == "user_message_sent":
            if self.terminal is not None:
                self.terminal.queued_prompt_sent(event.text)

        elif event.kind == "user_queue_reset":
            if self.terminal is not None:
                self.terminal.forget_queued_notices()

        elif event.kind == "tool_start" and event.tool_call is not None:
            self._tool_streamed = False
            if self.terminal is not None:
                self.terminal.track_call(event.tool_call, event.step)
            if event.tool_call.name == "answer":
                return
            self.emit(self.style.magenta(f"  ⚙ {event.tool_call.brief()}"), block=self._model_block)
            if self.terminal is not None:
                path = event.tool_call.arguments.get("path")
                if path is None and event.tool_call.name == "list_dir":
                    path = "."
                if isinstance(path, str):
                    self.terminal.write_directory_reference(path)
                    offset = event.tool_call.arguments.get("offset", 1)
                    self.terminal.write_file_reference(path, line=offset if isinstance(offset, int) and offset > 0 else 1)

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
            if self.terminal is not None and event.tool_call is not None:
                self.terminal.finish_call(event.tool_call, event.step, result.content, result.is_error)
            if self.terminal is not None and event.tool_call is not None and event.tool_call.name in COMMAND_OUTPUT_TOOLS:
                body = "Command failed; see Calls." if result.is_error else "Command completed."
                color = self.style.red if result.is_error else self.style.dim
                self.emit(color(f"  {'✗' if result.is_error else '✓'} {body}"), block=self._model_block)
            elif result.is_error:
                # Errors are the most useful signal; always surface them.
                self.emit(self.style.red(f"  ✗ {result.text()}"), block=self._model_block)
            elif event.tool_call is not None and event.tool_call.name == "answer":
                self.handle(AgentEvent(kind="assistant_text", step=event.step, text=result.content["text"]))
            elif self.terminal is not None and event.tool_call is not None and event.tool_call.name in {"write_file", "edit_file"} and self.terminal.has_diff(event.tool_call.id):
                self.emit(self.style.dim("    " + result.text().splitlines()[0]), block=self._model_block)
            elif getattr(self, "_tool_streamed", False):
                self.emit(self.style.dim("  ✓ Command completed."), block=self._model_block)
            elif self.verbose and result.text().strip():
                body = result.text().strip()
                for line in body.splitlines():
                    self.emit(self.style.dim(f"    {line}"), block=self._model_block)
            if self.terminal is not None and event.tool_call is not None and not result.is_error:
                if event.tool_call.name in {"write_file", "edit_file", "copy_path", "move_path"}:
                    for key in ("path", "destination"):
                        target = event.tool_call.arguments.get(key)
                        if isinstance(target, str):
                            self.terminal.write_file_reference(target)
                if event.tool_call.name == "grep":
                    for match in re.finditer(r"^(.+?):(\d+):", result.text(), re.MULTILINE):
                        self.terminal.write_file_reference(match[1], line=int(match[2]))

        elif event.kind == "artifact" and event.result is not None:
            reference = event.result.content
            if self.terminal is not None:
                store = self.terminal.__dict__.get("_artifact_store")
                if reference["kind"] == "diff" and reference["bytes"] <= 16_000 and store is not None:
                    try:
                        record = store.read(reference["record_id"])
                    except (OSError, ValueError, UnicodeError):
                        record = None
                        self.terminal.notify("Could not read saved diff.")
                    if record is not None and len(record["output"].splitlines()) <= 40:
                        for line in record["output"].splitlines():
                            color = self.style.cyan if line.startswith(("--- ", "+++ ", "@@")) else self.style.green if line.startswith("+") else self.style.red if line.startswith("-") else self.style.dim
                            self.emit(color(_literal_tool_output(line)), block=self._model_block)
                self.terminal.write_artifact_reference(reference)
            else:
                self.emit(self.style.dim(f"  Saved {reference['kind']} record {reference['record_id']}."), block=self._model_block)

        elif event.kind == "warning":
            if event.text == STOP_NOTICE and getattr(self, '_queued_settings_pause', False):
                self.emit(self.style.dim("  Current step finished; applying queued commands."))
            else:
                self.emit(self.style.red(f"  Warning: {event.text}"), block=self._model_block)
        elif event.kind == "notice":
            self.emit(self.style.dim(f"  {event.text}"))

        elif event.kind == "step_end":
            if self.verbose and event.usage is not None:
                self.emit(self.style.dim(f"    ({event.usage.summary()})"), block=self._model_block)


# --------------------------------------------------------------------------- #
# Session wiring
# --------------------------------------------------------------------------- #


async def build_session(args: argparse.Namespace) -> Session:
    if getattr(args, "no_session", False) and getattr(args, "resume", None):
        raise ConfigError("--resume cannot be combined with --no-session.")
    config = Config.from_env(
        api_key=args.api_key,
        provider=getattr(args, "provider", None),
        model=args.model,
        base_url=args.base_url,
        workspace=args.workspace,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        context_steps=args.context_steps,
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
        model_error: APIError | None = None
        try:
            await client.list_models()
            if await client.model_capabilities(config.model, refresh=True) is None:
                raise APIConfigError("The API did not supply capabilities to verify the configured model's compatibility.")
        except APIError as exc:
            model_error = exc
        registry = build_default_registry(workspace)
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
            context_tokens=getattr(args, "context_tokens", None),
            reasoning_history_steps=getattr(args, "reasoning_history_steps", 25),
            filtered_thoughts=not getattr(args, "raw_reasoning_history", False),
            exposed_tools=tuple(args.tools.split(',')) if getattr(args, "tools", None) else None,
            system_prompt=build_system_prompt(str(workspace.root)),
            on_event=lambda event: renderer.handle(event),
        )

        # Skip automatic MCP startup for one-shot runs unless --mcp is set.
        mcp: MCPManager | None = None
        if not args.no_mcp and (args.mcp or not args.prompt_flag and not args.prompt):
            manager = MCPManager(workspace.root, registry=registry)
            mcp = manager
            resources.push_async_callback(manager.aclose)
            # A server that cannot start must not block the session, so failures
            # are reported per server and the built-in tools stay available.
            try:
                states = await manager.connect_all()
            except MCPError as exc:
                print(style.red(f"  Warning: mcp: {exc}"), file=sys.stderr)
                states = []
            for state in states:
                if state.status == "error":
                    print(style.red(f"  Warning: mcp: {state.line()}"), file=sys.stderr)

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
        if model_error is None:
            try:
                await agent._context_view(registry.specs(), 0, preview=True)
            except APIError as exc:
                model_error = exc
        if model_error is not None:
            renderer.emit(f"  model check failed: {model_error} Select another model with /models or /model <slug>.")
        if not getattr(args, "no_session", False):
            journal = SessionJournal(str(config.workspace))
            registry.services["session_journal"] = journal
            if getattr(args, "resume", None):
                data = await asyncio.to_thread(journal.load, args.resume)
                journal.restore(agent, data)
                session.extensions["restore_transcript"] = True
                renderer.emit(f"  Resumed {journal.session_id}; no tools were replayed.")
        if not getattr(args, "no_reload", False):
            frame = RuntimeFrame(session, sys.modules[__name__])
            session.reloader = frame
            agent.on_boundary = lambda: frame.checkpoint(boundary=True)
        if args.model and model_error is None:
            try:
                await asyncio.to_thread(save_model_choice, config.model, config.provider)
            except (OSError, ConfigError) as exc:
                renderer.emit(f"  could not remember model choice: {exc}")
        resources.pop_all()
        return session


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


async def run_one_shot(session: Session, prompt: str) -> int:
    """Run a single task. Progress goes to stderr so stdout stays pipeable."""
    session.renderer.show_assistant_text = False
    try:
        answer = await session.agent.run(prompt)
        await session.agent.wait_for_compaction()
    except (APIError, WorkspaceError) as exc:
        print(f"slipagent: {exc}", file=sys.stderr)
        return 1
    finally:
        await _shutdown(session)

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
    return 1 if answer == STEP_LIMIT_NOTICE else 0


async def run_repl(session: Session) -> int:
    style = session.renderer.style
    mcp_line = ""
    if session.mcp is not None and session.mcp.servers:
        connected = sum(
            1 for state in session.mcp.servers.values() if state.status == "connected"
        )
        mcp_line = f"  MCP:       {connected}/{len(session.mcp.servers)} server(s) connected\n\n"
    # Prime the free-call count so the status bar is populated on the first
    # prompt rather than showing a dash until the first step finishes.
    await _refresh_quota(session)
    renderer = session.renderer
    terminal_task: asyncio.Task[None] | None = None
    if sys.stdin.isatty() and renderer.stream.isatty() and os.environ.get("TERM") != "dumb":
        terminal = TerminalUI(
            lambda columns: _status_bar(session, style, columns=columns),
            renderer.stream, color=style.enabled,
            status_suffix=lambda: _danger_suffix(session, style),
            project_directory=session.workspace.root,
        )
        renderer.terminal = terminal
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
        terminal.set_interrupt_handler(lambda: _request_interrupt(session))
        terminal_task = asyncio.create_task(terminal.run())
    renderer.show_banner(
        model=session.agent.model,
        workspace=session.workspace.root,
        tools=session.registry.names,
        mcp=mcp_line,
    )
    if session.extensions.pop("restore_transcript", False):
        await renderer.restore_transcript(session.agent.messages, session.agent.pending,
                                          queued_messages=session.agent.queued_messages)
    quota_task = asyncio.create_task(_poll_quota(session))
    reload_task = asyncio.create_task(session.reloader.watch()) if session.reloader is not None else None

    try:
        while True:
            try:
                line = await _read_line(session, style)
            except EOFError:
                renderer.emit()
                break
            if line is None:
                break
            await _wait_for_interrupt(session)
            line = line.strip()

            if not line:
                continue

            if line.startswith("/"):
                try:
                    if await _run_interactive_command(session, line):
                        break
                except Exception as exc:
                    if session.reloader is not None:
                        session.reloader.report_error(f"Command failed: {type(exc).__name__}: {exc}")
                    else:
                        renderer.emit(style.red(f"  ✗ Command failed: {type(exc).__name__}: {exc}"))
                continue

            # The prompt stays live while the agent works: anything typed in
            # the meantime is queued onto the step instead of being ignored.
            renderer.user_prompt(line)
            try:
                if await _run_request(session, line, style):
                    break
            except KeyboardInterrupt:
                # Preserve completed results; interrupted actions may be partial.
                renderer.emit(style.dim("  interrupted."))
                continue
            except Exception as exc:
                if session.reloader is not None:
                    session.reloader.report_error(str(exc))
                else:
                    renderer.emit(style.red(f"  ✗ {exc}"))
                continue
    finally:
        if reload_task is not None:
            reload_task.cancel()
            await _gather_quietly(reload_task)
        quota_task.cancel()
        await _gather_quietly(quota_task)
        try:
            await _shutdown(session)
        finally:
            if renderer.terminal is not None:
                renderer.terminal.close()
            if terminal_task is not None:
                await terminal_task
            renderer.terminal = None
    return 0


async def _read_line(session: Session, style: Style) -> str | None:
    """Read one line of input, drawing the prompt and readouts around it.

    Returns None at end of input, which the caller treats as EOF.
    """
    if "pending_input_line" in session.extensions:
        return cast(str | None, session.extensions.pop("pending_input_line"))
    renderer = session.renderer
    _refresh_title(session)
    renderer.draw_prompt(_status_bar(session, style), style.user("> "))
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


def _fit_readout(text: str, columns: int, *, keep_end: bool = False) -> str:
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


def _status_bar(session: Session, style: Style, *, columns: int | None = None) -> str:
    """Bottom readout: workspace CWD, active model, and free calls left."""
    cwd = _cwd_readout(session.workspace)
    quota = session.free_calls
    free = f"Free: {'—' if quota is None else quota}"
    model = f"Model: {session.agent.model}"
    if isinstance(getattr(session, "client", None), APIClient):
        model += f" ({session.client.provider})"
    if columns is None:
        columns = shutil.get_terminal_size(fallback=(0, 0)).columns
    danger = _danger_suffix(session, Style(False))
    available = max(0, columns - wcswidth(danger)) if columns else 0
    if columns:
        room = available - wcswidth(model + free + _READOUT_SEP * 2)
        cwd = _fit_readout(cwd, room, keep_end=True) if room >= 4 else ""
        if not cwd:
            model = _fit_readout(model, available - wcswidth(free + _READOUT_SEP))
    parts = [part for part in (cwd, model, free) if part]
    return (style.dim(_fit_readout(_READOUT_SEP.join(parts), available) if columns else _READOUT_SEP.join(parts))
            + style.red(_fit_readout(danger, columns, keep_end=True) if columns else danger))


def _danger_suffix(session: Session, style: Style) -> str:
    return style.red(" | Danger Mode") if session.workspace.access.danger else ""


def _cwd_readout(workspace: Workspace) -> str:
    """Show the tools' working directory, abbreviating the home directory."""
    cwd = workspace.root
    # Abbreviate the home prefix while retaining the workspace suffix.
    home = Path.home()
    if cwd == home:
        return "~"
    try:
        return "~/" + cwd.relative_to(home).as_posix()
    except ValueError:
        return str(cwd)


async def _refresh_quota(session: Session) -> None:
    frame = session.reloader
    if frame is not None:
        frame.busy += 1
    try:
        await _fetch_quota(session)
    finally:
        if frame is not None:
            frame.busy -= 1


async def _fetch_quota(session: Session) -> None:
    """Fetch the authoritative free-call quota at most every fifteen minutes."""
    now = time.monotonic()
    if session.quota_checked_at is not None and now - session.quota_checked_at < QUOTA_REFRESH_INTERVAL:
        return
    session.quota_checked_at = now
    client = session.client
    try:
        info = await client.key_info()
    except (APIError, ConfigError):
        return
    # A response for the previous key must not overwrite the new key's quota.
    if session.client is client and info is not None:
        _cache_quota(session, info)


async def _poll_quota(session: Session) -> None:
    """Refresh independently of user input and model requests."""
    while True:
        checked = session.quota_checked_at
        delay = 0.0 if checked is None else max(
            0.0, QUOTA_REFRESH_INTERVAL - (time.monotonic() - checked),
        )
        await asyncio.sleep(delay)
        await _refresh_quota(session)
        if session.renderer.terminal is not None:
            session.renderer.terminal.app.invalidate()


async def _run_request(session: Session, prompt: str, style: Style) -> bool:
    """Run a task while keeping the prompt usable in the background.

    A reader task stays alive for the duration: lines that arrive mid-step are
    queued onto the agent, and `/exit` or `/quit` finishes the step before exiting.
    Returns True when the user asked to exit.
    """
    reader: asyncio.Task[str | None] | None = asyncio.create_task(
        _read_line(session, style)
    )
    agent_task: asyncio.Task[str] = asyncio.create_task(session.agent.run(prompt))
    session.work.agent_task = agent_task
    renderer = session.renderer
    exiting = False
    if renderer.terminal is not None:
        renderer.terminal.set_interrupt_handler(lambda: _request_interrupt(session))
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
                else:
                    text = line.strip()
                    # Read-only commands stay available during the step.
                    # /exit lets the current work finish before shutdown.
                    if _background_command(text):
                        _start_background_command(session, text)
                    elif text.startswith("/") and await _handle_command(session, text):
                        renderer.emit(style.dim("  exiting."))
                        exiting = True
                        reader = None
                    elif text and not text.startswith("/"):
                        session.agent.enqueue(text)
                    if reader is not None:
                        reader = asyncio.create_task(_read_line(session, style))

            if agent_task in done:
                if session.extensions.get("interrupt_requested"):
                    await _wait_for_interrupt(session)
                    break
                answer = agent_task.result()
                resume, commands_exit = await _apply_deferred_commands(session, exiting=exiting)
                exiting = exiting or commands_exit
                if resume and not exiting and answer == STOP_NOTICE:
                    renderer.emit(style.dim("  Queued commands applied; continuing the current task."))
                    if renderer.terminal is not None:
                        renderer.terminal.set_working(True)
                    agent_task = asyncio.create_task(session.agent.run("", continue_run=True))
                    session.work.agent_task = agent_task
                    continue
                if not exiting and session.agent.pending and answer != STEP_LIMIT_NOTICE and not session.agent.stopped:
                    if renderer.terminal is not None:
                        renderer.terminal.set_working(True)
                    agent_task = asyncio.create_task(session.agent.run(""))
                    session.work.agent_task = agent_task
                    continue
                break
            if reader is None:
                # With no reader, wait for the active run to settle.
                if session.extensions.get("interrupt_requested"):
                    await _wait_for_interrupt(session)
                    break
                await agent_task
    except BaseException as exc:
        agent_task.cancel()
        await _gather_quietly(agent_task)
        # A failed model request still leaves an idle session where queued
        # settings can be applied. Cancellation/exit must not start new work.
        if not isinstance(exc, asyncio.CancelledError):
            await _apply_deferred_commands(session, exiting=exiting)
        if isinstance(exc, asyncio.CancelledError) and session.extensions.get("interrupt_requested"):
            await _wait_for_interrupt(session)
        else:
            raise
    finally:
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
        if exiting and session.agent.pending:
            count = len(session.agent.pending)
            subject = "message was" if count == 1 else "messages were"
            renderer.emit(style.dim(
                f"  {count} queued {subject} not run because the session is exiting."
            ))
        if session.agent.usage.cost is not None:
            renderer.emit(style.dim(f"    ({session.agent.usage.summary()})"), separate=False)

    return exiting


def _request_interrupt(session: Session) -> None:
    """Signal cancellation immediately; retain cleanup outside the input loop."""
    cleanup = session.extensions.get("interrupt_task")
    if cleanup is not None:
        return
    session.extensions["interrupt_requested"] = True
    session.extensions["interrupt_sequence"] = session.extensions.get("interrupt_sequence", 0) + 1
    session.registry.services["interrupt_requested"] = True
    session.extensions.pop("deferred_commands", None)
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


async def _wait_for_interrupt(session: Session) -> None:
    task = session.extensions.get("interrupt_task")
    if task is not None:
        await finish_cleanup(task)
        session.extensions.pop("interrupt_task", None)
        session.extensions.pop("interrupt_requested", None)


async def _run_interactive_command(session: Session, line: str) -> bool:
    """Keep commands such as model selection cancellable from the terminal."""
    task = asyncio.create_task(_handle_command(session, line))
    session.work.command_task = task
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if session.extensions.get("interrupt_requested"):
            await _wait_for_interrupt(session)
            return False
        task.cancel()
        await _gather_quietly(task)
        raise
    finally:
        session.work.command_task = None


def _command_help(argument: str) -> str:
    return command_help(argument)


def _changes_session(command: str, argument: str) -> bool:
    definition = BY_NAME.get(command)
    return definition.changes_session(argument) if definition is not None else False


def _queue_command(session: Session, line: str, command: str) -> None:
    commands = session.work.deferred_commands
    if commands is None:
        commands = []
        session.work.deferred_commands = commands
    if not commands:
        session.extensions["resume_after_commands"] = not session.agent.stop_requested
    commands.append(line)
    session.renderer._queued_settings_pause = bool(session.extensions.get("resume_after_commands"))
    session.agent.request_stop()
    if session.renderer.terminal is not None:
        session.renderer.terminal.set_working(True, stopping=True)
    session.renderer.emit(session.renderer.style.dim(
        f"  Queued /{command}; it will run after the current response and tool batch."
    ))


async def _apply_deferred_commands(session: Session, *, exiting: bool = False) -> tuple[bool, bool]:
    """Apply FIFO commands while idle; task replacements do not resume old work.

    Command text stays outside model history and journals, including credentials.
    The active response and every tool have settled before this is called.
    """
    commands = session.work.deferred_commands or []
    session.work.deferred_commands = None
    interrupt_sequence = session.extensions.get("interrupt_sequence", 0)
    resume = bool(session.work.resume_after_commands)
    session.work.resume_after_commands = None
    session.renderer._queued_settings_pause = False
    if exiting:
        if commands:
            session.renderer.emit(session.renderer.style.dim(
                f"  {len(commands)} queued command(s) were not run because the session is exiting."
            ))
        return False, True
    for line in commands:
        parts = line[1:].split(None, 1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        if BY_NAME.get(command) is not None and BY_NAME[command].replaces_task(argument):
            resume = False
        try:
            if session.renderer.terminal is not None:
                session.renderer.terminal.set_working(False)
                session.renderer.terminal.set_command_status(f"Applying /{command} (Esc to interrupt)")
            if await _run_interactive_command(session, line):
                return False, True
            if session.extensions.get("interrupt_sequence", 0) != interrupt_sequence:
                return False, False
        except Exception as exc:
            session.renderer.emit(session.renderer.style.red(f"  ✗ /{command} failed: {exc}"))
        finally:
            if session.renderer.terminal is not None:
                session.renderer.terminal.set_command_status("")
    return resume, False


def _background_command(text: str) -> bool:
    if not text.startswith("/"):
        return False
    parts = text.split(None, 1)
    name = parts[0].lower().removeprefix("/") if parts else ""
    argument = parts[1].strip().lower() if len(parts) > 1 else ""
    command = BY_NAME.get(name)
    return command.background(argument) if command is not None else False


def _start_background_command(session: Session, text: str) -> None:
    tasks = session.work.command_tasks
    if tasks is None:
        tasks = set()
        session.work.command_tasks = tasks
    async def run() -> None:
        try:
            await _handle_command(session, text)
        except Exception as exc:
            session.renderer.emit(session.renderer.style.red(f"  ✗ Command failed: {exc}"))
    task = asyncio.create_task(run())
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def _gather_quietly(*tasks: asyncio.Task[object]) -> None:
    for task in tasks:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def _shutdown(session: Session) -> None:
    """Release every resource the session owns, MCP subprocesses included.

    Order matters: MCP tools hold references to their clients, so the servers
    are stopped before the registry and HTTP client are closed.
    """
    await _wait_for_interrupt(session)
    tasks = list(session.extensions.get("command_tasks", ()))
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


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:10]}...{key[-4:]}"


def _price_cell(pricing: dict[str, Any]) -> str:
    """Render prompt/completion pricing as $/Mtok, or 'free'."""
    try:
        prompt = float(pricing.get("prompt") or 0.0) * 1e6
        completion = float(pricing.get("completion") or 0.0) * 1e6
    except (TypeError, ValueError):
        return "-"
    if prompt == 0 and completion == 0:
        return "free"
    return f"${prompt:.2f}/${completion:.2f}"


def _describe_model(model: ModelInfo) -> str:
    context = f"{model.context_length // 1000}k" if model.context_length else "-"
    price = _price_cell(model.pricing)
    return f"  {model.id:<44} {context:>8}  {price:>15}"


def _model_table(session: Session, models: list[ModelInfo]) -> str:
    terminal = session.renderer.terminal
    columns = terminal.output.get_size().columns if terminal is not None else shutil.get_terminal_size((80, 24)).columns
    rows = [[f"{model.provider}::\n{model.id}", f"{model.context_length:,}" if model.context_length else "-", _price_cell(model.pricing)] for model in models]
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
    return "\n".join("  " + line for line in table.splitlines())


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


async def _handle_command(session: Session, line: str) -> bool:
    terminal = session.renderer.terminal
    if terminal is not None and terminal.__dict__.get("_editor_active"):
        terminal.notify("Close the editor before running commands.")
        return False
    frame = session.reloader
    if frame is not None:
        await frame.checkpoint()
        frame.busy += 1
    try:
        return await _execute_command(session, line)
    finally:
        if frame is not None:
            frame.busy -= 1
            await frame.checkpoint()


def _refresh_title(session: Session) -> None:
    terminal = session.renderer.terminal
    if terminal is not None:
        journal = session.registry.services.get("session_journal")
        title = journal.title if journal is not None else session.extensions.get("title")
        if title is None:
            title = session_title(str(session.workspace.root))
        terminal.set_title(title)


async def _execute_command(session: Session, line: str) -> bool:
    """Dispatch definitions while preserving batch boundaries and finalization."""
    parts = line[1:].split(None, 1)
    name = parts[0].lower() if parts else ""
    argument = parts[1].strip() if len(parts) > 1 else ""
    command = BY_NAME.get(name)
    if command is not None and command.changes_session(argument) and (session.agent.running or session.extensions.get("deferred_commands")):
        _queue_command(session, line, name)
        return False
    if command is None:
        session.renderer.emit(session.renderer.style.red(f"  unknown command: /{name} (try /help)"))
    else:
        result = await command.run(CommandContext(session, session.renderer.style, _RendererStream(session.renderer)), argument)
        if result is not None:
            return result
    _refresh_title(session)
    session.agent._persist()
    return False


async def _model_command(
    session: Session, argument: str, style: Style, out: io.TextIOBase, *, require_free: bool = False,
) -> None:
    if not argument:
        if session.renderer.terminal is not None:
            await _models_command(session, "", style, out)
            return
        print(f"  current model: {style.cyan(session.agent.model)}", file=out)
        print(style.dim("  switch with: /model <slug>   browse with: /models"), file=out)
        return

    try:
        catalog = await session.catalog(refresh=True)
    except (APIError, ConfigError) as exc:
        print(style.red(f"  could not verify against the catalog: {exc}"), file=out)
        return

    matches = [model for model in catalog if argument in (model.id, model.selector)]
    if not matches:
        suggestions = [m for m in catalog if argument.lower() in m.id.lower()][:5]
        print(style.red(f"  unknown model: {argument}"), file=out)
        if suggestions:
            print(style.dim("  did you mean:"), file=out)
            print(_model_table(session, suggestions), file=out)
        return
    if len(matches) > 1:
        print(style.dim("  Select a provider explicitly with /model <provider>::<model>:"), file=out)
        print(_model_table(session, matches), file=out)
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
    print(f"  switched to {style.cyan(entry.selector)}  {_price_cell(entry.pricing)}/Mtok",
          file=out)
    if entry.provider == "openrouter" and _price_cell(entry.pricing) == "free":
        print(style.dim("  note: free models still work with a $0-limit key"), file=out)


async def _temperature_command(
    session: Session, argument: str, style: Style, out: io.TextIOBase,
) -> None:
    value = session.agent.temperature
    if argument:
        try:
            value = float(argument)
        except ValueError:
            value = float("nan")
        if not math.isfinite(value) or not 0 <= value <= 2:
            print(style.red("  temperature must be a number from 0 to 2."), file=out)
            return
    try:
        capabilities = await session.client.model_capabilities(session.agent.model)
    except APIError as exc:
        print(style.red(f"  could not verify temperature support: {exc}"), file=out)
        return
    if capabilities is None or "temperature" not in capabilities.parameters:
        print(style.red(
            f"  temperature control is unavailable for {session.agent.model}'s selected endpoints; "
            "the provider default is used."
        ), file=out)
        return
    if argument:
        session.agent.temperature = value
    from .inference import sampling_defaults
    effective = sampling_defaults(session.agent.model, capabilities).get("temperature", DEFAULT_TEMPERATURE) if value is None else value
    suffix = " (default)" if value is None else ""
    print(f"  temperature: {effective:g}{suffix}", file=out)


async def _models_command(
    session: Session, argument: str, style: Style, out: io.TextIOBase
) -> None:
    try:
        catalog = await session.catalog()
    except (APIError, ConfigError) as exc:
        print(style.red(f"  could not load the catalog: {exc}"), file=out)
        return

    for error in getattr(session, "extensions", {}).get("catalog_errors", []):
        print(style.red(f"  catalog unavailable: {error}"), file=out)

    needle = argument.strip().lower()
    if needle in ("free", "0"):
        matches = [m for m in catalog if _price_cell(m.pricing) == "free"]
    elif needle:
        words = needle.split()
        matches = [m for m in catalog
                   if all(word in f"{m.selector} {m.name or ''}".lower() for word in words)]
    else:
        matches = list(catalog)

    if not matches:
        print(f"  no models matching {argument!r}", file=out)
        return

    terminal = session.renderer.terminal
    if terminal is not None:
        options = [
            (model.selector, f"{model.selector}  {_price_cell(model.pricing)}/Mtok  "
             "context: " + (f"{model.context_length:,}" if model.context_length else "unknown") +
             ("  (current)" if model.id == session.agent.model and model.provider == getattr(getattr(session, "client", None), "provider", "openrouter") else ""))
            for model in sorted(matches, key=lambda model: (model.id, model.provider))
        ]
        selected = await terminal.choose("Select Model", options)
        if selected is not None:
            await _model_command(session, selected, style, out)
        return

    free_count = sum(1 for m in catalog if _price_cell(m.pricing) == "free")
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

    print(_model_table(session, sorted(matches, key=lambda m: m.id)), file=out)


def _cache_quota(session: Session, info: KeyInfo) -> None:
    """Remember the key's remaining free calls for the status bar."""
    session.free_calls = info.free_quota.remaining if info.free_quota is not None else None
    session.quota_checked_at = time.monotonic()


def _describe_key_info(info: KeyInfo, style: Style, out: io.TextIOBase) -> None:
    """Print a key's limits so the user knows what it can actually reach."""
    if info.label:
        print(f"    label      {info.label}", file=out)
    if info.limit is not None:
        remaining = info.limit_remaining
        detail = f"${info.limit:.2f} limit"
        if remaining is not None:
            detail += f", ${remaining:.2f} remaining"
        print(f"    spend      {detail}", file=out)
    print(f"    usage      ${info.usage:.2f} to date", file=out)
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


async def _key_command(
    session: Session, argument: str, style: Style, out: io.TextIOBase
) -> None:
    if argument.lower() in ("show", "status", ""):
        if argument == "" and sys.stdin.isatty():
            # Bare /key on a terminal starts the interactive replacement flow.
            await _replace_key(session, style, out)
            return
        print(f"  key: {style.cyan(_mask_key(session.api_key))}", file=out)
        try:
            info = await session.client.key_info()
        except APIError as exc:
            print(style.red(f"  could not verify key: {exc}"), file=out)
            return
        if info is None:
            print(style.dim("  This provider has no key-status API; authentication is checked on inference."), file=out)
        else:
            _cache_quota(session, info)
            _describe_key_info(info, style, out)
        return

    # Anything else is treated as a key supplied directly.
    await _replace_key(session, style, out, candidate=argument)


async def _replace_key(
    session: Session,
    style: Style,
    out: io.TextIOBase,
    candidate: str | None = None,
) -> None:
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
        print(style.dim(f"  Enter a new {session.client.provider} API key (input hidden)."), file=out)
        try:
            candidate = (await _ask_input(session, "  key: ", password=True)).strip()
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            return
        if not candidate:
            print(style.dim("  cancelled"), file=out)
            return

    previous = session.api_key
    try:
        await session.use_api_key(candidate)
    except APIConfigError as exc:
        print(style.red(f"  {exc}"), file=out)
        return

    # Verify through account metadata when the selected provider supports it.
    try:
        info = await session.client.key_info()
    except (APIError, ConfigError) as exc:
        await session.use_api_key(previous)
        print(style.red(f"  rejected: {exc}"), file=out)
        return

    if info is None:
        print(style.dim("  key updated for this session; authentication will be checked on inference."), file=out)
    else:
        print(f"  key accepted: {style.cyan(_mask_key(candidate))}", file=out)
        _cache_quota(session, info)
        _describe_key_info(info, style, out)

    # Persisting replaces a credential on disk, so require explicit consent.
    if not sys.stdin.isatty():
        print(style.dim("  not saved (no interactive terminal)"), file=out)
        return
    answer = await _ask_input(session, "  save to .env for future runs? [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        print(style.dim("  kept for this session only"), file=out)
        return

    path = dotenv_path()
    try:
        save_dotenv_value(path, session.client.key_env, candidate)
        print(style.dim(f"  saved to {path}"), file=out)
    except (OSError, ConfigError) as exc:
        print(style.red(f"  could not save to {path}: {exc}"), file=out)


async def _ask_input(session: Session, prompt: str, *, password: bool = False) -> str:
    if session.renderer is not None and session.renderer.terminal is not None:
        return await session.renderer.terminal.ask(prompt, password=password)
    reader = getpass.getpass if password else input
    return await asyncio.to_thread(reader, prompt)


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #


def _split_mcp_command(argument: str) -> tuple[str, ServerSpec | None, str | None]:
    """Parse `add|save|remove <name> <command> [args...]`.

    Returns the subcommand, the server spec to connect, and the name to remove.
    The command and its arguments are split on the first space after the name,
    so `npx -y @scope/server --flag` stays one command with flags intact.
    """
    parts = argument.split(None, 2)
    if not parts:
        return "", None, None

    action = parts[0].lower()
    if action == "remove" and len(parts) >= 2:
        return action, None, parts[1]

    if action not in ("add", "save") or len(parts) < 3:
        return action, None, None

    name, rest = parts[1], parts[2].strip()
    if not name or not rest:
        return action, None, None
    command_parts = shlex.split(rest)
    if not command_parts:
        return action, None, None

    return action, ServerSpec(name=name, command=command_parts[0], args=command_parts[1:]), None


async def _mcp_command(
    session: Session, argument: str, style: Style, out: io.TextIOBase
) -> None:
    if session.mcp is None:
        print(style.dim("  MCP is disabled for this run (--no-mcp)."), file=out)
        return

    try:
        action, spec, removal = _split_mcp_command(argument)
    except ValueError as exc:
        print(style.red(f"  invalid MCP command: {exc}"), file=out)
        return

    if not action:
        _describe_mcp(session, style, out)
        return

    if action == "remove":
        if removal is None:
            print(style.red("  usage: /mcp remove <name>"), file=out)
            return
        live = removal in session.mcp.servers
        try:
            saved = session.mcp.remove(removal)
        except (MCPError, OSError) as exc:
            print(style.red(f"  could not remove server: {exc}"), file=out)
            return
        await session.mcp.disconnect(removal)
        session.mcp.servers.pop(removal, None)
        if not live and not saved:
            print(style.red(f"  no MCP server named {removal!r}"), file=out)
        else:
            print(
                style.dim(f"  removed {removal} from {config_path(session.workspace.root)}"),
                file=out,
            )
        return

    if action not in ("add", "save") or spec is None:
        print(
            style.red(f"  usage: /mcp add <name> <command> [args...]  "
                      f"(or /mcp save ...)"),
            file=out,
        )
        return

    try:
        state = await session.mcp.connect(spec.name, spec)
    except MCPError as exc:
        print(style.red(f"  {exc}"), file=out)
        return

    if action == "save":
        try:
            session.mcp.add(spec.name, spec)
        except (OSError, MCPError) as exc:
            print(style.red(f"  connected, but could not save config: {exc}"), file=out)
        else:
            print(
                style.dim(f"  saved to {config_path(session.workspace.root)}"), file=out
            )

    if state.status == "error":
        print(style.red(f"  Warning: {state.line()}"), file=out)
        return

    print(f"  {style.green('connected')} {state.spec.name} — {state.detail}", file=out)
    for qualified in state.qualified:
        print(f"    {qualified}", file=out)
    if not state.qualified:
        print(style.dim("    (no tools)"), file=out)


async def _init_command(session: Session, style: Style, out: io.TextIOBase) -> None:
    """Create missing project guidance and apply it without restarting."""
    agents_md_path = session.workspace.resolve("AGENTS.md")
    content = """# Project Guidance

Repository-specific development notes. Unfilled sections are placeholders, not established facts or runnable commands.

## Purpose and Architecture
Not documented yet: project purpose, module responsibilities, and key invariants.

## Environment and Dependencies
Not documented yet: the project interpreter/toolchain and dependency setup.

## Build and Verification
Not documented yet: exact build, test, lint, and type-check commands.

## Local Conventions
Not documented yet: repository-specific coding conventions and restrictions.
"""
    try:
        # Exclusive creation preserves existing guidance and refuses new symlinks.
        with agents_md_path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        if not agents_md_path.is_file():
            raise WorkspaceError("AGENTS.md must be a regular file")
        action = "AGENTS.md already exists"
    else:
        action = "created AGENTS.md"

    load_project_instructions(session.workspace)
    session.agent.registry.services.setdefault("project_instructions", ProjectInstructions(session.workspace))
    prompt = build_system_prompt(str(session.workspace.root))
    session.agent.system_prompt = prompt
    if session.agent.messages and session.agent.messages[0].role == "system":
        session.agent.messages[0] = Message.system(prompt)
    else:
        session.agent.messages.insert(0, Message.system(prompt))
    # Reloads rebuild the harness prompt around this updated project guidance.
    if session.reloader is not None:
        session.reloader._project_instructions = prompt
    print(f"  {action}; project guidance loaded", file=out)


def _describe_mcp(session: Session, style: Style, out: io.TextIOBase) -> None:
    """List configured servers, live connections, and contributed tools."""
    manager = session.mcp
    assert manager is not None

    error = manager.config_error()
    if error:
        print(style.red(f"  {error}"), file=out)
        return

    try:
        configured = load_servers(manager.workspace)
    except MCPError as exc:
        print(style.red(f"  {exc}"), file=out)
        return

    live = {name: state for name, state in manager.servers.items()}
    if not configured and not live:
        print(
            style.dim(
                f"  no MCP servers configured ({config_path(manager.workspace)}).\n"
                f"  add one with: /mcp add <name> <command> [args...]"
            ),
            file=out,
        )
        return

    if configured:
        print(
            style.dim(
                f"  {len(live)}/{len(configured)} configured server(s) connected "
                f"({config_path(manager.workspace)})"
            ),
            file=out,
        )
    else:
        print(style.dim("  no servers in .mcp.json; showing this session only"), file=out)

    for name in sorted(set(configured) | set(live)):
        state = live.get(name)
        if state is not None:
            print(f"  {style.cyan(name)}: {state.status} — {state.detail}", file=out)
            for qualified in state.qualified:
                print(f"    {qualified}", file=out)
        else:
            spec = configured[name]
            marker = "not connected" if name in configured else "session only"
            print(style.dim(f"  {name}: {marker} — {spec.command_line()}"), file=out)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slipagent",
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
    parser.add_argument("--planning", action=argparse.BooleanOptionalAction, default=True,
                        help="Expose a persistent working-plan tool for multi-step tasks (default: enabled).")
    parser.add_argument("--raw-reasoning-history", action="store_true", help="Compare original thoughts instead of filtered thoughts in Overthinking history.")
    parser.add_argument("--context-tokens", type=int, help="Working context cap including output reserve (at least 4096; never expands the endpoint limit).")
    parser.add_argument("--reasoning-history-steps", type=int, choices=range(26), default=25,
                        metavar="0..25", help="Recent steps whose archived thoughts may be supplied; 0 keeps outcomes only (default 25).")
    parser.add_argument("--tools", help="Comma-separated tool allowlist; recall_history and enabled planning remain available.")
    parser.add_argument("prompt", nargs="*", help="Task to run, then exit.")
    parser.add_argument("-p", "--prompt", dest="prompt_flag",
                        help="Task to run, then exit.")
    parser.add_argument("-m", "--model", help="Select and remember a model (default: environment, saved choice, then Nemotron Ultra).")
    parser.add_argument("-w", "--workspace", default=None,
                        help="Project root and default path boundary (lifted by --danger). Default: cwd.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap on model requests per run, including response retries. "
                             "Default: 200.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default follows the known model profile, otherwise 1.0, when supported).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Cap on completion tokens per response.")
    parser.add_argument("--context-steps", type=int, default=None,
                        help="Recent model calls supplied in full (default: 50, minimum: 5 when context permits).")
    parser.add_argument("--python", default=None,
                        help="Project Python interpreter path, overriding .slipagent/project.json and venv discovery.")
    parser.add_argument("--provider", choices=list(PROVIDERS), help="API provider (or set SLIPAGENT_PROVIDER).")
    parser.add_argument("--api-key", help="Selected provider API key (or set its API_KEY environment variable).")
    parser.add_argument("--base-url", help="Override the API base URL.")
    parser.add_argument("--list-models", action="store_true",
                        help="List catalogs from providers with configured API keys and exit.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show full tool output and per-step token usage.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")
    parser.add_argument("--theme", choices=list(THEMES), default="dark", help="Markdown theme (default: dark).")
    parser.add_argument("--markdown", action="store_true", help="Preserve Markdown source instead of readable plain output.")
    parser.add_argument("--no-reload", action="store_true", help="Disable automatic component reloads.")
    parser.add_argument("--resume", nargs="?", const="latest", help="Restore a project session by full ID, or the latest saved session.")
    parser.add_argument("--no-session", action="store_true", help="Keep conversation and command logs only until reset or exit.")
    parser.add_argument("--danger", action="store_true",
                        help="Disable built-in workspace path confinement without a confirmation prompt.")
    parser.add_argument("--no-mcp", action="store_true",
                        help="Do not connect to MCP servers from .mcp.json.")
    parser.add_argument("--mcp", action="store_true",
                        help="Force connecting to MCP servers even in one-shot mode.")
    return parser


async def _list_models(args: argparse.Namespace) -> int:
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
                                    on_retry=renderer.retry_countdown) for name in names),
                                   return_exceptions=True)
    models: list[ModelInfo] = []
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            if not isinstance(result, (APIError, ConfigError)):
                raise result
            print(f"slipagent: {name}: {result}", file=sys.stderr)
        else:
            models.extend(coding_models(result))
    if not models:
        return 1

    style = Style(_use_color(sys.stdout, args.no_color))
    for model in sorted(models, key=lambda m: m.id):
        context = f"  {model.context_length:,} ctx" if model.context_length else ""
        print(f"{model.selector}{style.dim(context)}")
    print(style.dim(f"\n{len(models)} models"), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # argparse allows both a positional prompt and -p; prefer the explicit flag.
    prompt = args.prompt_flag or " ".join(args.prompt).strip() or None

    try:
        return asyncio.run(_dispatch(args, prompt))
    except PromptError as exc:
        print(f"slipagent: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


async def _dispatch(args: argparse.Namespace, prompt: str | None) -> int:
    """Run everything inside a single event loop.

    The httpx client must be created and used in the same loop, so setup and
    execution cannot live in separate `asyncio.run` calls.
    """
    if args.list_models:
        try:
            return await _list_models(args)
        except (ConfigError, APIConfigError) as exc:
            print(f"slipagent: {exc}", file=sys.stderr)
            return 2

    try:
        session = await build_session(args)
    except (ConfigError, APIError) as exc:
        print(f"slipagent: {exc}", file=sys.stderr)
        return 2

    if prompt:
        return await run_one_shot(session, prompt)
    return await run_repl(session)


if __name__ == "__main__":
    raise SystemExit(main())
