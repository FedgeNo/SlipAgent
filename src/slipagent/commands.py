"""Static slash-command definitions: behavior, policy, help, and menu syntax."""

from __future__ import annotations

import asyncio
import io
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .markdown import THEMES
from .providers import PROVIDERS, provider_class
from .sessions import SessionError, session_title
from .checkpoints import CheckpointError
from .task import TaskMemory
from .prompts import load_prompt

if TYPE_CHECKING:
    from .cli import Session, Style
    from .terminal import TerminalUI
    from .sessions import SessionJournal


@dataclass(frozen=True)
class CommandHelp:
    syntax: str
    description: str
    insertion: str

    def line(self) -> str:
        return f"  {self.syntax:<20}  {self.description}"


@dataclass(frozen=True)
class CommandContext:
    session: Session
    style: Style
    out: io.TextIOBase


class Command(ABC):
    name: str
    aliases: tuple[str, ...] = ()
    help: tuple[CommandHelp, ...] = ()
    banner = ""
    def replaces_task(self, argument: str) -> bool:
        return False

    def changes_session(self, argument: str) -> bool:
        return False

    def background(self, argument: str) -> bool:
        return False

    @abstractmethod
    async def run(self, context: CommandContext, argument: str) -> bool | None:
        """Return an explicit exit result, or None for common finalization."""
        raise NotImplementedError


class ExclusiveCommand(Command):
    def changes_session(self, argument: str) -> bool:
        return True


class PresentationCommand(Command):
    async def run(self, context: CommandContext, argument: str) -> bool | None:
        terminal = context.session.renderer.terminal
        if terminal is None:
            print(context.style.red(f"  /{self.name} requires an interactive terminal."), file=context.out)
        else:
            await self.apply(context, argument, terminal)
        return None

    @abstractmethod
    async def apply(self, context: CommandContext, argument: str, terminal: TerminalUI) -> None:
        raise NotImplementedError


class StoredSessionCommand(Command):
    async def run(self, context: CommandContext, argument: str) -> bool | None:
        journal = context.session.registry.resources.journal
        if journal is None:
            print(context.style.red("  session persistence is disabled; restart without --no-session to enable it."), file=context.out)
            return None
        return await self.execute_saved(context, argument, journal)

    @abstractmethod
    async def execute_saved(self, context: CommandContext, argument: str, journal: SessionJournal) -> bool | None:
        raise NotImplementedError


class BranchSessionCommand(Command):
    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        journal = session.registry.resources.journal
        if argument:
            print(style.red(f"  usage: /{self.name}"), file=out)
        elif journal is None:
            print(style.red("  session persistence is disabled; restart without --no-session to enable it."), file=out)
        elif journal.path is None:
            print("  no current saved session; enter a task prompt first.", file=out)
        else:
            if not await self.confirm(context, journal):
                return False
            try:
                await session.agent.wait_for_compaction()
                jobs = session.registry.resources.jobs
                if jobs is not None:
                    await jobs.stop_all()
                session.agent._persist()
                await self.change(context, journal)
            except SessionError as exc:
                print(style.red(f"  {exc}"), file=out)
                return False
        return None

    async def confirm(self, context: CommandContext, journal: SessionJournal) -> bool:
        return True

    @abstractmethod
    async def change(self, context: CommandContext, journal: SessionJournal) -> None:
        raise NotImplementedError


class HelpCommand(Command):
    name = 'help'
    banner = '/help'
    help = (
        CommandHelp('/help [command]', 'show all help, or help for one command (with or without /)', '/help '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        print(cli._command_help(argument), file=out, end="")
        return None


class ConfigShowCommand(Command):
    name = 'config-show'
    help = (
        CommandHelp('/config-show', 'show active interpreter, model, and workspace settings', '/config-show'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        print(f"Active interpreter: {sys.executable}", file=out)
        print(f"Model choice: {session.agent.model}", file=out)
        print(f"Workspace: {session.workspace.root}", file=out)
        print(f"Danger mode: {'on' if session.workspace.access.danger else 'off'}", file=out)
        print(f"Overthinking mode: {'on' if session.agent.overthinking else 'off'}", file=out)
        return None


class MenuCommand(Command):
    name = 'menu'
    banner = '/menu'
    help = (
        CommandHelp('/menu', 'browse commands and insert a selection into the input', '/menu'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        terminal = session.renderer.terminal
        if argument:
            print(style.red("  usage: /menu"), file=out)
        elif terminal is None:
            print(style.red("  /menu requires an interactive terminal; use /help for commands."), file=out)
        else:
            selected = await terminal.choose("Commands", cli.MENU_OPTIONS)
            if selected is not None:
                terminal.insert_command(selected)
        return False


class ViewCommand(Command):
    name = 'view'
    help = (
        CommandHelp('/view [record] [line]', 'inspect saved output or a diff in a read-only viewer', '/view '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        terminal = session.renderer.terminal
        store = session.registry.resources.artifacts
        if terminal is None or store is None:
            print(style.red("  /view requires an interactive terminal with inspection records."), file=out)
            return False
        try:
            values = argument.split()
            if len(values) > 2:
                raise ValueError("usage: /view [record] [line]")
            if values:
                identifier = int(values[0])
                line_number = int(values[1]) if len(values) == 2 else 1
                if line_number < 1:
                    raise ValueError("Line must be positive.")
            else:
                choices = [(str(record["record_id"]), f"{record['record_id']} — {record['kind']} — {record['command']}")
                           for record in store.records.values()]
                if not choices:
                    terminal.notify("No inspection records yet.")
                    return False
                selected = await terminal.choose("Inspection Records", choices)
                if selected is None:
                    return False
                identifier, line_number = int(selected), 1
            await terminal.open_artifact(identifier, line=line_number)
        except ValueError as exc:
            terminal.notify(str(exc))
        return False


class ThemeCommand(PresentationCommand):
    name = 'theme'
    help = (
        CommandHelp('/theme <name>', 'dark, light, monochrome, or ironbow Markdown palette', '/theme '),
    )

    async def apply(self, context: CommandContext, argument: str, terminal: TerminalUI) -> None:
        session, style, out = context.session, context.style, context.out
        if argument not in THEMES:
            print("  usage: /theme " + "|".join(THEMES), file=out)
        else:
            session.renderer.markdown_theme = argument
            terminal.configure_markdown(theme=argument)
            print(f"  Theme: {argument}", file=out)


class MarkdownCommand(PresentationCommand):
    name = 'markdown'
    help = (
        CommandHelp('/markdown [source|rendered]', 'toggle Markdown source view (F2)', '/markdown '),
    )

    async def apply(self, context: CommandContext, argument: str, terminal: TerminalUI) -> None:
        session, style, out = context.session, context.style, context.out
        if argument not in {"", "source", "rendered"}:
            print("  usage: /markdown [source|rendered]", file=out)
        else:
            source = argument == "source" if argument else not terminal._markdown_source
            terminal.configure_markdown(source=source)
            print("  Markdown source" if source else "  Rendered Markdown", file=out)


class CopyCommand(PresentationCommand):
    name = 'copy'
    help = (
        CommandHelp('/copy [auto|text|markdown|code [number]]', 'copy the latest reply (F3)', '/copy '),
    )

    async def apply(self, context: CommandContext, argument: str, terminal: TerminalUI) -> None:
        session, style, out = context.session, context.style, context.out
        copy_options = argument.split()
        mode = copy_options[0] if copy_options else "auto"
        try:
            if mode not in {"auto", "text", "markdown", "code"} or len(copy_options) > 2:
                raise ValueError("usage: /copy [auto|text|markdown|code [number]]")
            if len(copy_options) == 2 and mode != "code":
                raise ValueError("A block number is only valid with /copy code.")
            terminal.copy_response(mode, int(copy_options[1]) if len(copy_options) == 2 else 1)
        except ValueError as exc:
            print(style.red(f"  {exc}"), file=out)
        else:
            print("  Copied to clipboard; terminals without a local clipboard utility must support OSC 52.", file=out)


class DangerCommand(Command):
    name = 'danger'
    banner = '/danger [on|off|status]'
    help = (
        CommandHelp('/danger [on]', 'disable workspace path confinement (queues while working)', '/danger '),
        CommandHelp('/danger off', 'restore workspace path confinement (queues while working)', '/danger off'),
        CommandHelp('/danger status', 'show whether danger mode is active', '/danger status'),
    )

    def changes_session(self, argument: str) -> bool:
        return argument.lower() in {"", "on", "off"}

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if argument.lower() not in {"", "on", "off", "status"}:
            print(style.red("  usage: /danger [on|off|status]"), file=out)
        else:
            if argument.lower() != "status":
                session.workspace.access.danger = argument.lower() != "off"
            if session.workspace.access.danger:
                print(style.red("  Danger mode ON: workspace path confinement is disabled."), file=out)
            else:
                print("  Danger mode OFF: workspace path confinement is enabled.", file=out)
            if session.renderer.terminal is not None:
                session.renderer.terminal.app.invalidate()
        return None


class OverthinkingCommand(Command):
    name = 'overthinking'
    banner = '/overthinking [on|off]'
    help = (
        CommandHelp('/overthinking [on|off]', 'enable or disable recent archived thoughts, up to 25 steps (queues while working)', '/overthinking '),
    )

    def changes_session(self, argument: str) -> bool:
        return argument.lower() in {"", "on", "off"}

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if argument.lower() not in {"", "on", "off"}:
            print(style.red("  usage: /overthinking [on|off]"), file=out)
        else:
            session.agent.overthinking = argument.lower() != "off"
            print(f"  Overthinking Mode {'ON' if session.agent.overthinking else 'OFF'}.", file=out)
        return None


class PlanningCommand(Command):
    name = 'planning'
    banner = '/planning [on|off]'
    help = (
        CommandHelp('/planning [on|off]', 'enable planning (default); off disables it (queues while working)', '/planning '),
    )

    def changes_session(self, argument: str) -> bool:
        return argument.lower() in {"", "on", "off"}

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if argument.lower() not in {"", "on", "off"}:
            print(style.red("  usage: /planning [on|off]"), file=out)
        else:
            session.agent.set_planning(argument.lower() != "off")
            print(f"  Planning {'ON' if session.agent.planning else 'OFF'}.", file=out)
        return None


class HashlineCommand(Command):
    name = 'hashline'
    help = (
        CommandHelp('/hashline [on|off]', 'show or change optional hash-anchored editing (queues while working)', '/hashline '),
    )

    def changes_session(self, argument: str) -> bool:
        return argument.lower() in {'on', 'off'}

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from .tools.hashline import set_hashline

        session, style, out = context.session, context.style, context.out
        value = argument.lower()
        if value not in {'', 'on', 'off'}:
            print(style.red('  usage: /hashline [on|off]'), file=out)
            return False
        if value:
            set_hashline(session.registry, value == 'on')
            frame = session.reloader
            if frame is not None:
                frame._builtin_tools = tuple(tool for tool in session.registry.tools
                                             if tool.name in session.registry.builtin_names)
        enabled = session.registry.services.get('hashline_enabled', True)
        print(f"  Hashline editing {'ON' if enabled else 'OFF'}.", file=out)
        return None


class ToolsCommand(Command):
    name = 'tools'
    banner = '/tools'
    help = (
        CommandHelp('/tools', 'list the available tools', '/tools'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        print("  " + "\n  ".join(session.registry.names), file=out)
        return None


class ModelCommand(ExclusiveCommand):
    name = 'model'
    banner = '/model [slug]'
    help = (
        CommandHelp('/model', 'choose a model with Up/Down and Enter; Escape cancels', '/model'),
        CommandHelp('/model <selector>', 'switch model/provider; use provider::model when ambiguous', '/model '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        await cli._model_command(session, argument, style, out)
        return None


class ModelsCommand(ExclusiveCommand):
    name = 'models'
    banner = '/models [filter]'
    help = (
        CommandHelp('/models [filter]', 'search active provider catalogs (e.g. /models nemotron)', '/models '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        await cli._models_command(session, argument, style, out)
        return None


class OptimalCommand(ExclusiveCommand):
    name = 'optimal'
    banner = '/optimal [on|off|reset]'
    help = (
        CommandHelp('/optimal', 'show automatic free-model selection state', '/optimal'),
        CommandHelp('/optimal on', 'clear failed routes and select the best ranked free model', '/optimal on'),
        CommandHelp('/optimal reset', 'retry from the top rank and clear coding/compression failures', '/optimal reset'),
        CommandHelp('/optimal off', 'keep the current model under manual control', '/optimal off'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from .api import APIError
        routing = context.session.model_routing()
        action = argument.strip().lower()
        if action in {'on', 'reset'}:
            try:
                await routing.enable()
            except APIError as exc:
                print(context.style.red(f"  {exc}"), file=context.out)
        elif action == 'off':
            routing.enabled = False
            print("  Optimal off; the current model remains selected.", file=context.out)
        elif not action:
            session = context.session
            print(f"  Optimal {'on' if routing.enabled else 'off'}: "
                  f"{session.client.provider}::{session.agent.model}", file=context.out)
        else:
            print("  usage: /optimal [on|off|reset]", file=context.out)
        return None


class ProviderCommand(ExclusiveCommand):
    name = 'provider'
    help = (
        CommandHelp('/provider <provider> switch to a free default: openrouter/nvidia Ultra, kilo Auto Free', '', '/provider '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        provider = argument.lower()
        if provider not in PROVIDERS:
            print(style.red("  usage: /provider " + "|".join(PROVIDERS)), file=out)
        else:
            await cli._model_command(session, f"{provider}::{provider_class(provider).default_model}",
                                 style, out, require_free=True)
        return None


class TemperatureCommand(Command):
    name = 'temperature'
    banner = '/temperature [value]'
    help = (
        CommandHelp('/temperature', 'show the explicit temperature or provider default and model support', '/temperature'),
        CommandHelp('/temperature <value> set working-model temperature from 0 to 2 when supported', '', '/temperature '),
    )

    def changes_session(self, argument: str) -> bool:
        return bool(argument)

    def background(self, argument: str) -> bool:
        return not argument

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        await cli._temperature_command(session, argument, style, out)
        return None


class KeyCommand(Command):
    name = 'key'
    banner = '/key [show|status|[provider] key]'
    help = (
        CommandHelp('/key', 'enter an API key with hidden input and detect its provider', '/key'),
        CommandHelp('/key show', 'show the current key, masked', '/key show'),
        CommandHelp('/key status', 'same as /key show', '/key status'),
        CommandHelp('/key <key>', 'detect the provider, install its key, and offer to save it', '/key '),
        CommandHelp('/key <provider> <key>', 'install a key with an explicit provider', '/key '),
    )

    def changes_session(self, argument: str) -> bool:
        return argument.lower() not in {"show", "status"}

    def background(self, argument: str) -> bool:
        return argument.lower() in {"show", "status"}

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        await cli._key_command(session, argument, style, out)
        return None


class CostCommand(Command):
    name = 'cost'
    banner = '/cost'
    help = (
        CommandHelp('/cost', 'show token usage and cost for this session', '/cost'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        usage = session.agent.usage
        print(f"  {usage.summary()} across this session", file=out)
        return None


class QueueCommand(Command):
    name = 'queue'
    help = (
        CommandHelp('/queue', 'inspect pending messages and choose one to view or discard', '/queue'),
        CommandHelp('/queue <number>', 'view one pending message in full', '/queue '),
        CommandHelp('/queue discard <number>', 'discard one unsent message; preserve history and task', '/queue discard '),
    )

    def background(self, argument: str) -> bool:
        return True

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from .inputqueue import input_queue

        session, style, out = context.session, context.style, context.out
        terminal = session.renderer.terminal
        queue = input_queue(session.agent)
        entries = queue.snapshot()
        parts = argument.split()
        discard = len(parts) == 2 and parts[0] == 'discard'
        number = parts[-1] if parts else ''
        if parts and (not (discard or len(parts) == 1) or not number.isascii() or not number.isdecimal()):
            print(style.red('  usage: /queue [<number>|discard <number>]'), file=out)
            return False
        if not entries:
            print('  no pending user messages.', file=out)
            return False
        if parts:
            try:
                position = int(number) - 1
            except ValueError:
                print(style.red('  no pending message with that number; use /queue to list them.'), file=out)
                return False
            if not 0 <= position < len(entries):
                print(style.red('  no pending message with that number; use /queue to list them.'), file=out)
                return False
            identifier, text = entries[position]
        elif terminal is None:
            for position, (_, text) in enumerate(entries, 1):
                print(f'  {position}. {text}', file=out)
            print('  /queue <number> to view; /queue discard <number> to discard.', file=out)
            return False
        else:
            selected = await terminal.choose('Pending Messages', [
                (identifier, f'{position}. {text}') for position, (identifier, text) in enumerate(entries, 1)
            ])
            if selected is None:
                return False
            identifier, text = next(entry for entry in entries if entry[0] == selected)
            action = await terminal.choose('Pending Message', [('view', 'View Full Message'), ('discard', 'Discard Message')])
            if action is None:
                return False
            discard = action == 'discard'
        if discard:
            try:
                removed = queue.discard(identifier)
            except SessionError as exc:
                print(style.red(f'  {exc}'), file=out)
                return False
            notice = '  pending message discarded.' if removed else '  message is no longer pending; nothing discarded.'
            print(notice, file=out)
        elif terminal is not None:
            terminal.open_viewer('Pending Message', text)
        else:
            print(text, file=out)
        return False


class TaskCommand(Command):
    def replaces_task(self, argument: str) -> bool:
        return argument == "new"

    name = 'task'
    banner = '/task [new]'
    help = (
        CommandHelp('/task', 'show retained user input and its source step IDs', '/task'),
        CommandHelp('/task new', 'start a new task with your next prompt; retain history and logs', '/task new'),
    )

    def changes_session(self, argument: str) -> bool:
        return argument == "new"

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if argument == "new":
            if session.agent.pending or session.agent.queued_messages:
                print(style.red("  queued input is still pending; resume or use /queue to discard it before starting another task."), file=out)
            else:
                session.agent.history.task = TaskMemory(start_message=len(session.agent.messages))
                print("  next prompt starts a new task; previous history and command logs remain available.", file=out)
        elif argument:
            print(style.red("  usage: /task [new]"), file=out)
        else:
            task = session.agent.history.task
            print(f"  source steps: {', '.join(map(str, task.sources)) or '(none)'}", file=out)
            step_id = task.current_prompt_step
            for prompt in task.sources.get(step_id, []) if step_id is not None else []:
                print(f"  {prompt}", file=out)
        return None


class RenameCommand(Command):
    name = 'rename'
    banner = '/rename <name>'
    help = (
        CommandHelp('/rename <name>', 'name this saved conversation and its terminal title', '/rename '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if not argument:
            print(style.red("  usage: /rename <name>"), file=out)
        else:
            try:
                title = session_title(argument)
                journal = session.registry.resources.journal
                if journal is not None:
                    await asyncio.to_thread(journal.rename, argument)
                else:
                    session.extensions["title"] = title
                print(f"  title: {title}", file=out)
            except SessionError as exc:
                print(style.red(f"  {exc}"), file=out)
        return None


class SessionsCommand(StoredSessionCommand):
    name = 'sessions'
    banner = '/sessions'
    help = (
        CommandHelp('/sessions', 'list saved sessions for this project', '/sessions'),
    )

    async def execute_saved(self, context: CommandContext, argument: str, journal: SessionJournal) -> bool | None:
        session, style, out = context.session, context.style, context.out
        for entry in await asyncio.to_thread(journal.listing):
            print(f"  {entry['id']}  {entry['created']}  {entry['title']}" + ("  (current)" if entry["current"] == "True" else ""), file=out)
        print(f"  storage: {journal.directory}", file=out)
        return None


class ForkCommand(BranchSessionCommand):
    name = 'fork'
    banner = '/fork'
    help = (
        CommandHelp('/fork', 'copy the current saved session and continue in the copy (queues while working)', '/fork'),
    )

    def changes_session(self, argument: str) -> bool:
        return True

    async def change(self, context: CommandContext, journal: SessionJournal) -> None:
        session, style, out = context.session, context.style, context.out
        terminal = session.renderer.terminal
        parent = journal.session_id
        data = await asyncio.to_thread(journal.load, parent)
        journal.restore(session.agent, data, fork=True)
        print(f"  forked {parent} as {journal.session_id}; continuing in the new session.", file=out)


class DeleteCommand(BranchSessionCommand):
    name = 'delete'
    banner = '/delete'

    def replaces_task(self, argument: str) -> bool:
        return True

    help = (
        CommandHelp('/delete', 'confirm deletion of the session, logs, diagnostics, and checkpoints; Esc cancels (queues while working)', '/delete'),
    )

    def changes_session(self, argument: str) -> bool:
        return True

    async def change(self, context: CommandContext, journal: SessionJournal) -> None:
        session, style, out = context.session, context.style, context.out
        terminal = session.renderer.terminal
        session.agent.reset(new_session=False)
        await asyncio.to_thread(journal.delete_current)
        session.extensions.pop("title", None)
        if terminal is not None:
            terminal.clear_transcript()
        session.renderer.show_banner(
            model=session.agent.model, workspace=session.workspace.root,
        )

    async def confirm(self, context: CommandContext, journal: SessionJournal) -> bool:
        terminal = context.session.renderer.terminal
        if terminal is None:
            print(context.style.red("  /delete requires an interactive terminal for confirmation."), file=context.out)
            return False
        selected = await terminal.choose(
            f"Permanently Delete {journal.title}?",
            [("delete", "Delete Current Session and All Logs"), ("cancel", "Cancel")],
        )
        return selected == "delete"


class RewindCommand(ExclusiveCommand):
    name = 'rewind'
    banner = '/rewind'

    def replaces_task(self, argument: str) -> bool:
        return True

    help = (
        CommandHelp('/rewind', 'choose an edit checkpoint and confirm file restoration; Esc cancels', '/rewind'),
        CommandHelp('/rewind list', 'list available file checkpoints without restoring anything', '/rewind list'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        checkpoints = session.agent._checkpoints()
        terminal = session.renderer.terminal
        if argument not in {"", "list"}:
            print(style.red("  usage: /rewind [list]"), file=out)
        elif checkpoints is None or not checkpoints.listing():
            print("  no file-edit checkpoints in this session.", file=out)
        elif argument == "list":
            for batch in checkpoints.listing():
                print(f"  {batch['id']}  step {batch['step_id']}  {batch['created']}", file=out)
        elif terminal is None:
            print(style.red("  /rewind requires an interactive terminal for selection and confirmation."), file=out)
        else:
            batches = checkpoints.listing()
            selected = await terminal.choose("Rewind File Edits", [
                (batch["id"], f"Before Step {batch['step_id']} — {batch['created']}") for batch in batches
            ])
            if selected is not None:
                from .tools.blocking import run_blocking
                try:
                    preview = await run_blocking(checkpoints.preview, selected)
                except OSError as exc:
                    print(style.red(f"  {exc}"), file=out)
                    return False
                print(cli._literal_tool_output(preview), file=out)
                confirm = await terminal.choose("Restore These Files?", [("restore", "Restore Files"), ("cancel", "Cancel")])
                if confirm == "restore":
                    try:
                        jobs = session.registry.resources.jobs
                        if jobs is not None:
                            await jobs.stop_all()
                        restored = await run_blocking(checkpoints.rewind, selected)
                        notice = "Files restored by /rewind: " + ", ".join(restored)
                        session.registry.context_notes["file_rewind"] = load_prompt('file-rewind.md', notice=notice)
                        print("  " + notice, file=out)
                    except (OSError, CheckpointError) as exc:
                        print(style.red(f"  {exc}"), file=out)
        return None


class ResumeCommand(StoredSessionCommand):
    name = 'resume'
    banner = '/resume [id|latest]'

    def replaces_task(self, argument: str) -> bool:
        return True

    help = (
        CommandHelp('/resume', 'choose a saved session with Up/Down and Enter (queues while working)', '/resume'),
        CommandHelp('/resume <id|latest>', 'restore a saved conversation and scroll to its end (queues while working)', '/resume '),
    )

    def changes_session(self, argument: str) -> bool:
        return True

    async def execute_saved(self, context: CommandContext, argument: str, journal: SessionJournal) -> bool | None:
        session, style, out = context.session, context.style, context.out
        try:
            if not argument:
                terminal = session.renderer.terminal
                if terminal is None:
                    print(style.red("  /resume requires an interactive terminal to choose a session; use /resume <id|latest>."), file=out)
                    return False
                entries = await asyncio.to_thread(journal.listing)
                if not entries:
                    print("  no saved sessions for this project.", file=out)
                    return False
                options = [
                    (entry["id"], f"{entry['title']}  {entry['created']}" +
                     ("  (current)" if entry["current"] == "True" else ""))
                    for entry in entries
                ]
                selected = await terminal.choose("Resume Session", options)
                if selected is None:
                    return False
                argument = selected
            await session.agent.wait_for_compaction()
            data = await asyncio.to_thread(journal.load, argument)
            jobs = session.registry.resources.jobs
            if jobs is not None:
                await jobs.stop_all()
            journal.restore(session.agent, data)
            await session.renderer.restore_transcript(session.agent.messages, session.agent.pending,
                                                      queued_messages=session.agent.queued_messages)
            print(f"  Restored {journal.session_id}; no tools were replayed. Type a task or continue when ready.", file=out)
        except SessionError as exc:
            print(style.red(f"  {exc}"), file=out)
        return None


class RequestsCommand(Command):
    name = 'requests'
    banner = '/requests [attempt]'
    help = (
        CommandHelp('/requests [attempt]', 'list recent request attempts or inspect an exact saved request', '/requests '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        diagnostics = session.registry.resources.diagnostics
        if diagnostics is None:
            print(style.red("  Request diagnostics are unavailable in this session."), file=out)
        else:
            try:
                if argument:
                    print(diagnostics.read(int(argument)), file=out)
                else:
                    for entry in diagnostics.listing():
                        print(f"  {entry['attempt']}  step {entry['step']}  {entry['outcome']}  {entry['created']}", file=out)
                    print(f"  storage: {diagnostics.directory}", file=out)
            except (ValueError, OSError) as exc:
                print(style.red(f"  Cannot read request diagnostics: {exc}"), file=out)
        return None


class McpCommand(Command):
    name = 'mcp'
    banner = '/mcp [add|save|remove]'
    help = (
        CommandHelp('/mcp', 'show MCP servers and their tools', '/mcp'),
        CommandHelp('/mcp add <name> <command> [args...]', ' connect an MCP server over stdio for this session', '/mcp add '),
        CommandHelp('/mcp save <name> <command> [args...]', ' same, but also save it globally for future runs in every project', '/mcp save '),
        CommandHelp('/mcp remove <name>', 'disconnect a server and forget it', '/mcp remove '),
    )

    def changes_session(self, argument: str) -> bool:
        return bool(argument)

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        await cli._mcp_command(session, argument, style, out)
        return None


class ResetCommand(ExclusiveCommand):
    name = 'reset'
    banner = '/reset'

    def replaces_task(self, argument: str) -> bool:
        return True

    help = (
        CommandHelp('/reset', 'start a fresh conversation; saved sessions remain on disk', '/reset'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        jobs = session.registry.resources.jobs
        if jobs is not None:
            await jobs.stop_all()
        session.agent.reset()
        session.extensions.pop("title", None)
        print(style.dim("  conversation cleared"), file=out)
        return None


class ReloadCommand(Command):
    name = 'reload'
    banner = '/reload'
    help = (
        CommandHelp('/reload', 'apply component edits (also watched automatically)', '/reload'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if session.reloader is None:
            print(style.dim("  live reload is disabled; restart without --no-reload to enable it."), file=out)
        else:
            session.reloader.request()
            if session.agent.running:
                print(style.dim("  reload queued for the end of the current model/tool batch."), file=out)
        return None


class GenerationsCommand(Command):
    name = 'generations'
    banner = '/generations'
    help = (
        CommandHelp('/generations', 'show current generation count', '/generations'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if session.reloader is None:
            print(style.dim("  live reload is disabled; restart without --no-reload to enable it."), file=out)
        else:
            print(f"  current generation: {session.reloader.generation}", file=out)
        return None


class InitCommand(ExclusiveCommand):
    name = 'init'
    banner = '/init'
    help = (
        CommandHelp('/init', 'create AGENTS.md if missing and load project guidance', '/init'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        await cli._init_command(session, style, out)
        return None


class StopCommand(Command):
    name = 'stop'
    banner = '/stop'
    help = (
        CommandHelp('/stop', 'finish this step and its tools, then stop', '/stop'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        session.extensions["resume_after_commands"] = False
        session.renderer._queued_settings_pause = False
        if session.agent.request_stop():
            if session.renderer.terminal is not None:
                session.renderer.terminal.set_working(True, stopping=True)
            print(style.dim("  stopping after the current step and its tool results."), file=out)
        else:
            print(style.dim("  already idle."), file=out)
        return None


class ExitCommand(Command):
    name = 'exit'
    aliases = ("quit",)
    banner = '/exit /quit'
    help = (
        CommandHelp('/exit', 'quit (also ctrl-d)', '/exit'),
        CommandHelp('/quit', 'same as /exit', '/quit'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        session.extensions["resume_after_commands"] = False
        session.renderer._queued_settings_pause = False
        session.agent.request_stop()
        return True


COMMANDS: tuple[Command, ...] = (
    HelpCommand(),
    ConfigShowCommand(),
    MenuCommand(),
    ViewCommand(),
    ThemeCommand(),
    MarkdownCommand(),
    CopyCommand(),
    DangerCommand(),
    OverthinkingCommand(),
    PlanningCommand(),
    HashlineCommand(),
    ToolsCommand(),
    ModelCommand(),
    ModelsCommand(),
    OptimalCommand(),
    ProviderCommand(),
    TemperatureCommand(),
    KeyCommand(),
    CostCommand(),
    QueueCommand(),
    TaskCommand(),
    RenameCommand(),
    SessionsCommand(),
    ForkCommand(),
    DeleteCommand(),
    RewindCommand(),
    ResumeCommand(),
    RequestsCommand(),
    McpCommand(),
    ResetCommand(),
    ReloadCommand(),
    GenerationsCommand(),
    InitCommand(),
    StopCommand(),
    ExitCommand(),
)
BY_NAME = {name: command for command in COMMANDS for name in (command.name, *command.aliases)}
HELP_FOOTER = '\nCtrl+\\ toggles the context view. Scroll it with the mouse wheel, arrows,\nPage Up/Down, or Home/End. System prompts appear in yellow.\n\nOrdinary text continues the active task. Use /task new before a separate task.\nProject Python selection is shown in context; --python PATH overrides discovery.\nSessions, command logs, and request diagnostics are saved unless --no-session is used.\nCommand logs default to a shared 1 GiB quota; request diagnostics roll forward at 32 MiB,\nkeeping the newest attempt complete even when it exceeds that target.\nBackground commands use command_jobs and read_command_output for control/output.\n/stop leaves those jobs running; /reset and exit stop them.\nEsc immediately interrupts agent work, active tools, and background jobs.\nInside a chooser, Esc only closes the chooser. Completed results are preserved.'


def all_help() -> str:
    return "Commands\n\n" + "\n".join(entry.line() for command in COMMANDS for entry in command.help) + "\n" + HELP_FOOTER


def command_help(argument: str) -> str:
    topic = argument.strip().removeprefix("/").lower()
    if not argument.strip():
        return all_help()
    command = BY_NAME.get(topic)
    if command is None:
        return f"  No help found for {argument.strip()!r}. Use /help to list commands.\n"
    entries = [entry for entry in command.help if entry.syntax.split()[0] == "/" + topic]
    return f"Help: /{topic}\n\n" + "\n".join(entry.line() for entry in entries) + "\n"


def menu_options() -> list[tuple[str, str]]:
    return [(entry.insertion, entry.syntax + " — " + entry.description)
            for command in COMMANDS for entry in command.help]


def banner_commands() -> str:
    return "  ".join(command.banner for command in COMMANDS if command.banner)
