"""Static slash-command definitions: behavior, policy, help, and menu syntax."""

from __future__ import annotations

import asyncio
import io
import math
import shlex
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .markdown import THEMES, literal
from .mcp import MCPError, ServerSpec, config_path, load_servers
from .sessions import SessionError, session_title
from .requestfailures import failed_steps
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
    def background(self, argument: str) -> bool:
        return True

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

    def background(self, argument: str) -> bool:
        return True

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        text = command_help(argument)
        terminal = context.session.renderer.terminal
        if terminal is None:
            print(text, file=context.out, end="")
        else:
            # The viewer keeps help readable while streamed output continues.
            terminal.open_viewer("Help", text)
        return None


def current_settings(session: Session) -> list[tuple[str, str, str]]:
    """Each `/settings` entry as (command, current value, description)."""
    terminal = session.renderer.terminal
    temperature = session.agent.temperature

    def state(enabled: bool) -> str:
        return "on" if enabled else "off"

    return [
        ("overthinking", state(session.agent.overthinking), "supply archived thoughts from recent steps"),
        ("planning", state(session.agent.planning), "offer the working-plan tool for multi-step tasks"),
        ("hashline", state(session.registry.services.get("hashline_enabled", True)), "offer hash-anchored read/edit tools"),
        ("danger", state(session.workspace.access.danger), "lift workspace path confinement"),
        ("theme", session.renderer.markdown_theme, "Markdown colors"),
        ("markdown", "source" if terminal is not None and terminal._markdown_source else "rendered", "how replies display"),
        ("temperature", "provider default" if temperature is None else f"{temperature:g}", "working-model temperature"),
    ]


class StatusCommand(Command):
    name = 'status'
    aliases = ('config-show',)
    help = (
        CommandHelp('/status', 'show the model, workspace, interpreter, and current settings', '/status'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        routing = session.model_routing()
        print(f"Model: {session.client.provider}::{session.agent.model} (Optimal {'on' if routing.enabled else 'off'})", file=out)
        print(f"Workspace: {session.workspace.root}", file=out)
        print(f"Active interpreter: {sys.executable}", file=out)
        for name, value, _ in current_settings(session):
            print(f"{name.capitalize()}: {value}", file=out)
        return None


class SettingsCommand(Command):
    name = 'settings'
    help = (
        CommandHelp('/settings', 'choose a setting to change; each line shows its command and current value', '/settings'),
    )

    def background(self, argument: str) -> bool:
        return True

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        session, style, out = context.session, context.style, context.out
        settings = current_settings(session)
        terminal = session.renderer.terminal
        if terminal is None:
            for name, value, description in settings:
                print(f"  /{name:<13} {value:<17} {description}", file=out)
            return None
        choice = await terminal.choose("Settings", [(name, f"[{name}] {value} — {description}")
                                                    for name, value, description in settings])
        if choice is None:
            return None
        value = dict((name, current) for name, current, _ in settings)[choice]
        line = None
        if choice in {"overthinking", "planning", "hashline", "danger"}:
            line = f"/{choice} {'off' if value == 'on' else 'on'}"
        elif choice == "markdown":
            line = "/markdown " + ("rendered" if value == "source" else "source")
        elif choice == "theme":
            theme = await terminal.choose("Theme", [(name, name) for name in THEMES])
            line = f"/theme {theme}" if theme else None
        else:
            try:
                text = (await session.renderer.ask_input("  temperature (blank keeps it): ")).strip()
            except (EOFError, KeyboardInterrupt):
                text = ""
            line = f"/temperature {text}" if text else None
        if line is not None:
            # Typed commands and menu choices share queueing while work is active.
            await cli.CommandRunner(session).execute(line)
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
        CommandHelp('/danger on', 'disable workspace path confinement (queues while working)', '/danger on'),
        CommandHelp('/danger off', 'restore workspace path confinement (queues while working)', '/danger off'),
        CommandHelp('/danger', 'show whether danger mode is active (also /danger status)', '/danger'),
    )

    def changes_session(self, argument: str) -> bool:
        return argument.lower() in {"on", "off"}

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if argument.lower() not in {"", "on", "off", "status"}:
            print(style.red("  usage: /danger [on|off|status]"), file=out)
        else:
            # Lifting confinement takes the explicit word; a bare /danger only reports.
            if argument.lower() in {"on", "off"}:
                session.workspace.access.danger = argument.lower() == "on"
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
    banner = '/model [model|provider|free|optimal|next|reset|words]'
    help = (
        CommandHelp('/model', 'choose from every /model option, each shown as [subcommand]', '/model'),
        CommandHelp('/model <model>', 'switch to an exact model; use provider::model when ambiguous', '/model '),
        CommandHelp('/model <words>', 'choose from models matching every word; several words always search', '/model '),
        CommandHelp('/model provider', "choose a provider, then one of its models", '/model provider'),
        CommandHelp('/model <provider>', "choose from that provider's models, free ones first", '/model '),
        CommandHelp('/model free', 'choose from free models', '/model free'),
        CommandHelp('/model optimal', 'select the best ranked free model automatically', '/model optimal'),
        CommandHelp('/model next', 'skip the current model for the next ranked free model', '/model next'),
        CommandHelp('/model reset', 'retry from the best ranked free model, clearing failures', '/model reset'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        await cli.ModelPicker(context.session, context.style, context.out).menu(argument)
        return None


class ModelsCommand(ExclusiveCommand):
    name = 'models'
    banner = '/models [filter]'
    help = (
        CommandHelp('/models [filter]', 'search active provider catalogs (e.g. /models nemotron)', '/models '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        await cli.ModelPicker(context.session, context.style, context.out).browse(argument)
        return None


class ProviderCommand(ExclusiveCommand):
    name = 'provider'
    help = (
        CommandHelp('/provider <provider>', "switch to the provider's free default model", '/provider '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        await cli.ModelPicker(context.session, context.style, context.out).provider_default(argument.strip().lower())
        return None


class TemperatureCommand(Command):
    name = 'temperature'
    banner = '/temperature [value]'
    help = (
        CommandHelp('/temperature', 'show the explicit temperature or provider default and model support', '/temperature'),
        CommandHelp('/temperature <value>', 'set working-model temperature from 0 to 2 when supported', '/temperature '),
    )

    def changes_session(self, argument: str) -> bool:
        return bool(argument)

    def background(self, argument: str) -> bool:
        return not argument

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from .api import APIError
        session, style, out = context.session, context.style, context.out
        value = session.agent.temperature
        if argument:
            try:
                value = float(argument)
            except ValueError:
                value = float("nan")
            limit = session.client.max_temperature
            if not math.isfinite(value) or not 0 <= value <= limit:
                print(style.red(f"  temperature must be a number from 0 to {limit:g} for {session.client.provider}."), file=out)
                return None
        try:
            capabilities = await session.client.model_capabilities(session.agent.model)
        except APIError as exc:
            print(style.red(f"  could not verify temperature support: {exc}"), file=out)
            return None
        if capabilities is None or "temperature" not in capabilities.parameters:
            print(style.red(
                f"  temperature control is unavailable for {session.agent.model}'s selected endpoints; "
                "the provider default is used."
            ), file=out)
            return None
        if argument:
            session.agent.temperature = value
        if value is None:
            print("  temperature: provider default", file=out)
        else:
            print(f"  temperature: {value:g}", file=out)
        return None


class ContextCommand(Command):
    name = 'context'
    banner = '/context [tokens|reset]'
    help = (
        CommandHelp('/context', 'show the working-context cap and the selected model\'s limit', '/context'),
        CommandHelp('/context <tokens>', 'cap working context, such as 300k or 1.5M; the model\'s limit still applies (queues while working)', '/context '),
        CommandHelp('/context reset', 'clear the cap so the model\'s own limit applies (also /context default or off; queues while working)', '/context reset'),
    )

    def changes_session(self, argument: str) -> bool:
        return bool(argument)

    def background(self, argument: str) -> bool:
        return not argument

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from .api import APIError
        from .config import MIN_CONTEXT_TOKENS, parse_token_count
        session, style, out = context.session, context.style, context.out
        agent = session.agent
        if argument.lower() in {"reset", "default", "off"}:
            agent.context_tokens = None
        elif argument:
            try:
                count = parse_token_count(argument)
                if count < MIN_CONTEXT_TOKENS:
                    raise ValueError(f"the cap must be at least {MIN_CONTEXT_TOKENS:,} tokens")
            except ValueError as exc:
                print(style.red(f"  {exc}"), file=out)
                return None
            agent.context_tokens = count
        if argument:
            # The cap belongs to the session, so save it with the journal now.
            agent._persist()
        try:
            limit = await agent._context_length()
        except APIError:
            limit = None
        cap = agent.context_tokens
        if cap is None:
            print("  context cap: none" + (f"; the model's limit is {limit:,} tokens" if limit else ""), file=out)
        elif limit and cap >= limit:
            print(f"  context cap: {cap:,} tokens, above the model's limit of {limit:,}; the model's limit applies", file=out)
        else:
            print(f"  context cap: {cap:,} tokens" + (f" (the model's limit is {limit:,})" if limit else ""), file=out)
        return None


class KeyCommand(Command):
    name = 'key'
    banner = '/key [list|show|delete <provider>|[provider] key]'
    help = (
        CommandHelp('/key', 'choose a configured key to delete, after confirmation, or enter a new one', '/key'),
        CommandHelp('/key list', 'list configured keys, masked to their start and end, with where each is stored', '/key list'),
        CommandHelp('/key delete <provider>', "delete that provider's saved key and forget it for this session", '/key delete '),
        CommandHelp('/key show', 'show the current key, masked', '/key show'),
        CommandHelp('/key status', 'same as /key show', '/key status'),
        CommandHelp('/key <key>', 'detect the provider, install its key, and offer to save it', '/key '),
        CommandHelp('/key <provider> <key>', 'install a key with an explicit provider', '/key '),
    )

    def changes_session(self, argument: str) -> bool:
        return argument.lower() not in {"show", "status", "list"}

    def background(self, argument: str) -> bool:
        return argument.lower() in {"show", "status", "list"}

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        await cli.KeyEntry(context.session, context.style, context.out).command(argument)
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
        CommandHelp('/task new', 'treat your next prompt as a new task in this conversation; /new starts a fresh conversation', '/task new'),
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
    banner = '/sessions [list]'
    help = (
        CommandHelp('/sessions', 'choose a saved session to resume, rename, or delete (queues while working)', '/sessions'),
        CommandHelp('/sessions list', 'list saved session IDs and the storage path', '/sessions list'),
    )

    def changes_session(self, argument: str) -> bool:
        return not argument

    async def execute_saved(self, context: CommandContext, argument: str, journal: SessionJournal) -> bool | None:
        session, style, out = context.session, context.style, context.out
        terminal = session.renderer.terminal
        if argument not in {"", "list"}:
            print(style.red("  usage: /sessions [list]"), file=out)
            return None
        try:
            if argument or terminal is None:
                for entry in await asyncio.to_thread(journal.listing):
                    print(f"  {entry['id']}  {entry['created']}  {entry['title']}" + ("  (current)" if entry["current"] == "True" else ""), file=out)
                print(f"  storage: {journal.directory}", file=out)
                return None
            # Return to the list after changing another session, so several can be cleared in turn.
            while True:
                entries = {entry["id"]: entry for entry in await asyncio.to_thread(journal.listing)}
                if not entries:
                    print("  no saved sessions for this project.", file=out)
                    return None
                selected = await terminal.choose("Saved Sessions", [(entry["id"], journal.label(entry)) for entry in entries.values()])
                if selected is None:
                    return None
                entry = entries[selected]
                current = entry["current"] == "True"
                action = await terminal.choose(entry["title"], [
                    *([] if current else [("resume", "Resume Session")]),
                    ("rename", "Rename Session"), ("delete", "Delete Session"), ("back", "Back to Saved Sessions"),
                ])
                if action == "resume":
                    return await ResumeCommand().execute_saved(context, selected, journal)
                if action == "delete" and current:
                    return await DeleteCommand().run(context, "")
                if action == "delete":
                    confirm = await terminal.choose(f"Permanently Delete {entry['title']}?", [
                        ("delete", "Delete Session and All Logs"), ("cancel", "Cancel"),
                    ])
                    if confirm == "delete":
                        await asyncio.to_thread(journal.delete_saved, selected)
                        print(f"  deleted {entry['title']}", file=out)
                elif action == "rename":
                    name = (await terminal.ask("Session name › ")).strip()
                    if name and current:
                        await asyncio.to_thread(journal.rename, name)
                        terminal.set_title(journal.title)
                    elif name:
                        await asyncio.to_thread(journal.rename_saved, selected, name)
                    if name:
                        print(f"  title: {session_title(name)}", file=out)
        except SessionError as exc:
            print(style.red(f"  {exc}"), file=out)
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
                print(literal(preview), file=out)
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
                options = [(entry["id"], journal.label(entry)) for entry in entries]
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
                                                      queued_messages=session.agent.queued_messages,
                                                      failures=failed_steps(session.agent.history.steps))
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
        CommandHelp('/mcp add <name> <command> [args...]', 'connect an MCP server over stdio for this session', '/mcp add '),
        CommandHelp('/mcp save <name> <command> [args...]', 'same, but also save it globally for future runs in every project', '/mcp save '),
        CommandHelp('/mcp remove <name>', 'disconnect a server and forget it', '/mcp remove '),
    )

    def changes_session(self, argument: str) -> bool:
        return bool(argument)

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        session, style, out = context.session, context.style, context.out
        if session.mcp is None:
            print(style.dim("  MCP is disabled for this run (--no-mcp)."), file=out)
            return None

        try:
            action, spec, removal = self.split(argument)
        except ValueError as exc:
            print(style.red(f"  invalid MCP command: {exc}"), file=out)
            return None

        if not action:
            self.describe(context)
            return None

        if action == "remove":
            if removal is None:
                print(style.red("  usage: /mcp remove <name>"), file=out)
                return None
            live = removal in session.mcp.servers
            try:
                saved = session.mcp.remove(removal)
            except (MCPError, OSError) as exc:
                print(style.red(f"  could not remove server: {exc}"), file=out)
                return None
            await session.mcp.disconnect(removal)
            session.mcp.servers.pop(removal, None)
            if not live and not saved:
                print(style.red(f"  no MCP server named {removal!r}"), file=out)
            elif saved:
                print(
                    style.dim(f"  removed {removal} from {config_path(session.workspace.root)}"),
                    file=out,
                )
            else:
                print(style.dim(f"  disconnected {removal} for this session"), file=out)
            return None

        if action not in ("add", "save") or spec is None:
            print(
                style.red(f"  usage: /mcp add <name> <command> [args...]  "
                          f"(or /mcp save ...)"),
                file=out,
            )
            return None

        try:
            state = await session.mcp.connect(spec.name, spec)
        except MCPError as exc:
            print(style.red(f"  {exc}"), file=out)
            return None

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
            print(style.warning(f"  Warning: {state.line()}"), file=out)
            return None

        print(f"  {style.green('connected')} {literal(state.spec.name)} — {literal(state.detail)}", file=out)
        for qualified in state.qualified:
            print(literal(f"    {qualified}"), file=out)
        if not state.qualified:
            print(style.dim("    (no tools)"), file=out)
        return None

    def split(self, argument: str) -> tuple[str, ServerSpec | None, str | None]:
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

    def describe(self, context: CommandContext) -> None:
        """List configured servers, live connections, and contributed tools."""
        style, out = context.style, context.out
        manager = context.session.mcp
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
            connected = sum(name in configured and state.status == "connected" for name, state in live.items())
            print(
                style.dim(
                    f"  {connected}/{len(configured)} configured server(s) connected "
                    f"({config_path(manager.workspace)})"
                ),
                file=out,
            )
        else:
            print(style.dim("  no globally saved servers; showing this session only"), file=out)

        for name in sorted(set(configured) | set(live)):
            state = live.get(name)
            if state is not None:
                print(f"  {style.cyan(name)}: {literal(state.status)} — {literal(state.detail)}", file=out)
                for qualified in state.qualified:
                    print(literal(f"    {qualified}"), file=out)
            else:
                spec = configured[name]
                marker = "not connected" if name in configured else "session only"
                print(style.dim(f"  {name}: {marker} — {spec.command_line()}"), file=out)


class NewCommand(ExclusiveCommand):
    name = 'new'
    aliases = ('reset',)
    banner = '/new'

    def replaces_task(self, argument: str) -> bool:
        return True

    help = (
        CommandHelp('/new', 'start a fresh conversation; saved sessions remain on disk', '/new'),
        CommandHelp('/reset', 'same as /new', '/reset'),
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


class PluginsCommand(ExclusiveCommand):
    name = 'plugins'
    banner = '/plugins'
    help = (
        CommandHelp('/plugins', 'list plugins and the state of each override, then approve them or resolve conflicts', '/plugins'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        await cli.PluginMenu(context.session, context.style, context.out).menu()
        return None


class PluginCommand(ExclusiveCommand):
    name = 'plugin'
    help = (
        CommandHelp('/plugin copy [target] [file]', 'copy module.Class or module.Class.method into a plugin file, stamped and approved', '/plugin copy '),
        CommandHelp('/plugin diff [file]', 'show how application methods changed since plugins copied them', '/plugin diff '),
        CommandHelp('/plugin restamp [file [method]]', 'record the current fingerprints after reviewing a change', '/plugin restamp '),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        from . import cli
        await cli.PluginMenu(context.session, context.style, context.out).command(argument)
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


AGENTS_MD_SCAFFOLD = """# Project Guidance

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


class InitCommand(ExclusiveCommand):
    name = 'init'
    banner = '/init'
    help = (
        CommandHelp('/init', 'create AGENTS.md if missing and load project guidance', '/init'),
    )

    async def run(self, context: CommandContext, argument: str) -> bool | None:
        """Create missing project guidance and apply it without restarting."""
        from .agent import build_system_prompt
        from .instructions import ProjectInstructions, load_project_instructions
        from .types import Message
        from .workspace import WorkspaceError

        session, out = context.session, context.out
        agents_md_path = session.workspace.resolve("AGENTS.md")
        try:
            # Exclusive creation preserves existing guidance and refuses new symlinks.
            with agents_md_path.open("x", encoding="utf-8") as handle:
                handle.write(AGENTS_MD_SCAFFOLD)
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
    StatusCommand(),
    SettingsCommand(),
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
    ProviderCommand(),
    TemperatureCommand(),
    ContextCommand(),
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
    NewCommand(),
    ReloadCommand(),
    PluginsCommand(),
    PluginCommand(),
    GenerationsCommand(),
    InitCommand(),
    StopCommand(),
    ExitCommand(),
)
BY_NAME = {name: command for command in COMMANDS for name in (command.name, *command.aliases)}
HELP_FOOTER = '\nCtrl+\\ toggles the context view. Scroll it with the mouse wheel, arrows,\nPage Up/Down, or Home/End. System prompts appear in yellow.\n\nOrdinary text continues the active task. Use /task new before a separate task.\nProject Python selection is shown in context; --python PATH overrides discovery.\nSessions, command logs, and request diagnostics are saved unless --no-session is used.\nCommand logs default to a shared 1 GiB quota; request diagnostics roll forward at 32 MiB,\nkeeping the newest attempt complete even when it exceeds that target.\nBackground commands use command_jobs and read_command_output for control/output.\n/stop leaves those jobs running; /new and exit stop them.\nEsc immediately interrupts agent work, active tools, and background jobs.\nInside a chooser, Esc only closes the chooser. Completed results are preserved.'


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
