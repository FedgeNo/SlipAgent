"""Stable session frame and transactional source reloads.

Components execute in a fresh import namespace. Existing classes retain their
identity so live state, exception handlers, and transport references survive.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.abc
import importlib.util
import sys
import types
import uuid
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .tools.base import Tool, ToolRegistry
from .types import Message
from .lifecycle import Lifetime

if TYPE_CHECKING:
    from .cli import Session

# Wire/state contracts and resource owners stay in the running frame.
CORE_MODULES = frozenset({"", "runtime", "config", "types", "workspace", "tools.base", "mcp", "lifecycle"})
POLL_INTERVAL = .5
_CLASS_INTERNALS = frozenset({
    "__dict__", "__weakref__", "__slots__", "__module__", "__classcell__",
})


class ReloadError(Exception):
    """An edit cannot be applied to this live session."""


class _SourceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, prefix: str, sources: dict[str, bytes], root: Path) -> None:
        self.prefix = prefix
        self.sources = sources
        self.root = root

    def find_spec(
        self, fullname: str, path: Any = None, target: types.ModuleType | None = None,
    ) -> Any:
        if not fullname.startswith(self.prefix + "."):
            return None
        relative = fullname[len(self.prefix) + 1:].replace(".", "/")
        filename = relative + ".py"
        package = relative + "/__init__.py"
        if package in self.sources:
            filename = package
        elif filename not in self.sources:
            return None
        spec = importlib.util.spec_from_loader(fullname, self, is_package=filename == package)
        assert spec is not None
        spec.loader_state = filename
        return spec

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        assert module.__spec__ is not None
        filename = module.__spec__.loader_state
        module.__file__ = str(self.root / filename)
        module.__dict__["__source__"] = self.sources[filename]
        if filename.endswith("/__init__.py"):
            module.__path__ = [str((self.root / filename).parent)]
        # Compile bytes directly: timestamp-based .pyc files can miss quick edits.
        exec(compile(self.sources[filename], module.__file__, "exec"), module.__dict__)


def _module_name(filename: str) -> str:
    if filename == "__init__.py":
        return ""
    return filename.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _drop_namespace(prefix: str) -> None:
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def _class_member(name: str, value: Any) -> bool:
    return name not in _CLASS_INTERNALS and not isinstance(
        value, (types.MemberDescriptorType, types.GetSetDescriptorType),
    )


def _rebind(value: Any, classes: dict[type[Any], type[Any]]) -> Any:
    if isinstance(value, type):
        return classes.get(value, value)
    if isinstance(value, types.FunctionType):
        closure = value.__closure__
        if closure is None:
            return value
        cells = []
        for cell in closure:
            try:
                content = cell.cell_contents
            except ValueError:
                cells.append(cell)
                continue
            replacement = _rebind(content, classes) if isinstance(content, type) else content
            cells.append(types.CellType(replacement) if replacement is not content else cell)
        function = types.FunctionType(value.__code__, value.__globals__, value.__name__, value.__defaults__, tuple(cells))
        function.__kwdefaults__ = value.__kwdefaults__
        function.__annotations__ = value.__annotations__
        function.__dict__.update(value.__dict__)
        function.__qualname__ = value.__qualname__
        function.__module__ = value.__module__
        function.__doc__ = value.__doc__
        return function
    if isinstance(value, staticmethod):
        return staticmethod(_rebind(value.__func__, classes))
    if isinstance(value, classmethod):
        return classmethod(_rebind(value.__func__, classes))
    if isinstance(value, property):
        return property(_rebind(value.fget, classes), _rebind(value.fset, classes),
                        _rebind(value.fdel, classes), value.__doc__)
    return value


class RuntimeFrame:
    """Retain session resources; apply source edits only at protocol boundaries."""

    def __init__(self, session: Session, cli_module: types.ModuleType) -> None:
        self.session = session
        self.root = Path(__file__).resolve().parent
        self.cli_module = cli_module
        self.busy = 0
        self.reloading = False
        self.force = False
        self.generation = 0
        self.namespace: str | None = None
        self._applied = self._sources()
        self._attempted = self._signature(self._applied)
        self._observed = self._attempted
        self._project_instructions = session.agent.system_prompt
        self._read_error: str | None = None
        self._builtin_tools = tuple(
            tool for tool in session.registry.tools if tool.name in session.registry.builtin_names
        )
        self._terminal_write: Callable[[Any, str], None] = cli_module.TerminalUI.write
        self._component_classes: dict[tuple[str, str], type[Any]] = {}

    def _sources(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in sorted(self.root.rglob("*.py")) if "__pycache__" not in path.parts
        }

    @staticmethod
    def _signature(sources: dict[str, bytes]) -> bytes:
        digest = hashlib.sha256()
        for name, content in sorted(sources.items()):
            digest.update(name.encode() + b"\0" + content + b"\0")
        return digest.digest()

    def request(self) -> None:
        self.force = True

    def _notice(self, text: str, *, error: bool = False) -> None:
        # The model has no view of terminal notices. Carry current runtime
        # diagnostics into its next request without splicing a tool sequence.
        self.session.registry.context_notes["runtime"] = (
            f"Active component generation: {self.generation}. {text} "
            "Source on disk may differ from running behavior after a rejected reload."
        )
        renderer = self.session.renderer
        message = f"  {'✗' if error else '↻'} {text}"
        try:
            style = renderer.style.red if error else renderer.style.dim
            renderer.emit(style(message))
        except Exception:
            # The frame can report problems even if a renderer edit is broken.
            if renderer.terminal is not None:
                self._terminal_write(renderer.terminal, "\n" + message)
            else:
                print("\n" + message, file=renderer.stream, flush=True)

    def report_error(self, text: str) -> None:
        self._notice(text, error=True)

    async def watch(self) -> None:
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                sources = self._sources()
                self._read_error = None
                signature = self._signature(sources)
                # A second observation avoids most half-written editor saves.
                if signature == self._observed:
                    await self.checkpoint(sources=sources)
                self._observed = signature
            except OSError as exc:
                if str(exc) != self._read_error:
                    self._notice(f"Reload could not read source: {exc}", error=True)
                    self._read_error = str(exc)

    async def checkpoint(
        self, *, boundary: bool = False, sources: dict[str, bytes] | None = None,
    ) -> None:
        if self.reloading or self.busy or self.session.agent.running and not boundary:
            return
        try:
            sources = self._sources() if sources is None else sources
        except OSError as exc:
            self._notice(f"Reload could not read source: {exc}", error=True)
            return
        signature = self._signature(sources)
        if signature == self._attempted and not self.force:
            return
        self.force = False
        self._attempted = signature
        self.reloading = True
        try:
            await self._reload(sources)
        except Exception as exc:
            self._notice(f"Reload rejected; previous code remains active: {type(exc).__name__}: {exc}", error=True)
        finally:
            self.reloading = False

    def _stage(self, sources: dict[str, bytes], prefix: str) -> dict[str, types.ModuleType]:
        changed_core = [
            name for name in self._applied.keys() | sources.keys()
            if _module_name(name) in CORE_MODULES and self._applied.get(name) != sources.get(name)
        ]
        if changed_core:
            raise ReloadError(f"Restart required for frame/contract edits: {', '.join(sorted(changed_core))}. Core modules cannot be changed without restart as they contain shared state and wire-level contracts.")
        package = types.ModuleType(prefix)
        package.__path__ = [str(self.root)]
        sys.modules[prefix] = package
        for name in CORE_MODULES - {""}:
            sys.modules[f"{prefix}.{name}"] = importlib.import_module(f"slipagent.{name}")
        loader = _SourceLoader(prefix, sources, self.root)
        sys.meta_path.insert(0, loader)
        try:
            return {
                _module_name(filename): importlib.import_module(f"{prefix}.{_module_name(filename)}")
                for filename in sources if _module_name(filename) not in CORE_MODULES
            }
        finally:
            sys.meta_path.remove(loader)

    def _classes(
        self, modules: dict[str, types.ModuleType],
    ) -> tuple[dict[type[Any], type[Any]], dict[str, types.ModuleType], dict[tuple[str, str], type[Any]]]:
        classes: dict[type[Any], type[Any]] = {}
        originals: dict[str, types.ModuleType] = {}
        declared = {
            (name, symbol): value for name, module in modules.items()
            for symbol, value in module.__dict__.items()
            if isinstance(value, type) and value.__module__ == module.__name__
        }
        for key, old_class in self._component_classes.items():
            if key not in declared:
                raise ReloadError(f"Restart required: class {'.'.join(key)} was removed")
            classes[declared[key]] = old_class
        for name, candidate in modules.items():
            original = self.cli_module if name == "cli" else sys.modules.get(f"slipagent.{name}")
            if original is None:
                continue
            originals[name] = original
            for symbol, value in original.__dict__.items():
                if isinstance(value, type) and value.__module__ == original.__name__:
                    replacement = candidate.__dict__.get(symbol)
                    if not isinstance(replacement, type):
                        raise ReloadError(f"Restart required: class {name}.{symbol} was removed")
                    classes[replacement] = value
        for new_class, old_class in classes.items():
            bases = tuple(classes.get(base, base) for base in new_class.__bases__)
            fields = tuple(getattr(new_class, "__dataclass_fields__", ()))
            old_fields = tuple(getattr(old_class, "__dataclass_fields__", ()))
            if bases != old_class.__bases__ or new_class.__dict__.get("__slots__") != old_class.__dict__.get("__slots__") or fields != old_fields:
                raise ReloadError(f"Restart required: state layout changed for {old_class.__name__}")
        return classes, originals, declared

    async def _reload(self, sources: dict[str, bytes]) -> None:
        prefix = "_slipagent_generation_" + uuid.uuid4().hex
        candidate_registry: ToolRegistry | None = None
        old_tools: list[Tool] = []
        candidate_closers: list[Callable[[], Awaitable[None]]] = []
        retired_closers: list[Callable[[], Awaitable[None]]] = []
        previous_namespace: str | None = None
        committed = False
        try:
            modules = self._stage(sources, prefix)
            required = {
                "agent": ("Agent", "build_system_prompt"), "context": ("ConversationHistory", "RecallHistoryTool"),
                "cli": ("Renderer", "_execute_command", "_handle_command", "_read_line", "_run_turn", "_refresh_quota", "_shutdown"),
                "terminal": ("TerminalUI",),
                "tools": ("build_default_registry",),
            }
            for name, symbols in required.items():
                if name not in modules or any(not callable(getattr(modules[name], symbol, None)) for symbol in symbols):
                    raise ReloadError(f"Missing component API in {name}")
            methods: dict[str, dict[str, tuple[str, ...]]] = {
                "agent": {"Agent": ("run", "_step", "_stop_notice", "enqueue", "request_stop")},
                "cli": {"Renderer": ("emit", "handle", "user_prompt", "draw_prompt", "end_prompt")},
                "context": {"ConversationHistory": ("view", "sync", "save_response", "clear")},
                "openrouter": {"OpenRouterClient": ("chat", "key_info", "list_models", "aclose")},
                "terminal": {"TerminalUI": ("refresh", "_layout", "_bindings", "_style", "write", "read_line", "set_working", "close")},
            }
            for name, contracts in methods.items():
                for symbol, attributes in contracts.items():
                    component = getattr(modules[name], symbol)
                    if any(not callable(getattr(component, attribute, None)) for attribute in attributes):
                        raise ReloadError(f"Missing component API in {name}.{symbol}")
            for name, original in tuple(sys.modules.items()):
                if name.startswith("slipagent.") and original is not None:
                    relative = name[len("slipagent."):]
                    if relative not in CORE_MODULES and relative not in modules:
                        raise ReloadError(f"Restart required: loaded component {relative} was removed")

            classes, originals, declared = self._classes(modules)
            # New tool instances borrow the same environment/log owners. A
            # candidate must never erase prior logs or reset their quota usage.
            candidate_registry = modules["tools"].build_default_registry(
                self.session.workspace, services=self.session.registry.services,
            )
            replacement_tools = candidate_registry.tools
            candidate_closers = [getattr(tool, "aclose") for tool in replacement_tools if hasattr(tool, "aclose")]
            # Build with new constructors, then adopt the stable class identities.
            for tool in replacement_tools:
                tool.__class__ = classes.get(type(tool), type(tool))
            builtin_ids = {id(tool) for tool in self._builtin_tools}
            retained = [tool for tool in self.session.registry.tools if id(tool) not in builtin_ids]
            # Recall owns the current archive; remote MCP tools keep their clients.
            retained = [tool for tool in retained if tool.name != "recall_history"]
            replacement_tools += retained
            replacement_tools.append(modules["context"].RecallHistoryTool(self.session.agent.history))
            replacement_tools[-1].__class__ = classes.get(type(replacement_tools[-1]), type(replacement_tools[-1]))
            ToolRegistry(replacement_tools)  # Check collisions before touching live state.

            # Preserve explicitly supplied legacy suffixes. CLI project guidance
            # is refreshed separately by the session instruction tracker.
            old_builder = self.cli_module.build_system_prompt
            old_base = old_builder(str(self.session.workspace.root))
            pinned = self._project_instructions or ""
            suffix = pinned[len(old_base):] if pinned.startswith(old_base) else ""
            prompt = modules["agent"].build_system_prompt(str(self.session.workspace.root)) + suffix

            class_snapshots = {original: dict(original.__dict__) for original in classes.values()}
            module_snapshots = {name: dict(original.__dict__) for name, original in originals.items()}
            old_tools = self.session.registry.tools
            retired_closers = [
                getattr(tool, "aclose") for tool in self._builtin_tools if hasattr(tool, "aclose")
            ]
            old_prompt = self.session.agent.system_prompt
            old_system = self.session.agent.messages[:1]
            terminal = self.session.renderer.terminal
            presentation = (
                terminal.app, terminal.input, terminal.transcript, terminal._wrapped_columns,
                terminal.input.window.height, terminal.app.layout,
                terminal.app.key_bindings, terminal.app.style,
            ) if terminal is not None else None
            try:
                # New subclasses must inherit the same live parents as existing objects.
                for new_class in declared.values():
                    if new_class not in classes:
                        bases = tuple(classes.get(base, base) for base in new_class.__bases__)
                        if bases != new_class.__bases__:
                            new_class.__bases__ = bases
                for new_class, old_class in classes.items():
                    for symbol, value in tuple(old_class.__dict__.items()):
                        if _class_member(symbol, value) and symbol not in new_class.__dict__:
                            delattr(old_class, symbol)
                    for symbol, value in new_class.__dict__.items():
                        if _class_member(symbol, value):
                            setattr(old_class, symbol, _rebind(value, classes))
                for module in modules.values():
                    for symbol, value in tuple(module.__dict__.items()):
                        module.__dict__[symbol] = _rebind(value, classes)
                for name, original in originals.items():
                    metadata = {key: value for key, value in original.__dict__.items() if key in {"__name__", "__package__", "__spec__", "__loader__", "__file__", "__path__"}}
                    original.__dict__.clear()
                    original.__dict__.update(modules[name].__dict__)
                    original.__dict__.update(metadata)
                self.session.registry.replace(replacement_tools)
                self.session.agent.system_prompt = prompt
                if self.session.agent.messages and self.session.agent.messages[0].role == "system":
                    self.session.agent.messages[0] = Message.system(prompt)
                if terminal is not None:
                    terminal.refresh()
                    assert presentation is not None
                    if (terminal.app, terminal.input, terminal.transcript) != presentation[:3]:
                        raise ReloadError("Terminal reload must preserve its application, input, and transcript")
            except Exception:
                for old_class, snapshot in class_snapshots.items():
                    for symbol, value in tuple(old_class.__dict__.items()):
                        if _class_member(symbol, value) and symbol not in snapshot:
                            delattr(old_class, symbol)
                    for symbol, value in snapshot.items():
                        if _class_member(symbol, value):
                            setattr(old_class, symbol, value)
                for name, snapshot in module_snapshots.items():
                    originals[name].__dict__.clear()
                    originals[name].__dict__.update(snapshot)
                self.session.registry.replace(old_tools)
                self.session.agent.system_prompt = old_prompt
                self.session.agent.messages[:len(old_system)] = old_system
                if terminal is not None and presentation is not None:
                    (terminal.app, terminal.input, terminal.transcript, terminal._wrapped_columns,
                     terminal.input.window.height, terminal.app.layout,
                     terminal.app.key_bindings, terminal.app.style) = presentation
                    terminal.app.invalidate()
                raise
            committed = True
            previous_namespace = self.namespace
            self.namespace = prefix
            self._applied = sources
            self._project_instructions = prompt
            self._builtin_tools = tuple(candidate_registry.tools)
            self.session.registry.builtin_names = candidate_registry.builtin_names
            self._component_classes = {key: classes.get(value, value) for key, value in declared.items()}
            self.generation += 1
            self._notice(f"Applied component reload {self.generation}.")
        finally:
            # Each resource uses the cleanup implementation that created it.
            try:
                resources = Lifetime("retired generation" if committed else "rejected generation")
                for closer in reversed(retired_closers if committed else candidate_closers):
                    resources.defer(closer)
                try:
                    await resources.aclose()
                except Exception as exc:
                    self._notice(f"Could not close discarded tool resources: {exc}", error=True)
            finally:
                if not committed:
                    _drop_namespace(prefix)
                if previous_namespace is not None:
                    _drop_namespace(previous_namespace)

    def close(self) -> None:
        if self.namespace is not None:
            _drop_namespace(self.namespace)
            self.namespace = None


def main(argv: list[str] | None = None) -> int:
    """Stable process entry point; CLI components own commands and presentation."""
    from .cli import main as cli_main

    return cli_main(argv)
