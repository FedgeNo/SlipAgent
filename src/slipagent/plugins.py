"""User plugins: method overrides for reloadable classes, loaded from the data directory.

A plugin file marks a stand-in class with `@replaces("module.Class")`; every
member it defines replaces or adds to that class, and everything else stays the
application's. `original(self, ...)` calls the implementation an override
replaced. Plugin code resolves names from its own file, then the target module,
then builtins, so methods copied out of SlipAgent run without new imports.

Compatibility is tracked per method: `@overrides(fingerprint)` records a hash of
the application method an override was written against, and only a change to
that method warns. The runtime frame applies plugins at startup and to every
reload candidate before its checks, so plugin errors share the reload rollback.
This module is part of the stable frame.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import sys
import types
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import data_directory

PACKAGE_ROOT = Path(__file__).resolve().parent
PLUGIN_KEY = ":plugins"
APPROVALS_NAME = ".approvals.json"
NO_PLUGINS_ENV = "SLIPAGENT_NO_PLUGINS"
FINGERPRINT_LENGTH = 12
_PROPERTY_ACCESSORS = {"fget": "getter", "fset": "setter", "fdel": "deleter"}
# Class attributes Python creates for every class body; never treated as overrides.
_CLASS_BOOKKEEPING = frozenset({
    "__module__", "__qualname__", "__doc__", "__dict__", "__weakref__", "__firstlineno__",
    "__static_attributes__", "__annotations__", "__annotate__", "__annotate_func__",
    "__annotations_cache__", "__slipagent_replaces__", "__orig_bases__", "__type_params__",
})

# Set by `--no-plugins` for this process.
disabled = False
# Set by `--all-plugins`: apply every override, stacking in file-name order, without saved decisions.
all_plugins = False
# Live classes receive plugins once per process; reloads apply them to each candidate.
applied_to_live = False
# The latest status of every plugin file, from startup or the accepted reload.
reports: list[PluginReport] = []
# Code identity -> weak code reference, replaced member, and property accessor.
# Equal code from separate plugins or reload generations must keep separate chains.
_previous: dict[int, tuple[weakref.ReferenceType[types.CodeType], Any, str | None]] = {}
# Original dictionaries of live classes changed by `apply_to_live`.
_live: dict[type, dict[str, Any]] = {}


def active() -> bool:
    return not disabled and os.environ.get(NO_PLUGINS_ENV, "").strip() not in {"1", "true", "yes"}


def plugin_directory() -> Path:
    return data_directory() / "plugins"


def plugin_files() -> list[Path]:
    """Plugin files in load order; a leading `_` or `.` disables a file."""
    if not active():
        return []
    directory = plugin_directory()
    try:
        entries = sorted(directory.iterdir())
    except FileNotFoundError:
        return []
    return [path for path in entries
            if path.suffix == ".py" and path.is_file() and not path.name.startswith(("_", "."))]


def source_key(path: Path) -> str:
    """Key for a plugin file among the frame's sources, distinct from package paths."""
    return f"{PLUGIN_KEY}/{path.name}"


def _watched() -> list[Path]:
    """Plugin files plus saved decisions, so changing either starts a reload."""
    files = plugin_files()
    if files and (plugin_directory() / APPROVALS_NAME).is_file():
        files.append(plugin_directory() / APPROVALS_NAME)
    return files


def plugin_sources() -> dict[str, bytes]:
    return {source_key(path): path.read_bytes() for path in _watched()}


def plugin_stamps() -> dict[str, tuple[int, int, int]]:
    stamps = {}
    for path in _watched():
        metadata = path.stat()
        stamps[source_key(path)] = (metadata.st_ino, metadata.st_mtime_ns, metadata.st_size)
    return stamps


# Public API used inside plugin files.

def replaces(target: str) -> Callable[[type], type]:
    """Mark a class whose members override `target` ("module.Class")."""
    module, _, name = target.rpartition(".")
    if not module or not name:
        raise ValueError(f"@replaces needs 'module.Class', not {target!r}")

    def mark(cls: type) -> type:
        cls.__slipagent_replaces__ = (module, name)  # type: ignore[attr-defined]
        return cls
    return mark


def _function(member: Any) -> types.FunctionType | None:
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    elif isinstance(member, property):
        member = member.fget
    return member if isinstance(member, types.FunctionType) else None


def overrides(fingerprint: str) -> Callable[[Any], Any]:
    """Record the fingerprint of the application method this member replaces."""
    def mark(member: Any) -> Any:
        function = _function(member)
        if function is None:
            raise TypeError("@overrides applies to methods")
        function.__slipagent_fingerprint__ = fingerprint  # type: ignore[attr-defined]
        return member
    return mark


def _register_original(code: types.CodeType, previous: Any, accessor: str | None) -> None:
    key = id(code)

    def discard(reference: weakref.ReferenceType[types.CodeType]) -> None:
        registration = _previous.get(key)
        if registration is not None and registration[0] is reference:
            del _previous[key]

    _previous[key] = weakref.ref(code, discard), previous, accessor


def original(receiver: Any, *args: Any, **kwargs: Any) -> Any:
    """Call the implementation the calling override replaced."""
    frame = sys._getframe(1)
    while frame is not None:
        registration = _previous.get(id(frame.f_code))
        if registration is not None and registration[0]() is frame.f_code:
            _, previous, accessor = registration
            if accessor is not None:
                function = getattr(previous, accessor)
                if function is None:
                    raise AttributeError(f"original property has no {_PROPERTY_ACCESSORS[accessor]}")
                # Binding a property evaluates its getter; call the recorded
                # accessor directly so setters and deleters retain their role.
                return function(receiver, *args, **kwargs)
            owner = receiver if isinstance(receiver, type) else type(receiver)
            bound = previous.__get__(None if isinstance(receiver, type) else receiver, owner)
            return bound(*args, **kwargs)
        frame = frame.f_back  # type: ignore[assignment]
    raise RuntimeError("original() must be called from a plugin method that overrides an application method")


# Fingerprints.

def _method_node(source: bytes | str, class_name: str, method_name: str) -> ast.AST | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == method_name:
                    return member
    return None


def fingerprint(source: bytes | str, class_name: str, method_name: str) -> str | None:
    """Hash of a method's code, ignoring comments, formatting, and its docstring."""
    node = _method_node(source, class_name, method_name)
    if node is None:
        return None
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        node = type(node)(**{**{name: getattr(node, name) for name in node._fields}, "body": body[1:] or [ast.Pass()]})
    dump = ast.dump(node, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()[:FINGERPRINT_LENGTH]


# Loading and applying.

class _Names(dict[str, Any]):
    """Builtins for plugin code: the target module's names, then real builtins."""

    def __init__(self) -> None:
        super().__init__()
        self.module: types.ModuleType | None = None

    def __missing__(self, key: str) -> Any:
        if self.module is not None and key in self.module.__dict__:
            return self.module.__dict__[key]
        return builtins.__dict__[key]


@dataclass
class Override:
    target: str
    member: str
    status: str  # "ok", "changed", "unstamped", or "added"
    fingerprint: str | None = None
    # "on", "pending" (no decision), "rejected", "conflict" (unresolved), or "dropped" (another plugin chosen).
    state: str = "on"
    calls_original: bool = False
    # Plugins whose approved overrides target this member, when more than one does.
    contenders: list[str] = field(default_factory=list)
    # The plugin whose override runs first and never calls original(), so this one never runs.
    shadowed_by: str = ""

    @property
    def member_key(self) -> str:
        return f"{self.target.rpartition('.')[2]}.{self.member}"

    @property
    def label(self) -> str:
        return f"{self.target}.{self.member}"


@dataclass
class PluginReport:
    name: str
    path: str
    target: str = ""
    enabled: bool = False
    error: str = ""
    overrides: list[Override] = field(default_factory=list)
    digest: str = ""
    code_state: str = "on"

    def notices(self) -> Iterator[str]:
        if self.error:
            yield f"Plugin {self.name} disabled: {self.error}"
            return
        if self.code_state != "on":
            if self.code_state == "pending":
                yield f"{self.name} has unapproved Python code; review and approve this version with /plugins."
            return
        for override in self.overrides:
            if override.state != "on":
                continue
            if override.status == "changed":
                yield (f"{self.name} overrides {override.label}, which has changed since this override "
                       f"was written. Review it with /plugin diff {self.name}, then /plugin restamp {self.name}.")
            elif override.status == "unstamped":
                yield (f"{self.name} overrides {override.label} without @overrides; "
                       f"add @overrides(\"{override.fingerprint}\") to be warned when it changes.")
            if override.shadowed_by:
                yield (f"{self.name}'s override of {override.label} never runs: {override.shadowed_by} "
                       "runs before it and does not call original().")
        pending = sum(override.state == "pending" for override in self.overrides)
        if pending:
            yield (f"{self.name} has {pending} override{'' if pending == 1 else 's'} awaiting approval; "
                   "they stay off until approved with /plugins.")
        for override in self.overrides:
            if override.state == "conflict" and override.contenders[0] == self.name:
                yield (f"{override.label} is overridden by {', '.join(override.contenders)}; "
                       "none of them apply until one is chosen with /plugins.")


@dataclass
class Approvals:
    """Saved decisions: approval per plugin file and `Class.member`, and conflict choices.

    A conflict choice is the run order (first runs first) of the plugins kept for
    one `module.Class.member`, valid only while the same plugins contend for it.
    """

    approved: dict[str, dict[str, bool]] = field(default_factory=dict)
    conflicts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    code: dict[str, tuple[str, bool]] = field(default_factory=dict)
    error: str = ""

    @classmethod
    def parse(cls, content: bytes | None) -> Approvals:
        try:
            data = json.loads(content) if content else {}
            if not isinstance(data, dict):
                raise ValueError("expected an object")
            approved, conflicts, code = (data.get(name, {}) for name in ("approved", "conflicts", "code"))
            if not all(isinstance(value, dict) for value in (approved, conflicts, code)):
                raise ValueError("approved, conflicts, and code must be objects")
            for members in approved.values():
                if not isinstance(members, dict) or any(type(value) is not bool for value in members.values()):
                    raise ValueError("each approved entry must map member names to booleans")
            for choice in conflicts.values():
                if not isinstance(choice, dict) or any(
                        not isinstance(choice.get(name), list)
                        or any(not isinstance(value, str) for value in choice[name])
                        for name in ("plugins", "order")):
                    raise ValueError("each conflict needs plugins and order arrays of names")
                if (len(set(choice["plugins"])) != len(choice["plugins"])
                        or len(set(choice["order"])) != len(choice["order"])
                        or not set(choice["order"]) <= set(choice["plugins"])):
                    raise ValueError("conflict order must contain unique contenders")
            for choice in code.values():
                if (not isinstance(choice, dict) or type(choice.get("approved")) is not bool
                        or not isinstance(choice.get("sha256"), str) or len(choice["sha256"]) != 64
                        or any(character not in "0123456789abcdef" for character in choice["sha256"])):
                    raise ValueError("each code decision needs a SHA-256 digest and an approved boolean")
            return cls(approved, conflicts, {name: (choice["sha256"], choice["approved"]) for name, choice in code.items()})
        except (ValueError, UnicodeError) as exc:
            return cls(error=f"Invalid {APPROVALS_NAME}: {exc}. Saved plugin decisions are disabled.")

    def encode(self) -> bytes:
        return (json.dumps({"approved": self.approved, "conflicts": self.conflicts,
                            "code": {name: {"sha256": digest, "approved": approved}
                                     for name, (digest, approved) in self.code.items()}},
                           indent=2, sort_keys=True) + "\n").encode()

    def code_decision(self, plugin: str, digest: str) -> bool | None:
        saved = self.code.get(plugin)
        return saved[1] if saved is not None and saved[0] == digest else None

    def decide_code(self, plugin: str, digest: str, approved: bool) -> None:
        self.code[plugin] = (digest, approved)

    def decision(self, plugin: str, member: str) -> bool | None:
        return self.approved.get(plugin, {}).get(member)

    def decide(self, plugin: str, member: str, approved: bool) -> None:
        self.approved.setdefault(plugin, {})[member] = approved

    def order(self, label: str, contenders: list[str]) -> list[str] | None:
        choice = self.conflicts.get(label)
        if choice is None or sorted(choice["plugins"]) != sorted(contenders):
            return None
        return [plugin for plugin in choice["order"] if plugin in contenders] or None

    def choose(self, label: str, contenders: list[str], order: list[str]) -> None:
        self.conflicts[label] = {"plugins": sorted(contenders), "order": list(order)}


def approvals_path() -> Path:
    return plugin_directory() / APPROVALS_NAME


def load_approvals() -> Approvals:
    try:
        return Approvals.parse(approvals_path().read_bytes())
    except FileNotFoundError:
        return Approvals()
    except OSError as exc:
        return Approvals(error=f"Cannot read {approvals_path()}: {exc}. Saved plugin decisions are disabled.")


def save_approvals(approvals: Approvals) -> None:
    path = approvals_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(approvals.encode())
    os.replace(temporary, path)


def source_digest(source: bytes) -> str:
    """Approval covers every byte that Python will compile and execute."""
    return hashlib.sha256(source).hexdigest()


def _load(name: str, path: str, source: bytes) -> tuple[types.ModuleType, _Names]:
    module = types.ModuleType(f"slipagent_plugins.{Path(name).stem}")
    module.__file__ = path
    names = _Names()
    module.__dict__["__builtins__"] = names
    exec(compile(source, path, "exec"), module.__dict__)
    return module, names


def _stand_ins(module: types.ModuleType) -> list[type]:
    return [value for value in module.__dict__.values()
            if isinstance(value, type) and "__slipagent_replaces__" in value.__dict__]


def _check_layout(stand_in: type, target: type, label: str) -> None:
    bases = stand_in.__bases__
    if bases != (object,) and bases != target.__bases__:
        raise ValueError(f"{label} may omit base classes or repeat {target.__name__}'s exactly")
    if "__slots__" in stand_in.__dict__ or "__dataclass_fields__" in stand_in.__dict__:
        raise ValueError(f"{label} cannot change slots or dataclass fields")


def _retarget(value: Any, stand_in: type, target: type) -> Any:
    """Point zero-argument `super()` and `__class__` at the target instead of the stand-in."""
    if isinstance(value, staticmethod):
        return staticmethod(_retarget(value.__func__, stand_in, target))
    if isinstance(value, classmethod):
        return classmethod(_retarget(value.__func__, stand_in, target))
    if isinstance(value, property):
        return property(_retarget(value.fget, stand_in, target), _retarget(value.fset, stand_in, target),
                        _retarget(value.fdel, stand_in, target), value.__doc__)
    if not isinstance(value, types.FunctionType) or not value.__closure__:
        return value
    cells = []
    for cell in value.__closure__:
        try:
            content = cell.cell_contents
        except ValueError:
            cells.append(cell)
            continue
        cells.append(types.CellType(target) if content is stand_in else cell)
    function = types.FunctionType(value.__code__, value.__globals__, value.__name__, value.__defaults__, tuple(cells))
    function.__kwdefaults__ = value.__kwdefaults__
    function.__dict__.update(value.__dict__)
    function.__qualname__ = value.__qualname__
    function.__module__ = value.__module__
    function.__doc__ = value.__doc__
    return function


def _members(stand_in: type) -> list[tuple[str, Any]]:
    return [(symbol, value) for symbol, value in stand_in.__dict__.items()
            if symbol not in _CLASS_BOOKKEEPING]


@dataclass
class _Member:
    plugin: str
    stand_in: type
    target: type
    value: Any
    override: Override


def _calls_original(code: types.CodeType) -> bool:
    return "original" in code.co_names or any(
        isinstance(constant, types.CodeType) and _calls_original(constant) for constant in code.co_consts)


def _inspect(name: str, path: str, source: bytes, module_of: Callable[[str], types.ModuleType | None],
             source_of: Callable[[str], bytes | None], core: frozenset[str], report: PluginReport) -> list[_Member]:
    """Load one plugin and check every member against the application, without applying any."""
    module, names = _load(name, path, source)
    stand_ins = _stand_ins(module)
    if not stand_ins:
        raise ValueError("no class is marked with @replaces")
    targets = {stand_in.__slipagent_replaces__[0] for stand_in in stand_ins}  # type: ignore[attr-defined]
    if len(targets) != 1:
        raise ValueError(f"one file may target one module; found {', '.join(sorted(targets))}")
    component = targets.pop()
    report.target = component
    if component in core:
        raise ValueError(f"{component} is part of the stable frame and cannot be overridden")
    target_module = module_of(component)
    if target_module is None:
        raise ValueError(f"module {component} does not exist")
    names.module = target_module
    application = source_of(component)
    members = []
    for stand_in in stand_ins:
        class_name = stand_in.__slipagent_replaces__[1]  # type: ignore[attr-defined]
        target = target_module.__dict__.get(class_name)
        if not isinstance(target, type) or target.__module__ != target_module.__name__:
            raise ValueError(f"class {component}.{class_name} does not exist")
        label = f"{component}.{class_name}"
        _check_layout(stand_in, target, label)
        # Copied code names its own class, as in `StatusLine(session, style)`.
        # The stand-in has none of the target's members, so that name must
        # reach the target class once the file is loaded.
        for symbol in [symbol for symbol, value in module.__dict__.items() if value is stand_in]:
            module.__dict__[symbol] = target
        for symbol, value in _members(stand_in):
            function = _function(value)
            stamp = getattr(function, "__slipagent_fingerprint__", None)
            current = fingerprint(application, class_name, symbol) if application is not None else None
            if symbol not in target.__dict__:
                if stamp is not None:
                    raise ValueError(f"{label}.{symbol} no longer exists")
                status = "added"
            elif stamp is None:
                status = "unstamped"
            else:
                status = "ok" if stamp == current else "changed"
            override = Override(label, symbol, status, current,
                                calls_original=function is not None and _calls_original(function.__code__))
            members.append(_Member(name, stand_in, target, value, override))
    return members


def _restore(snapshots: dict[type, dict[str, Any]]) -> None:
    for target, snapshot in snapshots.items():
        for symbol in tuple(target.__dict__):
            if symbol not in snapshot and symbol not in _CLASS_BOOKKEEPING:
                delattr(target, symbol)
        for symbol, value in snapshot.items():
            if symbol not in _CLASS_BOOKKEEPING and target.__dict__.get(symbol) is not value:
                setattr(target, symbol, value)


def overlay(module_of: Callable[[str], types.ModuleType | None], source_of: Callable[[str], bytes | None],
            plugins: dict[str, bytes], core: frozenset[str], approvals: Approvals | None = None,
            snapshots: dict[type, dict[str, Any]] | None = None) -> list[PluginReport]:
    """Apply approved plugin members onto component classes; a failing plugin leaves no trace.

    `module_of` returns a component module (for example "cli") from the live
    package or a reload candidate. `source_of` returns its application source for
    fingerprints. Without `approvals`, every member applies and plugins that
    override the same member stack in file-name order. `snapshots` receives the
    original dictionary of every class changed.
    """
    reports = []
    if approvals is not None and approvals.error:
        reports.append(PluginReport(APPROVALS_NAME, str(approvals_path()), error=approvals.error))
    loaded: list[_Member] = []
    for key, source in sorted(plugins.items()):
        name = key.rpartition("/")[2]
        if not name.endswith(".py") or name.startswith(("_", ".")):
            continue
        path = str(plugin_directory() / name)
        report = PluginReport(name, path, digest=source_digest(source))
        reports.append(report)
        decision = True if approvals is None else approvals.code_decision(name, report.digest)
        if decision is not True:
            report.code_state = "pending" if decision is None else "rejected"
            continue
        try:
            members = _inspect(name, path, source, module_of, source_of, core, report)
        except Exception as exc:
            report.error = f"{type(exc).__name__}: {exc}"
            continue
        report.overrides = [member.override for member in members]
        report.enabled = True
        loaded.extend(members)

    contending: dict[str, list[_Member]] = {}
    for member in loaded:
        override = member.override
        decision = True if approvals is None else approvals.decision(member.plugin, override.member_key)
        override.state = "on" if decision else "pending" if decision is None else "rejected"
        if decision:
            contending.setdefault(override.label, []).append(member)
    applied: list[_Member] = []
    for label, members in contending.items():
        if len(members) > 1:
            contenders = [member.plugin for member in members]
            for member in members:
                member.override.contenders = contenders
            if approvals is not None:
                order = approvals.order(label, contenders)
                for member in members:
                    if order is None:
                        member.override.state = "conflict"
                    elif member.plugin not in order:
                        member.override.state = "dropped"
                by_plugin = {member.plugin: member for member in members}
                # The first in run order wraps the rest, so it is applied last.
                members = [by_plugin[plugin] for plugin in reversed(order or [])]
            # Running starts at the last applied and goes inward only through original().
            for outer, member in zip(reversed(members), reversed(members[:-1])):
                blocker = outer.override.shadowed_by or ("" if outer.override.calls_original else outer.plugin)
                member.override.shadowed_by = blocker
        applied.extend(members)

    touched = snapshots if snapshots is not None else {}
    registered: list[types.CodeType] = []
    try:
        for member in applied:
            target, symbol = member.target, member.override.member
            touched.setdefault(target, dict(target.__dict__))
            previous = target.__dict__.get(symbol)
            functions: list[tuple[str | None, Any]] = [(None, _function(member.value))]
            if isinstance(member.value, property) and isinstance(previous, property):
                functions = [(accessor, getattr(member.value, accessor)) for accessor in _PROPERTY_ACCESSORS]
            for accessor, function in functions:
                if previous is not None and isinstance(function, types.FunctionType):
                    _register_original(function.__code__, previous, accessor)
                    registered.append(function.__code__)
            setattr(target, symbol, _retarget(member.value, member.stand_in, target))
    except Exception as exc:
        _restore(touched)
        touched.clear()
        for code in registered:
            _previous.pop(id(code), None)
        for report in reports:
            if report.enabled:
                report.enabled = False
                report.error = f"could not apply plugins: {type(exc).__name__}: {exc}"
                report.overrides.clear()
    return reports


def pending(reports: list[PluginReport]) -> bool:
    """Whether file code, overrides, or conflicts await a decision."""
    return any(report.code_state == "pending" or any(override.state in {"pending", "conflict"}
               for override in report.overrides) for report in reports)


def component_file(component: str) -> str:
    """Package-relative source file of a component module name."""
    relative = component.replace(".", "/")
    return relative + "/__init__.py" if (PACKAGE_ROOT / relative / "__init__.py").exists() else relative + ".py"


def split_sources(sources: dict[str, bytes]) -> tuple[dict[str, bytes], Approvals | None]:
    """Plugin files and the saved decisions to apply, or None for `--all-plugins`."""
    approvals_key = f"{PLUGIN_KEY}/{APPROVALS_NAME}"
    files = {key: content for key, content in sources.items()
             if key.startswith(PLUGIN_KEY + "/") and key != approvals_key}
    return files, None if all_plugins else Approvals.parse(sources.get(approvals_key))


def apply_to_live(core: frozenset[str]) -> list[PluginReport]:
    """Apply plugins to the running package, first undoing any earlier application.

    Startup calls this before session objects exist. Without a reload frame it
    runs again after approvals change.
    """
    import importlib

    def module_of(component: str) -> types.ModuleType | None:
        if not (PACKAGE_ROOT / component_file(component)).is_file():
            return None
        return importlib.import_module(f"{__package__}.{component}")

    def source_of(component: str) -> bytes | None:
        try:
            return (PACKAGE_ROOT / component_file(component)).read_bytes()
        except OSError:
            return None

    global reports
    _restore(_live)
    _live.clear()
    files, approvals = split_sources(plugin_sources())
    reports = overlay(module_of, source_of, files, core, approvals, _live)
    return reports
