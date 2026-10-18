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
NO_PLUGINS_ENV = "SLIPAGENT_NO_PLUGINS"
FINGERPRINT_LENGTH = 12
# Class attributes Python creates for every class body; never treated as overrides.
_CLASS_BOOKKEEPING = frozenset({
    "__module__", "__qualname__", "__doc__", "__dict__", "__weakref__", "__firstlineno__",
    "__static_attributes__", "__annotations__", "__annotate__", "__annotate_func__",
    "__annotations_cache__", "__slipagent_replaces__", "__orig_bases__", "__type_params__",
})

# Set by `--no-plugins` for this process.
disabled = False
# Live classes receive plugins once per process; reloads apply them to each candidate.
applied_to_live = False
# The latest status of every plugin file, from startup or the accepted reload.
reports: list[PluginReport] = []
# Override code object -> the implementation it replaced, for `original`.
_previous: weakref.WeakKeyDictionary[types.CodeType, Any] = weakref.WeakKeyDictionary()


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


def plugin_sources() -> dict[str, bytes]:
    return {source_key(path): path.read_bytes() for path in plugin_files()}


def plugin_stamps() -> dict[str, tuple[int, int, int]]:
    stamps = {}
    for path in plugin_files():
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


def original(receiver: Any, *args: Any, **kwargs: Any) -> Any:
    """Call the implementation the calling override replaced."""
    frame = sys._getframe(1)
    while frame is not None:
        previous = _previous.get(frame.f_code)
        if previous is not None:
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


@dataclass
class PluginReport:
    name: str
    path: str
    target: str = ""
    enabled: bool = False
    error: str = ""
    overrides: list[Override] = field(default_factory=list)

    def notices(self) -> Iterator[str]:
        if self.error:
            yield f"Plugin {self.name} disabled: {self.error}"
            return
        for override in self.overrides:
            if override.status == "changed":
                yield (f"{self.name} overrides {override.target}.{override.member}, which has changed since "
                       "this override was written. Review it with /plugin diff, then restamp it.")
            elif override.status == "unstamped":
                yield (f"{self.name} overrides {override.target}.{override.member} without @overrides; "
                       f"add @overrides(\"{override.fingerprint}\") to be warned when it changes.")


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


def overlay(module_of: Callable[[str], types.ModuleType | None], source_of: Callable[[str], bytes | None],
            plugins: dict[str, bytes], core: frozenset[str],
            allow: Callable[[str, str], bool] | None = None) -> list[PluginReport]:
    """Apply plugin members onto component classes; one failing plugin leaves no trace.

    `module_of` returns a component module (for example "cli") from the live
    package or a reload candidate. `source_of` returns its application source for
    fingerprints. `allow(plugin, "Class.member")` filters overrides.
    """
    reports = []
    for key, source in sorted(plugins.items()):
        name = key.rpartition("/")[2]
        path = str(plugin_directory() / name)
        report = PluginReport(name, path)
        reports.append(report)
        snapshots: dict[type, dict[str, Any]] = {}
        registered: list[types.CodeType] = []
        try:
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
            for stand_in in stand_ins:
                class_name = stand_in.__slipagent_replaces__[1]  # type: ignore[attr-defined]
                target = target_module.__dict__.get(class_name)
                if not isinstance(target, type) or target.__module__ != target_module.__name__:
                    raise ValueError(f"class {component}.{class_name} does not exist")
                label = f"{component}.{class_name}"
                _check_layout(stand_in, target, label)
                for symbol, value in _members(stand_in):
                    if allow is not None and not allow(name, f"{class_name}.{symbol}"):
                        continue
                    function = _function(value)
                    stamp = getattr(function, "__slipagent_fingerprint__", None)
                    existing = target.__dict__.get(symbol)
                    current = fingerprint(application, class_name, symbol) if application is not None else None
                    if existing is None:
                        if stamp is not None:
                            raise ValueError(f"{label}.{symbol} no longer exists")
                        status = "added"
                    elif stamp is None:
                        status = "unstamped"
                    else:
                        status = "ok" if stamp == current else "changed"
                    report.overrides.append(Override(label, symbol, status, current))
                    snapshots.setdefault(target, dict(target.__dict__))
                    if existing is not None and function is not None:
                        _previous[function.__code__] = existing
                        registered.append(function.__code__)
                    setattr(target, symbol, _retarget(value, stand_in, target))
            report.enabled = True
        except Exception as exc:
            for target, snapshot in snapshots.items():
                for symbol in tuple(target.__dict__):
                    if symbol not in snapshot and symbol not in _CLASS_BOOKKEEPING:
                        delattr(target, symbol)
                for symbol, value in snapshot.items():
                    if symbol not in _CLASS_BOOKKEEPING and target.__dict__.get(symbol) is not value:
                        setattr(target, symbol, value)
            for code in registered:
                _previous.pop(code, None)
            report.error = f"{type(exc).__name__}: {exc}"
            report.overrides.clear()
    return reports


def component_file(component: str) -> str:
    """Package-relative source file of a component module name."""
    relative = component.replace(".", "/")
    return relative + "/__init__.py" if (PACKAGE_ROOT / relative / "__init__.py").exists() else relative + ".py"


def apply_to_live(core: frozenset[str]) -> list[PluginReport]:
    """Apply plugins to the running package at startup, before session objects exist."""
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
    reports = overlay(module_of, source_of, plugin_sources(), core)
    return reports
