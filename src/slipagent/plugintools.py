"""Plugin file tooling: copying application methods into plugins, baselines, diffs, and restamping.

Plugin files are read statically here, never executed. `/plugin copy` stamps
every copied method with its fingerprint and saves the application source it
copied into `plugins/.baseline/<file>.json`; `/plugin diff` compares that
baseline with the current application method, and `/plugin restamp` records
the current fingerprint and source once the change has been reviewed.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from . import plugins
from .runtime import CORE_MODULES

PLUGIN_IMPORT = "from slipagent.plugins import overrides, original, replaces"
# Decorators `@overrides` can sit above; anything else may not produce a function.
_STAMPABLE_DECORATORS = frozenset({"staticmethod", "classmethod", "property", "abstractmethod"})


class PluginToolError(Exception):
    """A request the plugin tools cannot carry out, with a message for the user."""


@dataclass
class Stamped:
    """One method defined in a plugin's stand-in class."""

    component: str
    class_name: str
    method: str
    fingerprint: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef
    stamp: ast.Constant | None

    @property
    def member_key(self) -> str:
        return f"{self.class_name}.{self.method}"

    @property
    def label(self) -> str:
        return f"{self.component}.{self.class_name}.{self.method}"


def _decorator_call(node: ast.expr, name: str) -> ast.Call | None:
    if isinstance(node, ast.Call) and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) \
            and isinstance(node.args[0].value, str):
        function = node.func
        called = function.id if isinstance(function, ast.Name) else \
            function.attr if isinstance(function, ast.Attribute) else ""
        if called == name:
            return node
    return None


def _replaces(node: ast.ClassDef) -> tuple[str, str] | None:
    for decorator in node.decorator_list:
        call = _decorator_call(decorator, "replaces")
        if call is not None:
            assert isinstance(call.args[0], ast.Constant)
            component, _, class_name = str(call.args[0].value).rpartition(".")
            if component and class_name:
                return component, class_name
    return None


def _first_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    return min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])


class PluginTools:
    """File operations behind `/plugin`; each returns text lines for the terminal."""

    copied_digest: str | None = None

    def directory(self) -> Path:
        return plugins.plugin_directory()

    def resolve_file(self, name: str) -> Path:
        name = name.strip()
        if not name:
            raise PluginToolError("name a plugin file")
        if "/" in name or os.sep in name or name.startswith((".", "_")):
            raise PluginToolError(f"{name} is not a plugin file name")
        return self.directory() / (name if name.endswith(".py") else name + ".py")

    # Application source.

    def application_source(self, component: str) -> str:
        if component in CORE_MODULES:
            raise PluginToolError(f"{component} is part of the stable frame and cannot be overridden")
        path = plugins.PACKAGE_ROOT / plugins.component_file(component)
        try:
            return path.read_text()
        except OSError:
            raise PluginToolError(f"module {component} does not exist") from None

    def targets(self) -> list[str]:
        """Every `module.Class` a plugin may replace."""
        found: list[str] = []
        for path in sorted(plugins.PACKAGE_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(plugins.PACKAGE_ROOT).as_posix()
            component = relative.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
            if relative == "__init__.py" or component in CORE_MODULES:
                continue
            try:
                tree = ast.parse(path.read_bytes())
            except SyntaxError:
                continue
            found.extend(f"{component}.{node.name}" for node in tree.body if isinstance(node, ast.ClassDef))
        return found

    def class_node(self, component: str, class_name: str) -> tuple[str, ast.ClassDef]:
        source = self.application_source(component)
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return source, node
        raise PluginToolError(f"class {component}.{class_name} does not exist")

    def methods(self, target: str) -> list[str]:
        component, _, class_name = target.rpartition(".")
        _, node = self.class_node(component, class_name)
        return list(dict.fromkeys(member.name for member in node.body
                                  if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))))

    def split_target(self, target: str) -> tuple[str, str, str | None]:
        """`module.Class` or `module.Class.method`, checked against the application."""
        parts = target.strip().split(".")
        for size in (len(parts) - 2, len(parts) - 1):
            component, class_name = ".".join(parts[:size]), parts[size] if size < len(parts) else ""
            path = plugins.PACKAGE_ROOT / plugins.component_file(component)
            if size < 1 or not path.is_file():
                continue
            try:
                classes = {node.name for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.ClassDef)}
            except SyntaxError:
                continue
            if class_name not in classes:
                continue
            self.application_source(component)
            method = parts[-1] if size == len(parts) - 2 else None
            if method is not None and method not in self.methods(f"{component}.{class_name}"):
                raise PluginToolError(f"{component}.{class_name}.{method} does not exist")
            return component, class_name, method
        raise PluginToolError(f"{target} names no module.Class or module.Class.method")

    def method_source(self, source: str, node: ast.ClassDef, method: str) -> str:
        """Every definition of `method` in the class (property setters included), dedented."""
        lines = source.splitlines()
        pieces = [textwrap.dedent("\n".join(lines[_first_line(member) - 1:member.end_lineno]))
                  for member in node.body
                  if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == method]
        return "\n\n".join(pieces)

    def stamped_source(self, source: str, node: ast.ClassDef, method: str) -> str:
        """The method's source with `@overrides` for its current fingerprint above the first definition."""
        text = self.method_source(source, node, method)
        first = next(member for member in node.body
                     if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == method)
        names = {decorator.id if isinstance(decorator, ast.Name) else
                 decorator.attr if isinstance(decorator, ast.Attribute) else "" for decorator in first.decorator_list}
        if not names <= _STAMPABLE_DECORATORS:
            return text
        stamp = plugins.fingerprint(source, node.name, method)
        return f'@overrides("{stamp}")\n{text}'

    # Baselines.

    def baseline_path(self, path: Path) -> Path:
        return self.directory() / ".baseline" / (path.name + ".json")

    def load_baseline(self, path: Path) -> dict[str, str]:
        try:
            data = json.loads(self.baseline_path(path).read_text())
        except (OSError, ValueError):
            return {}
        return {key: value for key, value in data.items() if isinstance(value, str)} if isinstance(data, dict) else {}

    def save_baseline(self, path: Path, entries: dict[str, str]) -> None:
        baseline = {**self.load_baseline(path), **entries}
        target = self.baseline_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)

    # Reading plugin files.

    def read_plugin(self, path: Path) -> str:
        try:
            return path.read_text()
        except FileNotFoundError:
            raise PluginToolError(f"{path.name} does not exist in {self.directory()}") from None

    def stand_ins(self, source: str) -> list[tuple[str, str, ast.ClassDef]]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise PluginToolError(f"the plugin has a syntax error: {exc}") from None
        return [(*target, node) for node in tree.body
                if isinstance(node, ast.ClassDef) and (target := _replaces(node)) is not None]

    def stamped(self, source: str) -> list[Stamped]:
        found: list[Stamped] = []
        for component, class_name, node in self.stand_ins(source):
            for member in node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) or \
                        any(item.method == member.name and item.class_name == class_name for item in found):
                    continue
                stamp = next((call.args[0] for decorator in member.decorator_list
                              if (call := _decorator_call(decorator, "overrides")) is not None), None)
                assert stamp is None or isinstance(stamp, ast.Constant)
                found.append(Stamped(component, class_name, member.name,
                                     None if stamp is None else str(stamp.value), member, stamp))
        return found

    # Commands.

    def copy(self, target: str, file_name: str = "") -> tuple[Path, list[str]]:
        """Write a class's methods, or one method, into a plugin file; returns the file and member keys."""
        self.copied_digest = None
        component, class_name, method = self.split_target(target)
        source, node = self.class_node(component, class_name)
        methods = [method] if method is not None else self.methods(f"{component}.{class_name}")
        if not methods:
            raise PluginToolError(f"{component}.{class_name} defines no methods")
        path = self.resolve_file(file_name or f"{component.replace('.', '-')}-{class_name}".lower())
        blocks = [self.stamped_source(source, node, name) for name in methods]
        existing = path.exists()
        if existing:
            text = self.append(path, component, class_name, methods, blocks)
        else:
            body = textwrap.indent("\n\n".join(blocks), "    ")
            text = f'{PLUGIN_IMPORT}\n\n\n@replaces("{component}.{class_name}")\nclass {class_name}:\n{body}\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_baseline(path, {f"{class_name}.{name}": self.method_source(source, node, name) for name in methods})
        content = text.encode("utf-8")
        path.write_bytes(content)
        if not existing:
            # Only a wholly generated file is approved automatically. Appending
            # copied methods cannot grant trust to pre-existing Python code.
            self.copied_digest = plugins.source_digest(content)
        return path, [f"{class_name}.{name}" for name in methods]

    def append(self, path: Path, component: str, class_name: str, methods: list[str], blocks: list[str]) -> str:
        """Add methods to the file's stand-in for the class, or a new stand-in for another class of its module."""
        source = self.read_plugin(path)
        stand_ins = self.stand_ins(source)
        others = {target for target, _, _ in stand_ins} - {component}
        if others:
            raise PluginToolError(f"{path.name} replaces classes of {', '.join(sorted(others))}; "
                                  "one file may target one module, so name another file")
        lines = source.splitlines()
        for _, name, node in stand_ins:
            if name != class_name:
                continue
            defined = {member.name for member in node.body if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if repeated := [method for method in methods if method in defined]:
                raise PluginToolError(f"{path.name} already defines {class_name}.{', '.join(repeated)}")
            assert node.end_lineno is not None
            lines[node.end_lineno:node.end_lineno] = ["", *textwrap.indent("\n\n".join(blocks), "    ").splitlines()]
            break
        else:
            body = textwrap.indent("\n\n".join(blocks), "    ")
            lines += ["", "", f'@replaces("{component}.{class_name}")', f"class {class_name}:", *body.splitlines()]
        return self.with_import("\n".join(lines) + "\n", "overrides")

    def with_import(self, source: str, name: str) -> str:
        """Add `name` to the file's `from slipagent.plugins import` line, or add that line."""
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "slipagent.plugins":
                if any(alias.name == name for alias in node.names):
                    return source
                assert node.end_lineno is not None
                lines.insert(node.end_lineno, f"from slipagent.plugins import {name}")
                return "\n".join(lines) + "\n"
        return f"from slipagent.plugins import {name}\n" + source

    def diff(self, file_name: str = "") -> list[str]:
        """Unified diffs from each changed method's saved baseline to the current application method."""
        paths = [self.resolve_file(file_name)] if file_name.strip() else plugins.plugin_files()
        output: list[str] = []
        for path in paths:
            source = self.read_plugin(path)
            baseline = self.load_baseline(path)
            for item in self.stamped(source):
                if item.fingerprint is None:
                    continue
                try:
                    application, node = self.class_node(item.component, item.class_name)
                except PluginToolError as exc:
                    output.append(f"{path.name}: {exc}")
                    continue
                current = plugins.fingerprint(application, item.class_name, item.method)
                if current == item.fingerprint:
                    continue
                if current is None:
                    output.append(f"{path.name} → {item.member_key}: the application no longer defines it")
                    continue
                saved = baseline.get(item.member_key)
                if saved is None:
                    output.append(f"{path.name} → {item.member_key} changed upstream; no baseline was saved "
                                  "(methods written by /plugin copy keep one)")
                    continue
                output.extend(difflib.unified_diff(
                    saved.splitlines(), self.method_source(application, node, item.method).splitlines(),
                    f"{path.name} → {item.member_key} (when copied)", f"{item.label} (now)", lineterm=""))
        return output

    def restamp(self, file_name: str, method: str = "") -> list[str]:
        """Record current fingerprints and baselines for changed or unstamped overrides; returns member keys."""
        path = self.resolve_file(file_name)
        source = self.read_plugin(path)
        wanted = method.strip()
        lines = source.splitlines()
        edits: list[tuple[int, Stamped, str]] = []
        baseline: dict[str, str] = {}
        for item in self.stamped(source):
            if wanted and wanted not in {item.method, item.member_key}:
                continue
            application, node = self.class_node(item.component, item.class_name)
            current = plugins.fingerprint(application, item.class_name, item.method)
            if current is None or current == item.fingerprint and not wanted:
                continue
            edits.append((item.node.lineno if item.stamp is None else item.stamp.lineno, item, current))
            baseline[item.member_key] = self.method_source(application, node, item.method)
        if wanted and not edits:
            raise PluginToolError(f"{path.name} has no override of an application method named {wanted}")
        # Bottom-up, so earlier line numbers stay valid.
        for _, item, current in sorted(edits, key=lambda edit: -edit[0]):
            if item.stamp is not None:
                assert item.stamp.end_col_offset is not None
                row = item.stamp.lineno - 1
                # AST columns count UTF-8 bytes.
                line = lines[row].encode()
                lines[row] = (line[:item.stamp.col_offset] + f'"{current}"'.encode()
                              + line[item.stamp.end_col_offset:]).decode()
            else:
                row = _first_line(item.node) - 1
                indent = lines[row][:len(lines[row]) - len(lines[row].lstrip())]
                lines.insert(row, f'{indent}@overrides("{current}")')
        if edits:
            text = "\n".join(lines) + "\n"
            if any(item.stamp is None for _, item, _ in edits):
                text = self.with_import(text, "overrides")
            self.save_baseline(path, baseline)
            path.write_text(text)
        return [item.member_key for _, item, _ in edits]
