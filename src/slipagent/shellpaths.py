"""Conservative, non-executing shell file-reference analysis for inspection."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_language


def shell_file_references(command: str, root: Path) -> dict[str, Any]:
    """Return inferred references and explicit gaps; never authorize an operation."""
    raw = command.encode("utf-8")
    tree = Parser(get_language("bash")).parse(raw)
    references: dict[tuple[str, str], dict[str, str]] = {}
    gaps: list[str] = []

    def gap(reason: str) -> None:
        if reason not in gaps:
            gaps.append(reason)

    def text(node: Node) -> str:
        return raw[node.start_byte:node.end_byte].decode("utf-8")

    def literal(node: Node) -> str | None:
        if node.type not in {"word", "number", "string", "raw_string", "concatenation", "command_name"}:
            return None
        pending = [node]
        while pending:
            current = pending.pop()
            if current.type in {"simple_expansion", "expansion", "command_substitution", "process_substitution"}:
                return None
            if current.type == "word" and any(char in text(current) for char in "*?[]{}~"):
                return None
            pending.extend(current.named_children)
        source = text(node)
        # Quoted wildcard characters are literal. Unquoted expansion needs the shell.
        if node.type == "word" and any(char in source for char in "*?[]{}~"):
            return None
        try:
            words = shlex.split(source)
        except ValueError:
            return None
        return words[0] if len(words) == 1 else None

    def add(value: str, mode: str, cwd: Path | None) -> None:
        if value in {"", "-", "/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr"}:
            return
        path = Path(value)
        if not path.is_absolute():
            if cwd is None:
                gap("Relative paths followed an unresolved working-directory change.")
                return
            path = cwd / path
        # Lexical normalization avoids filesystem I/O and follows no symlinks.
        name = os.path.normpath(str(path))
        references[(name, mode)] = {"path": name, "access": mode, "confidence": "inferred"}

    def classify(words: list[str], cwd: Path | None) -> Path | None:
        words = _unwrap(words)
        if not words:
            gap("A command wrapper or invocation could not be classified.")
            return cwd
        name, args = Path(words[0]).name, words[1:]
        if name == "cd":
            if len(args) != 1 or args[0].startswith("-") or cwd is None:
                gap("A working-directory change could not be resolved.")
                return None
            return Path(os.path.normpath(str(cwd / args[0])))
        if name in {"pushd", "popd", "eval", "source", "."}:
            gap(f"Working-directory effects of {name} were not classified.")
            return None
        if name in {"echo", "printf", "true", "false", ":", "pwd", "sleep"}:
            return cwd
        if name == "find":
            while args and args[0] in {"-H", "-L", "-P"}:
                args = args[1:]
            roots = []
            for value in args:
                if value.startswith("-") or value in {"!", "("}:
                    break
                roots.append(value)
            for value in roots or ["."]:
                add(value, "read", cwd)
            if any(value in {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"} for value in args):
                gap("find actions may have additional file effects.")
            return cwd
        operands, options, unknown = _operands(name, args)
        if unknown:
            gap(f"Unrecognized options or command semantics for {name}.")
            return cwd
        for value, mode in options:
            if mode in {"read", "write"}:
                add(value, mode, cwd)
        if name in {"cp", "mv", "ln", "install"}:
            destination = next((value for value, mode in options if mode == "write"), None)
            sources = operands if destination is not None else operands[:-1]
            for value in sources:
                add(value, "read" if name != "mv" else "read/write", cwd)
            if destination is None and operands:
                add(operands[-1], "write", cwd)
        elif name in {"rm", "rmdir", "mkdir", "touch", "truncate", "tee"}:
            for value in operands:
                add(value, "delete" if name in {"rm", "rmdir"} else "write", cwd)
        elif name in {"grep", "rg", "sed", "awk"}:
            has_script = any(mode == "pattern" for _, mode in options)
            files = operands if has_script else operands[1:]
            inplace = any(mode == "inplace" for _, mode in options)
            for value in files:
                if name == "awk" and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", value):
                    continue
                add(value, "read/write" if inplace else "read", cwd)
                for suffix, mode in options:
                    if mode == "backup" and suffix:
                        if "*" in suffix:
                            gap("A sed backup suffix required filename expansion.")
                        else:
                            add(value + suffix, "write", cwd)
        elif name == "dd":
            for value in operands:
                if value.startswith(("if=", "of=")):
                    add(value[3:], "read" if value.startswith("if=") else "write", cwd)
        elif name in {"cat", "head", "tail", "wc", "less", "more", "sort", "diff", "cmp", "ls", "stat", "file", "cut", "gcc", "g++", "cc", "clang", "clang++", "rustc"}:
            for value in operands:
                add(value, "read", cwd)
        else:
            gap(f"File effects of {name} were not classified.")
        return cwd

    def visit(node: Node, cwd: Path | None) -> Path | None:
        if node.type in {"program", "list"}:
            uncertain_flow = any(child.type in {"||", "&"} for child in node.children)
            for child in node.named_children:
                previous = cwd
                cwd = visit(child, cwd)
                if uncertain_flow and previous != cwd:
                    cwd = None
            # An OR branch can leave either cwd active.
            if node.type == "list" and any(child.type == "||" for child in node.children):
                gap("Conditional execution may leave a different working directory.")
                return None
            return cwd
        if node.type == "pipeline":
            for child in node.named_children:
                visit(child, cwd)
            return cwd
        if node.type == "subshell":
            local = cwd
            for child in node.named_children:
                local = visit(child, local)
            return cwd
        if node.type == "redirected_statement":
            # Redirection paths resolve before a command such as cd changes cwd.
            redirects = list(node.named_children)
            while redirects:
                child = redirects.pop(0)
                if child.type == "file_redirect":
                    target = child.child_by_field_name("destination")
                    operator = next((text(part) for part in child.children if not part.is_named), "")
                    if operator in {">&", "<&"}:
                        if target is not None and literal(target) not in {None, "-"} and not str(literal(target)).isdigit():
                            gap("A descriptor redirection could not be classified.")
                        continue  # Descriptor duplication is not a pathname.
                    value = literal(target) if target is not None else None
                    if value is None:
                        gap("A redirection target required shell expansion.")
                    else:
                        mode = "read/write" if operator == "<>" else "write" if ">" in operator else "read"
                        add(value, mode, cwd)
                elif child.type in {"heredoc_redirect", "herestring_redirect"}:
                    redirects.extend(part for part in child.named_children if part.type == "file_redirect")
                    pending = list(child.named_children)
                    while pending:
                        part = pending.pop()
                        if part.type in {"command_substitution", "process_substitution"}:
                            gap("A here-document or here-string contained executable expansion.")
                        pending.extend(part.named_children)
            body = node.child_by_field_name("body")
            return visit(body, cwd) if body is not None else cwd
        if node.type == "command":
            for child in node.named_children:
                if child.type == "variable_assignment":
                    visit(child, cwd)
            fields = [child for index, child in enumerate(node.children)
                      if node.field_name_for_child(index) in {"name", "argument"}]
            words = [literal(child) for child in fields]
            if any(word is None for word in words):
                gap("An invocation contained dynamic arguments or shell expansion.")
                if words and words[0] == "cd":
                    return None
                return cwd
            return classify([word for word in words if word is not None], cwd)
        if node.type == "variable_assignment":
            pending = list(node.named_children)
            while pending:
                part = pending.pop()
                if part.type in {"command_substitution", "process_substitution"}:
                    gap("A variable assignment contained executable expansion.")
                pending.extend(part.named_children)
            return cwd
        if node.type == "comment":
            return cwd
        gap(f"Shell syntax {node.type} was not classified.")
        return None  # Unsupported control flow may change cwd.

    if tree.root_node.has_error:
        gap("Shell syntax could not be parsed completely.")
    else:
        visit(tree.root_node, root)
    return {"references": list(references.values()), "incomplete": bool(gaps), "limitations": gaps,
            "method": "Static Bash analysis; references do not confirm execution or filesystem changes."}


def _unwrap(words: list[str]) -> list[str]:
    """Peel only wrappers whose literal argument boundaries are understood."""
    while words:
        name = Path(words[0]).name
        if name in {"command", "exec", "nohup"}:
            words = words[1:]
            if words and words[0] == "--":
                words = words[1:]
            if words and words[0].startswith("-"):
                return []
        elif name == "env":
            words = words[1:]
            while words and (re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", words[0]) or words[0] in {"--", "-i", "--ignore-environment"}):
                words = words[1:]
            if words and words[0].startswith("-"):
                return []
        elif name == "timeout":
            words = words[1:]
            if not words or words[0].startswith("-"):
                return []
            words = words[1:]
        elif name in {"nice", "sudo"}:
            wrapper = name
            words = words[1:]
            while words and words[0].startswith("-"):
                option = words.pop(0)
                if option == "--":
                    break
                takes_value = {"-n", "--adjustment"} if wrapper == "nice" else {"-u", "-g", "--user", "--group"}
                flags = set() if wrapper == "nice" else {"-n", "-E", "-H"}
                if option in takes_value and words:
                    words = words[1:]
                elif option not in flags:
                    return []
        else:
            break
    return words


def _operands(name: str, args: list[str]) -> tuple[list[str], list[tuple[str, str]], bool]:
    """Separate known option values from positional operands without guessing."""
    positional: list[str] = []
    options: list[tuple[str, str]] = []
    values: dict[str, str] = {
        "-t": "write", "--target-directory": "write",
    } if name in {"cp", "mv", "ln", "install"} else {}
    flag_options = {
        "cp": "abdfHilLnPpRrsTuvx", "mv": "bfinTuv", "ln": "bdfinPrsTv", "install": "bCdDpsTv",
        "rm": "dfirRv", "mkdir": "pv", "rmdir": "pv", "touch": "achm",
        "cat": "AbEenstTuv", "wc": "clmwL", "less": "EFfiMNmNqQrRsSX",
        "more": "dflpsu", "diff": "abBcdHinqrstTuwy", "cmp": "bls",
        "ls": "aAbBcCdDfFgGhHiLlnNopqQrRsStuUvXxZ1", "stat": "fLt",
        "file": "bchiLNrsvz", "tee": "aip", "sort": "bcdfigMnrsuVzhR",
    }
    flags = set(flag_options.get(name, ""))
    long_flags = {"--recursive", "--force", "--verbose", "--parents", "--no-clobber", "--interactive"} if flags else set()
    if name in {"mkdir", "install"}:
        values.update({"-m": "ignore", "--mode": "ignore"})
    command_values = {
        "touch": {"-r": "read", "--reference": "read", "-d": "ignore", "--date": "ignore", "-t": "ignore"},
        "stat": {"-c": "ignore", "--format": "ignore", "--printf": "ignore"},
        "diff": {"-L": "ignore", "--label": "ignore", "-D": "ignore", "-x": "ignore", "-X": "read"},
        "cmp": {"-i": "ignore", "--ignore-initial": "ignore", "-n": "ignore", "--bytes": "ignore"},
        "sort": {"-S": "ignore", "--buffer-size": "ignore", "-k": "ignore", "--key": "ignore", "-t": "ignore", "--field-separator": "ignore", "-T": "write"},
        "ls": {"-I": "ignore", "--ignore": "ignore", "-w": "ignore", "--width": "ignore", "-T": "ignore", "--tabsize": "ignore"},
        "file": {"-m": "read", "--magic-file": "read"},
        "install": {"-o": "ignore", "--owner": "ignore", "-g": "ignore", "--group": "ignore"},
    }
    values.update(command_values.get(name, {}))
    if name in {"grep", "rg", "sed", "awk"}:
        values.update({"-e": "pattern", "--regexp": "pattern", "-f": "read", "--file": "read"})
        if name in {"grep", "rg"}:
            long_flags = {"--line-number", "--ignore-case", "--fixed-strings", "--invert-match", "--files-with-matches", "--hidden", "--no-ignore", "--count", "--word-regexp"}
            values.update({"-g": "ignore", "--glob": "ignore", "-m": "ignore", "--max-count": "ignore", "-A": "ignore", "-B": "ignore", "-C": "ignore", "--type": "ignore", "-t": "ignore"})
            flags = set("rinvwlHhFsEqoxcUuLz")
        elif name == "sed":
            flags = set("nErz")
        else:
            values.update({"-F": "ignore", "-v": "ignore"})
    if name in {"head", "tail"}:
        values.update({"-n": "ignore", "--lines": "ignore", "-c": "ignore", "--bytes": "ignore"})
        flags = set("qvfF")
    if name in {"gcc", "g++", "cc", "clang", "clang++", "rustc", "sort"}:
        values.update({"-o": "write", "--output": "write"})
        if name != "sort":
            flags = set("cSgOsvrnu")
            values.update({"-I": "ignore", "-D": "ignore", "-L": "ignore", "-l": "ignore"})
    if name == "cut":
        values.update({"-d": "ignore", "-f": "ignore", "-b": "ignore", "-c": "ignore"})
    if name == "truncate":
        values.update({"-s": "ignore", "--size": "ignore", "-r": "read", "--reference": "read"})
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg == "--":
            positional.extend(args[index:])
            break
        if not arg.startswith("-") or arg == "-":
            positional.append(arg)
            continue
        if name == "sed" and (arg.startswith("-i") or arg == "--in-place" or arg.startswith("--in-place=")):
            options.append(("", "inplace"))
            suffix = arg[2:] if arg.startswith("-i") else arg.partition("=")[2]
            options.append((suffix, "backup"))
            continue
        if arg in long_flags:
            continue
        key, separator, value = arg.partition("=")
        if key in values:
            if not separator:
                if index == len(args):
                    return positional, options, True
                value = args[index]
                index += 1
            mode = values[key]
        elif len(arg) > 2 and arg[:2] in values and not arg.startswith("--"):
            value, mode = arg[2:], values[arg[:2]]
        elif not arg.startswith("--") and all(char in flags for char in arg[1:]):
            continue
        else:
            return positional, options, True
        options.append((value, mode))
        if mode == "read" and name in {"grep", "rg", "sed", "awk"}:
            options.append(("", "pattern"))
    return positional, options, False
