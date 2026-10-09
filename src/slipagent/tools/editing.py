"""Pure edit planning: validate against one original, then publish once.

Similarity is used only for error explanations. Replacement coordinates always
come from exact matches, with LF targets also matching CRLF or CR.
"""

from __future__ import annotations

import difflib
import heapq
import re
from collections.abc import Sequence

MAX_DIFF_CHARS = 32_000_000


def edit_pattern(old: str) -> re.Pattern[str]:
    pattern = re.escape(old)
    if "\r" not in old:
        pattern = "(?:\r\n|\r|\n)".join(re.escape(part) for part in old.split("\n"))
    return re.compile(pattern)


def missing_match_details(text: str, old: str, replacement: str) -> str:
    """Return bounded real source; never guess a replacement or another file."""
    notes = []
    existing = edit_pattern(replacement).search(text) if replacement else None
    if existing is not None:
        line_number = text.count("\n", 0, existing.start()) + 1
        notes.append(f"Replacement text already exists at line {line_number}; inspect before retrying.")
    lines = text.splitlines()
    anchor = max(old.splitlines(), key=len, default="")[:300].strip()
    words = set(re.findall(r"\w+|[^\w\s]", anchor))
    candidates = []
    # Cheap token overlap bounds the expensive character comparisons, even for
    # long generated files and repeated source lines.
    for index, line in enumerate(lines):
        sample = line[:300].strip()
        tokens = set(re.findall(r"\w+|[^\w\s]", sample))
        shared = len(words & tokens)
        if shared:
            candidates.append((shared / max(1, len(words | tokens)), index, sample))
    closest = heapq.nlargest(8, candidates)
    if closest:
        _, index, _ = max(closest, key=lambda item: difflib.SequenceMatcher(None, anchor, item[2]).ratio())
        start, end = max(0, index - 2), min(len(lines), index + 4)
        notes.append("Nearby actual source (line numbers are not part of the file):")
        for number in range(start, end):
            line = lines[number]
            notes.append(f"{number + 1:>6}\t{line}")
    notes.append("Read the file and copy the exact target text including whitespace.")
    return "\n".join(notes)


def apply_edits(text: str, edits: Sequence[tuple[str, str, bool]]) -> tuple[str, int, int]:
    """Plan non-overlapping replacements against the original file contents."""
    planned: list[tuple[int, int, str, int]] = []
    for index, (old, new, replace_all) in enumerate(edits):
        label = f"Edit {index + 1}: " if len(edits) > 1 else ""
        if not old:
            raise ValueError(label + "old_string must not be empty.")
        matches = list(edit_pattern(old).finditer(text))
        if not matches:
            raise ValueError(label + "No match for old_string.\n" + missing_match_details(text, old, new))
        if len(matches) > 1 and not replace_all:
            locations = ", ".join(str(text.count("\n", 0, match.start()) + 1) for match in matches[:5])
            raise ValueError(label + f"old_string matched {len(matches)} times (lines {locations}). "
                             "Add surrounding context to make it unique, or pass replace_all=true for a single edit.")
        for match in matches:
            newline = re.search(r"\r\n|\r|\n", match.group())
            if newline is None:
                newline = re.search(r"\r\n|\r|\n", text[match.start():])
            if newline is None:
                newline = re.search(r"\r\n|\r|\n", text)
            replacement = new
            if newline is not None and "\r" not in new:
                replacement = new.replace("\n", newline.group())
            planned.append((match.start(), match.end(), replacement, index + 1))
    planned.sort(key=lambda entry: entry[0])
    for previous, current in zip(planned, planned[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"Edits {previous[3]} and {current[3]} overlap. Merge them into one edit.")
    chunks: list[str] = []
    end = 0
    for start, stop, replacement, _ in planned:
        chunks.extend((text[end:start], replacement))
        end = stop
    chunks.append(text[end:])
    updated = "".join(chunks)
    if updated == text:
        raise ValueError("The replacements produce identical content; the file already has that text.")
    return updated, len(planned), text.count("\n", 0, planned[0][0]) + 1


def edit_diff(path: str, before: str, after: str) -> str:
    """Bound model/UI feedback; full file contents stay available to read_file."""
    lines = difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=path, tofile=path, lineterm="")
    chunks = []
    size = 0
    for line in lines:
        remaining = MAX_DIFF_CHARS - size
        if len(line) + 1 > remaining:
            chunks.append(line[:max(0, remaining)] + "\n… Diff truncated; read the file for remaining changes.")
            break
        chunks.append(line)
        size += len(line) + 1
    return "\n".join(chunks)
