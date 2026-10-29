"""Pure edit planning: validate against one original, then publish once.

Similarity is used only for error explanations. Replacement coordinates always
come from exact matches, with LF targets also matching CRLF or CR.
"""

from __future__ import annotations

import difflib
import heapq
import re
from collections.abc import Sequence
from itertools import zip_longest

MAX_DIFF_CHARS = 32_000_000


def source_lines(text: str) -> list[str]:
    """Split at LF, CRLF, or CR only, keeping each line's ending."""
    return [line for line in re.split(r"(?<=\n)|(?<=\r)(?!\n)", text) if line]


def split_lines(text: str) -> list[str]:
    """Lines numbered as editors and Git do; other Unicode separators stay in the line."""
    lines = re.split(r"\r\n|\r|\n", text)
    if lines[-1] == "":
        lines.pop()
    return lines


def line_at(text: str, offset: int) -> int:
    """The 1-based line number of an offset, counting CR, CRLF, and LF breaks."""
    return len(re.findall(r"\r\n|\r|\n", text[:offset])) + 1


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
        line_number = line_at(text, existing.start())
        notes.append(f"Replacement text already exists at line {line_number}; inspect before retrying.")
    lines = split_lines(text)
    anchor = max(split_lines(old), key=len, default="")[:300].strip()
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
            locations = ", ".join(str(line_at(text, match.start())) for match in matches[:5])
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
    # Compare logical lines lazily so unchanged matching context is skipped
    # without keeping two more copies of a large file in memory.
    lines = re.compile(r"[^\r\n]*(?:\r\n?|\n|$)")
    first_line = 1
    for first_line, (before, after) in enumerate(zip_longest(lines.finditer(text), lines.finditer(updated)), 1):
        if before is None or after is None or before.group() != after.group():
            break
    return updated, len(planned), first_line


def edit_result_line(result: str) -> int | None:
    """Locate the first change in the new file, including archived edit results."""
    hunk = re.search(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@[^\n]*\n", result, re.MULTILINE)
    if hunk is None:
        return None
    line = max(1, int(hunk[1]))
    for row in re.compile(r"[^\n]*(?:\n|$)").finditer(result, hunk.end()):
        text = row.group()
        if text.startswith(("-", "+")):
            return line
        if text.startswith(" "):
            line += 1
        elif not text.startswith("\\ No newline at end of file"):
            break
    return None


def edit_diff(path: str, before: str, after: str, *, limit: int | None = MAX_DIFF_CHARS) -> str:
    """Bound model/UI feedback; full file contents stay available to read_file."""
    diff = difflib.unified_diff(source_lines(before), source_lines(after),
                                fromfile=path, tofile=path)
    # A CR-only line is complete; like CRLF lines, it keeps its CR in the display.
    lines = (line.removesuffix("\n") if line.endswith(("\n", "\r"))
             else line + "\n\\ No newline at end of file" for line in diff)
    if limit is None:
        return "\n".join(lines)
    chunks = []
    size = 0
    for line in lines:
        remaining = limit - size
        if len(line) + 1 > remaining:
            chunks.append(line[:max(0, remaining)] + "\n… Diff truncated; read the file for remaining changes.")
            break
        chunks.append(line)
        size += len(line) + 1
    return "\n".join(chunks)
