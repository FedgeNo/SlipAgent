"""File walking and code chunking.

These helpers are pure-Python and have no third-party dependencies, so they can
be reused in any app that needs to enumerate and slice source files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Set


@dataclass
class Chunk:
    """A contiguous slice of a file.

    Attributes
    ----------
    path:
        Absolute or relative path to the source file.
    start_line:
        1-based index of the first line in the chunk.
    end_line:
        1-based index of the last line in the chunk (inclusive).
    text:
        The raw text content of the chunk.
    """

    path: str
    start_line: int
    end_line: int
    text: str


# Directories always skipped during traversal.
DEFAULT_EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".eggs",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "target",
    "bin",
    "obj",
}

# File extensions considered "indexable" by default.
DEFAULT_INDEX_EXTS: Set[str] = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".md",
    ".txt",
    ".sql",
    ".r",
    ".lua",
    ".pl",
    ".pm",
    ".tcl",
    ".vim",
}


def walk_files(
    root: str,
    exclude_dirs: Optional[List[str]] = None,
    index_exts: Optional[List[str]] = None,
) -> Iterator[str]:
    """Yield file paths under ``root``, skipping excluded dirs and non-indexed files.

    Parameters
    ----------
    root:
        Directory to traverse. May be relative or absolute.
    exclude_dirs:
        Directory names to skip. Defaults to :data:`DEFAULT_EXCLUDE_DIRS`.
    index_exts:
        File extensions (including the dot) to include. Defaults to
        :data:`DEFAULT_INDEX_EXTS`. Pass an empty list to index *all* files.

    Yields
    ------
    str
        Full path to each indexable file.
    """
    exclude = set(exclude_dirs) if exclude_dirs is not None else DEFAULT_EXCLUDE_DIRS
    exts = set(index_exts) if index_exts is not None else DEFAULT_INDEX_EXTS
    root = os.path.abspath(root)

    def walk_error(error: OSError) -> None:
        raise error

    for dirpath, dirnames, filenames in os.walk(root, onerror=walk_error):
        # Prune excluded directories in-place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if not exts or ext in exts:
                yield os.path.join(dirpath, fname)


def chunk_file(
    path: str,
    max_lines: int = 50,
    overlap: int = 5,
) -> List[Chunk]:
    """Split a file into overlapping chunks by line count.

    Parameters
    ----------
    path:
        Path to the file to read.
    max_lines:
        Maximum number of lines per chunk.
    overlap:
        Number of lines to carry over between consecutive chunks (provides
        context continuity at chunk boundaries).

    Returns
    -------
    List[Chunk]
        A list of chunks. Returns an empty list if the file cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return []

    return chunk_text(path, "".join(lines), max_lines=max_lines, overlap=overlap)


def chunk_text(path: str, text: str, max_lines: int = 50, overlap: int = 5, *,
               token_count: Callable[[str], int] | None = None,
               max_tokens: int | None = None) -> List[Chunk]:
    """Chunk a source snapshot, optionally fitting the encoder's full token budget.

    token_count includes special tokens. Whole lines are preferred; a line too
    large by itself is split into consecutive excerpts with the same line number.
    """
    if (token_count is None) != (max_tokens is None):
        raise ValueError("token_count and max_tokens must be supplied together")
    if token_count is not None and max_tokens is not None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        return _token_chunks(path, text, max_lines, overlap, token_count, max_tokens)
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    if max_lines <= 1:
        max_lines = 1
    if overlap < 0:
        overlap = 0
    if overlap >= max_lines:
        overlap = max_lines - 1

    chunks: List[Chunk] = []
    start = 0
    total = len(lines)
    while start < total:
        end = min(start + max_lines, total)
        text = "".join(lines[start:end])
        chunks.append(
            Chunk(
                path=path,
                start_line=start + 1,  # 1-based
                end_line=end,
                text=text,
            )
        )
        if end == total:
            break
        start = end - overlap

    return chunks


def _token_chunks(path: str, text: str, max_lines: int, overlap: int,
                  token_count: Callable[[str], int], max_tokens: int) -> List[Chunk]:
    max_lines = max(1, max_lines)
    overlap = min(max(0, overlap), max_lines - 1)
    pieces: list[tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        while line:
            if token_count(line) <= max_tokens:
                pieces.append((line, number))
                break
            # Count exact source prefixes, rather than decoding tokens, to retain
            # whitespace, Unicode, and line endings byte-for-byte in each excerpt.
            best = 1 if token_count(line[:1]) <= max_tokens else 0
            low, high = 2, len(line) - 1
            while low <= high:
                middle = (low + high) // 2
                if token_count(line[:middle]) <= max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if not best:
                raise ValueError(f"Cannot fit source text at {path}:{number} in {max_tokens} tokens")
            pieces.append((line[:best], number))
            line = line[best:]

    def fits(start: int, end: int) -> bool:
        return (pieces[end - 1][1] - pieces[start][1] + 1 <= max_lines
                and token_count("".join(piece[0] for piece in pieces[start:end])) <= max_tokens)

    chunks = []
    start = 0
    while start < len(pieces):
        end = start + 1
        while end < len(pieces) and fits(start, end + 1):
            end += 1
        chunks.append(Chunk(path, pieces[start][1], pieces[end - 1][1],
                            "".join(piece[0] for piece in pieces[start:end])))
        if end == len(pieces):
            break
        next_start = end
        first_overlap_line = pieces[end - 1][1] - overlap + 1
        while overlap and next_start > start + 1 and pieces[next_start - 1][1] >= first_overlap_line:
            next_start -= 1
        # Overlap must leave room for new source, otherwise it can emit repeated
        # windows containing only material from the preceding chunk.
        while next_start < end and not fits(next_start, end + 1):
            next_start += 1
        start = next_start
    return chunks
