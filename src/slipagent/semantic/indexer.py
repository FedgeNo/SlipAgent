"""File walking and code chunking.

These helpers are pure-Python and have no third-party dependencies, so they can
be reused in any app that needs to enumerate and slice source files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import StringIO
from typing import Callable, Iterator, List, Optional, Set, TextIO


SOURCE_BLOCK_CHARS = 64 * 1024
LINE_ENDINGS = ("\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


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
            return list(iter_chunks(path, f, max_lines=max_lines, overlap=overlap))
    except (OSError, UnicodeDecodeError):
        return []


def chunk_text(path: str, text: str, max_lines: int = 50, overlap: int = 5, *,
               token_count: Callable[[str], int] | None = None,
               max_tokens: int | None = None) -> List[Chunk]:
    """Materialize chunks for callers that already retain the complete source."""
    return list(iter_chunks(path, StringIO(text), max_lines, overlap,
                            token_count=token_count, max_tokens=max_tokens))


def _source_pieces(source: TextIO) -> Iterator[tuple[str, int]]:
    """Retain splitlines coordinates and line endings using bounded reads."""
    pending = ""
    number = 1
    while block := source.read(SOURCE_BLOCK_CHARS):
        lines = (pending + block).splitlines(keepends=True)
        pending = ""
        # A trailing CR may be the first half of CRLF in the next read.
        if lines and (not lines[-1].endswith(LINE_ENDINGS) or lines[-1].endswith("\r")):
            pending = lines.pop()
        for line in lines:
            for start in range(0, len(line), SOURCE_BLOCK_CHARS):
                yield line[start:start + SOURCE_BLOCK_CHARS], number
            number += 1
        while len(pending) > SOURCE_BLOCK_CHARS:
            yield pending[:SOURCE_BLOCK_CHARS], number
            pending = pending[SOURCE_BLOCK_CHARS:]
    if pending:
        yield pending, number


def _token_pieces(source: TextIO, path: str, token_count: Callable[[str], int] | None,
                  max_tokens: int | None, check_cancelled: Callable[[], None] | None) -> Iterator[tuple[str, int]]:
    for line, number in _source_pieces(source):
        while line:
            if check_cancelled is not None:
                check_cancelled()
            if token_count is None or max_tokens is None or token_count(line) <= max_tokens:
                yield line, number
                break
            # Exact source prefixes preserve whitespace, Unicode and line endings.
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
            yield line[:best], number
            line = line[best:]


def iter_chunks(path: str, source: TextIO, max_lines: int = 50, overlap: int = 5, *,
                token_count: Callable[[str], int] | None = None,
                max_tokens: int | None = None,
                check_cancelled: Callable[[], None] | None = None) -> Iterator[Chunk]:
    """Yield bounded source excerpts, retaining all text and overlapping lines.

    Token counts include special tokens. Whole lines are preferred; oversized
    lines produce consecutive excerpts with the same line number. The character
    bound also covers text, such as whitespace, that consumes very few tokens.
    """
    if (token_count is None) != (max_tokens is None):
        raise ValueError("token_count and max_tokens must be supplied together")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    max_lines = max(1, max_lines)
    overlap = min(max(0, overlap), max_lines - 1)

    def fits(pieces: list[tuple[str, int]]) -> bool:
        if (pieces[-1][1] - pieces[0][1] + 1 > max_lines
                or sum(len(piece[0]) for piece in pieces) > SOURCE_BLOCK_CHARS):
            return False
        return (token_count is None or max_tokens is None
                or token_count("".join(piece[0] for piece in pieces)) <= max_tokens)

    window: list[tuple[str, int]] = []
    for piece in _token_pieces(source, path, token_count, max_tokens, check_cancelled):
        if window and not fits([*window, piece]):
            yield Chunk(path, window[0][1], window[-1][1], "".join(item[0] for item in window))
            start = len(window)
            first_overlap_line = window[-1][1] - overlap + 1
            while overlap and start > 1 and window[start - 1][1] >= first_overlap_line:
                start -= 1
            # Overlap must leave room for new source, so each chunk advances.
            while start < len(window) and not fits([*window[start:], piece]):
                start += 1
            window = window[start:]
        window.append(piece)
    if window:
        yield Chunk(path, window[0][1], window[-1][1], "".join(item[0] for item in window))
