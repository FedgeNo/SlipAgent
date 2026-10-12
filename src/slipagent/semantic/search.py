"""Incremental workspace indexing and hybrid search over stored snapshots."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .embed import Embedder
from .indexer import DEFAULT_EXCLUDE_DIRS, DEFAULT_INDEX_EXTS, chunk_text, walk_files
from .retrieval import RETRIEVAL, source_header
from .storage import Store


@dataclass
class IndexStats:
    indexed_files: int = 0
    unchanged_files: int = 0
    deleted_files: int = 0
    indexed_chunks: int = 0


class Searcher:
    def __init__(self, store: Store, embedder: Embedder, *, embedding_id: str | None = None,
                 cancelled: Callable[[], bool] | None = None,
                 path_filter: Callable[[list[str]], set[str]] | None = None):
        self.store = store
        self.embedder = embedder
        self.embedding_id = embedding_id
        self.last_index_stats = IndexStats()
        self.cancelled = cancelled
        self.path_filter = path_filter

    def _check_cancelled(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("Semantic indexing interrupted")

    def _model_identity(self) -> str:
        if self.embedding_id is not None:
            return self.embedding_id
        identity = getattr(self.embedder, "identity", None)
        if identity is not None:
            return str(identity)
        model = getattr(self.embedder, "model_name", "")
        cls = type(self.embedder)
        return f"{cls.__module__}.{cls.__qualname__}:{model}:{self.embedder.dim()}"

    def index_path(self, path: str, max_lines: int = 50, overlap: int = 5, *, rebuild: bool = False,
                   changed_paths: Iterable[str | Path] | None = None) -> int:
        """Embed new/changed files, remove deleted files, and return updated chunk count.

        Timestamp and size skip unchanged files without reading them. Files with
        changed metadata are hashed before deciding whether to embed again.
        The complete update is atomic; a failed traversal/read/embedding keeps
        the previous searchable index. changed_paths restricts a watcher update
        to affected files/subtrees and hashes them even if metadata is unchanged.
        Rebuild explicitly replaces the contents and requires a full traversal.
        """
        root = Path(path).expanduser().resolve(strict=True)
        self._check_cancelled()
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        if rebuild and changed_paths is not None:
            raise ValueError("A rebuild requires a complete workspace traversal")
        if changed_paths is not None:
            normalized = {Path(os.path.abspath(root / name)) for name in changed_paths}
            changed_paths = {str(name) for name in normalized if name.is_relative_to(root)}
        max_lines = max(1, max_lines)
        overlap = min(max(0, overlap), max_lines - 1)
        token_count = getattr(self.embedder, "token_count", None)
        max_tokens = getattr(self.embedder, "max_tokens", None)
        configuration = {
            "workspace": str(root), "embedding_id": self._model_identity(),
            "dimensions": self.embedder.dim(), "max_lines": max_lines, "overlap": overlap,
            "retrieval": RETRIEVAL,
        }
        document_batch_size = getattr(self.embedder, "batch_size", None)
        if document_batch_size is not None:
            configuration["document_batch_size"] = document_batch_size
        if token_count is not None or max_tokens is not None:
            configuration["chunking"] = {"strategy": "token-lines-path-v1", "max_tokens": max_tokens}
        if configuration["dimensions"] != self.store.dim:
            raise ValueError("Embedder and store dimensions differ")
        stats = IndexStats()
        with self.store.transaction():
            if rebuild:
                self.store.clear()
            self.store.configure(configuration)
            previous = self.store.file_records(map(str, changed_paths) if changed_paths is not None else None)
            candidates = (self._changed_files(root, changed_paths, previous) if changed_paths is not None
                          else None)
            seen = set()
            names = sorted(candidates) if candidates is not None else walk_files(str(root))
            if self.path_filter is not None:
                names = sorted(self.path_filter(list(names)))
            for name in names:
                self._check_cancelled()
                source = Path(name)
                if source.is_symlink() or (candidates is not None and not self.accepts_path(root, source)):
                    continue
                try:
                    before = source.stat()
                except FileNotFoundError:
                    if candidates is None:
                        raise
                    continue
                seen.add(name)
                old = previous.get(name)
                if (candidates is None and old and old["mtime_ns"] == before.st_mtime_ns
                        and old["size"] == before.st_size):
                    stats.unchanged_files += 1
                    continue
                data = source.read_bytes()
                after = source.stat()
                if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                    raise OSError(f"File changed while indexing; retry: {source}")
                digest = hashlib.sha256(data).hexdigest()
                text = data.decode("utf-8", errors="replace")
                if old and old["digest"] == digest:
                    self.store.save_file(name, digest, after.st_size, after.st_mtime_ns, old["line_count"])
                    stats.unchanged_files += 1
                    continue
                header = source_header(source.relative_to(root).as_posix())
                reserve = token_count(header) - token_count("") + 8 if token_count is not None else 0
                if max_tokens is not None and reserve >= max_tokens:
                    raise ValueError(f"Filename context leaves no room for source in {max_tokens} tokens: {source}")
                chunks = chunk_text(name, text, max_lines=max_lines, overlap=overlap,
                                    token_count=token_count,
                                    max_tokens=max_tokens - reserve if max_tokens is not None else None)
                self.store.remove_file(name)
                batch_size = 1 if self.cancelled is not None else 64
                for start in range(0, len(chunks), batch_size):
                    self._check_cancelled()
                    batch = chunks[start:start + batch_size]
                    inputs = [header + chunk.text for chunk in batch]
                    if token_count is not None and any(token_count(value) > max_tokens for value in inputs):
                        raise ValueError(f"Filename context and source exceed the token limit: {source}")
                    vectors = self.embedder.embed_batch(inputs)
                    self.store.add_chunks(batch, vectors, keyword_texts=inputs)
                self.store.save_file(name, digest, after.st_size, after.st_mtime_ns, len(text.splitlines()))
                stats.indexed_files += 1
                stats.indexed_chunks += len(chunks)
            removed = previous.keys() - seen
            if candidates is not None:
                removed &= candidates
            for name in removed:
                self.store.remove_file(name)
                stats.deleted_files += 1
        self.last_index_stats = stats
        return stats.indexed_chunks

    @staticmethod
    def accepts_path(root: Path, path: str | Path, *, directory: bool = False,
                     check_symlinks: bool = True) -> bool:
        """Use the same source extensions and exclusions for events and traversal."""
        source = Path(os.path.abspath(path))
        try:
            relative = source.relative_to(root)
        except ValueError:
            return False
        parents = relative.parts if directory else relative.parts[:-1]
        if any(part.startswith(".") for part in relative.parts):
            return False
        if any(part in DEFAULT_EXCLUDE_DIRS for part in parents):
            return False
        if check_symlinks and any(parent.is_symlink() for parent in (source, *source.parents) if parent != root):
            return False
        return directory or source.suffix.lower() in DEFAULT_INDEX_EXTS

    def _changed_files(self, root: Path, paths: Iterable[str | Path],
                       previous: dict[str, dict[str, Any]]) -> set[str]:
        names = set()
        for path in paths:
            source = Path(os.path.abspath(path))
            # Deleted/renamed directories retain their former descendants in the
            # manifest, so they can be removed without traversing the whole tree.
            descendants = {name for name in previous if name.startswith(os.path.join(str(source), ""))}
            names.update(descendants)
            if str(source) in previous:
                names.add(str(source))
            directory = source.is_dir() or bool(descendants) or source == root
            if not self.accepts_path(root, source, directory=directory):
                continue
            if source.is_dir():
                names.update(walk_files(str(source)))
            elif not directory:
                names.add(str(source))
        return names

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Rank semantic and keyword candidates without traversing source files."""
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.store.snapshot():
            configuration = self.store.configuration()
            if configuration is None:
                if self.store.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
                    raise ValueError("Legacy index has no model identity; rebuild the semantic index.")
                return []
            if configuration["embedding_id"] != self._model_identity() or configuration["dimensions"] != self.embedder.dim():
                raise ValueError("Embedding model differs from the indexed model; rebuild or use the original model.")
            if configuration.get("retrieval") != RETRIEVAL:
                raise ValueError("Index predates filename context and hybrid search; rebuild the semantic index.")
            return self.store.search(self.embedder.embed(query), limit=limit, query_text=query)
