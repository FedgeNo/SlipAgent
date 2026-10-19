"""SQLite chunk storage and a transactionally persisted HNSW search graph.

The graph is derived from the stored embeddings. Each transaction appends its
additions and removals to `graph_changes`; the full graph is saved only as an
occasional checkpoint, after which the journal is truncated. Loading restores
the checkpoint and replays the journal, so every update stays atomic with its
rows without rewriting the whole graph.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence, cast

from .indexer import Chunk
from .cpu import default_threads
from .retrieval import hybrid_scores, words


class Store:
    """Keep source snapshots, file fingerprints, and an ANN graph in one database."""

    # Save a full checkpoint once the journal exceeds this many changes. Replaying
    # an addition costs about 1.5 ms on a 20,000-chunk graph, so this bounds the
    # extra load time while spreading each checkpoint over hundreds of changes.
    CHECKPOINT_CHANGES = 256

    def __init__(self, db_path: str, dim: int, *, threads: int | None = None):
        if dim < 1:
            raise ValueError("dim must be positive")
        self.db_path = db_path
        self.dim = dim
        if threads is not None and threads < 1:
            raise ValueError("threads must be positive")
        self.threads = default_threads() if threads is None else threads
        self._conn: sqlite3.Connection | None = None
        self._index: Any = None
        self._revision: int | None = None
        # Checkpoint generation and journal position reflected by `_index`.
        self._checkpoint: int | None = None
        self._applied = 0
        self._dirty = False
        self._reset = False
        self._keyword_average: float | None = None
        self._calibration_ids: list[int] = []

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # Another session of the same project may hold a long indexing transaction.
            connection = sqlite3.connect(self.db_path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            self._conn = connection
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                digest TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                line_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS configuration (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS search_graph (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                data BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS graph_state (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                revision INTEGER NOT NULL,
                checkpoint INTEGER NOT NULL,
                position INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS graph_changes (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,
                added INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS keyword_documents (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                length INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS keyword_terms (
                term TEXT NOT NULL,
                chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                frequency INTEGER NOT NULL,
                PRIMARY KEY (term, chunk_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS idx_keyword_chunk ON keyword_terms(chunk_id);
        """)

    def _new_index(self) -> Any:
        from usearch.index import Index

        return Index(ndim=self.dim, metric="cos", dtype="f32",
                     connectivity=16, expansion_add=128, expansion_search=64)

    def _graph(self) -> Any:
        row = self.conn.execute("SELECT revision, checkpoint, position FROM graph_state WHERE id = 1").fetchone()
        revision, checkpoint, position = row if row else (0, 0, 0)
        if self._index is not None and self._revision == revision:
            return self._index
        if row is None and self.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
            raise ValueError("Legacy index has no search graph; rebuild the semantic index.")
        try:
            if self._index is not None and self._checkpoint == checkpoint:
                # Same checkpoint: apply only the changes this copy has not seen.
                index = self._index
            else:
                index = self._new_index()
                blob = self.conn.execute("SELECT data FROM search_graph WHERE id = 1").fetchone()
                if blob is not None:
                    index.load(blob[0])
                    if index.ndim != self.dim:
                        raise ValueError("Embedding dimensions differ; rebuild the index with the selected model.")
                self._applied = position
            self._replay(index)
        except BaseException:
            self._index = None
            self._revision = None
            raise
        self._index = index
        self._revision = revision
        self._checkpoint = checkpoint
        self._keyword_average = None
        return index

    def _replay(self, index: Any) -> None:
        """Apply the net journal changes after `_applied` to an in-memory graph."""
        import numpy as np

        final: dict[int, bool] = {}
        for seq, chunk_id, added in self.conn.execute(
                "SELECT seq, chunk_id, added FROM graph_changes WHERE seq > ? ORDER BY seq", (self._applied,)):
            final[chunk_id] = bool(added)
            self._applied = seq
        if not final:
            return
        ids = np.asarray(list(final), dtype=np.uint64)
        present = index.contains(ids) if len(index) else np.zeros(len(ids), dtype=bool)
        removed = [int(key) for key, keep, there in zip(ids, final.values(), present) if not keep and there]
        if removed:
            index.remove(removed, threads=self.threads)
        added = [int(key) for key, keep, there in zip(ids, final.values(), present) if keep and not there]
        for start in range(0, len(added), 512):
            batch = added[start:start + 512]
            placeholders = ",".join("?" for _ in batch)
            vectors = dict(self.conn.execute(
                f"SELECT chunk_id, vector FROM embeddings WHERE chunk_id IN ({placeholders})", batch))
            if len(vectors) != len(batch):
                raise ValueError("Search graph journal references a missing vector; rebuild the semantic index.")
            matrix = np.stack([np.frombuffer(vectors[key], dtype=np.float32) for key in batch])
            index.add(np.asarray(batch, dtype=np.uint64), matrix, threads=self.threads)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Publish the graph and SQL rows together, or retain the previous index."""
        connection = self.conn
        if connection.in_transaction:
            raise RuntimeError("An index transaction is already active")
        connection.execute("BEGIN IMMEDIATE")
        self._dirty = self._reset = False
        try:
            yield
            revision, checkpoint, applied = self._revision, self._checkpoint, self._applied
            if self._dirty:
                row = connection.execute("SELECT revision, checkpoint, position FROM graph_state WHERE id = 1").fetchone()
                revision, checkpoint, position = row if row else (0, 0, 0)
                revision += 1
                latest = connection.execute("SELECT MAX(seq) FROM graph_changes").fetchone()[0]
                applied = position if latest is None else latest
                pending = connection.execute("SELECT COUNT(*) FROM graph_changes").fetchone()[0]
                if self._reset or pending > self.CHECKPOINT_CHANGES:
                    connection.execute("INSERT INTO search_graph VALUES (1, ?) "
                                       "ON CONFLICT(id) DO UPDATE SET data=excluded.data", (self._index.save(),))
                    connection.execute("DELETE FROM graph_changes")
                    checkpoint, position = checkpoint + 1, applied
                connection.execute("INSERT INTO graph_state VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                                   "revision=excluded.revision, checkpoint=excluded.checkpoint, "
                                   "position=excluded.position", (revision, checkpoint, position))
            connection.commit()
            self._revision, self._checkpoint, self._applied = revision, checkpoint, applied
        except BaseException:
            connection.rollback()
            self._index = None
            self._revision = None
            self._keyword_average = None
            raise
        finally:
            self._dirty = self._reset = False

    def configuration(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT data FROM configuration WHERE id = 1").fetchone()
        return cast(dict[str, Any], json.loads(row[0])) if row else None

    @contextmanager
    def snapshot(self) -> Iterator[None]:
        """Keep configuration, graph, and source rows on one committed revision."""
        connection = self.conn
        own_transaction = not connection.in_transaction
        if own_transaction:
            connection.execute("BEGIN")
        try:
            yield
        finally:
            if own_transaction:
                connection.rollback()

    def configure(self, configuration: dict[str, Any]) -> None:
        current = self.configuration()
        if current is not None and current != configuration:
            raise ValueError("Workspace, model, or chunk settings differ; use a separate database or rebuild the semantic index.")
        if current is None and self.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
            raise ValueError("Legacy index has no model identity; rebuild the semantic index.")
        self.conn.execute("INSERT OR IGNORE INTO configuration VALUES (1, ?)",
                          (json.dumps(configuration, sort_keys=True),))

    def add_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *,
                   keyword_texts: Sequence[str] | None = None) -> None:
        import numpy as np

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        if keyword_texts is None:
            keyword_texts = [chunk.text for chunk in chunks]
        if len(keyword_texts) != len(chunks):
            raise ValueError("keyword_texts and chunks must have the same length")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(chunks), self.dim):
            raise ValueError(f"Each vector must have {self.dim} dimensions")
        if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("Embedding vectors must be finite and nonzero")
        if not self.conn.in_transaction:
            with self.transaction():
                self.add_chunks(chunks, vectors, keyword_texts=keyword_texts)
            return
        index = self._graph()
        ids = []
        for chunk, vector, text in zip(chunks, matrix, keyword_texts):
            cursor = self.conn.execute(
                "INSERT INTO chunks(path, start_line, end_line, text) VALUES (?, ?, ?, ?)",
                (chunk.path, chunk.start_line, chunk.end_line, chunk.text),
            )
            ids.append(cursor.lastrowid)
            self.conn.execute("INSERT INTO embeddings VALUES (?, ?)", (cursor.lastrowid, vector.tobytes()))
            counts = Counter(words(text))
            self.conn.execute("INSERT INTO keyword_documents VALUES (?, ?)",
                              (cursor.lastrowid, sum(counts.values())))
            self.conn.executemany("INSERT INTO keyword_terms VALUES (?, ?, ?)",
                                  [(term, cursor.lastrowid, count) for term, count in counts.items()])
        index.add(np.asarray(ids, dtype=np.uint64), matrix, threads=self.threads)
        self.conn.executemany("INSERT INTO graph_changes(chunk_id, added) VALUES (?, 1)", [(key,) for key in ids])
        self._dirty = True
        self._keyword_average = None

    def remove_file(self, path: str) -> None:
        index = self._graph()
        ids = [row[0] for row in self.conn.execute("SELECT id FROM chunks WHERE path = ?", (path,))]
        if ids:
            index.remove(ids, threads=self.threads)
            self.conn.executemany("INSERT INTO graph_changes(chunk_id, added) VALUES (?, 0)", [(key,) for key in ids])
            self._dirty = True
            self._keyword_average = None
        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def file_records(self, paths: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        if paths is None:
            return {row["path"]: dict(row) for row in self.conn.execute("SELECT * FROM files")}
        records: dict[str, dict[str, Any]] = {}
        for path in set(paths):
            prefix = os.path.join(path, "")
            upper = prefix[:-1] + chr(ord(os.sep) + 1)
            rows = self.conn.execute("SELECT * FROM files WHERE path=? OR (path>=? AND path<?)",
                                     (path, prefix, upper))
            records.update((row["path"], dict(row)) for row in rows)
        return records

    def save_file(self, path: str, digest: str, size: int, mtime_ns: int, line_count: int) -> None:
        self.conn.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?, ?) ON CONFLICT(path) DO UPDATE SET "
            "digest=excluded.digest, size=excluded.size, mtime_ns=excluded.mtime_ns, line_count=excluded.line_count",
            (path, digest, size, mtime_ns, line_count),
        )

    def search(self, query_vector: Sequence[float], limit: int = 10, *,
               query_text: str | None = None) -> list[dict[str, Any]]:
        import numpy as np

        if limit < 1:
            raise ValueError("limit must be positive")
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.shape != (self.dim,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError(f"Query must be a finite, nonzero vector of {self.dim} dimensions")
        with self.snapshot():
            index = self._graph()
            if not len(index):
                return []
            if query_text is not None:
                return self._hybrid_search(index, vector, query_text, limit)
            matches = index.search(vector, count=min(limit, len(index)), exact=False, threads=self.threads)
            results = []
            for key, distance in zip(matches.keys, matches.distances):
                if distance >= 0.5:
                    continue
                row = self.conn.execute(
                    "SELECT id, path, start_line, end_line, text FROM chunks WHERE id = ?", (int(key),)
                ).fetchone()
                if row is not None:
                    results.append({**dict(row), "distance": float(distance)})
            return results

    def _keyword_query(self, query: str, count: int) -> tuple[list[tuple[str, float]], float]:
        import numpy as np

        if self._keyword_average is None:
            row = self.conn.execute("SELECT AVG(length), MIN(chunk_id), MAX(chunk_id) FROM keyword_documents").fetchone()
            self._keyword_average = max(float(row[0] or 0), 1)
            self._calibration_ids = []
            if count > 256 and row[1] is not None:
                # A small ID-spread sample stabilizes normalization when an
                # approximate graph query misses isolated or duplicate vectors.
                for key in np.linspace(row[1], row[2], 32, dtype=np.int64):
                    sample = self.conn.execute("SELECT chunk_id FROM keyword_documents WHERE chunk_id>=? "
                                               "ORDER BY chunk_id LIMIT 1", (int(key),)).fetchone()
                    self._calibration_ids.append(sample[0])
        terms = sorted(set(words(query)))
        if not terms:
            return [], self._keyword_average
        placeholders = ",".join("?" for _ in terms)
        rows = self.conn.execute(
            f"SELECT term, COUNT(*) FROM keyword_terms WHERE term IN ({placeholders}) GROUP BY term", terms)
        weights = [(row[0], math.log(1 + (count - row[1] + 0.5) / (row[1] + 0.5))) for row in rows]
        return weights, self._keyword_average

    def _keyword_scores(self, weights: list[tuple[str, float]], average: float, *,
                        limit: int | None = None, ids: Sequence[int] | None = None) -> dict[int, float]:
        if not weights:
            return {}
        values = ",".join("(?, ?)" for _ in weights)
        parameters = [value for pair in weights for value in pair]
        parameters.append(average)
        where = ""
        if ids is not None:
            where = " WHERE terms.chunk_id IN (" + ",".join("?" for _ in ids) + ")"
            parameters.extend(ids)
        sql = f"""
            WITH query_terms(term, idf) AS (VALUES {values})
            SELECT terms.chunk_id,
                   SUM(query_terms.idf * terms.frequency * 2.2 /
                       (terms.frequency + 1.2 * (0.25 + 0.75 * documents.length / ?))) AS score
            FROM query_terms
            JOIN keyword_terms AS terms ON terms.term = query_terms.term
            JOIN keyword_documents AS documents ON documents.chunk_id = terms.chunk_id
            {where}
            GROUP BY terms.chunk_id
            ORDER BY score DESC, terms.chunk_id
        """
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        return {row[0]: float(row[1]) for row in self.conn.execute(sql, parameters)}

    def _hybrid_search(self, index: Any, vector: Any, query: str, limit: int) -> list[dict[str, Any]]:
        import numpy as np

        # Retrieve independently so a lexical match can enter without clearing
        # a semantic cutoff. The farthest candidate estimates the full-corpus
        # cosine minimum used by the benchmark's per-query normalization.
        count = min(len(index), max(256, limit * 8))
        nearest = index.search(vector, count=count, exact=False, threads=self.threads)
        farthest = index.search(-vector, count=1, exact=False, threads=self.threads)
        weights, average = self._keyword_query(query, len(index))
        lexical = self._keyword_scores(weights, average, limit=count)
        ids = sorted(set(map(int, nearest.keys)) | set(map(int, farthest.keys)) | lexical.keys()
                     | set(self._calibration_ids))
        # Chunk IDs give deterministic tie-breaking across reopening the graph.
        matrix = np.asarray(index.get(ids, dtype=np.float32), dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        semantic = matrix @ (vector / np.linalg.norm(vector))
        lexical = self._keyword_scores(weights, average, ids=ids)
        sparse = np.asarray([lexical.get(key, 0) for key in ids], dtype=np.float32)
        scores = hybrid_scores(semantic, sparse)
        order = np.argsort(-scores, kind="stable")[:limit]
        results = []
        for position in order:
            row = self.conn.execute(
                "SELECT id, path, start_line, end_line, text FROM chunks WHERE id = ?", (ids[position],)
            ).fetchone()
            results.append({**dict(row), "distance": float(1 - np.clip(semantic[position], -1, 1)),
                            "score": float(scores[position])})
        return results

    def clear(self) -> None:
        if not self.conn.in_transaction:
            with self.transaction():
                self.clear()
            return
        row = self.conn.execute("SELECT revision, checkpoint, position FROM graph_state WHERE id = 1").fetchone()
        revision, checkpoint, position = row if row else (0, 0, 0)
        latest = self.conn.execute("SELECT MAX(seq) FROM graph_changes").fetchone()[0]
        # The empty graph is current; the commit saves it as a checkpoint and drops the journal.
        self._index = self._new_index()
        self._revision, self._checkpoint = revision, checkpoint
        self._applied = position if latest is None else latest
        self._dirty = self._reset = True
        self._keyword_average = None
        for table in ("embeddings", "chunks", "files", "configuration"):
            self.conn.execute(f"DELETE FROM {table}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._index = None
        self._revision = None
        self._checkpoint = None
        self._keyword_average = None

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
