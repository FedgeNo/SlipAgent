"""SQLite chunk storage and a transactionally persisted HNSW search graph."""

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
        self._dirty = False
        self._keyword_average: float | None = None
        self._calibration_ids: list[int] = []

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            connection = sqlite3.connect(self.db_path)
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
                revision INTEGER NOT NULL,
                data BLOB NOT NULL
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
        row = self.conn.execute("SELECT revision FROM search_graph WHERE id = 1").fetchone()
        revision = row[0] if row else 0
        if self._index is not None and self._revision == revision:
            return self._index
        index = self._new_index()
        if row:
            data = self.conn.execute("SELECT data FROM search_graph WHERE id = 1").fetchone()[0]
            index.load(data)
            if index.ndim != self.dim:
                raise ValueError("Embedding dimensions differ; rebuild the index with the selected model.")
        elif self.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
            raise ValueError("Legacy index has no search graph; rebuild the semantic index.")
        self._index = index
        self._revision = revision
        self._keyword_average = None
        return index

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Publish the graph and SQL rows together, or retain the previous index."""
        connection = self.conn
        if connection.in_transaction:
            raise RuntimeError("An index transaction is already active")
        connection.execute("BEGIN IMMEDIATE")
        self._dirty = False
        try:
            yield
            revision = self._revision
            if self._dirty:
                row = connection.execute("SELECT revision FROM search_graph WHERE id = 1").fetchone()
                revision = (row[0] if row else 0) + 1
                connection.execute(
                    "INSERT INTO search_graph VALUES (1, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET revision=excluded.revision, data=excluded.data",
                    (revision, self._index.save()),
                )
            connection.commit()
            self._revision = revision
        except BaseException:
            connection.rollback()
            self._index = None
            self._revision = None
            self._keyword_average = None
            raise
        finally:
            self._dirty = False

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
        self._dirty = True
        self._keyword_average = None

    def remove_file(self, path: str) -> None:
        index = self._graph()
        ids = [row[0] for row in self.conn.execute("SELECT id FROM chunks WHERE path = ?", (path,))]
        if ids:
            index.remove(ids, threads=self.threads)
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
        self._index = self._new_index()
        row = self.conn.execute("SELECT revision FROM search_graph WHERE id = 1").fetchone()
        self._revision = row[0] if row else 0
        self._dirty = True
        self._keyword_average = None
        for table in ("embeddings", "chunks", "files", "configuration"):
            self.conn.execute(f"DELETE FROM {table}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._index = None
        self._revision = None
        self._keyword_average = None

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
