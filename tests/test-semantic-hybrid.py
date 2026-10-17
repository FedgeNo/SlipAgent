"""Production hybrid retrieval, context budgets, and physical CPU defaults."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from slipagent.semantic import Chunk, ONNXEmbedder, Searcher, Store
from slipagent.semantic.cpu import default_threads
from slipagent.semantic.retrieval import words


class Encoder:
    identity = "hybrid-test"
    max_tokens = 96

    def __init__(self):
        self.inputs = []

    def dim(self):
        return 3

    def token_count(self, text):
        return len(text) + 2

    def embed(self, text):
        return [1, 0, 0]

    def embed_batch(self, texts):
        self.inputs.extend(texts)
        return [self.embed(text) for text in texts]


class HybridTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.encoder = Encoder()
        self.database = str(self.root / "index.db")
        self.store = Store(self.database, 3)
        self.addCleanup(self.store.close)
        self.searcher = Searcher(self.store, self.encoder)

    def test_relative_filename_context_fits_without_altering_source_excerpts(self):
        source = "value = '" + "λ🌓" * 60 + "'\r\nreturn value\n"
        path = self.workspace / "engine.py"
        path.write_bytes(source.encode())
        self.searcher.index_path(str(self.workspace), overlap=0)
        chunks = list(self.store.conn.execute("SELECT text FROM chunks ORDER BY id"))
        self.assertEqual("".join(row["text"] for row in chunks), source)
        self.assertEqual(self.encoder.inputs, ["File: engine.py\n\n" + row["text"] for row in chunks])
        self.assertTrue(all(self.encoder.token_count(text) <= 96 for text in self.encoder.inputs))
        self.assertTrue(all(str(self.root) not in text for text in self.encoder.inputs))

    def test_filename_is_searchable_even_when_source_does_not_name_it(self):
        (self.workspace / "clipboard.py").write_text("return value\n")
        (self.workspace / "storage.py").write_text("return value\n")
        self.searcher.index_path(str(self.workspace))
        self.assertTrue(self.searcher.search("clipboard")[0]["path"].endswith("clipboard.py"))

    def test_keyword_candidate_can_win_from_outside_semantic_shortlist(self):
        chunks = [Chunk(f"a{i}.py", 1, 1, "ordinary text") for i in range(600)]
        vectors = [[0.8, 0.6, 0] for _ in chunks]
        chunks.extend([Chunk("needle.py", 1, 1, "needle"), Chunk("far.py", 1, 1, "far")])
        vectors.extend([[0.79, math.sqrt(1 - 0.79**2), 0], [-0.9, math.sqrt(0.19), 0]])
        self.store.add_chunks(chunks, vectors)
        graph = self.store._graph()
        counts = []
        original_get = graph.get

        def get(ids, **kwargs):
            counts.append(len(ids))
            return original_get(ids, **kwargs)

        with patch.object(graph, "get", side_effect=get):
            result = self.store.search([1, 0, 0], limit=1, query_text="needle")
        self.assertEqual(result[0]["path"], "needle.py")
        self.assertLessEqual(max(counts), 545)
        self.assertAlmostEqual(result[0]["distance"], 0.21, places=5)

    def test_keyword_matches_are_not_discarded_by_old_cosine_cutoff(self):
        self.store.add_chunks([Chunk("a.py", 1, 1, "needle"), Chunk("b.py", 1, 1, "other")],
                              [[-1, 0, 0], [1, 0, 0]])
        results = self.store.search([1, 0, 0], query_text="needle")
        self.assertEqual(len(results), 2)
        self.assertEqual(next(row for row in results if row["path"] == "a.py")["distance"], 2)

    def test_sqlite_bm25_and_fusion_match_frozen_experiment_rankings(self):
        benchmark = Path(__file__).resolve().parents[1] / "benchmarks/semantic/retrieval-experiments.py"
        spec = importlib.util.spec_from_file_location("hybrid_oracle", benchmark)
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        self.assertEqual(words("HTTPServer read_command_output déjà"),
                         experiment.words("HTTPServer read_command_output déjà"))
        texts = ["File: graph.py\n\ngraph rollback", "graph unrelated " * 40,
                 "HTTPServer read_command_output déjà", "clipboard copy", "a the and"]
        vectors = np.random.default_rng(54).normal(size=(len(texts), 3)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        chunks = [Chunk(f"{i}.py", 1, 1, text) for i, text in enumerate(texts)]
        self.store.add_chunks(chunks, vectors)
        oracle = experiment.KeywordIndex(texts)
        for query in ("rollback graph", "HTTPServer output déjà", "clipboard", "the and", "unmatched"):
            vector = np.array([1, 2, 3], dtype=np.float32)
            semantic = vectors @ (vector / np.linalg.norm(vector))
            order = experiment.rank_candidates(semantic, *oracle.score(query), "hybrid-25")
            result = self.store.search(vector, limit=5, query_text=query)
            self.assertEqual([row["path"] for row in result], [f"{index}.py" for index in order])

    def test_keyword_rows_refresh_after_update_delete_and_reopen(self):
        path = self.workspace / "a.py"
        path.write_text("old_identifier\n")
        self.searcher.index_path(str(self.workspace))
        with Store(self.database, 3) as reader:
            query = Searcher(reader, Encoder())
            self.assertEqual(query.search("old_identifier")[0]["text"], "old_identifier\n")
            path.write_text("new_identifier\n")
            self.searcher.index_path(str(self.workspace), changed_paths=[path])
            self.assertEqual(query.search("new_identifier")[0]["text"], "new_identifier\n")
            self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM keyword_terms WHERE term='old'").fetchone()[0], 0)
            path.unlink()
            self.searcher.index_path(str(self.workspace), changed_paths=[path])
            self.assertEqual(query.search("new_identifier"), [])
            self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM keyword_documents").fetchone()[0], 0)

    def test_failed_update_keeps_keyword_rows_and_source_together(self):
        path = self.workspace / "a.py"
        path.write_text("old_identifier\n")
        self.searcher.index_path(str(self.workspace))
        path.write_text("new_identifier\n")
        with patch("usearch.index.Index.save", side_effect=OSError("save failed")), \
                patch.object(Store, "CHECKPOINT_CHANGES", 0):
            with self.assertRaises(OSError):
                self.searcher.index_path(str(self.workspace), changed_paths=[path])
        self.assertEqual(self.searcher.search("old_identifier")[0]["text"], "old_identifier\n")
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM keyword_terms WHERE term='new'").fetchone()[0], 0)

    def test_old_context_contract_requires_rebuild_before_embedding_query(self):
        (self.workspace / "a.py").write_text("source\n")
        self.searcher.index_path(str(self.workspace))
        config = self.store.configuration()
        del config["retrieval"]
        self.store.conn.execute("UPDATE configuration SET data=?", (json.dumps(config),))
        self.store.conn.commit()
        with patch.object(self.encoder, "embed", side_effect=AssertionError("model queried")):
            with self.assertRaisesRegex(ValueError, "rebuild"):
                self.searcher.search("source")
        self.searcher.index_path(str(self.workspace), rebuild=True)
        self.assertTrue(self.searcher.search("source"))

    def test_document_batch_changes_require_explicit_rebuild(self):
        (self.workspace / "a.py").write_text("source\n")
        self.encoder.batch_size = 1
        self.searcher.index_path(str(self.workspace))
        self.encoder.batch_size = 8
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.searcher.index_path(str(self.workspace))
        self.searcher.index_path(str(self.workspace), rebuild=True)

    def test_exact_ties_keep_chunk_ids_after_reopening(self):
        self.store.add_chunks([Chunk(f"{i}.py", 1, 1, "identical") for i in range(20)], [[1, 0, 0]] * 20)
        for query in ("nothing", "identical"):
            with self.subTest(query=query):
                first = self.store.search([1, 0, 0], query_text=query)
                self.store.close()
                self.assertEqual(self.store.search([1, 0, 0], query_text=query), first)
                ids = [row["id"] for row in first]
                self.assertEqual(len(ids), 10)
                self.assertEqual(ids, sorted(ids))
                # Approximate search can omit duplicate vectors. Keyword hits
                # supply every candidate for the assertion over the full tie set.
                if query == "identical":
                    self.assertEqual(ids, list(range(1, 11)))


class CPUTests(unittest.TestCase):
    def test_physical_cores_and_two_thread_fallback_respect_affinity(self):
        for physical, allowed, expected in ((2, [0, 1, 2, 3], 2), (4, [0], 1),
                                            (None, [0, 1, 2, 3], 2), (1, [0], 1)):
            with self.subTest(physical=physical, allowed=allowed), \
                    patch("slipagent.semantic.cpu.psutil.cpu_count", return_value=physical), \
                    patch("slipagent.semantic.cpu.psutil.Process") as process:
                process.return_value.cpu_affinity.return_value = allowed
                self.assertEqual(default_threads(), expected)

    def test_unavailable_affinity_and_explicit_overrides(self):
        with patch("slipagent.semantic.cpu.psutil.cpu_count", return_value=None), \
                patch("slipagent.semantic.cpu.psutil.Process", side_effect=NotImplementedError):
            self.assertEqual(default_threads(), 2)
            self.assertEqual(ONNXEmbedder(threads=1).threads, 1)
        with self.assertRaises(ValueError):
            ONNXEmbedder(threads=0)

    def test_failed_core_detection_uses_two_threads(self):
        with patch("slipagent.semantic.cpu.psutil.cpu_count", side_effect=OSError("unavailable")), \
                patch("slipagent.semantic.cpu.psutil.Process", side_effect=NotImplementedError):
            self.assertEqual(default_threads(), 2)


if __name__ == "__main__":
    unittest.main()
