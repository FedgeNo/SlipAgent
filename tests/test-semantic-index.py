"""Exercise real SQLite/HNSW storage using disposable workspaces and embeddings."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slipagent.semantic import Chunk, Searcher, Store


class FakeEmbedder:
    identity = "test-encoder-v1"

    def __init__(self):
        self.batches = []

    def dim(self):
        return 3

    def embed(self, text):
        if "auth" in text:
            return [1, 0, 0]
        if "storage" in text:
            return [0, 1, 0]
        return [0, 0, 1]

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        return [self.embed(text) for text in texts]


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source = self.workspace / "auth.py"
        self.original = "def auth():\n    return True\n"
        self.source.write_text(self.original)
        self.database = str(self.root / "index.db")
        self.store = Store(self.database, dim=3)
        self.addCleanup(self.store.close)
        self.embedder = FakeEmbedder()
        self.searcher = Searcher(self.store, self.embedder)

    def index(self, **kwargs):
        return self.searcher.index_path(str(self.workspace), **kwargs)

    def test_unchanged_files_keep_ids_without_reads_or_embeddings(self):
        self.assertEqual(self.index(), 1)
        original_id = self.searcher.search("auth")[0]["id"]
        self.embedder.batches.clear()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("source read")):
            self.assertEqual(self.index(), 0)
        self.assertEqual(self.embedder.batches, [])
        self.assertEqual(self.searcher.last_index_stats.unchanged_files, 1)
        self.assertEqual(self.searcher.search("auth")[0]["id"], original_id)

    def test_touch_without_content_change_does_not_embed_again(self):
        self.index()
        old = self.store.file_records()[str(self.source)]
        self.embedder.batches.clear()
        os.utime(self.source, ns=(old["mtime_ns"] + 1_000_000, old["mtime_ns"] + 1_000_000))
        self.assertEqual(self.index(), 0)
        self.assertEqual(self.embedder.batches, [])
        self.assertGreater(self.store.file_records()[str(self.source)]["mtime_ns"], old["mtime_ns"])

    def test_only_changed_and_added_files_are_embedded(self):
        other = self.workspace / "storage.py"
        other.write_text("def storage(): pass\n")
        self.index()
        storage_id = self.searcher.search("storage")[0]["id"]
        self.source.write_text("def auth():\n    return 'updated'\n")
        (self.workspace / "new.py").write_text("new code\n")
        self.embedder.batches.clear()
        self.assertEqual(self.index(), 2)
        embedded = [text for batch in self.embedder.batches for text in batch]
        self.assertEqual(len(embedded), 2)
        self.assertNotIn("def storage(): pass\n", embedded)
        self.assertEqual(self.searcher.last_index_stats.unchanged_files, 1)
        self.assertEqual(self.searcher.search("storage")[0]["id"], storage_id)
        self.assertIn("updated", self.searcher.search("auth")[0]["text"])

    def test_deleted_files_are_removed_from_sql_and_graph(self):
        self.index()
        self.source.unlink()
        self.assertEqual(self.index(), 0)
        self.assertEqual(self.searcher.last_index_stats.deleted_files, 1)
        self.assertEqual(self.store.file_records(), {})
        self.assertEqual(self.searcher.search("auth"), [])
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 0)
        self.store.close()
        self.assertEqual(self.searcher.search("auth"), [])

    def test_empty_files_are_tracked_and_do_not_generate_vectors(self):
        self.source.write_text("")
        self.assertEqual(self.index(), 0)
        self.assertEqual(self.store.file_records()[str(self.source)]["line_count"], 0)
        self.assertEqual(self.searcher.last_index_stats.indexed_files, 1)
        self.assertEqual(self.searcher.search("auth"), [])
        self.assertEqual(self.index(), 0)
        self.assertEqual(self.searcher.last_index_stats.unchanged_files, 1)

    def test_reopening_reuses_graph_and_file_manifest(self):
        self.index()
        result = self.searcher.search("auth")[0]
        self.store.close()
        self.embedder.batches.clear()
        self.assertEqual(self.searcher.search("auth")[0], result)
        self.assertEqual(self.index(), 0)
        self.assertEqual(self.embedder.batches, [])

    def test_search_does_not_walk_or_read_source_or_scan_vector_rows(self):
        self.index()
        statements = []
        self.store.conn.set_trace_callback(statements.append)
        self.addCleanup(lambda: self.store.conn.set_trace_callback(None))
        with patch("slipagent.semantic.search.walk_files", side_effect=AssertionError("workspace walk")), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("source read")):
            result = self.searcher.search("auth")[0]
        self.assertEqual(result["path"], str(self.source))
        self.assertEqual((result["start_line"], result["end_line"]), (1, 2))
        self.assertEqual(result["text"], self.original)
        self.assertFalse(any("FROM embeddings" in statement for statement in statements))
        source_queries = [statement for statement in statements if "FROM chunks" in statement]
        self.assertTrue(source_queries)
        self.assertTrue(all("WHERE id =" in statement for statement in source_queries))

    def test_missing_workspace_keeps_existing_index(self):
        self.index()
        with self.assertRaises(FileNotFoundError):
            self.searcher.index_path(str(self.workspace / "missing"))
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)

    def test_failed_embedding_restores_old_file_and_graph(self):
        self.index()
        previous = self.store.file_records()
        self.source.write_text("def auth(): return 'new content'\n")
        with patch.object(self.embedder, "embed_batch", side_effect=RuntimeError("embedding failed")):
            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                self.index()
        self.assertEqual(self.store.file_records(), previous)
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)
        self.store.close()
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)

    def test_failed_graph_save_rolls_back_sql_and_in_memory_graph(self):
        self.index()
        self.source.write_text("def auth(): return 'updated'\n")
        with patch("usearch.index.Index.save", side_effect=OSError("graph save failed")):
            with self.assertRaisesRegex(OSError, "graph save failed"):
                self.index()
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)

    def test_failed_rebuild_keeps_previous_index(self):
        self.index()
        with patch.object(self.embedder, "embed_batch", side_effect=RuntimeError("embedding failed")):
            with self.assertRaises(RuntimeError):
                self.index(rebuild=True)
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)

    def test_failed_walk_does_not_remove_previous_files(self):
        self.index()
        with patch("slipagent.semantic.search.walk_files", side_effect=PermissionError("unreadable directory")):
            with self.assertRaises(PermissionError):
                self.index()
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)

    def test_file_changing_during_read_aborts_update(self):
        self.index()
        self.source.write_text("def auth(): return 'updated'\n")
        read_bytes = Path.read_bytes
        def changing_read(path):
            data = read_bytes(path)
            path.write_bytes(data + b"# changed again\n")
            return data
        with patch.object(Path, "read_bytes", changing_read):
            with self.assertRaisesRegex(OSError, "File changed while indexing"):
                self.index()
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)

    def test_large_file_embeddings_are_bounded(self):
        self.source.write_text("auth\n" * 150)
        self.assertEqual(self.index(max_lines=1, overlap=0), 150)
        self.assertEqual([len(batch) for batch in self.embedder.batches], [64, 64, 22])

    def test_changed_model_is_rejected_even_with_same_dimensions(self):
        self.index()
        self.embedder.identity = "different-test-encoder"
        with self.assertRaisesRegex(ValueError, "model differs"):
            self.searcher.search("auth")
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.index()
        self.assertEqual(self.index(rebuild=True), 1)
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)

    def test_chunking_changes_require_rebuild(self):
        self.index()
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.index(max_lines=1, overlap=0)
        self.assertEqual(self.index(max_lines=1, overlap=0, rebuild=True), 2)

    def test_enabling_token_aware_chunking_requires_rebuild_even_for_unchanged_files(self):
        self.index()
        previous = self.store.file_records()
        self.source.write_text("auth\n" * 20)
        self.embedder.max_tokens = 50
        self.embedder.token_count = lambda text: len(text) + 2
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.index()
        self.assertEqual(self.store.file_records(), previous)
        self.assertGreater(self.index(rebuild=True), 1)
        rows = list(self.store.conn.execute("SELECT text FROM chunks ORDER BY id"))
        self.assertTrue(all(self.embedder.token_count("File: auth.py\n\n" + row["text"]) <= 50 for row in rows))

    def test_different_workspace_cannot_silently_replace_existing_index(self):
        self.index()
        other = self.root / "other-workspace"
        other.mkdir()
        (other / "storage.py").write_text("storage\n")
        with self.assertRaisesRegex(ValueError, "separate database"):
            self.searcher.index_path(str(other))
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)
        self.assertEqual(self.searcher.index_path(str(other), rebuild=True), 1)
        self.assertEqual(self.searcher.search("auth")[0]["path"], str(other / "storage.py"))

    def test_legacy_index_requires_explicit_rebuild(self):
        self.store.conn.execute("INSERT INTO chunks(path,start_line,end_line,text) VALUES ('legacy.py',1,1,'old')")
        self.store.conn.commit()
        with self.assertRaisesRegex(ValueError, "Legacy index"):
            self.searcher.search("auth")
        with self.assertRaisesRegex(ValueError, "Legacy index"):
            self.index()
        self.assertEqual(self.store.conn.execute("SELECT text FROM chunks").fetchone()[0], "old")
        self.assertEqual(self.index(rebuild=True), 1)
        self.assertEqual(self.searcher.search("auth")[0]["text"], self.original)

    def test_other_connection_refreshes_cached_graph_after_update(self):
        self.index()
        with Store(self.database, dim=3) as other:
            reader = Searcher(other, FakeEmbedder())
            self.assertEqual(reader.search("auth")[0]["text"], self.original)
            self.source.write_text("def auth(): return 'updated'\n")
            self.index()
            self.assertIn("updated", reader.search("auth")[0]["text"])

    def test_query_pins_model_graph_and_text_during_concurrent_rebuild(self):
        self.index()
        with Store(self.database, dim=3) as other:
            reader_embedder = FakeEmbedder()
            reader = Searcher(other, reader_embedder)
            def query_and_rebuild(text):
                self.source.write_text("def auth(): return 'new model'\n")
                self.embedder.identity = "new-model"
                self.index(rebuild=True)
                return [1, 0, 0]
            with patch.object(reader_embedder, "embed", side_effect=query_and_rebuild):
                self.assertEqual(reader.search("auth")[0]["text"], self.original)
            with self.assertRaisesRegex(ValueError, "model differs"):
                reader.search("auth")

    def test_invalid_vectors_do_not_publish_partial_chunks(self):
        self.index()
        chunk = Chunk("invalid.py", 1, 1, "invalid")
        for vector in ([1, 2], [0, 0, 0], [float("nan"), 1, 0], [float("inf"), 1, 0]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.store.add_chunks([chunk], [vector])
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)

    def test_limit_is_enforced_and_invalid_limits_are_rejected(self):
        (self.workspace / "other.py").write_text("auth again\n")
        self.index()
        self.assertEqual(len(self.searcher.search("auth", limit=1)), 1)
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.searcher.search("auth", limit=limit)

    def test_symlink_files_outside_workspace_are_not_indexed(self):
        outside = self.root / "outside.py"
        outside.write_text("storage secret\n")
        (self.workspace / "linked.py").symlink_to(outside)
        self.index()
        self.assertEqual(set(self.store.file_records()), {str(self.source)})


if __name__ == "__main__":
    unittest.main()
