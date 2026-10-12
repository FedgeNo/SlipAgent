"""Real filesystem observation and targeted updates use disposable workspaces."""

import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from watchdog.events import FileModifiedEvent

from slipagent.semantic import IndexWatcher, Searcher, Store


class Encoder:
    identity = "watcher-test"

    def __init__(self):
        self.batches = []
        self.during_embed = None

    def dim(self):
        return 3

    def embed(self, text):
        return [1, 0, 0]

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        if self.during_embed is not None:
            callback, self.during_embed = self.during_embed, None
            callback()
        return [[1, 0, 0] for text in texts]


class TargetedTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source = self.workspace / "a.py"
        self.source.write_text("original\n")
        self.encoder = Encoder()
        self.store = Store(str(self.root / "index.db"), 3)
        self.addCleanup(self.store.close)
        self.searcher = Searcher(self.store, self.encoder)
        self.searcher.index_path(str(self.workspace))
        self.encoder.batches.clear()

    def update(self, *paths):
        return self.searcher.index_path(str(self.workspace), changed_paths=paths)

    def test_single_file_update_does_not_walk_or_hash_other_sources(self):
        other = self.workspace / "other.py"
        other.write_text("other\n")
        self.searcher.index_path(str(self.workspace))
        self.encoder.batches.clear()
        self.source.write_text("updated\n")
        original = Path.read_bytes

        def read(path):
            self.assertEqual(path, self.source)
            return original(path)

        with patch("slipagent.semantic.search.walk_files", side_effect=AssertionError("full traversal")), \
                patch.object(Path, "read_bytes", side_effect=read, autospec=True):
            self.assertEqual(self.update("a.py"), 1)
        self.assertEqual(len(self.encoder.batches), 1)
        self.assertEqual(self.store.file_records()[str(other)]["size"], len("other\n"))

    def test_event_hashes_even_when_size_and_timestamp_are_unchanged(self):
        before = self.source.stat()
        self.source.write_text("modified\n")
        self.assertEqual(self.source.stat().st_size, before.st_size)
        os.utime(self.source, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.update(self.source), 1)
        self.assertEqual(self.searcher.search("modified")[0]["text"], "modified\n")

    def test_created_moved_and_deleted_subtrees_preserve_other_files(self):
        folder = self.workspace / "sub"
        folder.mkdir()
        (folder / "new.py").write_text("new\n")
        self.assertEqual(self.update(folder), 1)
        renamed = folder.rename(self.workspace / "renamed")
        self.assertEqual(self.update(folder, renamed), 1)
        self.assertEqual(self.searcher.last_index_stats.deleted_files, 1)
        shutil.rmtree(renamed)
        self.update(renamed)
        self.assertEqual(set(self.store.file_records()), {str(self.source)})

    def test_replacing_a_file_with_an_external_symlink_removes_stale_source(self):
        outside = self.root / "outside.py"
        outside.write_text("secret\n")
        self.source.unlink()
        self.source.symlink_to(outside)
        self.update(self.source)
        self.assertEqual(self.store.file_records(), {})

    def test_ignored_and_outside_paths_do_not_enter_index(self):
        ignored = self.workspace / "node_modules"
        ignored.mkdir()
        (ignored / "ignored.py").write_text("ignored\n")
        outside = self.root / "outside.py"
        outside.write_text("outside\n")
        self.assertEqual(self.update(ignored, outside), 0)
        self.assertEqual(set(self.store.file_records()), {str(self.source)})

    def test_outside_parent_event_does_not_read_tracked_files(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("outside event read source")):
            self.assertEqual(self.update(self.root), 0)

    def test_directory_replaced_with_external_symlink_removes_stale_descendants(self):
        folder = self.workspace / "sub"
        folder.mkdir()
        (folder / "nested.py").write_text("tracked\n")
        self.update(folder)
        shutil.rmtree(folder)
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "nested.py").write_text("secret\n")
        folder.symlink_to(outside, target_is_directory=True)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("read through symlink")):
            self.update(folder)
        self.assertEqual(set(self.store.file_records()), {str(self.source)})


class WatcherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source = self.workspace / "a.py"
        self.source.write_text("original\n")
        self.database = str(self.root / "index.db")
        self.encoder = Encoder()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.errors = []
        self.failures = []
        self.updates = []
        self.watcher = None

    def launch(self, *, reconcile=30, prepare=None):
        def run():
            try:
                with Store(self.database, 3) as store:
                    searcher = Searcher(store, self.encoder)
                    self.watcher = IndexWatcher(searcher, str(self.workspace), debounce=0.15,
                                                reconcile_interval=reconcile,
                                                on_update=lambda stats: (self.updates.append(stats), self.ready.set()),
                                                on_error=self.errors.append)
                    if prepare:
                        prepare(self.watcher)
                    self.watcher.run(stop=self.stop)
            except BaseException as error:
                self.failures.append(error)
                self.ready.set()

        thread = threading.Thread(target=run)
        thread.start()

        def shutdown():
            self.stop.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "watcher did not shut down")
            self.assertEqual(self.failures, [])

        self.addCleanup(shutdown)
        self.assertTrue(self.ready.wait(timeout=5), "initial indexing did not finish")
        self.assertEqual(self.failures, [])

    def wait_for(self, expected, *, timeout=6):
        deadline = time.monotonic() + timeout
        with Store(self.database, 3) as reader:
            while time.monotonic() < deadline:
                with reader.snapshot():
                    texts = [row[0] for row in reader.conn.execute("SELECT text FROM chunks ORDER BY path")]
                if texts == expected:
                    return
                time.sleep(0.03)
        self.fail(f"expected {expected}, got {texts}; errors={self.errors}, failures={self.failures}")

    def test_rapid_saves_are_coalesced_and_renames_and_deletions_update_live(self):
        self.launch()
        for number in range(10):
            self.source.write_text(f"edit {number}\n")
        self.wait_for(["edit 9\n"])
        self.assertEqual(len(self.encoder.batches), 2)
        renamed = self.source.rename(self.workspace / "b.py")
        self.wait_for(["edit 9\n"])
        deadline = time.monotonic() + 5
        with Store(self.database, 3) as reader:
            while str(renamed) not in reader.file_records() and time.monotonic() < deadline:
                time.sleep(0.03)
            self.assertIn(str(renamed), reader.file_records())
        renamed.unlink()
        self.wait_for([])

    def test_edit_during_embedding_is_processed_in_a_later_batch(self):
        self.launch()
        self.encoder.during_embed = lambda: self.source.write_text("latest edit\n")
        self.source.write_text("first edit\n")
        self.wait_for(["latest edit\n"])
        self.assertGreaterEqual(len(self.encoder.batches), 3)

    def test_observation_starts_before_initial_embedding(self):
        self.encoder.during_embed = lambda: self.source.write_text("edit during startup\n")
        self.launch()
        self.wait_for(["edit during startup\n"])
        self.assertGreaterEqual(len(self.encoder.batches), 2)

    def test_periodic_reconciliation_recovers_a_missed_event(self):
        self.launch(reconcile=0.1, prepare=lambda watcher: setattr(watcher, "on_any_event", lambda event: None))
        self.source.write_text("missed event\n")
        self.wait_for(["missed event\n"])

    def test_failed_batch_retries_without_another_edit(self):
        self.launch()
        original = self.watcher.searcher.index_path
        calls = []

        def fail_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("file changed while indexing")
            return original(*args, **kwargs)

        with patch.object(self.watcher.searcher, "index_path", side_effect=fail_once):
            self.source.write_text("retry this edit\n")
            self.wait_for(["retry this edit\n"])
        self.assertEqual(len(self.errors), 1)
        self.assertGreaterEqual(len(calls), 2)

    def test_event_queue_deduplicates_paths_and_filters_ignored_files(self):
        with Store(self.database, 3) as store:
            watcher = IndexWatcher(Searcher(store, self.encoder), str(self.workspace))
            for _ in range(1000):
                watcher.on_any_event(FileModifiedEvent(str(self.source)))
            watcher.on_any_event(FileModifiedEvent(str(self.workspace / "node_modules/ignored.py")))
            self.assertEqual(watcher._pending, {str(self.source)})

    def test_successful_retry_reports_recovery_without_source_changes(self):
        self.launch()
        self.ready.clear()
        original = self.watcher.searcher.index_path
        calls = []

        def fail_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("temporary indexing failure")
            return original(*args, **kwargs)

        with patch.object(self.watcher.searcher, "index_path", side_effect=fail_once):
            self.watcher.on_any_event(FileModifiedEvent(str(self.source)))
            self.assertTrue(self.ready.wait(timeout=4), "successful retry did not report recovery")
        self.assertEqual(len(self.errors), 1)
        self.assertEqual(self.updates[-1].indexed_files, 0)


if __name__ == "__main__":
    unittest.main()
