"""Coalesce filesystem events and publish atomic incremental index updates."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .search import IndexStats, Searcher


class IndexWatcher(FileSystemEventHandler):
    """Run in the Store's owning thread; observer threads only enqueue paths.

    Observation begins before the initial scan. Taking a batch removes it from
    the queue before indexing, so edits arriving during inference remain pending.
    Periodic metadata reconciliation catches missed events and branch switches;
    its interval doubles while scans find nothing, up to `MAX_RECONCILE_INTERVAL`.
    Watches cover the root and its accepted top-level directories, so excluded
    trees such as `.venv` and `node_modules` use no inotify watches.
    """

    MAX_RECONCILE_INTERVAL = 600.0

    def __init__(self, searcher: Searcher, path: str, *, max_lines: int = 50, overlap: int = 5,
                 debounce: float = 0.25, max_delay: float = 1,
                 reconcile_interval: float = 30,
                 on_update: Callable[[IndexStats], None] | None = None,
                 on_error: Callable[[Exception], None] | None = None):
        super().__init__()
        if debounce < 0 or max_delay <= 0 or reconcile_interval <= 0:
            raise ValueError("Watcher delays must be positive, with a nonnegative debounce")
        self.searcher = searcher
        self.root = Path(path).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(str(self.root))
        self.max_lines = max_lines
        self.overlap = overlap
        self.debounce = debounce
        self.max_delay = max_delay
        self.reconcile_interval = reconcile_interval
        self.on_update = on_update
        self.on_error = on_error
        self._condition = threading.Condition()
        self._pending: set[str] = set()
        self._first_event: float | None = None
        self._last_event: float | None = None
        # Top-level directory paths whose watch must be added (True) or removed (False).
        self._directories: dict[str, bool] = {}

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type not in {"created", "modified", "deleted", "moved", "closed"}:
            return
        if event.is_directory and event.event_type in {"modified", "closed"}:
            return
        for value in (event.src_path, getattr(event, "dest_path", "")):
            if not value:
                continue
            path = str(value)
            if event.is_directory and os.path.dirname(path) == str(self.root):
                with self._condition:
                    self._directories[path] = event.event_type == "created" or value is not event.src_path
            if not self.searcher.accepts_path(self.root, path, directory=event.is_directory, check_symlinks=False):
                continue
            with self._condition:
                self._pending.add(path)
                now = time.monotonic()
                if self._first_event is None:
                    self._first_event = now
                self._last_event = now
                self._condition.notify()

    def refresh(self) -> None:
        """Publish already queued edits before a lookup on the owning thread."""
        with self._condition:
            batch = self._pending
            self._pending = set()
            self._first_event = self._last_event = None
        if not batch:
            return
        try:
            self.searcher.index_path(str(self.root), max_lines=self.max_lines,
                                     overlap=self.overlap, changed_paths=batch)
        except Exception:
            with self._condition:
                self._pending.update(batch)
                now = time.monotonic()
                self._first_event = self._last_event = now
            raise

    def run(self, *, stop: threading.Event | None = None, rebuild: bool = False,
            on_idle: Callable[[], None] | None = None) -> None:
        """Watch until stop is set or interrupted; retain failed batches for retry."""
        stop = stop if stop is not None else threading.Event()
        observer = Observer(timeout=0.1)
        observer.schedule(self, str(self.root), recursive=False)
        watched: dict[str, Any] = {}

        def watch(directories: dict[str, bool]) -> None:
            for directory, present in directories.items():
                if directory in watched:
                    try:
                        observer.unschedule(watched.pop(directory))
                    except KeyError:
                        pass
                if (present and os.path.isdir(directory) and not os.path.islink(directory)
                        and self.searcher.accepts_path(self.root, directory, directory=True, check_symlinks=False)):
                    watched[directory] = observer.schedule(self, directory, recursive=True)

        with os.scandir(self.root) as entries:
            watch({entry.path: True for entry in entries if entry.is_dir(follow_symlinks=False)})
        observer.start()
        full_scan = True
        interval = self.reconcile_interval
        next_scan = next_attempt = 0.0
        retry: set[str] = set()
        last_error = None
        initial = True
        try:
            while not stop.is_set():
                if on_idle is not None:
                    on_idle()
                now = time.monotonic()
                with self._condition:
                    directories, self._directories = self._directories, {}
                if directories:
                    # Files created before the watch exists are found by walking the directory.
                    watch(directories)
                    with self._condition:
                        self._pending.update(directory for directory in directories if directory in watched)
                        self._first_event = self._first_event or now
                        self._last_event = now
                with self._condition:
                    full_scan = full_scan or now >= next_scan
                    due = (min(self._first_event + self.max_delay, self._last_event + self.debounce)
                           if self._pending and self._first_event is not None and self._last_event is not None
                           else float("inf"))
                    if now < next_attempt or not (full_scan or retry or now >= due):
                        self._condition.wait(timeout=0.1)
                        continue
                    batch = self._pending | retry
                    self._pending = set()
                    self._first_event = self._last_event = None
                try:
                    self.searcher.index_path(str(self.root), max_lines=self.max_lines, overlap=self.overlap,
                                             rebuild=rebuild,
                                             changed_paths=None if full_scan else batch)
                except (OSError, sqlite3.OperationalError) as error:
                    retry = batch
                    next_attempt = time.monotonic() + 1
                    message = str(error)
                    if message != last_error:
                        if self.on_error is not None:
                            self.on_error(error)
                        else:
                            logging.getLogger(__name__).warning("Index update failed; retrying: %s", error)
                    last_error = message
                    continue
                stats = self.searcher.last_index_stats
                if full_scan:
                    changed = stats.indexed_files or stats.deleted_files
                    interval = self.reconcile_interval if changed else min(interval * 2, self.MAX_RECONCILE_INTERVAL)
                    next_scan = time.monotonic() + interval
                full_scan = rebuild = False
                retry = set()
                if self.on_update is not None and (initial or last_error is not None
                                                  or stats.indexed_files or stats.deleted_files):
                    self.on_update(stats)
                last_error = None
                initial = False
        finally:
            observer.stop()
            observer.join()
