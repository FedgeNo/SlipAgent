"""Session-owned actor for lazy model loading, indexing, watching, and lookup."""

from __future__ import annotations

from concurrent.futures import Future
import hashlib
import logging
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
from typing import Any, Callable

from ..config import command_environment, data_directory
from ..lifecycle import Lifetime
from ..tools.blocking import run_blocking
from ..workspace import Workspace


class SemanticSearch:
    """Keep all model and SQLite operations on one lazily started worker."""

    def __init__(self, workspace: Workspace, *,
                 encoder_factory: Callable[[], Any] | None = None) -> None:
        self.workspace = workspace
        identity = hashlib.sha256(os.fsencode(workspace.root)).hexdigest()
        self.state_directory = data_directory().resolve()
        self.database = self.state_directory / "semantic" / identity / "index-v2.db"
        self.encoder_factory = encoder_factory
        self._requests: Queue[tuple[str, int, Future[dict[str, Any]]]] = Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._closing: Lifetime | None = None

    def lookup(self, query: str, limit: int) -> dict[str, Any]:
        request: Future[dict[str, Any]] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("Semantic search is closed")
            self._requests.put((query, limit, request))
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, name="slipagent-semantic")
                self._thread.start()
        try:
            return request.result()
        finally:
            if self._stop.is_set():
                thread = self._thread
                if thread is not None:
                    thread.join()

    def interrupt(self) -> None:
        """Stop between inference batches; the awaiting caller drains the worker."""
        self._stop.set()

    def _paths(self, names: list[str]) -> set[str]:
        """Apply workspace confinement and Git's own ignore rules before reads."""
        accepted: dict[str, str] = {}
        for name in names:
            source = Path(name)
            if (self.workspace.contains(source) and not source.is_symlink()
                    and self.workspace.contains(source.resolve())
                    and not any(part.startswith(".") for part in source.relative_to(self.workspace.root).parts)
                    and not source.is_relative_to(self.state_directory)):
                accepted[self.workspace.relative(source)] = name
        relative = list(accepted)
        for start in range(0, len(relative), 1000):
            if self._stop.is_set():
                raise InterruptedError("Semantic indexing interrupted")
            batch = relative[start:start + 1000]
            try:
                result = subprocess.run(
                    ["git", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}",
                     "check-ignore", "-z", "--stdin"],
                    input=("\0".join(batch) + "\0").encode(), cwd=self.workspace.root,
                    capture_output=True, timeout=2, env=command_environment(),
                )
            except FileNotFoundError:
                break
            if result.returncode == 128 and b"not a git repository" in result.stderr:
                break
            if result.returncode not in {0, 1}:
                raise OSError("Cannot apply Git ignore rules: " + result.stderr.decode(errors="replace"))
            for ignored in result.stdout.decode(errors="replace").split("\0"):
                accepted.pop(ignored, None)
        return set(accepted.values())

    def _run(self) -> None:
        # Imports and native resources stay out of ordinary startup and discovery.
        failure: Exception = InterruptedError("Semantic search interrupted")
        try:
            from .embed import ONNXEmbedder
            from .search import Searcher
            from .storage import Store
            from .watcher import IndexWatcher

            from .cpu import background_threads

            encoder = (self.encoder_factory() if self.encoder_factory is not None
                       else ONNXEmbedder(threads=background_threads()))
            dimensions = encoder.dim()
            if self._stop.is_set():
                return
            self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Version one numbered lines with str.splitlines; its chunks cannot be reused.
            for suffix in ("", "-wal", "-shm", "-journal"):
                (self.database.parent / f"index-v1.db{suffix}").unlink(missing_ok=True)
            with Store(str(self.database), dimensions, threads=getattr(encoder, "threads", None)) as store:
                searcher = Searcher(store, encoder, cancelled=self._stop.is_set, path_filter=self._paths)
                ready = False
                update_error: Exception | None = None

                def updated(stats: Any) -> None:
                    nonlocal ready, update_error
                    ready = True
                    update_error = None

                def update_failed(error: Exception) -> None:
                    nonlocal update_error
                    update_error = error
                    logging.getLogger(__name__).warning("Semantic index update failed; retrying: %s", error)
                    while True:
                        try:
                            _, _, request = self._requests.get_nowait()
                        except Empty:
                            break
                        request.set_exception(error)

                watcher = IndexWatcher(searcher, str(self.workspace.root),
                                       on_error=update_failed, on_update=updated)

                def lookup() -> None:
                    if self._stop.is_set():
                        return
                    if not ready and update_error is None:
                        return
                    try:
                        query, limit, request = self._requests.get_nowait()
                    except Empty:
                        return
                    try:
                        if update_error is not None:
                            raise update_error
                        watcher.refresh()
                        hits = searcher.search(query, limit=limit)
                        matches = []
                        for hit in hits:
                            path = self.workspace.resolve(hit["path"])
                            if not self.workspace.contains(path):
                                raise OSError("Indexed path escaped the workspace; retry after reconciliation")
                            matches.append({"path": self.workspace.relative(path),
                                            "start_line": hit["start_line"], "end_line": hit["end_line"],
                                            "text": hit["text"], "score": hit["score"]})
                        request.set_result({"query": query, "matches": matches,
                                            "indexed_files": store.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                                            "snapshot": "Indexed source excerpts; read current source before editing.",
                                            "ranking": "75% semantic and 25% keyword; scores are relative, not confidence."})
                    except Exception as error:
                        request.set_exception(error)

                watcher.run(stop=self._stop, on_idle=lookup)
        except Exception as error:
            failure = error
            logging.getLogger(__name__).warning("Semantic search stopped: %s", error)
        finally:
            # Serialize settlement against enqueuing and restarting a failed worker.
            with self._lock:
                while True:
                    try:
                        _, _, request = self._requests.get_nowait()
                    except Empty:
                        break
                    request.set_exception(failure)
                self._thread = None

    def _close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop.set()
            thread = self._thread
        if thread is not None:
            thread.join()

    async def aclose(self) -> None:
        if self._closing is None:
            self._closing = Lifetime("semantic search")
            self._closing.defer(lambda: run_blocking(self._close))
        await self._closing.aclose()
