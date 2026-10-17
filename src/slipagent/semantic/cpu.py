"""Conservative CPU worker defaults with an explicit two-thread fallback."""

from __future__ import annotations

import psutil


def default_threads() -> int:
    try:
        physical = psutil.cpu_count(logical=False)
    except (OSError, NotImplementedError, psutil.Error):
        physical = None
    threads = physical
    if threads is None:
        threads = 2
    try:
        threads = min(threads, len(psutil.Process().cpu_affinity()))
    except (AttributeError, NotImplementedError, psutil.Error):
        pass
    return max(1, int(threads))


def background_threads() -> int:
    """Leave half the cores to the user's own work during background indexing."""
    return max(1, default_threads() // 2)
