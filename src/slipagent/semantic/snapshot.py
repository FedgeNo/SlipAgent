"""Hash checked source descriptors into temporary, immutable indexing snapshots."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from ..fileio import open_regular

SOURCE_BLOCK_BYTES = 64 * 1024


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


@contextmanager
def source_snapshot(path: Path, before: os.stat_result,
                    check_cancelled: Callable[[], None], *, directory: Path) -> Iterator[tuple[BinaryIO, str]]:
    """Keep one disk snapshot; reject reads racing a source edit or replacement."""
    digest = hashlib.sha256()
    # Keep snapshots beside the index: the system temporary directory may be RAM-backed.
    with tempfile.TemporaryFile(dir=directory) as snapshot:
        with open_regular(path) as source:
            if _identity(before) != _identity(os.fstat(source.fileno())):
                raise OSError(f"File changed while indexing; retry: {path}")
            while True:
                check_cancelled()
                block = source.read(SOURCE_BLOCK_BYTES)
                if not block:
                    break
                digest.update(block)
                snapshot.write(block)
            after = os.fstat(source.fileno())
        if _identity(before) != _identity(after) or _identity(before) != _identity(path.stat()):
            raise OSError(f"File changed while indexing; retry: {path}")
        snapshot.seek(0)
        yield snapshot, digest.hexdigest()
