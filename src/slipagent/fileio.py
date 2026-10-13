"""Regular-file descriptors that cannot wait for a pipe or follow swapped links."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import BinaryIO


def open_regular(path: Path) -> BinaryIO:
    path = path.absolute()
    if not stat.S_ISREG(path.stat().st_mode):
        raise OSError("Only regular files can be read.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    if os.name == "posix":
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory = os.open(path.anchor, directory_flags)
        try:
            for part in path.parts[1:-1]:
                child = os.open(part, directory_flags, dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(path.name, flags, dir_fd=directory)
        finally:
            os.close(directory)
    else:
        descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Only regular files can be read.")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def read_regular(path: Path, limit: int | None = None) -> bytes:
    with open_regular(path) as source:
        if limit is not None and os.fstat(source.fileno()).st_size > limit:
            raise OSError(f"File exceeds the read memory limit ({limit} bytes).")
        data = source.read() if limit is None else source.read(limit + 1)
    if limit is not None and len(data) > limit:
        raise OSError(f"File exceeds the read memory limit ({limit} bytes).")
    return data
