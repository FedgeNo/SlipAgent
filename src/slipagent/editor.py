"""Paged, file-backed editing with explicit disk-conflict checks."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from pathlib import Path
from collections.abc import Iterator

from .inspection import IndexedText, CHUNK_SIZE

PAGE_SEGMENTS = 32


def signature(path: Path) -> tuple[int, int, int, int, int] | None:
    try:
        value = path.stat()
    except OSError:
        return None
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class EditConflict(ValueError):
    pass


class FileEditor:
    """Keep only one bounded page in the UI and changed pages on disk."""

    def __init__(self, path: Path) -> None:
        self.edit_id = uuid.uuid4().hex
        self.path = path.resolve(strict=True)
        self.loaded = signature(self.path)
        if not stat.S_ISREG(self.path.stat().st_mode):
            raise ValueError("Only regular text files can be edited")
        self.original = IndexedText()
        self.overlay = tempfile.TemporaryDirectory(prefix="slipagent-editor-")
        self.pages: dict[int, Path] = {}
        try:
            with self.path.open(encoding="utf-8", newline="") as source:
                while text := source.read(CHUNK_SIZE):
                    if "\x00" in text:
                        raise ValueError("Binary files cannot be edited as text")
                    self.original.text.write(text)
            self.original.finish()
            if signature(self.path) != self.loaded:
                raise EditConflict("File changed while it was being loaded")
        except BaseException:
            self.close()
            raise

    @property
    def page_count(self) -> int:
        return max(1, (len(self.original) + PAGE_SEGMENTS - 1) // PAGE_SEGMENTS)

    def _original_chunks(self, page: int) -> Iterator[str]:
        for index in range(page * PAGE_SEGMENTS, min((page + 1) * PAGE_SEGMENTS, len(self.original))):
            yield self.original.segment(index)

    def page_text(self, page: int) -> str:
        snapshot = self.pages.get(page)
        if snapshot is None:
            return "".join(self._original_chunks(page))
        with snapshot.open(encoding="utf-8", newline="") as source:
            return source.read()

    def original_page_text(self, page: int) -> str:
        return "".join(self._original_chunks(page))

    def set_page(self, page: int, text: str) -> None:
        if text == "".join(self._original_chunks(page)):
            previous = self.pages.pop(page, None)
            if previous is not None:
                previous.unlink()
            return
        snapshot = Path(self.overlay.name) / str(page)
        descriptor, name = tempfile.mkstemp(dir=self.overlay.name)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
                target.write(text)
            temporary.replace(snapshot)
            self.pages[page] = snapshot
        finally:
            temporary.unlink(missing_ok=True)

    def chunks(self) -> Iterator[str]:
        for page in range(self.page_count):
            snapshot = self.pages.get(page)
            if snapshot is None:
                yield from self._original_chunks(page)
            else:
                with snapshot.open(encoding="utf-8", newline="") as source:
                    while chunk := source.read(CHUNK_SIZE):
                        yield chunk

    def save(self, *, overwrite: bool = False,
             confirmed_signature: tuple[int, int, int, int, int] | None = None) -> None:
        expected = confirmed_signature if overwrite else self.loaded
        if signature(self.path) != expected:
            raise EditConflict("File changed on disk; reload, cancel, or explicitly overwrite")
        mode = stat.S_IMODE(self.path.stat().st_mode) if expected is not None else 0o600
        descriptor, name = tempfile.mkstemp(prefix=".slipagent-edit-", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
                for chunk in self.chunks():
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            temporary.chmod(mode)
            if signature(self.path) != expected:
                raise EditConflict("File changed again before save; nothing was overwritten")
            os.replace(temporary, self.path)
            self.loaded = signature(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def diff(self) -> str:
        """Use Git's native diff engine; capture its output on disk first."""
        git = shutil.which("git")
        if git is None:
            raise ValueError("Git is required to record user-edit diffs")
        with tempfile.TemporaryDirectory(prefix="slipagent-user-edit-") as directory:
            before, after = Path(directory) / "before", Path(directory) / "after"
            with before.open("w", encoding="utf-8", newline="") as target:
                for index in range(len(self.original)):
                    target.write(self.original.segment(index))
            with after.open("w", encoding="utf-8", newline="") as target:
                for chunk in self.chunks():
                    target.write(chunk)
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
                result = subprocess.run([git, "--no-pager", "diff", "--no-index", "--no-ext-diff", "--no-textconv", "--text", "--", str(before), str(after)],
                                        stdout=output, stderr=subprocess.PIPE, check=False)
                if result.returncode not in (0, 1):
                    raise ValueError(result.stderr.decode("utf-8", errors="replace"))
                output.seek(0)
                text = output.read()
            # Keep user-facing paths rather than ephemeral private snapshots.
            return text.replace(str(before), str(self.path)).replace(str(after), str(self.path))

    def close(self) -> None:
        self.pages.clear()
        self.overlay.cleanup()
        self.original.close()
