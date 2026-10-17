"""File editing preserves untouched data and refuses external conflicts."""

import pytest

from slipagent.editor import FileEditor, EditConflict, signature


@pytest.mark.parametrize("control", ["\x01", "\x07", "\x1b", "\x7f", "\x9b"])
def test_unsupported_control_characters_open_read_only(tmp_path, control):
    path = tmp_path / "controls.txt"
    original = ("plain\n" * 100 + control + "tail\n").encode()
    path.write_bytes(original)
    editor = FileEditor(path)
    try:
        assert editor.read_only
        assert "".join(editor.chunks()).encode() == original
        with pytest.raises(ValueError, match="read-only"):
            editor.set_page(0, "changed")
        with pytest.raises(ValueError, match="read-only"):
            editor.save()
        assert path.read_bytes() == original
    finally:
        editor.close()


@pytest.mark.parametrize("original,edited,expected", [
    ("unchanged\n", "unchanged\n", ""),
    ("old", "new", "-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"),
    ("x" * 5000 + "\r\nold\r\n", "x" * 5000 + "\r\nnew\r\n", "-old\r\n+new\r\n"),
])
def test_editor_diff_and_save_without_git(tmp_path, monkeypatch, original, edited, expected):
    from slipagent import editor as module
    from unittest.mock import Mock
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    run = Mock(side_effect=AssertionError("No external diff process should run"))
    monkeypatch.setattr(module.subprocess, "run", run)
    path = tmp_path / "file.txt"
    path.write_bytes(original.encode())
    editor = FileEditor(path)
    try:
        editor.set_page(0, edited)
        diff = editor.diff()
        if expected:
            assert f"--- {path}\n+++ {path}\n" in diff
            assert expected in diff
        else:
            assert diff == ""
        editor.save()
        assert path.read_bytes() == edited.encode()
        run.assert_not_called()
    finally:
        editor.close()


def test_editing_pages_preserves_large_file_and_line_endings(tmp_path):
    path = tmp_path / "large.txt"
    # A megabyte spans many pages without making the diff index 100,000 lines.
    text = ("original" + "x" * 1024 + "\r\n") * 1025
    path.write_bytes(text.encode())
    editor = FileEditor(path)
    try:
        assert editor.page_count > 1
        assert len(editor.page_text(0)) < 140_000
        editor.set_page(0, editor.page_text(0).replace("original", "edited", 1))
        editor.set_page(editor.page_count - 1, editor.page_text(editor.page_count - 1) + "last\n")
        editor.save()
        assert path.read_bytes() == (text.replace("original", "edited", 1) + "last\n").encode()
        diff = editor.diff()
        assert "+edited" in diff and "+last" in diff
        assert str(path) in diff
    finally:
        editor.close()


@pytest.mark.parametrize("change", ["replace", "modify", "delete"])
def test_save_requires_explicit_overwrite_on_disk_change(tmp_path, change):
    path = tmp_path / "file.txt"
    path.write_text("original\n")
    editor = FileEditor(path)
    try:
        editor.set_page(0, "user\n")
        if change == "replace":
            replacement = tmp_path / "replacement"
            replacement.write_text("external\n")
            replacement.replace(path)
        elif change == "modify":
            path.write_text("external\n")
        else:
            path.unlink()
        with pytest.raises(EditConflict):
            editor.save()
        assert editor.page_text(0) == "user\n"
        editor.save(overwrite=True, confirmed_signature=signature(path))
        assert path.read_text() == "user\n"
        assert not list(tmp_path.glob(".slipagent-edit-*"))
    finally:
        editor.close()


def test_save_rechecks_disk_immediately_before_publish(tmp_path, monkeypatch):
    path = tmp_path / "file.txt"
    path.write_text("original")
    editor = FileEditor(path)
    chmod = type(path).chmod
    def race(temporary, mode, **kwargs):
        chmod(temporary, mode, **kwargs)
        path.write_text("external")
    try:
        editor.set_page(0, "user")
        monkeypatch.setattr(type(path), "chmod", race)
        with pytest.raises(EditConflict):
            editor.save()
        assert path.read_text() == "external"
        assert editor.page_text(0) == "user"
    finally:
        editor.close()


def test_binary_files_are_not_opened_for_text_editing(tmp_path):
    path = tmp_path / "binary"
    path.write_bytes(b"abc\x00def")
    with pytest.raises(ValueError, match="Binary"):
        FileEditor(path)


def test_overwrite_consent_does_not_cover_later_external_changes(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("original")
    editor = FileEditor(path)
    try:
        editor.set_page(0, "user")
        path.write_text("external")
        confirmed = signature(path)
        path.write_text("external again")
        with pytest.raises(EditConflict):
            editor.save(overwrite=True, confirmed_signature=confirmed)
        assert path.read_text() == "external again"
    finally:
        editor.close()


@pytest.mark.parametrize("operation", ["diff", "save"])
@pytest.mark.parametrize("legacy_snapshot", [False, True])
def test_redraw_during_worker_read_preserves_diff_and_file(tmp_path, monkeypatch, operation, legacy_snapshot):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import slipagent.inspection as inspection

    positioned, release, render_progress = (threading.Event() for _ in range(3))
    role = threading.local()

    class ObservedLock:
        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            if getattr(role, "render", False):
                render_progress.set()
            return self.lock.__enter__()

        def __exit__(self, *args):
            return self.lock.__exit__(*args)

    monkeypatch.setattr(inspection, "RLock", ObservedLock, raising=False)
    path = tmp_path / "long-lines.txt"
    original = "".join(f"line-{n:03} " + "long wrapped contents " * 12 + "\n" for n in range(70))
    path.write_text(original)
    editor = FileEditor(path)
    if legacy_snapshot:
        editor.original.__dict__.pop("_snapshot_lock", None)

    class PauseWorkerAfterSeek:
        def __init__(self, source):
            self.source = source
            self.paused = False

        def __getattr__(self, name):
            return getattr(self.source, name)

        def seek(self, *args):
            result = self.source.seek(*args)
            if getattr(role, "worker", False) and not self.paused:
                self.paused = True
                positioned.set()
                assert release.wait(5)
            return result

    editor.original.text = PauseWorkerAfterSeek(editor.original.text)

    def publish():
        role.worker = True
        return getattr(editor, operation)()

    def render():
        role.render = True
        try:
            return editor.page_text(1)
        finally:
            render_progress.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            worker = pool.submit(publish)
            try:
                assert positioned.wait(5)
                drawing = pool.submit(render)
                # Wait for either a blocked read or the unsynchronized redraw.
                assert render_progress.wait(5)
            finally:
                release.set()
            result = worker.result(timeout=5)
            assert drawing.result(timeout=5) == "".join(original.splitlines(keepends=True)[32:64])
        assert path.read_bytes() == original.encode()
        if operation == "diff":
            assert result == ""
    finally:
        editor.close()
