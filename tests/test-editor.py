"""File editing preserves untouched data and refuses external conflicts."""

import pytest

from slipagent.editor import FileEditor, EditConflict, signature


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
    text = "original\r\n" * 100_000
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
