"""Incremental wrapping must agree with complete text at every chunk boundary."""

from __future__ import annotations

import pytest
from wcwidth import width, wrap

from slipagent.transcript import AnsiStream, TranscriptFile, WrappedTranscript


def lines(flow):
    tail = flow.content_rows()
    return [flow.get_row(i) for i in range(len(flow.rows))] + tail


def expected(text, columns):
    result = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.expandtabs()
        indent = paragraph[:len(paragraph) - len(paragraph.lstrip(" "))]
        if len(indent) >= columns:
            indent = ""
        result.extend([paragraph] if width(paragraph) <= columns else wrap(
            paragraph, columns, subsequent_indent=indent, replace_whitespace=False,
            break_on_hyphens=False, propagate_sgr=False,
        ) or [""])
    return result


@pytest.mark.parametrize("columns", [8, 20, 40])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_streamed_rows_match_complete_unicode_and_indented_text(columns, chunk_size):
    text = ("    界界界 a " + "long" * 25 + " e\u0301 👩\u200d💻 🇨🇦 word " * 8
            + "\n\nName    Count\nalpha   12\n    one two three four five")
    flow = WrappedTranscript(columns)
    try:
        for start in range(0, len(text), chunk_size):
            end = start + chunk_size
            flow.append(text[start:end], first=start == 0)
            actual = ["".join(fragment[1] for fragment in row) for row in lines(flow)]
            assert actual == expected(text[:end], columns)
    finally:
        flow.close()


def test_streamed_ansi_escapes_and_styles_survive_chunk_boundaries():
    decoder = AnsiStream()
    text = "\x1b[38;2;255;0;255mcolored\x1b[0m plain \x1b[1mbold"
    result = [fragment for char in text for fragment in decoder.feed(char)]
    assert "".join(text for _, text in result) == "colored plain bold"
    assert all(style == "#ff00ff" for style, _ in result[:7])
    assert all(style == "" for style, _ in result[7:14])
    assert all(style == "bold" for style, _ in result[14:])


def test_explicit_blocks_reset_style_but_newlines_keep_it():
    flow = WrappedTranscript(20)
    try:
        flow.append("\x1b[1mfirst\nsecond", first=True)
        flow.append("third", first=True)
        rows = lines(flow)
        assert rows == [[("bold", "first")], [("bold", "second")], [("", "third")]]
    finally:
        flow.close()


def test_large_whitespace_runs_and_tab_alignment_remain_bounded(monkeypatch):
    import slipagent.transcript as transcript
    flow = WrappedTranscript(20)
    examined = []
    original = transcript.wrap
    def measure(text, *args, **kwargs):
        examined.append(len(text))
        return original(text, *args, **kwargs)
    monkeypatch.setattr(transcript, "wrap", measure)
    try:
        text = "first" + " " * 4000 + "\tlast"
        for start in range(0, len(text), 17):
            flow.append(text[start:start + 17], first=start == 0)
            flow.content_rows()
        assert ["".join(fragment[1] for fragment in row) for row in lines(flow)] == expected(text, 20)
        assert max(examined) < 100
    finally:
        flow.close()


def test_file_records_support_random_access_and_release_storage():
    archive = TranscriptFile()
    try:
        for record in ["", "first\nsecond", "界\x1b[1m", "last"]:
            archive.append(record)
        assert archive[2] == "界\x1b[1m"
        assert archive[0] == ""
        assert archive[-1] == "last"
        assert archive[1:3] == ["first\nsecond", "界\x1b[1m"]
        assert len(archive) == 4
        with pytest.raises(IndexError):
            archive[4]
    finally:
        archive.close()
    with pytest.raises(ValueError, match="closed"):
        archive[1]
