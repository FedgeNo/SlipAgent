"""Keep every source character inside a chunk that fits the encoder budget."""

import unittest
import tempfile
from pathlib import Path

from slipagent.semantic.indexer import chunk_text, walk_files


class TokenChunkingTests(unittest.TestCase):
    def test_empty_extension_filter_indexes_all_nonhidden_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("source.py", "custom.extension", "extensionless", ".hidden"):
                (root / name).write_text("text")
            self.assertEqual({Path(path).name for path in walk_files(directory, index_exts=[])},
                             {"source.py", "custom.extension", "extensionless"})

    def chunks(self, text, **kwargs):
        return chunk_text("source.py", text, token_count=lambda value: len(value) + 2,
                          max_tokens=kwargs.pop("max_tokens", 10), **kwargs)

    def test_lines_are_kept_whole_when_they_fit(self):
        chunks = self.chunks("aa\nbb\ncc\ndd\n", max_tokens=8, overlap=1)
        self.assertEqual([chunk.text for chunk in chunks], ["aa\nbb\n", "bb\ncc\n", "cc\ndd\n"])
        self.assertEqual([(chunk.start_line, chunk.end_line) for chunk in chunks], [(1, 2), (2, 3), (3, 4)])

    def test_long_single_lines_split_without_losing_unicode_or_line_endings(self):
        source = "value = '" + "λ🌓" * 30 + "'\r\nreturn value\n"
        chunks = self.chunks(source, max_tokens=11, overlap=0)
        self.assertEqual("".join(chunk.text for chunk in chunks), source)
        self.assertTrue(all(len(chunk.text) + 2 <= 11 for chunk in chunks))
        self.assertTrue(all(chunk.start_line == chunk.end_line for chunk in chunks))
        self.assertEqual(chunks[-1].end_line, 2)

    def test_line_limit_still_applies_when_token_budget_is_large(self):
        chunks = self.chunks("a\nb\nc\nd\ne\n", max_tokens=100, max_lines=3, overlap=1)
        self.assertEqual([chunk.text for chunk in chunks], ["a\nb\nc\n", "c\nd\ne\n"])

    def test_large_overlap_still_advances_and_covers_all_lines(self):
        source = "".join(f"{number:02}\n" for number in range(50))
        chunks = self.chunks(source, max_tokens=8, overlap=5)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[-1].end_line, 50)
        self.assertTrue(all(left.start_line < right.start_line for left, right in zip(chunks, chunks[1:])))
        self.assertEqual({line for chunk in chunks for line in range(chunk.start_line, chunk.end_line + 1)},
                         set(range(1, 51)))

    def test_empty_source_has_no_chunks(self):
        self.assertEqual(self.chunks(""), [])

    def test_impossible_budget_reports_error_instead_of_dropping_text(self):
        with self.assertRaisesRegex(ValueError, "fit"):
            self.chunks("abc", max_tokens=2)

    def test_token_counter_and_limit_must_be_supplied_together(self):
        with self.assertRaises(ValueError):
            chunk_text("source.py", "abc", token_count=len)
        with self.assertRaises(ValueError):
            chunk_text("source.py", "abc", max_tokens=10)


if __name__ == "__main__":
    unittest.main()
