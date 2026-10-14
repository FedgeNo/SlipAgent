# `read_hashlines`

Read optional hash-anchored source for `edit_hashlines`, while ordinary `read_file` and exact `edit_file` remain available. `path` selects a regular UTF-8 file, `offset` is the first 1-based line (default 1), and `limit` is the maximum number of lines (default 200, maximum 2000). CRLF, LF, and CR are line endings; other Unicode characters remain source text. Files are limited to 32 MB.

The result contains a complete-file SHA-256 `revision` and `lines` entries with an `anchor` such as `12:8f01ab23cd` and literal `text`. Copy these values; do not calculate or invent hashes. `next_offset` identifies another page. An empty file has no anchors; use `write_file` for an empty or new file.

Load `edit_hashlines` separately if its full definition is not supplied in this request. The same response cannot load and execute a deferred tool.
