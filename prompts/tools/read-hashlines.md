# `read_hashlines`

Read hash-anchored source for `edit_hashlines`; ordinary `read_file` and `edit_file` remain available. `path` selects a regular UTF-8 file up to 32 MB. `offset` is the first 1-based line (default 1); `limit` caps lines (default 200, maximum 2000). CRLF, LF, and CR are line endings; other Unicode remains source text.

Results include a complete-file SHA-256 `revision`, `lines` containing `anchor` and literal `text`, and `next_offset` for paging. Copy revisions and anchors exactly from results. Empty files have no anchors; use `write_file` for empty or new files.

Load `edit_hashlines` separately when needed and call it after receiving its full definition.
