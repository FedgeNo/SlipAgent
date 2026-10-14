# `edit_hashlines`

Edit existing source with copied anchors; exact `edit_file` remains available. First use `read_hashlines`; supply its `path`, complete-file `revision`, and 1–100 `edits` against that same original revision. Copy anchors exactly; the harness computes and validates hashes.

Each edit has `op`, `start`, and `lines`. `start` is a copied anchor; `lines` is an array of replacement source lines WITHOUT anchor labels or CR/LF. Tabs and other Unicode characters are literal source text.

- `replace` replaces the whole start line, or the inclusive range through optional `end`. An empty `lines` array deletes the range.
- `insert_before` and `insert_after` insert nonempty `lines` beside the start line; omit `end`.

All anchors are checked against the ORIGINAL file before any changes. Overlapping ranges, insertions inside replacements, duplicate insertion boundaries, and copied display anchors in replacement text are rejected. Multiple edits apply together in one atomic write and share ordinary checkpoints, instruction checks, and post-edit checks.

Any file revision change, including moved blocks or changed line endings, rejects stale edits. Read again and use fresh anchors. Short line hashes serve only as matching aids. New lines use the file's first line-ending style; replacement ranges preserve their final terminator and insertion at EOF preserves whether the original ended in a newline.

Successful output includes a diff, fresh revision, and a bounded anchored excerpt near the first edit; read other changed locations before chaining more edits. A no-op writes nothing. Validation errors make no changes. A storage/checkpoint error can occur after publication; inspect before retrying. Use `write_file` for empty or new files.
