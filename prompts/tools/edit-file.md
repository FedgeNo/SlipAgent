# `edit_file`

## Exact Matching

Replace an exact substring within a file. `old_string` contains the text to replace with `new_string`; it does not need to contain the entire file. It must occur exactly once unless `replace_all=true`; include surrounding lines to make it unique.

Prefer this tool over `write_file` for changes to existing code. Copy `old_string` from a real file read, including indentation.

When copied text contains no CR, its LF also matches CRLF or CR. Existing line endings are preserved outside the replacement.

Omit `read_file`'s line numbers.

## Multiple Replacements

Alternatively supply `edits=[{old_string,new_string}, ...]` for several unique, non-overlapping replacements in this file.

Every edit matches the ORIGINAL file; all are validated before one atomic write.

Choose either `edits` or the single `old_string`/`new_string` pair, never both.

## Failure and Review

A matching or validation failure applies no edits; nearby source is diagnostic evidence, not an applied fuzzy match. A later storage or checkpoint error may report an already-written file; inspect that result before retrying.

Read the returned diffs to check the batch's changes.
