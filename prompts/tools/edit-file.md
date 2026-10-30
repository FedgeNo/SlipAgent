# `edit_file`

Replace an exact substring using `old_string` and `new_string`. Copy source and indentation from a current read, excluding displayed line numbers. The target must occur exactly once unless `replace_all=true`; include context to make it unique. Prefer this tool for changes to existing code.

Copied LF text containing no CR also matches CRLF or CR. Line endings outside the replacement are preserved.

Alternatively, supply `edits=[{old_string,new_string}, …]` for unique, non-overlapping replacements, each matching the ORIGINAL file. Choose either `edits` or the single pair. All edits validate before one atomic write.

The original must be a regular UTF-8 file up to 32 MB. Unsafe path changes are rejected. Matching or validation failures apply no edits; nearby source is diagnostic only. Storage or checkpoint errors may follow a successful write; inspect results before retrying.

Use supplied diffs to assess earlier edits; requested edits remain pending verification.
