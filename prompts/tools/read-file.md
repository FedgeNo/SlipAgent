# `read_file`

Read a regular UTF-8 text file up to 32 MB (32,000,000 bytes), subject to Workspace Access. Returns numbered lines with LF display endings; copy source text without line numbers into `edit_file`.

Read intended change locations before editing. Batch all known needed files and ranges. Omit `offset` and `limit` for the whole file, or `limit` for all lines from `offset`.

Combined results remain subject to the context budget.
