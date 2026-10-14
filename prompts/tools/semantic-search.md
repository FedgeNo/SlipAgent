# `semantic_search`

Find code by behavior or purpose when the repository map leaves its location unclear.

Use five or six meaningful query words describing the behavior, concept, or identifier. Returns complete source excerpts with workspace-relative paths and line numbers, up to `limit` (default five, maximum ten). Ranking combines local embeddings and exact keyword evidence, including filenames. Scores are relative to the query.

First use loads a small CPU model and builds a persistent workspace index. Cached model files are reused; missing files download. Source and queries stay local. Filesystem events update changed files, periodic metadata scans reconcile missed events, and queued edits are indexed before lookup. Dependencies, hidden files/directories, symlinks, and Git-ignored files are excluded.

Assess ranked snapshots for relevance and completeness; read current source before editing. Use `grep` for known identifiers and `grep`, `glob`, or `read_file` when results are weak.
