# `semantic_search`

Find code by behavior or purpose when the repository map does not show where to look.

Use a short query, preferably five or six meaningful words describing the behavior, concept, or identifier you need. Returns up to `limit` complete source excerpts (default five, maximum ten) with workspace-relative paths and line numbers. Ranking combines local semantic embeddings with exact keyword evidence, including filenames. Scores compare results for this query; they are not probabilities.

The first call loads a small CPU model and builds a persistent workspace index, which can take time on a large repository. Existing locally cached model files are reused; missing files download on first use. Source and query text stay local. After first use, filesystem events update changed files and a periodic metadata scan reconciles missed events. Queued edits are indexed before each lookup. Dependencies, hidden files/directories, symlinks, and Git-ignored files are excluded.

These are ranked indexed snapshots, not proof that the first result is correct or that all relevant code was found. Read current source before editing. If results are weak, use `grep`, `glob`, or `read_file`; exact known identifiers usually suit `grep`. This tool supplements the repository map and ordinary file tools.
