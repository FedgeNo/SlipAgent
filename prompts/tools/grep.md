# `grep`

## Search Results

Search file contents with a regular expression.

Returns matching lines as `path:line: text`.

Searches text files up to 32 MB and returns complete matching lines, up to 100,000 matches. Use `max_results` for a smaller selection.

Searches have a 120-second timeout. Timeout or cancellation stops the scan and waits for any filesystem operation already in progress to return.

## Choosing a Search

Use `include` as a glob to narrow the file set (for example '*.py' or 'src/**/*.ts').

Use `glob` for filenames and `grep` for definitions or uses.
