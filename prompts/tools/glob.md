# `glob`

## Patterns

Find files and directories by name pattern.

Patterns are matched against the workspace-relative path (or relative to the searched directory for permitted external paths), so '*.py', 'src/**/*.ts', and '**/test_*.py' all work.

## Directory Results

Directories are shown with a trailing '/'. Use this with `list_dir` to understand the layout before reading code with `read_file`.
