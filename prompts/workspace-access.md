# Workspace Access

${access}

## Path Anchor

Relative paths and shell cwd remain anchored to `${workspace}`.

## Temporary Files

Use absolute paths under `${temporary_directory}` for scratch files with built-in filesystem tools, including when Danger Mode is OFF.

## Process Permissions

OS permissions still apply. Shell and MCP processes use the harness process's permissions.
