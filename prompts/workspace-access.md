# Workspace Access

${access}

## Path Anchor

Relative paths and shell cwd remain anchored to `${workspace}`.

## Temporary Files

The configured system temporary directory is `${temporary_directory}`. You can use it
for scratch files with the built-in filesystem tools, including when Danger Mode is OFF.
Use absolute paths to access it. OS permissions still apply.

## Process Permissions

OS permissions still apply. Shell and MCP processes use the harness process's permissions.
