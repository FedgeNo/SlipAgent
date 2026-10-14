# `git_diff`

## Changes and Paths

Show unstaged changes, or staged changes with staged=true.

Paths are literal and workspace-relative or absolute.

## External Helpers

External diff/text conversion helpers are disabled.

## Output Recovery

Long output previews retain the beginning and end with a truncation marker. Treat omitted text as unknown; retrieve it with `read_command_output` using the returned `log_id` and recovery call.
