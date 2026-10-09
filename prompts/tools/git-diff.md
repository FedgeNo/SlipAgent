# `git_diff`

## Changes and Paths

Show unstaged changes, or staged changes with staged=true.

Paths are literal and workspace-relative or absolute.

## External Helpers

External diff/text conversion helpers are disabled.

## Output Recovery

Output is a captured preview; long streams retain the beginning and end with an explicit truncation marker. Omitted output is not evidence of an empty result or a successful command. Use `read_command_output` with the returned `log_id` to inspect omitted text without rerunning the command; the observation includes a concrete recovery call.
