# `git_add`

## Staging and Paths

Stage literal workspace-relative or absolute paths, including deletions; paths=['.'] stages all in the workspace-root repository.

Paths follow the current Workspace Access mode.

## External Helpers

Active clean/process filters are rejected instead of running external helpers.

## Output Recovery

Output is a captured preview; long streams retain the beginning and end with an explicit truncation marker. Omitted output is not evidence of an empty result or a successful command. Use `read_command_output` with the returned `log_id` to inspect omitted text without rerunning the command; the observation includes a concrete recovery call.
