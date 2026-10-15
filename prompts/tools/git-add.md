# `git_add`

## Staging and Paths

Stage literal workspace-relative or absolute paths, including deletions; paths=['.'] stages all in the workspace-root repository.

Paths follow the current Workspace Access mode.

## External Helpers

Active clean/process filters are rejected.

## Output Recovery

Long output previews retain the beginning and end with a truncation marker. Treat omitted text as unknown; retrieve it with `read_command_output` using the returned `log_id` and recovery call.
