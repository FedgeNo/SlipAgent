# `git_status`

## Repository Status

Show branch and short Git status for a repository allowed by the current Workspace Access mode.

## Output Recovery

Long output previews retain the beginning and end with a truncation marker. Treat omitted text as unknown; retrieve it with `read_command_output` using the returned `log_id` and recovery call.
