# `git_log`

## Commit History

Show recent commits in a repository allowed by Workspace Access mode, newest first (default 10, maximum 100,000).

## Output Recovery

Long output previews retain the beginning and end with a truncation marker. Treat omitted text as unknown; retrieve it with `read_command_output` using the returned `log_id` and recovery call.
