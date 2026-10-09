# `git_log`

## Commit History

Show recent commits in a repository allowed by Workspace Access mode, newest first (default 10, maximum 100).

## Output Recovery

Output is a captured preview; long streams retain the beginning and end with an explicit truncation marker. Omitted output is not evidence of an empty result or a successful command. Use `read_command_output` with the returned `log_id` to inspect omitted text without rerunning the command; the observation includes a concrete recovery call.
