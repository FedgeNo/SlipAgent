# `git_commit`

## Committing

Commit already staged changes with a message.

Does not stage files, amend, or push.

## External Helpers

Hooks and signing are disabled to keep external helpers from bypassing workspace checks.

## Output Recovery

Long output previews retain the beginning and end with a truncation marker. Treat omitted text as unknown; retrieve it with `read_command_output` using the returned `log_id` and recovery call.
