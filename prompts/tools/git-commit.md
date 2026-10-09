# `git_commit`

## Committing

Commit already staged changes with a message.

Does not stage files, amend, or push.

## External Helpers

Hooks and signing are disabled to keep external helpers from bypassing workspace checks.

## Output Recovery

Output is a captured preview; long streams retain the beginning and end with an explicit truncation marker. Omitted output is not evidence of an empty result or a successful command. Use `read_command_output` with the returned `log_id` to inspect omitted text without rerunning the command; the observation includes a concrete recovery call.
