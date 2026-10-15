# `run_command`

Run a shell command in the workspace root.

Use this for tests, linters, type checks, builds, and operations outside dedicated tools' capabilities. **Use dedicated tools when they suffice** for files, search, navigation, and Git. Verify actual effects from results. Identify intended file-modification targets in your reply.

## Environment and Access

Commands use a non-login shell with inherited `PATH` and the harness process's permissions, including access outside the workspace. Use explicit project interpreter paths. Commands are killed at their timeout.

## Output

Each stream returns up to 32,000,000 characters, retaining beginning and end for larger output. Combined results remain subject to the context budget. Treat omitted output as unknown.

Session logs retain decoded output within the disk quota. Use `read_command_output` with the returned log ID and recovery call to page or tail it. Lost bytes are reported.

## Polling and Background Jobs

Set `poll=true` only for intentional external-state polling.

Set `background=true` for a managed job; the result supplies `job_id` and `log_id`. Use `command_jobs` to check/wait/stop known jobs and `read_command_output` for live output. Up to four jobs run concurrently. Execution timeout defaults to 120 seconds, capped at 600. Establish success from the outcome.
