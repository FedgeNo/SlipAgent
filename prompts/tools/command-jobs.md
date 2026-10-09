# `command_jobs`

## Managed Commands

Manage commands started with `run_command` and `background=true`.

## Inspecting Jobs

Use `action="list"` to list jobs, or `action="status"` with `job_id` for one job. List/status returns `job_id`, state, exit status and `log_id`; `read_command_output` retrieves live pages or tails of either stream.

## Waiting

Use `action="wait"` with `job_id` to wait at most 30 seconds without cancelling the job.

## Stopping

Use `action="stop"` with `job_id` to kill its process group and drain cleanup.

## Lifecycle and Evidence

Starting a job is not evidence that it succeeded. Completion does not wake the model; request status/wait when you need the outcome. `/stop` leaves jobs running; reset, resume, fork, deletion, and exit stop them.
