# `run_command`

## Purpose

Run a shell command in the workspace root and return its output.

Use it to run tests, linters, type checkers, and builds, and to inspect files that dedicated tools cannot read.

## Environment and Access

The command runs under a non-login shell and inherits `PATH`; it does not activate a project environment. Prefer explicit paths and use the selected project interpreter.

Commands have the harness process's permissions; their access is not confined to the workspace.

## Timeouts

Long-running commands are killed at the timeout.

## Output and Recovery

The observation includes up to 32,000,000 characters per stream. Larger output retains the beginning and end. Omitted output is not evidence of an empty result or a successful command. The model's context budget still applies to the combined results.

Session command logs retain full decoded output within the configured disk quota; `read_command_output` retrieves pages or tails using the returned log ID without rerunning the command. The observation includes a concrete recovery call.

Lost bytes are reported explicitly.

## Polling

For intentional polling of external state, set `poll=true`. Do not mark ordinary failed retries as polling.

## Background Jobs

Set `background=true` to request a managed job. Its result supplies a `job_id` and `log_id`. For a job already identified in supplied results, use `command_jobs` to check/wait/stop it and `read_command_output` for live output.

At most four jobs run concurrently. The same execution timeout still applies (default 120 seconds, maximum 600). Starting a job is not evidence that it succeeded.
