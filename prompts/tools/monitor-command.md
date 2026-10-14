# `monitor_command`

Watch a managed background command's literal output or completion. Request prompts include active state, ready results, and the initialization turn.

## Initialize

Start a command with `run_command` and `background=true`, then use its returned `job_id`: `{"action":"start","job_id":"ID_FROM_RESULT","pattern":"ready","timeout":120}`. Omit `pattern` for completion. Patterns are case-sensitive literal substrings of 1–256 characters. `stream` selects `stdout`, `stderr`, or `both` (default). Matches span chunks within each stream.

Initialization replays up to 65,536 retained bytes per stream, reports earlier omissions, then observes live output. Completed jobs can be watched from retained tails and exit status. `repeat=false` completes on the first match; `repeat=true` continues until job completion, cancellation, or timeout. Watch lifetime defaults to 30 seconds, capped at 600, independently of the command timeout.

Notices coalesce to one per five seconds per monitor with suppressed counts. Raw output remains in the quota-limited command archive; suppression leaves the command running.

## Results and Control

`start` returns `monitor_id` and `initialized_turn`. Ready results supplied directly in prompts include reason, job state, exit code, match counts, latest excerpt, and up to the last 1,000 observed characters per stream with omission/retention metadata.

- `list`: return retained watches.
- `wait` with `monitor_id`: wait up to `min(timeout,30)` seconds and return state or result. Cancelling the wait leaves the watch and command running.
- `cancel`: complete the watch with a cancellation result, leaving the job running.
- `acknowledge`: return a ready result and remove its prompt reminder; logs and result artifacts remain. Cancel active watches first.

At most eight active or unacknowledged watches are retained. Capacity exhaustion returns an error.

## Lifetime

Watches survive compatible reloads and `/stop`. Completion emits a notice; model requests occur through the normal agent flow. Reset, resume, fork, deletion, and shutdown stop jobs and clear watches. Saved result artifacts remain available. Use `job_id`, `monitor_id`, and `log_id` for their respective operations.
