# `monitor_command`

Subscribe to a managed background command's literal output or completion, with results delivered directly in request prompts and initialization-turn references.

## Initialize

Start a command with `run_command` and `background=true`, then call `{"action":"start","job_id":"ID_FROM_RESULT","pattern":"ready","timeout":120}`. Omit `pattern` for job completion. Patterns are case-sensitive literal substrings of 1–256 characters, not regexes; `stream` is `stdout`, `stderr`, or `both` (default). Matches span output chunks but never streams. Initialization replays at most the last 65,536 retained bytes per stream and reports earlier omissions. Live output is observed without polling. A completed job can be monitored immediately from its retained tail and exit status.

`repeat=false` completes on a match. `repeat=true` keeps watching matches until job completion, cancellation or timeout. Notices are coalesced to one per five seconds per monitor, with suppressed counts; complete raw output remains in the normal quota-limited command archive. Suppression never terminates a command. `timeout` defaults to 30 seconds and is limited to 600. Watching never extends the underlying command's timeout.

## Results and Control

`start` returns a `monitor_id` and `initialized_turn`. Active state and ready results are injected into subsequent request prompts. Results include the reason, job state, exit code, match counts, latest excerpt, and the last 1,000 observed characters per stream with explicit omission/retention metadata. No extra result-fetch call is needed. `action="list"` returns all retained monitors; `action="wait"` with `monitor_id` waits up to `min(timeout,30)` seconds and returns current state or the ready result. Waiting cancellation never cancels the watch or command.

`action="cancel"` completes the watch with a cancellation result, leaving the job running. `action="acknowledge"` removes a ready result from prompt reminders, returning that result once more. Logs and result artifacts remain retrievable. Cancel before acknowledging an active monitor. At most eight monitors (active plus unacknowledged) are retained; capacity exhaustion returns an error instead of silently dropping results.

## Lifetime

Monitors survive compatible reloads and `/stop`; jobs continue independently. Completion produces a notice but never launches a model request. Reset, resume, fork, deletion, and shutdown stop jobs and clear live monitors; saved results use the existing artifact store, and prior processes' monitors never restart automatically. Job IDs, monitor IDs, and log IDs are distinct tool fields even when a job and its log share an identifier. Tool availability restrictions also apply to monitor instructions.
