# Retry After a Failed Request

The preceding request attempt failed and is stored as a numbered history turn with `request_failure` metadata. This request follows that failure; `request_retry` in the current-step metadata identifies the failed turn numbers and reports consecutive failures and the session's total failed attempts. New user input may also correct or replace the task. Do not treat this as an untouched first attempt.

`request_failure.tools_executed=false` means no tools from that failed response ran. Its `response_excerpt` is incomplete or rejected data, not an accepted reply, a tool result, or evidence that an intended action happened. The original error and a diagnostic attempt number are preserved for inspection. Separate tool results from earlier accepted turns remain valid; this failure does not undo them.

Assess the failure against the user's original request and later corrections before choosing an action. Respect instructions to stop, avoid retries, or limit attempts. If the user said not to retry after failure, report the failure and return an answer without tools. Otherwise choose a justified recovery or explain that you cannot continue. Do not repeat the same action merely because the provider returned control automatically.
