# Retry After a Failed Request

The preceding failed request is a numbered history turn with `request_failure` metadata. Current-step `request_retry` identifies failed turn numbers, consecutive failures, and the session's total failed attempts.

`request_failure.tools_executed=false` means no tools from that response ran. Treat its incomplete or rejected `response_excerpt` as diagnostic data. The original error and diagnostic attempt number remain available; earlier accepted tool results remain valid.

Assess the failure against the original request and later user corrections, including retry and attempt limits. When instructed to stop after failure, report it with no tools. Otherwise choose an evidence-based recovery or explain the blocker.
