# `read_command_output`

## Retained Output

Read retained shell/Git stdout or stderr without rerunning the command.

## Paging and Streams

Use `log_id` from the result; `offset` counts UTF-8 bytes and `limit` counts characters.

Omit `limit` to read up to 32,000,000 characters, also the maximum permitted. Use `stream="stderr"` for errors or `tail=true` for the end. If a supplied page has a non-null `next_offset` and needed output is missing, request that offset. A null `next_offset` marks the end.

Batch independent output reads together. Polling retained logs is exempt from unchanged-batch detection.

## Finding Logs

Omit `log_id` to list logs, optionally filtered by `step_id` and `call_id`.

## Retention Limits

Quota/disk failures report `lost_bytes` and `retention_error`; missing bytes cannot be retrieved.

Logs are saved with persistent sessions.

With `--no-session`, `/reset` and exit delete them.
