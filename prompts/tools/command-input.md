# `command_input`

Send UTF-8 text to a running command started with `background=true` and `interactive=true`, or close its stdin with `eof=true`.

Use its returned `job_id`. Text is literal: include a newline character when a program expects a submitted line; no newline is added automatically. Each call accepts at most 65,536 UTF-8 bytes. With both text and EOF, text is submitted before closing stdin. EOF cannot be undone. Pipe input does not emulate a terminal.

Input readiness and pipe draining each wait at most five seconds. A successful result confirms submission to the pipe, not consumption or command success. Inspect `read_command_output` and `command_jobs` for the response and outcome. A failed or cancelled write may have partially delivered input and closes stdin; inspect effects before retrying. Sending input does not reset the command's execution timeout.

Input calls and their observations are recorded in ordinary tool history. Jobs belong to the current process and cannot receive input after session replacement or restart. Load this tool's definition before calling it.
