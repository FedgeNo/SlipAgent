# Replies and Tool Calls

Write useful Markdown reply text. To request tools, follow the text with one `<tool_call>` element per call in the format under Tools; several calls run as one batch in order. Write string argument values as plain text and other values as JSON.

Each call's outcome appears under `result` or `error` in the next request's `history`.
