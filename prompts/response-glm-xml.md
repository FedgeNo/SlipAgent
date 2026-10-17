# Replies and Tool Calls

Write useful Markdown reply text. To request tools, follow the text with one `<tool_call>` element per call in the format under Tools; several calls run as one batch in order. Write string argument values as plain text and other values as JSON.

Finish with a useful, nonblank final answer and no tool calls, or a successful batch containing only the available `answer` tool. Other tool batches normally continue the run. Each call's outcome appears under `result` or `error` in the next request's `history`.
