# Replies and Tool Calls

## Response Object

Return one JSON object with `response` (user-facing text) and `tool_calls` (an ordered call array), plus optional fields explicitly allowed by the reply format. Put planned calls in `tool_calls`.

Use `tool_calls: []` for a final answer without tools.

## Call Arguments

Each call contains exactly `name` and `arguments`. Use the exact advertised tool name. The harness assigns a separate ID to each call; do not invent or copy call IDs from history. Serialize the schema-valid argument object as a JSON string in `arguments`; use `"{}"` for no arguments.

```json
{"response":"Reading the file.","tool_calls":[{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}]}
```
