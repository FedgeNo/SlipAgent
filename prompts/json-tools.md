# Replies and Tool Calls

## Response Object

Return one JSON object with `response` (user-facing text) and `tool_calls` (an ordered call array), plus optional fields explicitly allowed by the reply format.

Prefer brief reply text recording findings from supplied results and the purpose of pending calls. Empty text is allowed with calls. Finish with a nonblank answer and `tool_calls: []`, or a successful batch containing only the available `answer` tool. Other tool batches normally continue the run.

## Call Arguments

Each call contains exactly `id`, `name`, and `arguments`. Use a nonempty ID unique within the batch and the exact advertised tool name. Serialize the schema-valid argument object as a JSON string in `arguments`; use `"{}"` for no arguments.

```json
{"response":"Reading the file.","tool_calls":[{"id":"read-1","name":"read_file","arguments":"{\"path\":\"README.md\"}"}]}
```
