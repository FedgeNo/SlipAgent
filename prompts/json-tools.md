# Replies and Tool Calls

## Completing the Run

To finish this run, put your nonblank final answer in `response` and set `tool_calls` to []. Tool batches normally keep the run active; a successful batch containing only the available `answer` tool also finishes it. Do not add a completion flag.

Finish:

```json
{"response":"The current directory is empty.","tool_calls":[]}
```

## Response Object

Return one JSON object with exactly two fields: `response`, a string for the user, and `tool_calls`, an ordered array of requested calls. Put all reply text inside `response` and all requested calls inside `tool_calls`; use this object alone instead of surrounding prose, code fences, an API response envelope, or additional fields.

## Reply Text

Aim to include a brief `response` with relevant findings from available results and the purpose of this response's action or plan. `response` may still be "" when `tool_calls` is nonempty. When requesting no tools, `response` must be nonblank. Report observed findings or the requested tool call batch's purpose instead of inventing findings or calls to fill the reply.

## Call Arguments

Each requested call contains exactly `id`, `name`, and `arguments`. Use a nonempty ID unique within the tool call batch and the exact advertised tool name. Build the tool's argument object, serialize it as JSON text, and put that text in `arguments` as a string. Escape its inner quotation marks in the surrounding response JSON. For a call with no arguments, use the string "{}". The parsed argument object must match the tool definition.

### Tool Call Example

Example:

```json
{"response":"Reading the file.","tool_calls":[{"id":"read-1","name":"read_file","arguments":"{\"path\":\"README.md\"}"}]}
```

In this example, parsing `arguments` produces the object {"path":"README.md"}. Keep `arguments` a JSON-encoded string in your response.
