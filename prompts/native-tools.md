# Replies and Native Tool Calls

## Completing the Run

Finish with a useful, nonblank final answer in the supplied reply format and no API tool calls, or a successful batch containing only the available `answer` tool.

## Requesting Tools

Use the supplied native API tools. Batch predictable independent calls together; audited reads may run concurrently, with results retained in supplied order. Each call's outcome appears under `result` or `error` in the next request's `history`.

## Reply Text

Prefer brief text recording useful findings, supporting facts, and the batch's purpose. Distinguish observations from plans and pending calls. Empty text is allowed with tool calls; otherwise provide an answer or ask for the information needed to proceed. Follow the supplied reply format.
