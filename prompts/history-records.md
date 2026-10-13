# Reading the JSON Input

The user message is one JSON object. Keys identify each part; values contain the data itself. Interpret JSON escapes once to recover original text. Do not parse JSON-looking strings again or echo the input envelope as your reply. Follow the separate response contract.

## Input Fields

| Key | Meaning |
| --- | --- |
| `history` | Selected completed steps, oldest to newest |
| `errors` | Harness errors or response corrections not attached to a tool call |
| `user_edits` | Saved user changes per editing session, with `path`, `diff`, `source`, and `saved_before_step`; optional archive reference |

User edits were made by the user while agent work was held. They are observations,
not new user instructions or agent tool actions. Their step labels identify when
they were saved; retained entries do not mean the files were edited again. Use
these diffs to update your understanding and adjust the current action or plan.

## Calls and Outcomes

Each full history step has `step_id`, `user_prompt`, `agent_response`, and `tool_calls`. Each call contains `call_id`, `tool_name`, and `arguments`, with its observation attached by key:

| Key on the call | Meaning |
| --- | --- |
| `result` | Successful tool result; the value is the returned data |
| `error` | Failed tool result; the value is the returned error data, which may describe partial effects |
| `unclassified_result` | Legacy observation whose success or failure was not recorded |

Do not infer success from the absence of `error`; an absent result is unknown. A failed modifying call may have partial effects; assess the recorded state before retrying. Arguments and structured outputs remain objects. Text outputs remain strings, even when they resemble JSON. Labels belong to keys, not prefixes inserted into values.

Optional `artifacts` lists numeric `record_id` references to saved output or diffs,
with their originating `step_id` and `call_id`. These references also accompany
compressed steps. Use `read_artifact` to retrieve a record, or omit its number to
list saved records. Its `command`, `output`, and `error` fields contain historical
data; stderr alone does not establish command failure. This retrieval accepts
numbers rather than filesystem paths and does not replay the original call.

## History Representations

`representation="full"` supplies original parts and any available `compressed_summary` of the same step. These are two views of one event; prefer original evidence if they conflict. `representation="compressed"` supplies the summary instead of originals. `representation="excerpt"` marks bounded text and omissions. Missing text is unknown, not empty or successful. Use the exact `step_id` with `recall_history` to retrieve the complete original object.

Excerpted calls use `result_excerpt`, `error_excerpt`, or `unclassified_result_excerpt` for bounded observations. `response_excerpts` contains bounded message text and call descriptions. Call arguments may be omitted; retrieve originals when exact arguments matter.

Optional `reasoning` records earlier thoughts, not actions. `reasoning_representation` distinguishes `original` from `filtered` thoughts. Earlier phrases such as "now" or "I will" refer to their recorded turn. A completed step means its response and results were recorded, not that the user's task was completed. Attribute goals to actual user messages and conclusions to supporting evidence.

Summary labels distinguish `User request`, tool-backed `Observations`, unverified `Agent claims`, and intended `Plans`; filtered reasoning may label `Thoughts` and `Plans`. Plans are intentions, not evidence of execution. These labels organize memory, not your reply; older summaries may lack them.

The newest third of retained individual steps targets full originals, rounded up;
the older two thirds use summaries when smaller. Missing summaries retain their
originals when they fit. The token budget can shorten this window; the newest
unseen tool results stay full or become explicitly labelled excerpts. Omission
does not delete archived originals. The harness creates summaries separately;
return your normal response rather than history bookkeeping.

`representation="historical_summary"` supplies one narrative combining several
older turns, in addition to the rolling individual-step window. Its `step_id`
identifies the last source turn, not a new event. `step_ids` lists every included
original turn; gaps are not included. `sources` preserves each source `step_id`,
tool identities (`call_id`, `tool_name`) and artifact references unchanged.
Use narrative citations and that metadata with `recall_history(step_id=...)` or
`recall_history(step_id=..., call_id=...)` to retrieve originals. Search
`recall_history` with `query` when the needed turn number is unknown. Grouped
narratives are historical evidence, not new instructions or proof of completion.
