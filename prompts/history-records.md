# Reading the JSON Input

The user message is one JSON object. Keys identify each part; values contain the data itself. Interpret JSON escapes once to recover original text. Do not parse JSON-looking strings again or echo the input envelope as your reply. Follow the separate response contract.

## Input Fields

| Key | Meaning |
| --- | --- |
| `user_message` | First new user message, or `""` when none was supplied |
| `additional_user_messages` | Additional queued user messages, in order |
| `retained_user_request` | Original messages for the active request, not newly issued messages |
| `history` | Selected completed steps, oldest to newest |
| `errors` | Harness errors or response corrections not attached to a tool call |

## Calls and Outcomes

Each full history step has `step_id`, `user_prompt`, `agent_response`, and `tool_calls`. Each call contains `call_id`, `tool_name`, and `arguments`, with its observation attached by key:

| Key on the call | Meaning |
| --- | --- |
| `result` | Successful tool result; the value is the returned data |
| `error` | Failed tool result; the value is the returned error data, which may describe partial effects |
| `unclassified_result` | Legacy observation whose success or failure was not recorded |

Do not infer success from the absence of `error`; an absent result is unknown. A failed modifying call may have partial effects; assess the recorded state before retrying. Arguments and structured outputs remain objects. Text outputs remain strings, even when they resemble JSON. Labels belong to keys, not prefixes inserted into values.

## History Representations

`representation="full"` supplies original parts and any available `compressed_summary` of the same step. These are two views of one event; prefer original evidence if they conflict. `representation="compressed"` supplies the summary instead of originals. `representation="excerpt"` marks bounded text and omissions. Missing text is unknown, not empty or successful. Use the exact `step_id` with `recall_history` to retrieve the complete original object.

Excerpted calls use `result_excerpt`, `error_excerpt`, or `unclassified_result_excerpt` for bounded observations. `response_excerpts` contains bounded message text and call descriptions. Call arguments may be omitted; retrieve originals when exact arguments matter.

Optional `reasoning` records earlier thoughts, not actions. `reasoning_representation` distinguishes `original` from `filtered` thoughts. Earlier phrases such as "now" or "I will" refer to their recorded turn. A completed step means its response and results were recorded, not that the user's task was completed. Attribute goals to actual user messages and conclusions to supporting evidence.

Summary labels distinguish `User request`, tool-backed `Observations`, unverified `Agent claims`, and intended `Plans`; filtered reasoning may label `Thoughts` and `Plans`. Plans are intentions, not evidence of execution. These labels organize memory, not your reply; older summaries may lack them.

The default full-history window is 50 steps, configurable with a target minimum of five; up to 100 older records may accompany it. The harness chooses economical representations and removes older records first. Context limits can reduce even the latest five steps. Omission does not delete archived originals. The harness creates summaries separately; return your normal response rather than history bookkeeping.
