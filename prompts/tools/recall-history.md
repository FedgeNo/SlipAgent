# `recall_history`

## Search and Selection

Search full session history or retrieve an original step with both user input and assistant/tool messages.

Omit step_id to search/list steps; total_matches counts all matching steps.

## Complete Objects

Returns the complete selected object or array in content. Plain-text selections remain strings. There is no character limit or paging.

## Selecting Results and Fields

Add call_id to retrieve one tool observation from that step, including status and content.

Use section to select prompt, response, reasoning, tool_calls, tool_results, user_edits, or background_commands; sections selects several. reasoning returns the thoughts generated in that step, when the model produced any. user_edits returns the saved-edit notifications originally delivered with that turn, including artifact references for full diffs. background_commands returns background command completions delivered with that turn. section='user' retrieves only the original new user messages in that step.
