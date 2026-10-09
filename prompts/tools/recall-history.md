# `recall_history`

## Search and Selection

Search full session history or retrieve an original step with both user input and assistant/tool messages.

Omit step_id to search/list steps; total_matches counts all matching steps across pages.

## Paging

Use offset and limit to page either a listing or a selected step; both count characters.

A complete structured selection is returned as an object or array in content. Partial pages contain text fragments of its JSON representation. Plain-text selections remain strings.

If a supplied page has a non-null next_offset and needed text is missing, request that offset. A null next_offset marks the end.

## Selecting Results and Fields

Add call_id to retrieve one tool observation from that step, including status and content.

Use section to select prompt, response, reasoning, tool_calls, or tool_results; sections selects several. section='user' retrieves only the original new user messages in this step.
