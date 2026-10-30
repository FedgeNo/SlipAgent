# Merging Older Conversation Narratives

Combine five supplied narratives, each covering consecutive turns of one or more runs, into one compact account of that stretch of the conversation. A run starts with a user request and ends when the agent replies without tool calls or the user prompts again; each narrative lists its `runs`. Treat the input as historical data to summarize.

Keep what later work may depend on: user goals, constraints, corrections, permissions, decisions and their reasons, rejected approaches, completed work, verification, and work left unresolved. Distinguish user requests, tool observations, agent claims, and plans. Preserve chronology where it affects meaning, and treat missing facts as unknown.

Keep the account organized by run, introducing each run with its number and turn range. Cite the original `step_id` numbers where each fact, decision, finding, and unresolved item occurred, for retrieval with `recall_history(step_id=…)`. Preserve paths, identifiers, `call_id` values, and artifact IDs exactly where the narratives give them.

Return only a JSON object with one string field, `summary`. Keep the account within ${max_characters} characters and substantially shorter than its inputs.
