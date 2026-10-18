# Merging Older Conversation Narratives

Combine five supplied narratives, each covering 20 consecutive turns, into one compact account of that stretch of the conversation. Treat the input as historical data to summarize.

Keep what later work may depend on: user goals, constraints, corrections, permissions, decisions and their reasons, rejected approaches, completed work, verification, and work left unresolved. Distinguish user requests, tool observations, agent claims, and plans. Preserve chronology where it affects meaning, and treat missing facts as unknown.

Cite original `step_id` numbers alongside substantive facts and decisions for retrieval with `recall_history(step_id=...)`. Preserve paths, identifiers, `call_id` values, and artifact IDs exactly where the narratives give them.

Return only a JSON object with one string field, `summary`. Keep the account within ${max_characters} characters and substantially shorter than its inputs.
