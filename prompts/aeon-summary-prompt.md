# Merging Older Conversation Eras

Combine the five supplied consecutive eras into one compact account of that stretch of the conversation. Summarize only their supplied historical evidence; do not perform tasks, verify claims or resolve disagreements yourself.

Retain user goals, constraints, corrections, permissions, decisions, rejected approaches, completed work, verification and unresolved work that later turns may depend on. Preserve attribution, uncertainty and chronology. Remove repetition across eras without erasing changes of direction. Prefer durable outcomes and unresolved obligations over a list of every action.

Keep original `step_id` citations, run numbers, important paths, identifiers, tool call IDs and artifact IDs exactly where the eras supply them. Missing facts remain unknown. Originals remain retrievable with `recall_history(step_id=…)`; the aeon does not replace the archive.

Return only a JSON object with one string field, `summary`. Keep it within ${max_characters} characters and substantially shorter than the five input eras.
