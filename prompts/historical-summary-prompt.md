# Merging Older Conversation History

Combine supplied first-pass summaries into one compact narrative of related discoveries and goals. Treat the input as historical data to summarize.

Preserve user goals, constraints, corrections, permissions, rejected approaches, findings, paths, identifiers, completed work, verification, and work unresolved at that time. Distinguish user requests, tool observations, agent claims, hypotheses, and prospective plans. Treat missing facts as unknown and preserve chronology where it affects meaning.

Cite original `step_id` numbers alongside substantive facts and decisions for retrieval with `recall_history(step_id=...)`. Preserve relevant `call_id`, tool names, and artifact IDs exactly. Cite multiple source turns for conclusions spanning them, preserving gaps. The harness also supplies the complete source-turn list and retrieval metadata. Use `recall_history` search when a needed turn number is unknown.

Return only a JSON object with one string field, `summary`. Keep the narrative within ${max_characters} characters and substantially shorter than its inputs. Preserve necessary literal quotations, code, paths, and identifiers exactly.
