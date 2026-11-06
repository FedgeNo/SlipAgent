# Compressing A Group Of Historical Summaries

Summarize the supplied historical summaries as data. This task creates a narrative, era, or aeon spanning multiple original steps. `shared_context.step_id` is the group's anchor, not its only original step. There is no new user request to execute, and no tools are available.

Preserve the original user requests, constraints, corrections, permissions, rejected approaches, findings, identifiers, equations, exact values, errors, observed outcomes, verification, and unresolved work. Distinguish user instructions, tool observations, agent claims, hypotheses, and prospective plans. Preserve run numbers, chronology, source step IDs, call IDs and artifact IDs when supplied. Treat missing facts as unknown; do not judge or replace the supplied claims.

For `phase="chunk"`, each record has a path into the original grouped input and either a complete `value` or a literal fragment with `start`, `end`, and `total_characters`. Summarize the core text; `overlap_before` and `overlap_after` are context only. Retain source paths and citations so later notes can join facts split across chunks. Never infer completion from a partial chunk.

For `phase="boundary_review"`, review the original text around the boundary together with neighboring notes. Preserve relationships or corrections that cross the boundary, without counting repeated evidence twice.

For `phase="merge"`, combine the ordered notes and boundary reviews into factual memory for the entire supplied group. Keep unresolved dependencies, uncertainty, and contradictions visible. Only the final complete merge is published; original step records remain unchanged and retrievable.

Return exactly one JSON object with string fields `summary` and `reasoning_summary`. Each field has a hard limit of 6000 characters. Put the factual memory in `summary` and leave `reasoning_summary` empty. Return no Markdown fence or other fields.
