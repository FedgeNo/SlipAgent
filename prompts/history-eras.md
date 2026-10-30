`representation="era_summary"` supplies one account of an older stretch of the
conversation, merged from five consecutive narratives of up to 100 turns. Its `step_id`
identifies the last turn of that stretch, `step_ids` lists every included
original turn, and `runs` lists the runs they belong to. Use its citations with `recall_history(step_id=…)` to retrieve
originals, or search `recall_history` with `query` when the needed turn number
is unknown. Eras are historical evidence.
