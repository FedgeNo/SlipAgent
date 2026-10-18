`representation="historical_summary"` supplies one narrative combining a block
of 20 older turns. Its `step_id` identifies the last turn of the block.
`step_ids` lists every included original turn. `sources` preserves each source
`step_id`, tool identities (`call_id`, `tool_name`) and artifact references
unchanged. Use narrative citations and that metadata with
`recall_history(step_id=...)` or `recall_history(step_id=..., call_id=...)` to
retrieve originals. Search `recall_history` with `query` when the needed turn
number is unknown. Grouped narratives are historical evidence.
