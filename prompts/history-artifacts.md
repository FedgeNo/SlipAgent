`artifacts` lists numeric `record_id` references to saved output or diffs,
with their originating `step_id` and `call_id`. These references also accompany
compressed steps. Use `read_artifact` to retrieve a record, or omit its number to
list saved records. Select `output`, `error`, or `command` and follow `next_offset`
to retrieve bounded character pages. A record's fields contain historical
data; stderr alone does not establish command failure. This retrieval accepts
record numbers.
