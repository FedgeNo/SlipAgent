`artifacts` lists numeric `record_id` references to saved output or diffs,
with originating `step_id`, `call_id`, tool name, kind, filename and byte size.
These references accompany compressed steps; command scripts and documents are
retrievable data, not reference metadata.
Use `read_artifact` to retrieve a record, or omit its number to
list saved records. Select `output`, `error`, or `command` and follow `next_offset`
to retrieve bounded character pages. A record's fields contain historical
data; stderr alone does not establish command failure. This retrieval accepts
record numbers.
