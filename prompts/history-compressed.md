`representation="compressed"` replaces the turn's prompt, reply, calls and results with `compressed_summary`; originals are not also supplied.

Labels distinguish `User request`, tool-backed `Observations`, `Agent claims` and intended `Plans`; filtered reasoning uses `Thoughts` and `Plans`. Older summaries may lack labels.

Rolling compression preserves recent history as space allows; it is not a context reset. Missing summaries keep originals. Unseen results stay full or explicitly excerpted. All originals remain retrievable.
