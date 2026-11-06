`representation="compressed"` replaces the turn's prompt, reply, calls and results with `compressed_summary`; originals are not also supplied.

Labels distinguish `User request`, tool-backed `Observations`, `Agent claims` and intended `Plans`; filtered reasoning uses `Thoughts` and `Plans`. Older summaries may lack labels.

Rolling compression is not a context reset. Missing aggregates use lower-level summaries, then excerpts if needed. Unseen results stay full or excerpted. Originals remain retrievable.
