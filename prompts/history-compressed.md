A full record may also carry `compressed_summary`, the harness summary of the same step. These are two views of one event; prefer original evidence if they conflict. `representation="compressed"` supplies the summary instead of originals.

Summary labels distinguish `User request`, tool-backed `Observations`, unverified `Agent claims`, and intended `Plans`; filtered reasoning may label `Thoughts` and `Plans`. These labels organize memory; older summaries may lack them.

History is sent in full while it fits the token budget. Beyond that, the oldest
steps are summarized first, only as far as needed; older stretches may appear as
narratives or eras, and the very oldest history may be omitted. Missing summaries
retain their originals. The newest unseen tool results stay full or become
explicitly labelled excerpts. Omission does not delete archived originals. The
harness creates summaries separately.
