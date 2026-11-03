`representation="compressed"` supplies `compressed_summary`, the harness summary replacing that step's original prompt, reply, calls and results. Full records supply those originals without appending the step summary. Each selected representation replaces the other; they are not duplicate events.

Summary labels distinguish `User request`, tool-backed `Observations`, unverified `Agent claims`, and intended `Plans`; filtered reasoning may label `Thoughts` and `Plans`. These labels organize memory; older summaries may lack them.

History is sent in full while it fits the token budget. Beyond that, the oldest
steps are summarized first, only as far as needed; older stretches may appear as
narratives or eras, and the very oldest history may be omitted. Missing summaries
retain their originals. The newest unseen tool results stay full or become
explicitly labelled excerpts. Omission does not delete archived originals. The
harness creates summaries separately.
