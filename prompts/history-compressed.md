A full record may also carry `compressed_summary`, the harness summary of the same step. These are two views of one event; prefer original evidence if they conflict. `representation="compressed"` supplies the summary instead of originals.

Summary labels distinguish `User request`, tool-backed `Observations`, unverified `Agent claims`, and intended `Plans`; filtered reasoning may label `Thoughts` and `Plans`. These labels organize memory; older summaries may lack them.

The newest third of retained individual steps targets full originals, rounded up;
the older two thirds use summaries when smaller. Missing summaries retain their
originals when they fit. The token budget can shorten this window; the newest
unseen tool results stay full or become explicitly labelled excerpts. Omission
does not delete archived originals. The harness creates summaries separately.
