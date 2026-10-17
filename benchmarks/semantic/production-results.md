# Production Retrieval Verification

The backend is integrated in `slipagent.semantic`; reproduction commands below
run from the SlipAgent repository root. Earlier comparison reports retain the
original Dots source fingerprints and frozen corpus labels.

The production backend uses quantized BGE with mean pooling and plain queries,
512-token line chunks, relative filename context, and a normalized 25% BM25 /
75% cosine blend. Source excerpts remain complete and unchanged. The default
encoder now runs one source chunk per forward pass, making vectors independent
of batch neighbors and keeping inference memory lower.

## Quality

All runs use the same frozen 19 source files, 294 queries, and file-level split.
There are 216 development queries and 78 internal validation queries. The source
corpus produces 95 chunks with filename context. The production check runs the
actual indexer, ONNX encoder, SQLite keyword postings, and HNSW search path.

| Run | Development MRR@10 | Validation Answer First | Validation Within Five |
| --- | ---: | ---: | ---: |
| Original semantic baseline | 0.439 | 19.2% (15/78) | 59.0% (46/78) |
| Matrix-selected hybrid, shared sorted batches | 0.601 | 35.9% (28/78) | 78.2% (61/78) |
| Production hybrid, document batch size 8 | 0.585 | 30.8% (24/78) | 78.2% (61/78) |
| Production default, document batch size 1 | 0.583 | 33.3% (26/78) | 80.8% (63/78) |

Quantized inference depends on batch composition, so the original matrix's shared
length-sorted batches do not reproduce production's per-file batches exactly.
Single-document inference is chosen for low memory and stable indexing, with
near-equal development quality to per-file batches of eight. The 25/75 ranking
recipe remains the development-selected choice; validation outcomes do not
retroactively select another keyword weight. Production finds 11 more validation
answers first and 17 more within five than the original semantic baseline.

The [final machine-readable report](production-results.json) records every query's
top ten, full split/style metrics, dependency versions, and the exact source and
model fingerprints. [Batch-one evaluation](production-batch-1-results.json) and
[batch-eight evaluation](production-batch-8-results.json) retain the comparison
runs. The original [1,152-condition results](retrieval-results.md) remain separate.

The integrated SlipAgent backend reproduces the batch-one split metrics on all
294 queries. Its final report records the current `src/slipagent/semantic/`
fingerprints rather than the former Dots source paths.

This is an internal Python pilot, not an external benchmark. The corpus fits
inside the 256-chunk semantic shortlist. Production uses a bounded union of ANN,
keyword, and calibration candidates for larger indexes; these measurements do
not establish large-corpus candidate recall. Background CPU load varied during
production checks, so their elapsed indexing times are not speed comparisons.

## Cold Thread Measurements

The development computer is an Intel Core i3-5010U with two physical cores,
four logical CPUs, and about 7.7 GiB RAM. Each measurement starts a fresh process
and ONNX session without inference warmup; operating-system file caches remain.
Medians of three runs per workload/thread count:

| Threads | First Query | Eight Long Chunks | Model Load Before Query |
| --- | ---: | ---: | ---: |
| 1 | 14.4 ms | 5.77 s | 0.605 s |
| 2 | 9.6 ms | 3.58 s | 0.634 s |
| 4 | 17.7 ms | 4.62 s | 0.633 s |

These thread comparisons use document batches of eight. Two threads win here;
defaults use physical-core count, with a two-thread fallback and an explicit
override. [Raw measurements](thread-results.json) include individual processes,
input token counts, memory observations, and the model hash. A later batch-size
speed test was interrupted when unrelated CPU-heavy work made timing unreliable;
no speed claim is derived from that incomplete run.

## Reproduce

```bash
.venv/bin/python -B benchmarks/semantic/production-retrieval.py \
    --output /tmp/production-results.json

.venv/bin/python -B benchmarks/semantic/threads.py \
    --output /tmp/thread-results.json
```

Real inference should run in a memory-limited systemd scope on constrained
machines. Offline regressions separately exercise native observation, coalesced
saves, renames/deletions, startup and in-flight edit races, periodic reconciliation,
retries, atomic publication, code-word tokenization, hybrid ranking, source
coverage, model identity, and CPU selection. Earlier Dots verification checked
Hugging Face client 2.2.0 with offline fixtures and a fresh anonymous config
download followed by an offline cache hit. The cached weights matched the
upstream artifact at that check.
