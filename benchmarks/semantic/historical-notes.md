# Historical Dots Embedding Experiments

These notes describe the standalone Dots experiments before integration into
SlipAgent. Paths and commands below refer to that historical layout. Current
reproduction commands are in [README.md](README.md).

The [production verification](production-results.md) covers the applied hybrid
defaults, native watcher, physical-core settings, current Hugging Face client,
and differences between the original matrix's batching and production inference.

For the larger code-search experiment, see [the retrieval results](retrieval-results.md).
It evaluates 294 realistic queries and combinations of source chunking, context,
pooling, query instructions, precision, and keyword/semantic ranking. The original
39-query pilot below remains available for historical comparisons.

## Retrieval Experiment Matrix

[realistic-cases.json](realistic-cases.json) contains 98 reviewed intents with
three hand-written queries each: a task, an ordinary question, and a symbol,
configuration, or error-message lookup. Queries include short requests and longer
bug symptoms. They cover Dots indexing, embedding, storage, CLI, GUI, and tests,
plus the SlipAgent confinement, instruction, editing, cleanup, progress, clipboard,
and output code used in the original pilot. They are plausible examples written
for this benchmark, not a collection of actual user search logs.

[realistic-search.json](realistic-search.json) freezes 19 complete Python files,
including test files and unqueried code as distractors. It stores exact source
text, SHA-256 hashes, and character/line anchors. All three variants of an intent
stay together. Entire target files are reserved for internal validation: 216
development queries and 78 validation queries. All source files remain in the
searchable corpus; the holdout applies to labeled queries, not to excluding their
answers from search. The initial catalog and source
snapshots are fixed before measurement. Editing the harness during a run cannot
change the evaluated corpus. To deliberately capture a new corpus:

```bash
.venv/bin/python -B benchmarks/freeze-realistic.py \
    --dots /path/to/Dots --slipagent /path/to/SlipAgent \
    --output /tmp/new-realistic-search.json
```

The matrix crosses these settings:

| Factor | Values |
| --- | --- |
| Total input token budget | 192, 320, 512 |
| Boundaries | Production token/line splitting; AST function/class boundaries with token splitting for oversized symbols |
| Source context | Raw source; relative filename; relative filename and enclosing qualified symbol |
| Pooling | Attention-masked mean; CLS |
| Query instruction | Plain; BGE retrieval prefix |
| Precision | Existing quantized ONNX; optional full-precision ONNX of the same BGE model |
| Ranking | Semantic; BM25; reciprocal rank fusion; weighted hybrid at 25%, 50%, 75% lexical share; words first; identifiers first |

This gives 1,152 combinations with both precision artifacts, or 576 using only
the existing quantized model. Each chunking/context combination covers every
source character and full answer anchor. Source excerpts retain original text
and line ranges. Context consumes input space; the chunker reserves header space
and checks the final complete input. No input is truncated.

BM25 and semantic ranking share the same chunks. Keyword tokenization splits
snake_case and camelCase identifiers, retains Unicode words, and removes common
English stop words. BM25 uses `k1=1.2`, `b=0.75`; fusion uses RRF with `k=60`.
Weighted hybrids normalize cosine scores by their per-query range and BM25 by
its per-query maximum. Words-first prioritizes IDF-weighted query-word coverage
before cosine. Identifiers-first prioritizes explicit code-like lexemes and
quoted phrases before RRF and cosine. A whole-identifier match must satisfy word
boundaries; substrings such as `token_counter` do not count as `token_count`.
Zero keyword matches retain semantic order in every hybrid.

The runner uses exact full-corpus scoring, not HNSW or the application's 0.5
distance cutoff. Mean and CLS vectors come from the same forward pass and are
cached by exact input text, model/configuration/tokenizer hashes, and encoder
source hashes. All unique inputs are sorted by token length and then text before
batching, reducing padding and giving identical text the same vector in every
configuration. Queries are encoded individually, matching `Searcher.search`.
Cache keys include batch size because quantized inference can depend on batch
composition. Answer labels never enter ranking. Many
conditions share vectors; keyword-only rankings repeat across pooling and prefix
settings. The condition count is not a count of independent observations.

Run with the existing cached quantized weights:

```bash
.venv/bin/python -B benchmarks/retrieval-experiments.py \
    --output /tmp/retrieval-results.json
```

To include full precision, supply a **local** `--fp32 /path/to/onnx/model.onnx`
from the same BGE model. The runner never downloads model assets. The measured
full-precision artifact comes from `Xenova/bge-small-en-v1.5`, pinned revision
`ea104dacec62c0de699686887e3f920caeb4f3e3`. Its SHA-256 is
`828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35`;
the installed quantized artifact matches that revision's SHA-256
`6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4`.
This compares two precision artifacts of one model, without a second model or
reranker. The full-precision file is about 133 MB and stays in the model cache.

Inference uses two intra-op threads, one inter-op thread, document batches of
eight, and individual queries.
On a memory-constrained Linux workstation, run the command inside a user systemd
scope with an appropriate `MemoryMax`. `--cache` changes the vector-cache
directory; the default is `~/.cache/dots-retrieval-experiments`. `--budgets`,
`--boundaries`, and `--contexts` can restrict a run. Cached work is reusable after
an interruption when inputs and encoder source hashes are unchanged.

```bash
.venv/bin/python -B benchmarks/summarize-retrieval.py \
    --report benchmarks/retrieval-results.json \
    --output benchmarks/retrieval-results.md
.venv/bin/python -B tests/test-retrieval-experiments.py -v
```

Selections use **development MRR@10**, breaking ties with Hit@5 and Hit@1.
Validation metrics never select configurations. The report includes all query
ranks, top-ten chunk indices, source spans, relevance mappings, split/style/project
metrics, and fingerprints. It compares factors while holding the other selected
settings fixed and shows validation regressions as well as improvements.
The pilot's related queries and limited Python corpus do not establish a general
winner or measure absent-answer rejection and large-repository throughput.
Production indexing and search defaults remain unchanged.

### Recorded Retrieval Findings

The saved 1,152-condition run selects quantized BGE, mean pooling, plain queries,
512-token line chunks, filename context, and a normalized 25% keyword / 75%
semantic mixture using development scores. On the 78 held-out queries, it places
28 answers first versus 15 for the original semantic baseline, and finds 61 within
five versus 46. Ranks improve for 47 queries and worsen for 13.

Keyword-only search places more held-out answers first (41.0%), while the selected
hybrid retrieves more within five (78.2% versus 73.1%). Broad word-first priority
does not consistently beat weighted hybrids. The controlled 50/50 mixture reaches
83.3% within five on validation, but it is not retroactively selected using those
held-out scores. Ordinary questions remain harder than direct identifier lookups.

Filename context helps in the controlled comparison. Smaller chunks, the query
instruction, and full precision do not consistently improve the measured metrics.
Mean and CLS are close for the selected hybrid. These are observations about the
recorded corpus, not universal model settings.

The tested AST boundary strategy retains interstitial segments as separate chunks:
166 of 424 chunks contain only whitespace at 512 tokens without context, versus
zero of 92 line chunks. This is a limitation of that experimental splitter.
Its poor scores do not rule out better function chunking that attaches those gaps
to neighboring code while retaining complete source coverage.

## Original Pooling Pilot

This pilot measures whether an embedding configuration retrieves the intended
implementation for natural-language code-search queries. It compares four
conditions using the same cached, quantized BGE weights:

| Condition | Document And Query Pooling | Query Instruction |
| --- | --- | --- |
| `mean-plain` | Attention-masked mean, L2 normalized | None; production baseline |
| `mean-prefix` | Attention-masked mean, L2 normalized | BGE retrieval prefix |
| `cls-plain` | First token (CLS), L2 normalized | None |
| `cls-prefix` | First token (CLS), L2 normalized | BGE retrieval prefix |

The experimental CLS encoder lives only in the benchmark. The mean condition
calls the production encoder directly. The prefix is
`Represent this sentence for searching relevant passages: `
and is applied only to queries. BGE documents CLS pooling and this query
instruction in its [model card](https://huggingface.co/BAAI/bge-small-en-v1.5).

## Run

From the Dots root, with the existing model cache:

```bash
.venv/bin/python -B benchmarks/embedding-quality.py \
    --queries benchmarks/short-queries.json \
    --output benchmarks/bge-short-query-results.json
```

The short variants contain five or six words each. The original wording remains
in the frozen dataset; rerun it by omitting `--queries`:

```bash
.venv/bin/python -B benchmarks/embedding-quality.py \
    --output benchmarks/bge-token-quality-results.json
```

Or supply the directory containing `config.json`, `tokenizer.json`, and
`onnx/model_quantized.onnx`:

```bash
.venv/bin/python -B benchmarks/embedding-quality.py \
    --model-dir /path/to/local/model \
    --output /tmp/bge-quality-results.json
```

This is an explicit local-model evaluation: it runs inference but never downloads
weights, modifies the model cache, opens a live codebase, or creates an index.
Missing local assets are an error. `MEMORY_MODEL_CACHE_PATH` selects the default
cache root. `--batch-size` defaults to eight. Reports overwrite the supplied
output path; its parent directory must already exist.

## Dataset And Scoring

`code-search.json` freezes 47 source chunks from 12 Dots and SlipAgent Python
files, with 39 hand-written queries in 12 categories. Source SHA-256 hashes and
paths identify the snapshots. The fixture retains the initial 50-line chunks
with five overlapping lines. Before inference, the benchmark reconstructs the
complete frozen files and splits them using the production token-aware chunker.
The line limit remains 50, with up to five overlapping lines when they fit.
Paths are metadata and are not added to embedding inputs. No current checkout
is needed to run the benchmark.

`short-queries.json` replaces only query wording, matching all 39 original IDs.
The source snapshots, categories, and expected code locations stay identical.
The runner rejects missing or duplicated IDs and wording outside five or six
whitespace-separated words. Counts apply before the optional BGE query prefix.

Each query identifies an implementation using a source anchor and line number.
Every new chunk containing the anchor text and its line is labeled relevant. These
are judgments about the intended implementation, not exhaustive judgments of
every potentially useful result. Queries and labels were fixed before measuring
the four conditions. Keep this fixture stable for comparisons; review labels
when deliberately adding or replacing cases.

The benchmark scores every chunk with **exact cosine similarity**, without an
HNSW graph or distance cutoff. This isolates embedding behavior from approximate
retrieval and filtering. Equal scores preserve fixture order.

- **Hit@1 / Hit@5:** fraction of queries with any labeled answer in those ranks.
- **Recall@5:** fraction of each query's labeled chunks found in the first five,
  averaged across queries. Overlapping answers can make this differ from Hit@5.
- **MRR@10:** average reciprocal rank of the first labeled answer; zero past ten.
- **nDCG@10:** discounted binary relevance, normalized by the ideal ranking.

JSON reports include overall and category metrics, every answer's rank, the top
ten results for each query, changes from the baseline, token lengths and truncated
chunk IDs, dataset/query-set/evaluated-corpus/benchmark/model fingerprints, query
word counts, and library versions. Timings
are informational warm timings, not controlled performance measurements.

## Historical Line-Only Results

The saved [initial report](bge-quality-results.json) uses the cached quantized
`Xenova/bge-small-en-v1.5` model:

| Condition | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean, plain | 38.5% | 74.4% | 70.5% | 0.533 | 0.576 |
| Mean, prefix | 35.9% | 69.2% | 65.4% | 0.495 | 0.530 |
| CLS, plain | 38.5% | 71.8% | 67.9% | 0.532 | 0.574 |
| CLS, prefix | 33.3% | 69.2% | 65.4% | 0.505 | 0.553 |

This pilot does not establish an improvement from changing pooling or adding the
prefix. Mean and CLS both find the intended answer first for 15 of 39 queries;
their plain-query Hit@5 counts differ by one query. The queries share files and
chunks, so they are not independent samples and cannot establish a general winner.

**17 of 47 original chunks exceeded 512 tokens and were truncated.** Fifty source
lines can contain considerably more than 512 tokens. That historical behavior
could hide labeled code and limits interpretation of these scores. The current
benchmark fits inputs in full and fails if any document or query exceeds the
limit. The historical report is preserved; the current script does not recreate
its silent truncation.

The pilot covers only a small Python corpus and compares one quantized weight
artifact; it does not measure full-precision parity, conversion loss, or other
languages. Queries share source files and are not independent samples.

## Token-Aware Results

The [complete-input report](bge-token-quality-results.json) uses the same frozen
files, queries, answer anchors, and quantized weights. It splits the files into
52 chunks, with **zero truncated documents or queries**. Every answer anchor
occurs in at least one complete embedding input.

| Condition | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean, plain | 25.6% | 64.1% | 59.0% | 0.420 | 0.502 |
| Mean, prefix | 28.2% | 69.2% | 65.4% | 0.433 | 0.503 |
| CLS, plain | 30.8% | 61.5% | 56.4% | 0.439 | 0.516 |
| CLS, prefix | 30.8% | 71.8% | 66.7% | 0.448 | 0.519 |

Complete coverage does not by itself improve ranking. Mean/plain retrieves an
intended answer first for 10 of 39 queries and within five for 25, compared with
15 and 29 in the historical truncated run. Chunk boundaries, the number of
candidates, and overlapping relevance labels change between those runs, so
their score difference measures the combined chunking change. The four current
conditions share identical complete chunks. Mean pooling remains the application
default; these pilot scores do not automatically select a new embedding policy.

## Short Query Results

The [short-query report](bge-short-query-results.json) uses five or six word
queries from [short-queries.json](short-queries.json). The 52 complete source
chunks, 39 answer locations, model weights, and scoring are unchanged from the
long-query token-aware run. No documents or queries are truncated.

Examples include "Skip unchanged files when refreshing index", "Finish cleanup
despite waiter cancellation", and "Match edits across different line endings".
The variants were fixed before this measurement; wording was not selected by
trying alternatives and keeping whichever scored best.

| Condition | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean, plain | 35.9% | 66.7% | 62.8% | 0.503 | 0.576 |
| Mean, prefix | 35.9% | 64.1% | 60.3% | 0.510 | 0.583 |
| CLS, plain | 33.3% | 71.8% | 69.2% | 0.488 | 0.545 |
| CLS, prefix | 33.3% | 71.8% | 69.2% | 0.489 | 0.557 |

First-result accuracy improves in all four conditions on this pilot. Mean/plain
finds an intended answer first for 14 queries instead of 10, and within five for
26 instead of 25. CLS/plain improves from 24 to 28 queries within five. Top-five
changes are mixed: mean/prefix drops from 27 to 25, while CLS/prefix stays at 28.
These are paired query rewrites in a small corpus, not evidence that every short
query will outperform its longer version. Defaults were unchanged for that pilot;
the production verification above records the later hybrid configuration.

## Offline Harness Checks

```bash
.venv/bin/python -B tests/test-quality.py -v
```

These checks use synthetic vectors and a tiny tokenizer with a stub ONNX session.
They verify hand-calculated metrics, exact ranking, stable ties, invalid-vector
handling, source hashes and answer anchors, production-baseline parity, CLS
pooling, query-only prefixing, and local-only loading. They do not run or download
the real model. Real-model quality scores are measurements, not unit-test gates.
Token-chunking checks also verify source coverage, Unicode and line endings,
oversized single lines, and overlap that leaves room for new source:

```bash
.venv/bin/python -B tests/test-chunking.py -v
```
