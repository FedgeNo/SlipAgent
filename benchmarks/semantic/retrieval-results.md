# Retrieval Experiment Results

Measured **1,152 conditions**, **294 hand-written queries**, **98 intents**, and **19 complete source files**.

Development: 216 queries. Internal validation: 78 queries from entirely held-out target files. Query styles: 98 lookup, 98 question, 98 task.

Every corpus covers every source character and complete answer anchor. All embedding inputs fit in full. The same BGE model is evaluated as quantized and full-precision ONNX; no second model or reranker is used.

## Comparisons

Each row below is chosen using development MRR@10, then Hit@5 and Hit@1. Validation scores do not select configurations. Rows optimize several factors and therefore are not isolated estimates of one factor's effect. The controlled comparisons below hold other settings fixed.

| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original mean/plain semantic baseline | 32.4% | 59.7% | 0.439 | 19.2% | 59.0% | 0.339 |
| Selected combination | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |
| Best q8 combination | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |
| Semantic only | 41.7% | 68.5% | 0.528 | 26.9% | 66.7% | 0.427 |
| Keywords only (BM25) | 46.3% | 70.4% | 0.556 | 41.0% | 73.1% | 0.544 |
| Semantic + keywords (RRF) | 39.8% | 72.7% | 0.543 | 38.5% | 78.2% | 0.543 |
| Weighted semantic + keywords | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |
| Exact words before semantic | 44.4% | 68.5% | 0.549 | 37.2% | 76.9% | 0.527 |
| Exact identifiers before fusion | 40.7% | 73.6% | 0.551 | 35.9% | 80.8% | 0.528 |

## Findings

The development-selected combination finds an answer first for 28 of 78 validation queries versus 15 for the baseline, and within five for 61 versus 46. Its first-answer rank improves on 47 queries and worsens on 13.

The development-selected configuration is `q8/lines-512-path/mean/plain/hybrid-25`. The controlled tables below expose other settings' validation outcomes without retroactively using those scores to select a different configuration. See the [recorded interpretation](historical-notes.md#recorded-retrieval-findings) for the original run's findings.

The experimental AST splitter keeps interstitial source as separate chunks, including whitespace-only segments. Its results describe that specific implementation and do not rule out improved function chunking that attaches gaps to neighboring code while retaining coverage.

BM25 does not use neural vectors: precision and pooling labels in its condition name are tied matrix metadata, not requirements for keyword-only search.

## Selected Settings

- **Original mean/plain semantic baseline:** `q8/lines-512-none/mean/plain/semantic`
- **Selected combination:** `q8/lines-512-path/mean/plain/hybrid-25`
- **Best q8 combination:** `q8/lines-512-path/mean/plain/hybrid-25`
- **Semantic only:** `fp32/lines-512-path/mean/plain/semantic`
- **Keywords only (BM25):** `fp32/lines-512-none/cls/plain/bm25`
- **Semantic + keywords (RRF):** `q8/lines-512-path/mean/plain/rrf`
- **Weighted semantic + keywords:** `q8/lines-512-path/mean/plain/hybrid-25`
- **Exact words before semantic:** `q8/lines-512-symbols/cls/plain/words-first`
- **Exact identifiers before fusion:** `q8/lines-512-path/cls/plain/identifiers-first`

## Controlled Comparisons

These comparisons vary one setting around the development-selected combination. Its corpus name encodes boundary, total token budget, and context.

### Ranker

| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 45.8% | 69.4% | 0.548 | 37.2% | 71.8% | 0.519 |
| hybrid-25 | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |
| hybrid-50 | 45.8% | 75.5% | 0.588 | 41.0% | 83.3% | 0.564 |
| hybrid-75 | 47.2% | 72.7% | 0.573 | 39.7% | 75.6% | 0.554 |
| identifiers-first | 40.3% | 73.1% | 0.546 | 39.7% | 78.2% | 0.550 |
| rrf | 39.8% | 72.7% | 0.543 | 38.5% | 78.2% | 0.543 |
| semantic | 39.4% | 69.9% | 0.523 | 25.6% | 65.4% | 0.419 |
| words-first | 44.0% | 68.5% | 0.543 | 37.2% | 70.5% | 0.517 |

### Boundary

| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lines | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |
| symbols | 37.0% | 70.4% | 0.512 | 30.8% | 69.2% | 0.466 |

### Budget

| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 192 | 37.0% | 65.3% | 0.489 | 25.6% | 51.3% | 0.354 |
| 320 | 41.2% | 71.8% | 0.541 | 37.2% | 65.4% | 0.493 |
| 512 | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |

### Context

| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 41.2% | 74.5% | 0.546 | 32.1% | 73.1% | 0.485 |
| path | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |
| symbols | 43.1% | 77.3% | 0.575 | 35.9% | 79.5% | 0.539 |

### Pooling

| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cls | 46.8% | 78.2% | 0.601 | 37.2% | 78.2% | 0.540 |
| mean | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |

### Prefix

| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |
| True | 44.4% | 77.8% | 0.583 | 35.9% | 75.6% | 0.519 |

### Precision

| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 44.4% | 79.6% | 0.586 | 34.6% | 80.8% | 0.523 |
| q8 | 46.3% | 78.7% | 0.601 | 35.9% | 78.2% | 0.525 |

## Validation By Query Style

| Style | Queries | Baseline First | Selected First | Baseline Top Five | Selected Top Five |
| --- | ---: | ---: | ---: | ---: | ---: |
| lookup | 26 | 34.6% | 50.0% | 84.6% | 96.2% |
| question | 26 | 7.7% | 30.8% | 46.2% | 61.5% |
| task | 26 | 15.4% | 26.9% | 46.2% | 76.9% |

## Paired Validation Examples

Ranks below refer to complete labeled source anchors. Improvement and regression examples are shown; the JSON retains every query's first-answer rank and top ten chunk indices.

### Improved

- Display complete code excerpts in the interface — rank 39 → 5 (`Dots/gui/app.py:47`).
- Find fnmatchcase instructions rules directory walk — rank 35 → 1 (`SlipAgent/src/slipagent/instructions.py:84`).
- Can I set a token limit without a token counter? — rank 55 → 24 (`Dots/codeindex/indexer.py:178`).
- Discover nested editor rule files — rank 33 → 6 (`SlipAgent/src/slipagent/instructions.py:84`).
- Find test_info_does_not_load_an_embedding_model — rank 32 → 5 (`Dots/tests/test-cli.py:62`).
- How is context retained across line-based chunk boundaries? — rank 39 → 14 (`Dots/codeindex/indexer.py:210`).

### Regressed

- Where do I add another source file extension? — rank 11 → 21 (`Dots/codeindex/indexer.py:134`).
- Can a new command evict an older command's output? — rank 6 → 9 (`SlipAgent/src/slipagent/tools/output.py:57`).
- Find CommandLog.finish fsync — rank 5 → 8 (`SlipAgent/src/slipagent/tools/output.py:92`).
- How does an archive learn the current step and call identifiers? — rank 5 → 8 (`SlipAgent/src/slipagent/tools/output.py:167`).
- Read the last characters of saved output — rank 1 → 4 (`SlipAgent/src/slipagent/tools/output.py:240`).
- Find max_lines overlap line chunking loop — rank 3 → 5 (`Dots/codeindex/indexer.py:210`).

## Interpretation Limits

This is an internal Python pilot with related query variants and shared source files. The three variants of an intent are not independent observations. Labels identify intended implementation anchors, not every useful helper or test. The set contains answerable queries; it does not measure refusing nonexistent features or confidence thresholds. The initial code snapshots and query wording were frozen before this run. The validation set remains internal and should not be repeatedly used to tune settings.

Retrieval uses exact cosine and full-corpus keyword scoring. It does not measure HNSW candidate recall, the application's distance cutoff, or large-repository throughput. Many matrix conditions share vectors or identical keyword rankings. Timings are informational because host load is uncontrolled and cached embeddings are reused. Production defaults are unchanged.

## Artifacts

- [Reviewed query cases](realistic-cases.json)
- [Frozen source and labels](realistic-search.json)
- [Complete machine-readable results](retrieval-results.json)
- [Runner](retrieval-experiments.py)

Dataset SHA-256: `fe76812b0b59d36a9db0c4d5f7faadeff9a9da4a2823a2e772c680db8dbab6c8`.

Runner SHA-256: `1920e6d783d1c39e5ac2aaad010610a0ee68b502c3c1b55413e255fdc2b693d7`.
