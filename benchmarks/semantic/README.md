# Semantic Retrieval Benchmarks

This folder retains the complete Dots search experiments now that the backend
lives in `slipagent.semantic`. Runtime integration is documented in
[`docs/semantic-search.md`](../../docs/semantic-search.md).

The frozen corpus contains 19 complete source files and 294 realistic queries:
216 development queries and 78 internal validation queries, split by target file.
The historical project names and code paths inside fixtures remain unchanged so
their source hashes, answer anchors, and comparisons stay reproducible.

- [Production quality and thread results](production-results.md)
- [Original 1,152-condition retrieval experiment](retrieval-results.md)
- [Original detailed experiment notes](historical-notes.md)
- [Frozen source and query corpus](realistic-search.json)
- [Original complete matrix results](retrieval-results.json)

The applied recipe is quantized BGE, masked mean pooling, plain queries,
512-token line chunks with relative filename context, and a normalized
75% semantic / 25% BM25 blend. Documents embed individually, independently of
batch neighbors. Results beyond 256 chunks use approximate candidate retrieval.
The small internal pilot does not establish large-corpus recall or agent turn savings.

Run the actual integrated backend on all frozen queries from the repository root:

```bash
.venv/bin/python -B benchmarks/semantic/production-retrieval.py \
    --output /tmp/semantic-production-results.json
```

This explicit benchmark uses the model bundled in `src/slipagent/semantic/model/`
(or `--model-dir`) and temporary source/database
files. It does not call a provider or download another model. Reports record
source and model hashes, dependency versions, and every top-ten result. The
historical batch-one/batch-eight reports retain their original fingerprints.

The other scripts also import `slipagent.semantic`: `embedding-quality.py`,
`retrieval-experiments.py`, `summarize-retrieval.py`, and `threads.py`. Use `--help`
for their arguments. Thread measurements start each sample in a fresh process
without inference warmup; the OS file cache is not flushed. Run timing comparisons
while unrelated CPU-heavy work is idle. No real inference runs in pytest.
