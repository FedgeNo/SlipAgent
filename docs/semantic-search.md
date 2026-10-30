# Semantic Search

`semantic_search` is a deferred built-in tool. The additional-tool catalog exposes
its purpose on each request, and the model loads its full definition through
`load_tools` before calling it. It supplements the repository map, `grep`, and
file reads. Loading the definition does not load the encoder or start indexing.

The implementation lives in `src/slipagent/semantic/`; `tools/semantic.py` adapts
it to the normal tool contract. The former standalone Dots CLI, GUI, and optional
SentenceTransformer backend are not part of the harness. The tested quantized
BGE model, token-aware line chunking, complete excerpts, relative filename
context, SQLite BM25 postings, and HNSW candidate retrieval are retained.

## Ownership And Persistence

The first lookup creates a `SemanticSearch` service in `ToolRegistry.services`.
One dedicated worker owns the model, SQLite connection, graph, watcher, and
lookup queue. Native filesystem callbacks only queue paths. Searches and writes
run serially on that worker; terminal input stays responsive. Reload candidates
borrow the service and do not close it. Registry teardown stops observation,
drains indexing, and closes SQLite through a retained `Lifetime` cleanup.

Model and backend modules are restart boundaries: a live worker keeps one
implementation generation, including its embedding and chunking contracts.
The tool adapter and editable tool guidance can reload normally. Restart once
after installing this integration or changing its backend.

Each workspace's canonical root hashes to its own folder under
`data_directory()/semantic/`. The database stores original source chunks,
file metadata and content digests, model identity, keyword postings, and the
search graph. Each update appends its graph additions and removals to a journal;
the full graph is saved as a checkpoint once the journal passes 256 changes, and
loading replays the journal over the checkpoint. Vectors, source rows and journal
entries publish in one SQLite transaction; failed or interrupted updates retain
the previous committed index. A model or index-configuration mismatch is reported
rather than mixing incompatible vectors. Index state is generated data and is not
shipped in the source; the default model is bundled.

The observer starts before the initial scan. Save bursts coalesce for 250 ms,
with a one-second maximum delay. Queued edits are hashed even when timestamp
and size are unchanged. Renames and deleted directories remove old descendants.
A metadata reconciliation runs every 30 seconds, and every new session starts
with reconciliation before answering its first lookup. A lookup flushes queued
edits first; ongoing external edits can still arrive after that snapshot.

## Retrieval And Boundaries

The default query returns five excerpts; `limit` accepts 1–10. The encoder uses
quantized BGE, masked mean pooling, L2 normalization, plain queries, a 512-token
budget, and one document per inference call. Chunk headers consume input budget;
source text is never silently truncated. Background indexing uses half the
physical core count clamped to available CPU affinity, with a minimum of one
thread. If the physical core count is unavailable, its starting count is two
before affinity clamping and halving.

Ranking blends normalized cosine (75%) and BM25 (25%). Independent semantic and
keyword shortlists allow exact word matches to enter without a cosine cutoff.
Candidate retrieval and normalization become approximate beyond 256 chunks.
The frozen pilot's quality figures do not establish large-codebase recall or
prove a reduction in agent turns. Scores are not confidence probabilities.

Traversal skips dependency/generated directories, hidden paths, symlinks,
Git-ignored files, and harness data. Workspace confinement is applied before
reading or returning paths; danger mode does not expand this index beyond the
current workspace. File tools retain their normal visibility and access rules.
The default model (quantized `bge-small-en-v1.5`, MIT) ships with SlipAgent in
`src/slipagent/semantic/model/`, so it needs no download. Embedding and query
text are processed locally.

Unavailable models, watcher failures, or invalid indexes return a tool error with
ordinary-search guidance. Cancellation stops indexing between chunk embeddings
and waits for owned work to settle. A later lookup can restart an interrupted
worker. Loading no semantic tool incurs no model inference or index scan.

## Verification

`tests/test-semantic-*.py` covers the migrated backend and harness integration
using disposable workspaces and stub encoders. Full query fixtures, historical
reports, and explicit real-model benchmark scripts live in
[`benchmarks/semantic/`](../benchmarks/semantic/README.md). Pytest never downloads
models or performs real model inference.
