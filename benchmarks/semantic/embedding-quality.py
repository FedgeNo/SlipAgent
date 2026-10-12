"""Compare pooling and query instructions on a frozen, labeled code corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from slipagent.semantic import DEFAULT_MODEL, ONNXEmbedder
from slipagent.semantic.indexer import chunk_text


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
METRICS = ("hit_at_1", "hit_at_5", "recall_at_5", "mrr_at_10", "ndcg_at_10")


class BenchmarkEmbedder(ONNXEmbedder):
    """Keep the production mean path; isolate experimental CLS pooling here."""

    def __init__(self, *, model_dir: Path, pooling: str, batch_size: int = 8):
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be mean or cls")
        super().__init__(model_dir=model_dir, batch_size=batch_size)
        self.pooling = pooling

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.pooling == "mean":
            return super().embed_batch(texts)
        if not texts:
            return []
        self._load()
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self._tokenizer.encode_batch(texts[start:start + self.batch_size])
            if any(len(token.ids) > self.max_tokens for token in tokens):
                raise ValueError("CLS input exceeds the token limit; split it before embedding")
            inputs = {
                "input_ids": np.asarray([token.ids for token in tokens], dtype=np.int64),
                "attention_mask": np.asarray([token.attention_mask for token in tokens], dtype=np.int64),
                "token_type_ids": np.asarray([token.type_ids for token in tokens], dtype=np.int64),
            }
            feed = {item.name: inputs[item.name] for item in self._session.get_inputs()}
            hidden = self._session.run(["last_hidden_state"], feed)[0]
            pooled = hidden[:, 0, :]
            normalized = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            vectors.extend(normalized.tolist())
        return vectors


def load_dataset(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema_version")
    documents, queries = data["documents"], data["queries"]
    if not documents or not queries:
        raise ValueError("Dataset needs documents and labeled queries")
    ids = set()
    for document in documents:
        if not isinstance(document["id"], str) or not document["id"] or document["id"] in ids:
            raise ValueError("Document IDs must be unique nonempty strings")
        ids.add(document["id"])
        if not isinstance(document["text"], str) or not document["text"].strip():
            raise ValueError("Document text must be nonempty")
        if not 1 <= document["start_line"] <= document["end_line"]:
            raise ValueError("Invalid document line range")
    query_ids = set()
    for query in queries:
        if not isinstance(query["id"], str) or not query["id"] or query["id"] in query_ids:
            raise ValueError("Query IDs must be unique nonempty strings")
        query_ids.add(query["id"])
        if not isinstance(query["text"], str) or not query["text"].strip():
            raise ValueError("Query text must be nonempty")
        relevant = query["relevant"]
        if not relevant or len(set(relevant)) != len(relevant) or not set(relevant) <= ids:
            raise ValueError(f"Invalid relevance labels for {query['id']}")
    return data, hashlib.sha256(raw).hexdigest()


def replace_queries(dataset: dict, path: Path) -> tuple[dict, str]:
    """Replace query wording by ID while preserving source and relevance labels."""
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported query set schema_version")
    texts = {}
    for query in data["queries"]:
        key, text = query["id"], query["text"]
        if key in texts or not isinstance(text, str) or not 5 <= len(text.split()) <= 6:
            raise ValueError("Replacement queries need unique IDs and five or six words")
        texts[key] = text
    if set(texts) != {query["id"] for query in dataset["queries"]}:
        raise ValueError("Replacement query IDs must match the dataset exactly")
    return {**dataset, "queries": [{**query, "text": texts[query["id"]]} for query in dataset["queries"]]}, \
        hashlib.sha256(raw).hexdigest()


def ranking_metrics(ranking: list[str], relevant: set[str]) -> dict[str, float]:
    """Binary relevance; reciprocal rank and nDCG are truncated at ten."""
    if not relevant or len(set(ranking)) != len(ranking) or not relevant <= set(ranking):
        raise ValueError("Ranking must contain every relevant ID exactly once")
    first = next(index for index, key in enumerate(ranking, 1) if key in relevant)
    dcg = sum(1 / math.log2(index + 1) for index, key in enumerate(ranking[:10], 1) if key in relevant)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(10, len(relevant)) + 1))
    return {
        "hit_at_1": float(first == 1),
        "hit_at_5": float(first <= 5),
        "recall_at_5": len(set(ranking[:5]) & relevant) / len(relevant),
        "mrr_at_10": 1 / first if first <= 10 else 0.0,
        "ndcg_at_10": dcg / ideal,
    }


def unit_vectors(vectors: list[list[float]], rows: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] < 1:
        raise ValueError("Encoder returned unexpected vector shape")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Encoder returned invalid vectors")
    return matrix / norms


def evaluate(documents: list[dict], queries: list[dict], document_vectors: list[list[float]],
             query_vectors: list[list[float]]) -> dict:
    corpus = unit_vectors(document_vectors, len(documents))
    questions = unit_vectors(query_vectors, len(queries))
    if corpus.shape[1] != questions.shape[1]:
        raise ValueError("Query and document dimensions differ")
    scores = questions @ corpus.T
    rows = []
    for query, similarities in zip(queries, scores):
        # Stable ties retain fixture order rather than favoring labeled answers.
        order = np.argsort(-similarities, kind="stable")
        ranking = [documents[int(index)]["id"] for index in order]
        relevant = set(query["relevant"])
        rows.append({
            "id": query["id"], "query": query["text"], "category": query["category"],
            "relevant": query["relevant"],
            "relevant_ranks": {key: ranking.index(key) + 1 for key in query["relevant"]},
            "metrics": ranking_metrics(ranking, relevant),
            "top_10": [{"id": documents[int(index)]["id"],
                        "path": documents[int(index)]["path"],
                        "start_line": documents[int(index)]["start_line"],
                        "end_line": documents[int(index)]["end_line"],
                        "cosine_similarity": float(similarities[index]),
                        "relevant": documents[int(index)]["id"] in relevant}
                       for index in order[:10]],
        })

    def average(items: list[dict]) -> dict[str, float]:
        return {key: sum(row["metrics"][key] for row in items) / len(items) for key in METRICS}

    return {"metrics": average(rows),
            "categories": {category: average([row for row in rows if row["category"] == category])
                           for category in sorted({row["category"] for row in rows})},
            "queries": rows}


def run_comparison(dataset: dict, embedder: BenchmarkEmbedder) -> list[dict]:
    documents, queries = dataset["documents"], dataset["queries"]
    start = time.perf_counter()
    document_vectors = embedder.embed_batch([document["text"] for document in documents])
    document_seconds = time.perf_counter() - start
    results = []
    for instruction in (False, True):
        prefix = QUERY_PREFIX if instruction else ""
        start = time.perf_counter()
        query_vectors = embedder.embed_batch([prefix + query["text"] for query in queries])
        query_seconds = time.perf_counter() - start
        results.append({"name": embedder.pooling + ("-prefix" if instruction else "-plain"),
                        "pooling": embedder.pooling, "query_prefix": prefix,
                        "document_seconds": document_seconds, "query_seconds": query_seconds,
                        **evaluate(documents, queries, document_vectors, query_vectors)})
    return results


def token_dataset(dataset: dict, embedder: BenchmarkEmbedder) -> dict:
    """Rebuild full frozen files, split to fit, and retain the same answer anchors."""
    files: dict[str, dict[int, str]] = {}
    for document in dataset["documents"]:
        lines = files.setdefault(document["path"], {})
        for number, text in enumerate(document["text"].splitlines(keepends=True), document["start_line"]):
            if number in lines and lines[number] != text:
                raise ValueError("Frozen source excerpts disagree")
            lines[number] = text
    documents = []
    for path, lines in files.items():
        if sorted(lines) != list(range(1, max(lines) + 1)):
            raise ValueError("Token-aware evaluation needs complete frozen source files")
        source = "".join(lines[number] for number in sorted(lines))
        chunks = chunk_text(path, source, **dataset["chunking"], token_count=embedder.token_count,
                            max_tokens=embedder.max_tokens)
        for number, chunk in enumerate(chunks, 1):
            documents.append({"id": f"{path}:{chunk.start_line}-{chunk.end_line}:part{number}",
                              "path": path, "start_line": chunk.start_line,
                              "end_line": chunk.end_line, "text": chunk.text})
    queries = []
    for query in dataset["queries"]:
        target = query["target"]
        relevant = [doc["id"] for doc in documents if doc["path"] == target["path"]
                    and doc["start_line"] <= target["line"] <= doc["end_line"]
                    and target["anchor"] in doc["text"]]
        if not relevant:
            raise ValueError(f"Answer anchor is not visible in a complete chunk: {query['id']}")
        queries.append({**query, "relevant": relevant})
    return {**dataset, "documents": documents, "queries": queries}


def truncation_counts(dataset: dict, embedder: BenchmarkEmbedder) -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(embedder._tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    limit = embedder.max_tokens

    def count(texts: list[str]) -> int:
        return sum(len(tokens.ids) > limit for tokens in tokenizer.encode_batch(texts))

    lengths = [len(tokens.ids) for tokens in tokenizer.encode_batch(
        [document["text"] for document in dataset["documents"]])]
    truncated = [document["id"] for document, length in zip(dataset["documents"], lengths) if length > limit]
    return {"max_tokens": limit, "documents": len(truncated), "document_ids": truncated,
            "document_token_lengths": {document["id"]: length
                                       for document, length in zip(dataset["documents"], lengths)},
            "plain_queries": count([query["text"] for query in dataset["queries"]]),
            "prefixed_queries": count([QUERY_PREFIX + query["text"] for query in dataset["queries"]])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("code-search.json"))
    parser.add_argument("--queries", type=Path, help="Five or six word query variants keyed by dataset query ID")
    cache = Path(os.environ.get("MEMORY_MODEL_CACHE_PATH", str(Path.home() / ".conventions-mcp/models")))
    parser.add_argument("--model-dir", type=Path, default=cache / DEFAULT_MODEL,
                        help="Local model directory; missing assets fail without downloading")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    args = parser.parse_args(argv)
    try:
        dataset, digest = load_dataset(args.dataset)
        query_digest = None
        if args.queries is not None:
            dataset, query_digest = replace_queries(dataset, args.queries)
        embedder = BenchmarkEmbedder(model_dir=args.model_dir, pooling="mean", batch_size=args.batch_size)
        identity = embedder.identity
        original_documents = len(dataset["documents"])
        dataset = token_dataset(dataset, embedder)
        truncation = truncation_counts(dataset, embedder)
        if truncation["documents"] or truncation["plain_queries"] or truncation["prefixed_queries"]:
            raise ValueError("Quality evaluation inputs must fit in full")
        print(f"{len(dataset['documents'])} chunks, {len(dataset['queries'])} labeled queries; local model loaded",
              flush=True)
        conditions = []
        for pooling in ("mean", "cls"):
            embedder.pooling = pooling
            # Warm up before informational timings; use the same loaded weights in all conditions.
            embedder.embed("warmup")
            print(f"Evaluating {pooling} pooling…", flush=True)
            conditions.extend(run_comparison(dataset, embedder))
        baseline = conditions[0]
        for condition in conditions:
            condition["delta_from_mean_plain"] = {
                key: condition["metrics"][key] - baseline["metrics"][key] for key in METRICS}
            condition["rank_changes_from_mean_plain"] = {
                "improved": [], "worsened": [], "unchanged": []}
            for current, old in zip(condition["queries"], baseline["queries"]):
                rank, old_rank = min(current["relevant_ranks"].values()), min(old["relevant_ranks"].values())
                bucket = "improved" if rank < old_rank else "worsened" if rank > old_rank else "unchanged"
                condition["rank_changes_from_mean_plain"][bucket].append(current["id"])
        report = {
            "schema_version": 1, "dataset_sha256": digest, "dataset_name": dataset["name"],
            "query_set": args.queries.name if args.queries is not None else "original",
            "query_set_sha256": query_digest,
            "query_words": {"min": min(len(query["text"].split()) for query in dataset["queries"]),
                            "max": max(len(query["text"].split()) for query in dataset["queries"])},
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluated_corpus_sha256": hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest(),
            "input_documents": original_documents,
            "chunking": {**dataset["chunking"], "strategy": "token-lines-v1", "max_tokens": embedder.max_tokens},
            "documents": len(dataset["documents"]), "queries": len(dataset["queries"]),
            "model": DEFAULT_MODEL, "model_identity": identity,
            "batch_size": args.batch_size, "truncation": truncation,
            "ranking": "exact cosine; no HNSW approximation or distance cutoff",
            "timings": "Informational warm timings, not a controlled performance benchmark",
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("onnxruntime", "numpy", "tokenizers")},
            "conditions": conditions,
        }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Benchmark failed: {exc}\n")
    print("Condition          Hit@1   Hit@5   Recall@5  MRR@10  nDCG@10")
    for condition in conditions:
        values = condition["metrics"]
        print(f"{condition['name']:<18} " + "  ".join(f"{values[key]:.3f}" for key in METRICS))
    print(f"Truncated chunks: {truncation['documents']}/{len(dataset['documents'])}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
