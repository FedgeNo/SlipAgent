"""Factorial code retrieval experiments using one BGE model, frozen code, and labels."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import gc
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from slipagent.semantic.embed import DEFAULT_MODEL, ONNXEmbedder
from slipagent.semantic.indexer import chunk_text


spec = importlib.util.spec_from_file_location("embedding_quality", Path(__file__).with_name("embedding-quality.py"))
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)

STOP_WORDS = frozenset("a an and are as at be before besides by can could do does for from get gets "
                       "how i if in into is it its me of on or our that the their them these this "
                       "to use used using was we what when where which why will with without would "
                       "find show explain does not than then too any all have has".split())
RANKERS = ("semantic", "bm25", "rrf", "hybrid-25", "hybrid-50", "hybrid-75",
           "words-first", "identifiers-first")


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_dataset(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 2:
        raise ValueError("Expected a complete-source schema version 2 dataset")
    files = {item["path"]: item for item in data["files"]}
    if len(files) != len(data["files"]) or not files:
        raise ValueError("Source paths must be unique and nonempty")
    for item in files.values():
        if hashlib.sha256(item["text"].encode()).hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {item['path']}")
    ids, families, file_splits = set(), {}, {}
    for query in data["queries"]:
        if query["id"] in ids or not query["text"].strip() or query["split"] not in {"development", "validation"}:
            raise ValueError("Invalid or duplicate query")
        ids.add(query["id"])
        for key, mapping in ((query["family"], families), *[(t["path"], file_splits) for t in query["targets"]]):
            if key in mapping and mapping[key] != query["split"]:
                raise ValueError("A family or target file crosses evaluation splits")
            mapping[key] = query["split"]
        if not query["targets"]:
            raise ValueError("Queries need source targets")
        for target in query["targets"]:
            source = files[target["path"]]["text"]
            if (source[target["start"]:target["end"]] != target["anchor"]
                    or source.count("\n", 0, target["start"]) + 1 != target["line"]):
                raise ValueError(f"Invalid source anchor: {query['id']}")
    if not ids or set(file_splits.values()) != {"development", "validation"}:
        raise ValueError("Need nonempty development and validation query sets")
    return data, hashlib.sha256(raw).hexdigest()


def symbols(source: str) -> list[dict]:
    """Derive symbol ranges and qualified names from source alone, never labels."""
    result = []

    def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            names = parents
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names = (*parents, child.name)
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                result.append({"start_line": start, "end_line": child.end_lineno,
                               "name": ".".join(names)})
            visit(child, names)

    visit(ast.parse(source))
    return result


def source_header(path: str, ranges: list[dict], line: int, context: str) -> str:
    if context == "none":
        return ""
    header = f"File: {path}\n"
    if context == "symbols":
        enclosing = [item for item in ranges if item["start_line"] <= line <= item["end_line"]]
        if enclosing:
            item = max(enclosing, key=lambda item: (item["name"].count("."), item["start_line"]))
            header += f"Symbol: {item['name']}\n"
    return header + "\n"


def build_corpus(dataset: dict, counter, *, budget: int, boundary: str, context: str) -> dict:
    if boundary not in {"lines", "symbols"} or context not in {"none", "path", "symbols"}:
        raise ValueError("Unknown source chunking or context policy")
    if budget < 32:
        raise ValueError("Token budget is too small")
    documents = []
    for file in dataset["files"]:
        source, path = file["text"], file["path"]
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        ranges = symbols(source)
        boundaries = {1, len(lines) + 1}
        if boundary == "symbols":
            for item in ranges:
                boundaries.update((item["start_line"], item["end_line"] + 1))
        boundaries = sorted(boundaries)
        coverage = []
        for start, end in zip(boundaries, boundaries[1:]):
            # Reserve enough space for any enclosing symbol in this segment.
            # Count complete inputs again below; metadata can never truncate code.
            overhead = max(counter(source_header(path, ranges, line, context)) - counter("")
                           for line in range(start, end)) if context != "none" else 0
            reserve = overhead + (8 if context != "none" else 0)
            segment = source[offsets[start - 1]:offsets[end - 1]]
            chunks = chunk_text(path, segment, max_lines=50, overlap=5,
                                token_count=counter, max_tokens=budget - reserve)
            cursor = offsets[start - 1]
            for chunk in chunks:
                first_line = start + chunk.start_line - 1
                last_line = start + chunk.end_line - 1
                low = max(cursor, offsets[first_line - 1])
                position = source.find(chunk.text, low, offsets[last_line])
                if position < 0:
                    raise ValueError(f"Cannot locate complete source chunk: {path}:{first_line}")
                cursor = position + 1
                stop = position + len(chunk.text)
                header = source_header(path, ranges, first_line, context)
                text = header + chunk.text
                length = counter(text)
                if length > budget:
                    raise ValueError(f"Input exceeds full {budget}-token budget: {path}:{first_line}")
                documents.append({"id": f"{path}:{position}-{stop}", "path": path,
                                  "start": position, "end": stop,
                                  "start_line": first_line, "end_line": last_line,
                                  "header": header, "text": chunk.text, "tokens": length})
                coverage.append((position, stop))
        covered = 0
        for first, last in sorted(coverage):
            if first > covered:
                raise ValueError(f"Dropped source text at {path}:{covered}")
            covered = max(covered, last)
        if covered != len(source):
            raise ValueError(f"Missing source tail: {path}")
    queries = []
    for query in dataset["queries"]:
        relevant = [index for index, document in enumerate(documents)
                    if any(document["path"] == target["path"] and document["start"] <= target["start"]
                           and document["end"] >= target["end"] for target in query["targets"])]
        if not relevant:
            raise ValueError(f"No complete answer anchor in chunks: {query['id']}")
        queries.append({**query, "relevant": relevant})
    if len({item["id"] for item in documents}) != len(documents):
        raise ValueError("Duplicate source chunk")
    return {"name": f"{boundary}-{budget}-{context}", "budget": budget, "boundary": boundary,
            "context": context, "documents": documents, "queries": queries,
            "coverage": "Every source character covered; every answer anchor fully present; zero truncation"}


def words(text: str) -> list[str]:
    """Split code identifiers and prose consistently; keep whole lexemes separately."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)
    return [word.casefold() for word in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
            if word.casefold() not in STOP_WORDS]


class KeywordIndex:
    def __init__(self, texts: list[str]):
        self.texts = [text.casefold() for text in texts]
        self.counts = [Counter(words(text)) for text in texts]
        self.lengths = np.asarray([sum(count.values()) for count in self.counts], dtype=np.float32)
        self.average = max(float(self.lengths.mean()), 1)
        frequencies = Counter(word for count in self.counts for word in count)
        self.idf = {word: math.log(1 + (len(texts) - count + 0.5) / (count + 0.5))
                    for word, count in frequencies.items()}

    def score(self, query: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = set(words(query))
        scores = np.zeros(len(self.texts), dtype=np.float32)
        coverage = scores.copy()
        denominator = sum(self.idf.get(word, 0) for word in tokens) or 1
        for word in sorted(tokens):
            frequency = np.asarray([count[word] for count in self.counts], dtype=np.float32)
            inverse = self.idf.get(word, 0)
            scores += inverse * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * self.lengths / self.average))
            coverage += (frequency > 0) * inverse / denominator
        # Only explicitly written code-like lexemes or quoted phrases get an
        # identifier tier. Ordinary prose words are tested separately below.
        lexemes = re.findall(r"[\w./-]+", query, flags=re.UNICODE)
        exact = [word for word in lexemes if ("_" in word or "." in word
                 or re.search(r"[a-z][A-Z]", word) or re.search(r"[A-Z]{2,}", word))]
        exact.extend(match[1] for match in re.findall(r"([`\"'])(.+?)\1", query))
        identifiers = np.zeros(len(self.texts), dtype=np.float32)
        for term in sorted(set(exact)):
            pattern = re.compile(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)")
            identifiers += np.asarray([bool(pattern.search(text)) for text in self.texts], dtype=np.float32)
        return scores, coverage, identifiers


def reciprocal_ranks(scores: np.ndarray, *, positive_only: bool = False) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    result = np.zeros(len(scores), dtype=np.float32)
    # Equal scores receive equal rank credit, including all-zero keyword rows.
    previous, rank = None, 0
    for position, index in enumerate(order, 1):
        score = float(scores[index])
        if previous is None or score != previous:
            rank = position
        previous = score
        if not positive_only or score > 0:
            result[index] = 1 / (60 + rank)
    return result


def rank_candidates(semantic: np.ndarray, lexical: np.ndarray, coverage: np.ndarray,
                    identifiers: np.ndarray, method: str) -> np.ndarray:
    fused = reciprocal_ranks(semantic) + reciprocal_ranks(lexical, positive_only=True)
    if method == "semantic":
        return np.argsort(-semantic, kind="stable")
    if method == "bm25":
        return np.argsort(-lexical, kind="stable")
    if method == "rrf":
        return np.lexsort((np.arange(len(semantic)), -semantic, -fused))
    if method.startswith("hybrid-"):
        weight = int(method.split("-")[1]) / 100
        span = float(semantic.max() - semantic.min())
        dense = (semantic - semantic.min()) / max(span, 1e-12)
        sparse = lexical / max(float(lexical.max()), 1e-12)
        return np.argsort(-((1 - weight) * dense + weight * sparse), kind="stable")
    if method == "words-first":
        return np.lexsort((np.arange(len(semantic)), -semantic, -coverage))
    if method == "identifiers-first":
        return np.lexsort((np.arange(len(semantic)), -semantic, -fused, -identifiers))
    raise ValueError(f"Unknown ranker: {method}")


def measure(corpus: dict, orders: list[np.ndarray]) -> dict:
    rows = []
    first_ranks, top_10 = [], []
    for query, order in zip(corpus["queries"], orders):
        ranking = [int(index) for index in order]
        relevant = set(query["relevant"])
        metrics = quality.ranking_metrics(ranking, relevant)
        first = min(ranking.index(index) + 1 for index in relevant)
        paths = {target["path"] for target in query["targets"]}
        metrics["file_hit_at_1"] = float(corpus["documents"][ranking[0]]["path"] in paths)
        metrics["hit_at_10"] = float(first <= 10)
        first_ranks.append(first)
        top_10.append(ranking[:10])
        rows.append({"metrics": metrics, "split": query["split"], "style": query["style"],
                     "project": query["targets"][0]["path"].split("/", 1)[0]})

    def average(items: list[dict]) -> dict:
        return {"count": len(items), **{key: sum(row["metrics"][key] for row in items) / len(items)
                                       for key in rows[0]["metrics"]}}

    return {"metrics": average(rows),
            "splits": {key: average([row for row in rows if row["split"] == key])
                       for key in sorted({row["split"] for row in rows})},
            "styles": {key: average([row for row in rows if row["style"] == key])
                       for key in sorted({row["style"] for row in rows})},
            "projects": {key: average([row for row in rows if row["project"] == key])
                         for key in sorted({row["project"] for row in rows})},
            "first_ranks": first_ranks, "top_10": top_10}


class ExperimentEncoder(ONNXEmbedder):
    """Get mean and CLS from one forward pass over exactly the same weights."""

    def __init__(self, model_dir: Path, weights: Path, batch_size: int = 8):
        super().__init__(model_dir=model_dir, batch_size=batch_size)
        self.weights = weights

    def _files(self) -> dict[str, Path]:
        files = super()._files()
        if not self.weights.is_file():
            raise FileNotFoundError(f"Missing local weights: {self.weights}")
        files["onnx/model_quantized.onnx"] = self.weights
        return files

    def both(self, texts: list[str]) -> dict[str, np.ndarray]:
        self._load()
        output = {"mean": [], "cls": []}
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self._tokenizer.encode_batch(batch)
            if any(len(item.ids) > self.max_tokens for item in encoded):
                raise ValueError("Experiment input exceeds model limit; no truncation is allowed")
            inputs = {"input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                      "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                      "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64)}
            feed = {item.name: inputs[item.name] for item in self._session.get_inputs()}
            hidden = self._session.run(["last_hidden_state"], feed)[0]
            mask = inputs["attention_mask"][..., None].astype(hidden.dtype)
            pooled = {"mean": (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1),
                      "cls": hidden[:, 0, :]}
            for name, vectors in pooled.items():
                output[name].extend((vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).tolist())
            if start and start % 128 == 0:
                print(f"    Embedded {min(start + self.batch_size, len(texts))}/{len(texts)}", flush=True)
        return {name: quality.unit_vectors(vectors, len(texts)) for name, vectors in output.items()}


def cached_vectors(encoder: ExperimentEncoder, texts: list[str], cache: Path, identity: dict) -> dict:
    key = hashlib.sha256(json.dumps({"identity": identity, "texts": texts,
                                    "batch_size": encoder.batch_size}, sort_keys=True).encode()).hexdigest()
    path = cache / (key + ".npz")
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            return {name: quality.unit_vectors(saved[name], len(texts)) for name in ("mean", "cls")}
    vectors = encoder.both(texts)
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **vectors)
    temporary.replace(path)
    return vectors


def unique_inputs(corpora: list[dict], queries: list[dict], counter) -> list[str]:
    """Give identical inputs one vector and form deterministic length-sorted batches."""
    texts = {doc["header"] + doc["text"] for corpus in corpora for doc in corpus["documents"]}
    texts.update(query["text"] for query in queries)
    texts.update(quality.QUERY_PREFIX + query["text"] for query in queries)
    return sorted(texts, key=lambda text: (counter(text), text))


def choose(conditions: list[dict]) -> dict:
    """Lock selection using development metrics only; never inspect validation."""
    return min(conditions, key=lambda item: (-item["splits"]["development"]["mrr_at_10"],
                                           -item["splits"]["development"]["hit_at_5"],
                                           -item["splits"]["development"]["hit_at_1"], item["name"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("realistic-search.json"))
    default = Path(os.environ.get("MEMORY_MODEL_CACHE_PATH", str(Path.home() / ".conventions-mcp/models"))) / DEFAULT_MODEL
    parser.add_argument("--model-dir", type=Path, default=default)
    parser.add_argument("--fp32", type=Path, help="Optional local full-precision ONNX of the same BGE model")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/dots-retrieval-experiments")
    parser.add_argument("--budgets", type=int, nargs="+", default=[192, 320, 512])
    parser.add_argument("--boundaries", nargs="+", choices=["lines", "symbols"], default=["lines", "symbols"])
    parser.add_argument("--contexts", nargs="+", choices=["none", "path", "symbols"], default=["none", "path", "symbols"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset, digest = load_dataset(args.dataset)
    script_hash = sha256(Path(__file__))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    counter = lambda text: len(tokenizer.encode(text).ids)
    maximum = json.loads((args.model_dir / "config.json").read_text())["max_position_embeddings"]
    if any(budget > maximum for budget in args.budgets):
        raise ValueError("Chunk budget exceeds model limit")
    for query in dataset["queries"]:
        if counter(quality.QUERY_PREFIX + query["text"]) > maximum:
            raise ValueError(f"Query exceeds model limit: {query['id']}")
    corpora = [build_corpus(dataset, counter, budget=budget, boundary=boundary, context=context)
               for budget in args.budgets for boundary in args.boundaries for context in args.contexts]
    print(f"Frozen {len(dataset['files'])} files, {len(dataset['queries'])} queries; {len(corpora)} complete corpora", flush=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    weights = [("q8", args.model_dir / "onnx/model_quantized.onnx")]
    if args.fp32:
        weights.append(("fp32", args.fp32))
    inputs = unique_inputs(corpora, [], counter)
    question_inputs = unique_inputs([], dataset["queries"], counter)
    input_indices = {text: index for index, text in enumerate(inputs)}
    question_indices = {text: index for index, text in enumerate(question_inputs)}
    print(f"{len(inputs)} unique document inputs and {len(question_inputs)} unique query inputs; "
          "deterministic length-sorted batches, queries encoded individually", flush=True)
    model_files = {name: sha256(args.model_dir / name) for name in ("config.json", "tokenizer.json")}
    results, artifacts = [], {}
    for precision, path in weights:
        identity = {"weights_sha256": sha256(path), **model_files, "encoder_sha256": script_hash,
                    "production_encoder_sha256": sha256(Path(sys.modules[ONNXEmbedder.__module__].__file__))}
        artifacts[precision] = identity
        encoder = ExperimentEncoder(args.model_dir, path, args.batch_size)
        print(f"Embedding {precision} inputs…", flush=True)
        all_vectors = cached_vectors(encoder, inputs, args.cache, identity)
        encoder.batch_size = 1
        question_vectors = cached_vectors(encoder, question_inputs, args.cache, identity)
        encoder._session = None
        del encoder
        gc.collect()
        queries = {}
        for prefix in (False, True):
            texts = [(quality.QUERY_PREFIX if prefix else "") + query["text"] for query in dataset["queries"]]
            indices = [question_indices[text] for text in texts]
            queries[prefix] = {name: vectors[indices] for name, vectors in question_vectors.items()}
        for corpus in corpora:
            start = time.perf_counter()
            print(f"{precision}: {corpus['name']} ({len(corpus['documents'])} chunks)", flush=True)
            texts = [doc["header"] + doc["text"] for doc in corpus["documents"]]
            indices = [input_indices[text] for text in texts]
            document_vectors = {name: vectors[indices] for name, vectors in all_vectors.items()}
            keyword_index = KeywordIndex(texts)
            lexical = [keyword_index.score(query["text"]) for query in corpus["queries"]]
            for pooling in ("mean", "cls"):
                for prefix in (False, True):
                    scores = queries[prefix][pooling] @ document_vectors[pooling].T
                    for method in RANKERS:
                        orders = [rank_candidates(row, *terms, method) for row, terms in zip(scores, lexical)]
                        name = f"{precision}/{corpus['name']}/{pooling}/{'prefix' if prefix else 'plain'}/{method}"
                        results.append({"name": name, "precision": precision, "corpus": corpus["name"],
                                        "pooling": pooling, "prefix": prefix, "ranker": method,
                                        **measure(corpus, orders)})
            print(f"  {time.perf_counter() - start:.1f}s; {len(results)} conditions evaluated", flush=True)
        del all_vectors
        gc.collect()
    baseline_name = "q8/lines-512-none/mean/plain/semantic"
    baseline = next((item["name"] for item in results if item["name"] == baseline_name), None)
    selected = choose(results)["name"]
    selections = {ranker: choose([item for item in results if item["ranker"] == ranker])["name"]
                  for ranker in RANKERS}
    report = {"schema_version": 1, "dataset_sha256": digest, "benchmark_sha256": script_hash,
              "quality_helpers_sha256": sha256(Path(quality.__file__)), "model": DEFAULT_MODEL,
              "execution": {"provider": "CPUExecutionProvider", "batch_size": args.batch_size,
                            "intra_op_threads": 2, "inter_op_threads": 1,
                            "unique_document_inputs": len(inputs), "unique_query_inputs": len(question_inputs),
                            "query_batch_size": 1,
                            "batch_order": "Unique document inputs sorted by token length, then text; queries encoded individually; shared across conditions",
                            "versions": {name: importlib.metadata.version(name)
                                         for name in ("onnxruntime", "tokenizers", "numpy")},
                            "timings": "Informational; cached vectors and uncontrolled host load preclude speed comparisons"},
              "artifacts": artifacts, "files": len(dataset["files"]), "query_count": len(dataset["queries"]),
              "queries": dataset["queries"], "baseline": baseline, "selected": selected,
              "selection_policy": "Development MRR@10, then Hit@5, then Hit@1, then name. Entire target files held out.",
              "selected_by_ranker": selections, "ranking": "Exact full-corpus scoring; no ANN or cutoff. "
                  "BM25 k1=1.2 b=0.75; RRF k=60; hybrid weight is lexical share after per-query normalization. "
                  "words-first sorts IDF-weighted word coverage before cosine. identifiers-first sorts explicit "
                  "code lexeme or quoted phrase matches before RRF and cosine.",
              "limits": "Hand-written internal Python pilot; anchor judgments are not exhaustive. "
                  "File-level holdout is internal validation, not an independent external test. "
                  "Many configurations share embeddings or rankings; condition count is not independent evidence.",
              "corpora": [{**{key: value for key, value in corpus.items() if key not in {"queries", "documents"}},
                           "documents": [{key: value for key, value in doc.items() if key != "text"}
                                         for doc in corpus["documents"]],
                           "relevance": [query["relevant"] for query in corpus["queries"]]}
                          for corpus in corpora], "conditions": results}
    # Compact per-condition arrays retain every query rank and top ten IDs without
    # repeating source text hundreds of thousands of times.
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    print(f"Selected on development only: {selected}", flush=True)
    print(f"Saved {len(results)} conditions to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
