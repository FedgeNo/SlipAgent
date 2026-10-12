"""Validate the production SQLite/HNSW hybrid on the frozen source/query corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile
import time

from slipagent.semantic import ONNXEmbedder, Searcher, Store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    names = ("src/slipagent/semantic/embed.py", "src/slipagent/semantic/search.py", "src/slipagent/semantic/storage.py",
             "src/slipagent/semantic/retrieval.py", "src/slipagent/semantic/cpu.py", "src/slipagent/semantic/indexer.py")
    project = Path(__file__).resolve().parents[2]
    fingerprints = {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in names}
    spec = importlib.util.spec_from_file_location("production_oracle", Path(__file__).with_name("retrieval-experiments.py"))
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    data, digest = experiment.load_dataset(Path(__file__).with_name("realistic-search.json"))
    original = json.loads(Path(__file__).with_name("retrieval-results.json").read_text())
    selected = next(row for row in original["conditions"] if row["name"] == original["selected"])
    encoder = ONNXEmbedder(model_dir=args.model_dir,
                           **({"batch_size": args.batch_size} if args.batch_size is not None else {}))
    corpus = experiment.build_corpus(data, encoder.token_count, budget=512, boundary="lines", context="path")
    document_indices = {doc["id"]: index for index, doc in enumerate(corpus["documents"])}
    sources = {file["path"]: file["text"] for file in data["files"]}
    results = []
    with tempfile.TemporaryDirectory(prefix="dots-production-retrieval-") as directory:
        root = Path(directory) / "workspace"
        root.mkdir()
        for file in data["files"]:
            path = root / file["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(file["text"].encode())
        with Store(str(Path(directory) / "index.db"), encoder.dim(), threads=encoder.threads) as store:
            searcher = Searcher(store, encoder)
            start = time.perf_counter()
            count = searcher.index_path(str(root))
            index_seconds = time.perf_counter() - start
            rows = store.conn.execute("SELECT * FROM chunks ORDER BY path,id").fetchall()
            keys, cursors = {}, {}
            for row in rows:
                path = Path(row["path"]).relative_to(root).as_posix()
                position = sources[path].find(row["text"], cursors.get(path, 0))
                if position < 0:
                    raise ValueError("Stored source chunk differs from the frozen source")
                cursors[path] = position + 1
                identity = f"{path}:{position}-{position + len(row['text'])}"
                keys[row["id"]] = document_indices[identity]
            if len(keys) != len(document_indices) or len(set(keys.values())) != len(document_indices):
                raise ValueError("Production chunk coverage differs from the measured recipe")
            print(f"Indexed {count} complete chunks from {len(sources)} files in {index_seconds:.2f}s", flush=True)
            orders = []
            for number, query in enumerate(data["queries"]):
                start = time.perf_counter()
                rows = searcher.search(query["text"], limit=10)
                latency = time.perf_counter() - start
                top = [keys[row["id"]] for row in rows]
                # Only top ten are measured. Appended IDs are unranked and cannot
                # change any of the benchmark's metrics truncated at ten.
                orders.append(top + [index for index in range(len(keys)) if index not in top])
                results.append({"id": query["id"], "top_10": top, "seconds": latency})
                if (number + 1) % 50 == 0:
                    print(f"Checked {number + 1}/{len(data['queries'])} queries", flush=True)
            measured = experiment.measure(corpus, orders)
            measured.pop("first_ranks")
    if fingerprints != {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in names}:
        raise ValueError("Production source changed during evaluation; rerun against one source snapshot")
    report = {"dataset_sha256": digest, "selected_recipe": original["selected"], "threads": encoder.threads,
              "batch_size": encoder.batch_size,
              "model_identity": encoder.identity, "chunks": count, "files": len(sources),
              "source_sha256": fingerprints,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("onnxruntime", "usearch", "numpy", "huggingface-hub", "tokenizers")},
              "original_selected_splits": selected["splits"], "production": measured,
              "queries": results, "index_seconds": index_seconds,
              "top_10_orders_identical": sum(row["top_10"] == expected for row, expected in
                                            zip(results, selected["top_10"])),
              "limits": "Frozen internal Python pilot. Production candidate union is approximate beyond 256 chunks; "
                  "this corpus fits in the semantic shortlist. Results do not establish large-corpus ANN recall. "
                  "Original matrix embeddings used shared length-sorted batches; production embeds per file."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"production": measured["splits"], "original": selected["splits"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
