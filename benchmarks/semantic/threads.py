"""Compare cold-process ONNX worker counts on frozen query and source inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


def worker(payload: dict) -> dict:
    import psutil
    from slipagent.semantic import ONNXEmbedder

    start = time.perf_counter()
    encoder = ONNXEmbedder(model_dir=payload["model_dir"], threads=payload["threads"],
                           batch_size=payload.get("batch_size", 8))
    encoder.dim()
    load = time.perf_counter() - start
    start = time.perf_counter()
    vectors = encoder.embed_batch(payload["texts"])
    inference = time.perf_counter() - start
    process = psutil.Process()
    return {"pid": os.getpid(), "load_seconds": load, "inference_seconds": inference,
            "total_seconds": load + inference, "rss_bytes": process.memory_info().rss,
            "cpu_seconds": sum(process.cpu_times()[:2]),
            "vector_sha256": hashlib.sha256(json.dumps(vectors).encode()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--batch-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return 0
    if args.model_dir is None or args.output is None or args.repeats < 1:
        parser.error("--model-dir, --output, and positive --repeats are required")

    import psutil
    from tokenizers import Tokenizer
    from slipagent.semantic.indexer import chunk_text
    from slipagent.semantic.retrieval import source_header

    dataset_path = Path(__file__).with_name("realistic-search.json")
    dataset = json.loads(dataset_path.read_text())
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    counter = lambda text: len(tokenizer.encode(text).ids)
    texts = []
    for file in dataset["files"]:
        header = source_header(file["path"])
        reserve = counter(header) - counter("") + 8
        chunks = chunk_text(file["path"], file["text"], token_count=counter, max_tokens=512 - reserve)
        texts.extend(header + chunk.text for chunk in chunks)
    texts.sort(key=lambda text: (-counter(text), text))
    query = next(item["text"] for item in dataset["queries"] if item["style"] == "task")
    mixed = [texts[index] for index in (0, 10, 20, 30, 40, 50, 70, len(texts) - 1)]
    workloads = {"query": [query], "batch-8": texts[:8], "mixed-8": mixed}
    if args.batch_only:
        del workloads["query"]
    physical, logical = psutil.cpu_count(logical=False), psutil.cpu_count()
    counts = sorted(set(args.threads)) if args.threads else sorted({1, 2, physical or 2, logical or 2})
    schedule = [(repeat, mode, count, batch_size) for repeat in range(args.repeats)
                for mode in workloads for count in counts for batch_size in args.batch_sizes]
    random.Random(20261011).shuffle(schedule)
    results = []
    print(f"Physical cores: {physical}; logical CPUs: {logical}; thread counts: {counts}", flush=True)
    for repeat, mode, count, batch_size in schedule:
        payload = {"model_dir": str(args.model_dir), "threads": count, "texts": workloads[mode],
                   "batch_size": batch_size}
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
                                input=json.dumps(payload), capture_output=True, text=True, check=True,
                                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
        row = {"repeat": repeat, "workload": mode, "threads": count, "batch_size": batch_size,
               "wall_seconds": time.perf_counter() - start, **json.loads(result.stdout)}
        results.append(row)
        print(f"{mode}, {count} threads, batch size {batch_size}: load {row['load_seconds']:.3f}s; "
              f"first inference {row['inference_seconds']:.3f}s; RSS {row['rss_bytes'] / 2**20:.0f} MiB", flush=True)
    aggregates = []
    for mode in workloads:
        for count in counts:
            for batch_size in args.batch_sizes:
                rows = [row for row in results if row["workload"] == mode and row["threads"] == count
                        and row["batch_size"] == batch_size]
                aggregates.append({"workload": mode, "threads": count, "batch_size": batch_size,
                                   **{name: statistics.median(row[name] for row in rows) for name in
                                      ("load_seconds", "inference_seconds", "total_seconds", "wall_seconds", "rss_bytes")}})
    report = {"physical_cores": physical, "logical_cpus": logical,
              "cpu_affinity": psutil.Process().cpu_affinity(), "repeats": args.repeats,
              "cold_contract": "Fresh subprocess and ONNX session per measurement; no inference warmup. "
                  "OS filesystem caches are retained; this is model/process cold, not cold disk.",
              "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256((args.model_dir / "onnx/model_quantized.onnx").read_bytes()).hexdigest(),
              "input_tokens": {name: [counter(text) for text in items] for name, items in workloads.items()},
              "inputs": workloads, "runs": results, "medians": aggregates}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(aggregates, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
