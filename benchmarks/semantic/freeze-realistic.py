"""Freeze reviewed code-search cases and complete local source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def freeze(catalog: dict, roots: dict[str, Path]) -> dict:
    paths = sorted({case["path"] for case in catalog["cases"]} | {
        "Dots/codeindex/__init__.py", "Dots/tests/test-quality.py",
    })
    files = []
    for path in paths:
        project, relative = path.split("/", 1)
        source = roots[project] / relative
        raw = source.read_bytes()
        text = raw.decode("utf-8")
        if source.read_bytes() != raw:
            raise ValueError(f"Source changed during capture; retry: {path}")
        files.append({"path": path, "text": text, "sha256": hashlib.sha256(raw).hexdigest()})
    sources = {item["path"]: item["text"] for item in files}
    queries = []
    for case in catalog["cases"]:
        source = sources[case["path"]]
        anchor = case["anchor"]
        if not anchor:
            raise ValueError(f"Empty source anchor: {case['id']}")
        positions = []
        start = 0
        while (found := source.find(anchor, start)) >= 0:
            positions.append(found)
            start = found + len(anchor)
        if not positions or (len(positions) != 1 and "occurrence" not in case):
            raise ValueError(f"Need an unambiguous anchor for {case['id']}: {len(positions)} matches")
        position = positions[case.get("occurrence", 1) - 1]
        target = {"path": case["path"], "anchor": anchor, "start": position,
                  "end": position + len(anchor), "line": source.count("\n", 0, position) + 1}
        if len(case["queries"]) != len(catalog["query_styles"]):
            raise ValueError(f"Missing query styles: {case['id']}")
        for style, text in zip(catalog["query_styles"], case["queries"]):
            queries.append({"id": f"{case['id']}/{style}", "family": case["id"],
                            "style": style, "text": text, "targets": [target],
                            "split": "validation" if case["path"] in catalog["validation_files"] else "development"})
    return {"schema_version": 2, "name": "Realistic Dots and SlipAgent code search",
            "description": "Hand-written task, question, and lookup queries. Complete frozen source; "
                           "anchor judgments identify intended implementations, not exhaustive relevance.",
            "split_policy": "Entire target files held out; all variants of an intent stay together. "
                            "This is an internal validation set, not an independent external benchmark.",
            "files": files, "queries": queries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dots", type=Path, required=True)
    parser.add_argument("--slipagent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path(__file__).with_name("realistic-cases.json")
    raw = path.read_bytes()
    result = freeze(json.loads(raw), {"Dots": args.dots, "SlipAgent": args.slipagent})
    result["catalog_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen {len(result['files'])} files and {len(result['queries'])} queries")


if __name__ == "__main__":
    main()
