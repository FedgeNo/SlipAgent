"""Render development-selected retrieval comparisons and held-out outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def selection(items: list[dict]) -> dict:
    return min(items, key=lambda item: (-item["splits"]["development"]["mrr_at_10"],
                                      -item["splits"]["development"]["hit_at_5"],
                                      -item["splits"]["development"]["hit_at_1"], item["name"]))


def table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = ["| Configuration | Dev First | Dev Top Five | Dev MRR | Validation First | Validation Top Five | Validation MRR |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, condition in rows:
        development, validation = (condition["splits"][key] for key in ("development", "validation"))
        cells = [f"{values['hit_at_1']:.1%} | {values['hit_at_5']:.1%} | {values['mrr_at_10']:.3f}"
                 for values in (development, validation)]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    return lines


def summarize(report: dict) -> str:
    conditions = report["conditions"]
    by_name = {item["name"]: item for item in conditions}
    baseline = by_name[report["baseline"]]
    winner = by_name[report["selected"]]
    splits = Counter(query["split"] for query in report["queries"])
    styles = Counter(query["style"] for query in report["queries"])
    rows = [("Original mean/plain semantic baseline", baseline), ("Selected combination", winner)]
    groups = {"Best q8 combination": [item for item in conditions if item["precision"] == "q8"],
              "Semantic only": [item for item in conditions if item["ranker"] == "semantic"],
              "Keywords only (BM25)": [item for item in conditions if item["ranker"] == "bm25"],
              "Semantic + keywords (RRF)": [item for item in conditions if item["ranker"] == "rrf"],
              "Weighted semantic + keywords": [item for item in conditions if item["ranker"].startswith("hybrid-")],
              "Exact words before semantic": [item for item in conditions if item["ranker"] == "words-first"],
              "Exact identifiers before fusion": [item for item in conditions if item["ranker"] == "identifiers-first"]}
    rows.extend((label, selection(items)) for label, items in groups.items())
    lines = ["# Retrieval Experiment Results", "",
             f"Measured **{len(conditions):,} conditions**, **{report['query_count']} hand-written queries**, "
             f"**{len(set(query['family'] for query in report['queries']))} intents**, and **{report['files']} complete source files**.", "",
             f"Development: {splits['development']} queries. Internal validation: {splits['validation']} queries from entirely held-out target files. "
             f"Query styles: {', '.join(f'{count} {style}' for style, count in sorted(styles.items()))}.", "",
             "Every corpus covers every source character and complete answer anchor. All embedding inputs fit in full. "
             "The same BGE model is evaluated as quantized and full-precision ONNX; no second model or reranker is used.", "",
             "## Comparisons", "",
             "Each row below is chosen using development MRR@10, then Hit@5 and Hit@1. Validation scores do not select configurations. "
             "Rows optimize several factors and therefore are not isolated estimates of one factor's effect. "
             "The controlled comparisons below hold other settings fixed.", "", *table(rows), "",
             "## Findings", ""]
    indices = [i for i, query in enumerate(report["queries"]) if query["split"] == "validation"]
    counts = {name: {cutoff: sum(condition["first_ranks"][i] <= cutoff for i in indices)
                     for cutoff in (1, 5)} for name, condition in (("baseline", baseline), ("selected", winner))}
    improved = sum(winner["first_ranks"][i] < baseline["first_ranks"][i] for i in indices)
    worsened = sum(winner["first_ranks"][i] > baseline["first_ranks"][i] for i in indices)
    lines.extend([f"The development-selected combination finds an answer first for {counts['selected'][1]} "
                  f"of {len(indices)} validation queries versus {counts['baseline'][1]} for the baseline, "
                  f"and within five for {counts['selected'][5]} versus {counts['baseline'][5]}. "
                  f"Its first-answer rank improves on {improved} queries and worsens on {worsened}.", "",
                  f"The development-selected configuration is `{winner['name']}`. "
                  "The controlled tables below expose other settings' validation outcomes without "
                  "retroactively using those scores to select a different configuration. "
                  "See the [recorded interpretation](historical-notes.md#recorded-retrieval-findings) for the original run's findings.", "",
                  "The experimental AST splitter keeps interstitial source as separate chunks, including "
                  "whitespace-only segments. Its results describe that specific implementation and do not "
                  "rule out improved function chunking that attaches gaps to neighboring code while retaining coverage.", "",
                  "BM25 does not use neural vectors: precision and pooling labels in its condition name "
                  "are tied matrix metadata, not requirements for keyword-only search.", "",
                  "## Selected Settings", ""])
    for label, condition in rows:
        lines.append(f"- **{label}:** `{condition['name']}`")
    lines.extend(["", "## Controlled Comparisons", "",
                  "These comparisons vary one setting around the development-selected combination. "
                  "Its corpus name encodes boundary, total token budget, and context.", ""])
    corpora = {item["name"]: item for item in report["corpora"]}
    selected_corpus = corpora[winner["corpus"]]
    settings = {"precision": winner["precision"], "pooling": winner["pooling"], "prefix": winner["prefix"],
                "ranker": winner["ranker"], **{key: selected_corpus[key] for key in ("boundary", "budget", "context")}}
    for axis in ("ranker", "boundary", "budget", "context", "pooling", "prefix", "precision"):
        neighbors = []
        for item in conditions:
            corpus = corpora[item["corpus"]]
            candidate = {key: item[key] for key in ("precision", "pooling", "prefix", "ranker")}
            candidate.update({key: corpus[key] for key in ("boundary", "budget", "context")})
            if all(candidate[key] == value for key, value in settings.items() if key != axis):
                neighbors.append((str(candidate[axis]), item))
        lines.extend([f"### {axis.title()}", "", *table(sorted(neighbors, key=lambda row: row[0])), ""])
    lines.extend(["## Validation By Query Style", "",
                  "| Style | Queries | Baseline First | Selected First | Baseline Top Five | Selected Top Five |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for style in sorted(styles):
        indices = [i for i, query in enumerate(report["queries"]) if query["split"] == "validation" and query["style"] == style]
        rates = [sum(condition["first_ranks"][index] <= cutoff for index in indices) / len(indices)
                 for cutoff in (1, 5) for condition in (baseline, winner)]
        lines.append(f"| {style} | {len(indices)} | " + " | ".join(f"{rate:.1%}" for rate in rates) + " |")
    lines.extend(["", "## Paired Validation Examples", "",
                  "Ranks below refer to complete labeled source anchors. Improvement and regression examples are shown; "
                  "the JSON retains every query's first-answer rank and top ten chunk indices.", ""])
    indices = [i for i, q in enumerate(report["queries"]) if q["split"] == "validation"]
    changed = [i for i in indices if baseline["first_ranks"][i] != winner["first_ranks"][i]]
    improvements = sorted([i for i in changed if winner["first_ranks"][i] < baseline["first_ranks"][i]],
                          key=lambda i: baseline["first_ranks"][i] - winner["first_ranks"][i], reverse=True)[:6]
    regressions = sorted([i for i in changed if winner["first_ranks"][i] > baseline["first_ranks"][i]],
                         key=lambda i: winner["first_ranks"][i] - baseline["first_ranks"][i], reverse=True)[:6]
    for label, examples in (("Improved", improvements), ("Regressed", regressions)):
        lines.extend([f"### {label}", ""])
        for index in examples:
            query = report["queries"][index]
            lines.append(f"- {query['text']} — rank {baseline['first_ranks'][index]} → {winner['first_ranks'][index]} "
                         f"(`{query['targets'][0]['path']}:{query['targets'][0]['line']}`).")
        if not examples:
            lines.append("No changed ranks in this direction.")
        lines.append("")
    lines.extend(["## Interpretation Limits", "",
                  "This is an internal Python pilot with related query variants and shared source files. "
                  "The three variants of an intent are not independent observations. Labels identify intended "
                  "implementation anchors, not every useful helper or test. The set contains answerable queries; "
                  "it does not measure refusing nonexistent features or confidence thresholds. "
                  "The initial code snapshots and query wording were frozen before this run. "
                  "The validation set remains internal and should not be repeatedly used to tune settings.", "",
                  "Retrieval uses exact cosine and full-corpus keyword scoring. It does not measure HNSW "
                  "candidate recall, the application's distance cutoff, or large-repository throughput. "
                  "Many matrix conditions share vectors or identical keyword rankings. Timings are informational "
                  "because host load is uncontrolled and cached embeddings are reused. Production defaults are unchanged.", "",
                  "## Artifacts", "",
                  "- [Reviewed query cases](realistic-cases.json)",
                  "- [Frozen source and labels](realistic-search.json)",
                  "- [Complete machine-readable results](retrieval-results.json)",
                  "- [Runner](retrieval-experiments.py)",
                  "", f"Dataset SHA-256: `{report['dataset_sha256']}`.", "",
                  f"Runner SHA-256: `{report['benchmark_sha256']}`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(summarize(json.loads(args.report.read_text())), encoding="utf-8")


if __name__ == "__main__":
    main()
