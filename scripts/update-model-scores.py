"""Calculate the curated text-coding scores from raw published results, without network I/O.

Run with --help. This maintenance command fits every model and benchmark offset from
model-score-measurements.json and rewrites the score table and calibration; provider
catalogs and pricing independently determine automatic eligibility.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import stat
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = Path(__file__).with_name("model-score-calibration.json")
DEFAULT_MEASUREMENTS = Path(__file__).with_name("model-score-measurements.json")
DEFAULT_TABLE = ROOT / "src/slipagent/modelscores.py"


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def load_calibration(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported calibration format")
    clip = number(data.get("probability_clip"), "probability_clip")
    delta = number(data.get("huber_delta"), "huber_delta")
    if not 0 < clip < 0.5 or delta <= 0:
        raise ValueError("Invalid calibration bounds")
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, dict) or data.get("anchor") not in benchmarks:
        raise ValueError("Calibration needs benchmarks and an anchor")
    for name, row in benchmarks.items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid calibration for {name}")
        number(row.get("offset"), f"{name} offset")
        minimum = number(row.get("min_score", 0), f"{name} min_score")
        maximum = number(row.get("max_score", 100), f"{name} max_score")
        if not 0 <= minimum < maximum:
            raise ValueError(f"Invalid score range for {name}")
        if number(row.get("overlap_models"), f"{name} overlap_models") < 2:
            raise ValueError(f"{name} needs at least two models with shared benchmark results")
    if benchmarks[data["anchor"]]["offset"] != 0:
        raise ValueError("The reference benchmark must have zero offset")
    return data


def sigmoid(value: float) -> float:
    """Stable logistic function; also handles extreme finite abilities."""
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def read_measurements(path: Path) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    """Raw published results per model: the scales used and validated model records."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {"reviewed", "scales", "models"}:
        raise ValueError("Expected an object with reviewed, scales, and models")
    scales: dict[str, tuple[float, float]] = {}
    for name, row in (raw.get("scales") or {}).items():
        if not isinstance(row, dict) or set(row) - {"min_score", "max_score"}:
            raise ValueError(f"Invalid scale for {name}")
        minimum, maximum = number(row.get("min_score"), f"{name} min_score"), number(row.get("max_score"), f"{name} max_score")
        if not 0 <= minimum < maximum:
            raise ValueError(f"Invalid score range for {name}")
        scales[name] = (minimum, maximum)
    rows = raw.get("models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Provide at least one model")
    records, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"model", "benchmarks", "aliases", "sources", "notes", "reference"}:
            raise ValueError("Expected model, benchmarks, sources, and optional aliases/notes/reference")
        model = identity(row.get("model"))
        if model in seen:
            raise ValueError(f"Repeated model {model}")
        seen.add(model)
        sources = row.get("sources")
        if not isinstance(sources, list) or not sources or any(
                not isinstance(source, str) or not source.startswith("https://") for source in sources):
            raise ValueError(f"{model}: provide HTTPS source links for manual verification")
        benchmarks = row.get("benchmarks")
        if not isinstance(benchmarks, dict) or not benchmarks:
            raise ValueError(f"{model}: provide at least one benchmark result")
        for name, value in benchmarks.items():
            minimum, maximum = scales.get(name, (0.0, 100.0))
            if not minimum <= number(value, f"{model} {name}") <= maximum:
                raise ValueError(f"{model}: {name} must be between {minimum:g} and {maximum:g}")
        aliases = row.get("aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(f"{model}: aliases must be a list")
        reference = row.get("reference", False)
        if not isinstance(reference, bool):
            raise ValueError(f"{model}: reference must be true or false")
        records.append({"model": model, "benchmarks": benchmarks, "reference": reference,
                        "aliases": [identity(alias) for alias in aliases]})
    return scales, records


def observations(records: list[dict[str, Any]], scales: dict[str, tuple[float, float]], clip: float) -> list[tuple[str, str, float]]:
    """Each published result as (model, benchmark, log-odds)."""
    result = []
    for record in records:
        for name, value in record["benchmarks"].items():
            probability = min(1 - clip, max(clip, value / scales.get(name, (0.0, 100.0))[1]))
            result.append((record["model"], name, math.log(probability / (1 - probability))))
    return result


def check_connected(rows: list[tuple[str, str, float]], anchor: str) -> None:
    """Every benchmark needs a chain of shared models to the anchor, and two models of its own."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    benchmarks_of: dict[str, set[str]] = defaultdict(set)
    for model, benchmark, _ in rows:
        benchmarks_of[model].add(benchmark)
        parent[find("m:" + model)] = find("b:" + benchmark)
    benchmarks = {benchmark for _, benchmark, _ in rows}
    if anchor not in benchmarks:
        raise ValueError(f"No model has a {anchor} result to anchor the scale")
    for benchmark in sorted(benchmarks):
        if find("b:" + benchmark) != find("b:" + anchor):
            raise ValueError(f"{benchmark} shares no chain of models with {anchor}; add reference models measured on it "
                             "and on a calibrated benchmark")
        overlap = sum(1 for names in benchmarks_of.values() if benchmark in names and len(names) > 1)
        if overlap < 2:
            raise ValueError(f"{benchmark} needs at least two models with shared benchmark results; has {overlap}")


def solve(rows: list[tuple[str, str, float]], anchor: str, delta: float,
          start: tuple[dict[str, float], dict[str, float]] | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Huber fit of logit(result) = ability - offset, with the anchor's offset fixed at zero."""
    check_connected(rows, anchor)
    ability = {model: 0.0 for model, _, _ in rows}
    offset = {benchmark: 0.0 for _, benchmark, _ in rows}
    if start is not None:
        ability.update({key: value for key, value in start[0].items() if key in ability})
        offset.update({key: value for key, value in start[1].items() if key in offset})
    offset[anchor] = 0.0

    def weights() -> list[float]:
        return [min(1.0, delta / max(1e-12, abs(y - (ability[model] - offset[benchmark])))) for model, benchmark, y in rows]

    for _ in range(100000):
        moved = 0.0
        total, weight = defaultdict(float), defaultdict(float)
        for (model, benchmark, y), w in zip(rows, weights()):
            total[model] += w * (y + offset[benchmark])
            weight[model] += w
        for model in ability:
            updated = total[model] / weight[model]
            moved, ability[model] = max(moved, abs(updated - ability[model])), updated
        total, weight = defaultdict(float), defaultdict(float)
        for (model, benchmark, y), w in zip(rows, weights()):
            total[benchmark] += w * (ability[model] - y)
            weight[benchmark] += w
        for benchmark in offset:
            if benchmark == anchor:
                continue
            updated = total[benchmark] / weight[benchmark]
            moved, offset[benchmark] = max(moved, abs(updated - offset[benchmark])), updated
        if moved < 1e-10:
            return ability, offset
    raise ValueError("Benchmark calibration did not converge")


def leave_one_out(rows: list[tuple[str, str, float]], fitted: tuple[dict[str, float], dict[str, float]],
                  anchor: str, delta: float) -> list[float]:
    """Absolute errors, in percentage points, predicting each result of a multi-result model without it."""
    counts: dict[str, int] = defaultdict(int)
    for model, _, _ in rows:
        counts[model] += 1
    errors = []
    for index, (model, benchmark, y) in enumerate(rows):
        if counts[model] < 2:
            continue
        rest = rows[:index] + rows[index + 1:]
        try:
            ability, offset = solve(rest, anchor, delta, fitted)
        except ValueError:
            continue
        if benchmark in offset:
            errors.append(abs(100 * (sigmoid(ability[model] - offset[benchmark]) - sigmoid(y))))
    return errors


def plain(value: float) -> float | int:
    """Whole numbers stay integers in saved JSON, as in `1 and 10`."""
    return int(value) if value == int(value) else value


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def fit_scores(records: list[dict[str, Any]], scales: dict[str, tuple[float, float]], base: dict[str, Any],
               reviewed: str, *, validate: bool = True) -> tuple[dict[str, float], dict[str, Any]]:
    """Scores for every measured model, and the calibration that produced them."""
    clip, delta, anchor = base["probability_clip"], base["huber_delta"], base["anchor"]
    rows = observations(records, scales, clip)
    fitted = solve(rows, anchor, delta)
    ability, offset = fitted
    scores = {model: round(100 * sigmoid(value), 1) for model, value in ability.items()}
    benchmarks_of: dict[str, set[str]] = defaultdict(set)
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for name, value in record["benchmarks"].items():
            benchmarks_of[record["model"]].add(name)
            values[name].append(value)
    calibration = {key: base[key] for key in ("schema_version", "scale", "anchor", "probability_clip", "huber_delta",
                                                "method", "limits")}
    calibration["reviewed"] = reviewed
    calibration["model_count"] = len(ability)
    calibration["reference_model_count"] = sum(1 for record in records if record["reference"])
    calibration["benchmarks"] = {}
    for name in sorted(offset):
        row: dict[str, Any] = {"offset": offset[name]}
        if name in scales:
            row.update(min_score=plain(scales[name][0]), max_score=plain(scales[name][1]))
        row["observations"] = len(values[name])
        row["overlap_models"] = sum(1 for names in benchmarks_of.values() if name in names and len(names) > 1)
        row["observed_score_range" if name in scales else "observed_percent_range"] = [min(values[name]), max(values[name])]
        calibration["benchmarks"][name] = row
    errors = leave_one_out(rows, fitted, anchor, delta) if validate else []
    if errors:
        calibration["validation"] = {"count": len(errors), "mean_absolute_error": statistics.fmean(errors),
                                     "median_absolute_error": statistics.median(errors),
                                     "p90_absolute_error": percentile(errors, 0.9),
                                     "maximum_absolute_error": max(errors)}
    return scores, calibration


def identity(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./:-]+", value):
        raise ValueError("Use a canonical vendor/model ID without whitespace")
    return value.removesuffix(":free")


def render_table(original: str, updates: list[tuple[str, float, list[str]]]) -> str:
    """Replace only literal score/alias assignments, without importing the app."""
    tree = ast.parse(original)
    assignments: dict[str, ast.Assign | ast.AnnAssign] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id in {"SCORES", "MODEL_ALIASES"}:
            assignments[target.id] = node
    if set(assignments) != {"SCORES", "MODEL_ALIASES"}:
        raise ValueError("Score table is missing its literal SCORES or MODEL_ALIASES assignment")
    score_value, alias_value = assignments["SCORES"].value, assignments["MODEL_ALIASES"].value
    if score_value is None or alias_value is None:
        raise ValueError("Score and alias tables need literal values")
    scores = ast.literal_eval(score_value)
    aliases = ast.literal_eval(alias_value)
    if not isinstance(scores, dict) or not isinstance(aliases, dict):
        raise ValueError("Score and alias tables must be dictionaries")
    for model, score, names in updates:
        if model in aliases:
            raise ValueError(f"{model} is an alias; update its canonical model {aliases[model]}")
        scores[model] = score
        for name in names:
            if name == model:
                continue
            if name in scores or aliases.get(name, model) != model:
                raise ValueError(f"Alias {name} already belongs to a different model")
            aliases[name] = model
    # Check all updates together, including aliases introduced earlier in a batch.
    if set(scores) & set(aliases):
        raise ValueError("A canonical model cannot also be an alias")
    ordered = sorted(scores.items(), key=lambda item: (item[1] is None, -(item[1] or 0), item[0]))
    replacements = {
        "SCORES": "SCORES: dict[str, float | None] = {\n" + "".join(
            f"    {json.dumps(model)}: {score!r},\n" for model, score in ordered) + "}\n",
        "MODEL_ALIASES": "MODEL_ALIASES = {\n" + "".join(
            f"    {json.dumps(name)}: {json.dumps(model)},\n" for name, model in sorted(aliases.items())) + "}\n",
    }
    lines = original.splitlines(keepends=True)
    for name, node in sorted(assignments.items(), key=lambda item: item[1].lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [replacements[name]]
    result = "".join(lines)
    ast.parse(result)
    return result


def save_table(path: Path, original: str, updated: str) -> None:
    """Publish a complete table while preserving permissions and concurrent edits."""
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        if path.read_text(encoding="utf-8") != original:
            raise ValueError("Score table changed while calculating; retry with the new contents")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def refit(args: argparse.Namespace, base: dict[str, Any]) -> int:
    """Recalculate scores and offsets from raw measurements."""
    scales, records = read_measurements(args.measurements)
    reviewed = args.reviewed or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    scores, calibration = fit_scores(records, scales, base, reviewed)
    updates = [(record["model"], scores[record["model"]], record["aliases"]) for record in records if not record["reference"]]
    original = args.table.read_text(encoding="utf-8")
    old = ast.literal_eval(next(node.value for node in ast.parse(original).body
                                if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "SCORES"))
    measured = {model for model, _, _ in updates}
    unmeasured = sorted(model for model, score in old.items() if score is not None and model not in measured)
    if unmeasured:
        raise ValueError("Scored without measurements: " + ", ".join(unmeasured)
                         + ". Add their published results to the measurements file, or remove the entries.")
    updated = render_table(original, updates)
    for model, score, _ in sorted(updates, key=lambda item: -item[1]):
        before = old.get(model)
        print(f"{model}: {'new' if model not in old else 'unscored' if before is None else f'{before:.1f}'} -> {score:.1f}")
    print("Benchmark offsets (SWE Verified = 0): " + ", ".join(
        f"{name} {row['offset']:+.2f}" for name, row in calibration["benchmarks"].items()))
    if "validation" in calibration:
        validation = calibration["validation"]
        print(f"Leave-one-out error over {validation['count']} results: mean {validation['mean_absolute_error']:.1f}, "
              f"median {validation['median_absolute_error']:.1f}, p90 {validation['p90_absolute_error']:.1f}, "
              f"max {validation['maximum_absolute_error']:.1f} points")
    if not args.dry_run:
        previous = args.calibration.read_text(encoding="utf-8")
        if updated != original:
            save_table(args.table, original, updated)
        save_table(args.calibration, previous, json.dumps(calibration, indent=2) + "\n")
    print("Preview only." if args.dry_run else "Score table and calibration updated.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, nargs="?", default=DEFAULT_MEASUREMENTS,
                        help="Raw published results (default: model-score-measurements.json)")
    parser.add_argument("--reviewed", default=None, help="Review date recorded in the calibration (default: today, UTC)")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Curated Python score table to update")
    parser.add_argument("--dry-run", action="store_true", help="Calculate and validate without changing any file")
    parser.add_argument("--list-benchmarks", action="store_true", help="Print calibrated benchmark names and exit")
    args = parser.parse_args(argv)
    try:
        calibration = load_calibration(args.calibration)
        if args.list_benchmarks:
            print("\n".join(calibration["benchmarks"]))
            return 0
        return refit(args, calibration)
    except (OSError, ValueError, SyntaxError, TypeError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
