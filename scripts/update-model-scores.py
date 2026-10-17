"""Add published text-coding results using saved calibration, without network I/O.

Run with --help. This maintenance command updates the curated score table;
provider catalogs and pricing independently determine automatic eligibility.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = Path(__file__).with_name("model-score-calibration.json")
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


def estimate(results: object, calibration: dict[str, Any]) -> float:
    """Fit one model's ability with the benchmark offsets held fixed."""
    if not isinstance(results, dict) or not results:
        raise ValueError("Provide at least one calibrated text-coding benchmark")
    values = []
    clip = calibration["probability_clip"]
    for name, raw in results.items():
        if name not in calibration["benchmarks"]:
            raise ValueError(f"Uncalibrated benchmark {name!r}; add shared-model evidence before using it")
        benchmark = calibration["benchmarks"][name]
        score = number(raw, name)
        minimum, maximum = benchmark.get("min_score", 0), benchmark.get("max_score", 100)
        if not minimum <= score <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        probability = min(1 - clip, max(clip, score / maximum))
        values.append(math.log(probability / (1 - probability)) + benchmark["offset"])
    ability = sum(values) / len(values)
    delta = calibration["huber_delta"]
    for _ in range(5000):
        weights = [min(1.0, delta / max(1e-12, abs(value - ability))) for value in values]
        updated = sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
        if abs(updated - ability) < 1e-9:
            # Stable logistic evaluation also handles extreme finite offsets.
            probability = 1 / (1 + math.exp(-updated)) if updated >= 0 else math.exp(updated) / (1 + math.exp(updated))
            return round(100 * probability, 1)
        ability = updated
    raise ValueError("Model score calculation did not converge")


def identity(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./:-]+", value):
        raise ValueError("Use a canonical vendor/model ID without whitespace")
    return value.removesuffix(":free")


def read_updates(path: Path, calibration: dict[str, Any]) -> list[tuple[str, float, list[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else [raw]
    if not rows:
        raise ValueError("Provide at least one model")
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"model", "benchmarks", "aliases", "sources", "notes"}:
            raise ValueError("Expected model, benchmarks, sources, and optional aliases/notes")
        model = identity(row.get("model"))
        if model in seen:
            raise ValueError(f"Repeated model {model}")
        seen.add(model)
        sources = row.get("sources")
        if not isinstance(sources, list) or not sources or any(
                not isinstance(source, str) or not source.startswith("https://") for source in sources):
            raise ValueError(f"{model}: provide HTTPS source links for manual verification")
        aliases = row.get("aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(f"{model}: aliases must be a list")
        result.append((model, estimate(row.get("benchmarks"), calibration), [identity(alias) for alias in aliases]))
    return result


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="JSON model record or array; see model-score-example.json")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Curated Python score table to update")
    parser.add_argument("--dry-run", action="store_true", help="Calculate and validate without changing the table")
    parser.add_argument("--list-benchmarks", action="store_true", help="Print calibrated benchmark names and exit")
    args = parser.parse_args(argv)
    try:
        calibration = load_calibration(args.calibration)
        if args.list_benchmarks:
            print("\n".join(calibration["benchmarks"]))
            return 0
        if args.input is None:
            parser.error("provide an input file or --list-benchmarks")
        updates = read_updates(args.input, calibration)
        original = args.table.read_text(encoding="utf-8")
        updated = render_table(original, updates)
        if not args.dry_run and updated != original:
            save_table(args.table, original, updated)
        for model, score, _ in updates:
            print(f"{model}: estimated coding score {score:.1f}/100")
        print("Preview only." if args.dry_run else "Score table updated; existing models retain their scores.")
        return 0
    except (OSError, ValueError, SyntaxError, TypeError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
