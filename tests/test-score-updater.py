"""Scores are calculated from raw published results; bad evidence changes nothing."""

import importlib.util
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_updater", ROOT / "scripts/update-model-scores.py")
assert spec is not None and spec.loader is not None
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


@pytest.fixture
def calibration():
    return updater.load_calibration(ROOT / "scripts/model-score-calibration.json")


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "modelscores.py"
    path.write_text((ROOT / "src/slipagent/modelscores.py").read_text())
    path.chmod(0o640)
    return path


@pytest.fixture
def small_table(tmp_path):
    path = tmp_path / "small-scores.py"
    path.write_text('SCORES: dict[str, float | None] = {\n    "vendor/gap": None,\n}\n\nMODEL_ALIASES = {}\n')
    path.chmod(0o640)
    return path


def test_saved_table_and_calibration_match_the_saved_measurements(calibration):
    from slipagent.modelscores import SCORES
    scales, records = updater.read_measurements(ROOT / "scripts/model-score-measurements.json")
    scores, fitted = updater.fit_scores(records, scales, calibration, calibration["reviewed"], validate=False)
    for record in records:
        if not record["reference"]:
            assert SCORES[record["model"]] == scores[record["model"]], record["model"]
    # Every score has recorded evidence; only reviewed gaps lack it.
    measured = {record["model"] for record in records if not record["reference"]}
    assert {model for model, score in SCORES.items() if score is not None} == measured
    assert {name: row["offset"] for name, row in fitted["benchmarks"].items()} == pytest.approx(
        {name: row["offset"] for name, row in calibration["benchmarks"].items()})


@pytest.mark.parametrize("minimum, maximum", [(1, 1), (-1, 10), (0, 0), (0, float("inf")), (0, True)])
def test_invalid_benchmark_scale_is_rejected(tmp_path, calibration, minimum, maximum):
    calibration["benchmarks"]["MT-Bench coding"].update(min_score=minimum, max_score=maximum)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError):
        updater.load_calibration(path)


def test_insufficient_shared_calibration_is_rejected(tmp_path, calibration):
    calibration["benchmarks"]["HumanEval"]["overlap_models"] = 1
    saved = tmp_path / "calibration.json"
    saved.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="at least two"):
        updater.load_calibration(saved)


def test_concurrent_table_change_is_preserved(table):
    original = table.read_text()
    concurrent = original + "\n# Another contributor's update\n"
    table.write_text(concurrent)
    with pytest.raises(ValueError, match="changed while calculating"):
        updater.save_table(table, original, original + "\n# Our update\n")
    assert table.read_text() == concurrent
    assert list(table.parent.iterdir()) == [table]


def test_list_benchmarks_requires_no_measurements(capsys):
    assert updater.main(["--list-benchmarks"]) == 0
    output = capsys.readouterr().out
    assert "SWE Verified\n" in output and "Terminal 2.1\n" in output


ABILITIES = {"vendor/a": 3.0, "vendor/b": 1.5, "vendor/c": 0.0, "vendor/d": -1.0, "vendor/e": 2.2}
OFFSETS = {"SWE Verified": 0.0, "Alpha Bench": 1.0, "Beta Bench": -0.6, "Gamma Bench": 2.0}


def measured(model, benchmark, scale=100):
    from math import exp
    return round(scale / (1 + exp(-(ABILITIES[model] - OFFSETS[benchmark]))), 6)


def measurement(model, *benchmarks, **changes):
    return {"model": model, "benchmarks": {name: measured(model, name) for name in benchmarks},
            "sources": ["https://example.invalid/results"], **changes}


def write_measurements(tmp_path, models, **extra):
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps({"reviewed": "2026-10-26", "models": models, **extra}))
    return path


def synthetic():
    return [measurement("vendor/a", "SWE Verified", "Alpha Bench"),
            measurement("vendor/b", "SWE Verified", "Alpha Bench", "Beta Bench"),
            measurement("vendor/c", "SWE Verified", "Beta Bench"),
            measurement("vendor/d", "Alpha Bench", "Beta Bench"),
            measurement("vendor/e", "Alpha Bench")]


def test_fit_recovers_scores_and_offsets_from_raw_results(tmp_path, calibration):
    scales, records = updater.read_measurements(write_measurements(tmp_path, synthetic()))
    scores, fitted = updater.fit_scores(records, scales, calibration, "2026-10-26")
    for model, ability in ABILITIES.items():
        assert scores[model] == round(100 / (1 + 2.718281828459045 ** -ability), 1)
    for name in ("Alpha Bench", "Beta Bench"):
        assert fitted["benchmarks"][name]["offset"] == pytest.approx(OFFSETS[name], abs=1e-3)
    assert fitted["benchmarks"]["SWE Verified"]["offset"] == 0
    assert fitted["benchmarks"]["Alpha Bench"]["overlap_models"] == 3
    assert fitted["model_count"] == 5


def test_reference_models_calibrate_a_benchmark_without_entering_the_table(tmp_path, calibration, small_table, capsys):
    models = synthetic() + [
        measurement("vendor/a", "SWE Verified", "Alpha Bench", "Gamma Bench", reference=True) | {"model": "paid/ref-1"},
        measurement("vendor/b", "SWE Verified", "Gamma Bench", reference=True) | {"model": "paid/ref-2"},
    ]
    models[-2]["benchmarks"] = {name: measured("vendor/a", name) for name in ("SWE Verified", "Alpha Bench", "Gamma Bench")}
    models[-1]["benchmarks"] = {name: measured("vendor/b", name) for name in ("SWE Verified", "Gamma Bench")}
    models.append({"model": "vendor/new", "benchmarks": {"Gamma Bench": measured("vendor/c", "Gamma Bench")},
                   "sources": ["https://example.invalid/results"]})
    path = write_measurements(tmp_path, models)
    args = updater.argparse.Namespace(measurements=path, reviewed="2026-10-26", table=small_table, dry_run=True,
                                       calibration=tmp_path / "calibration.json")
    (tmp_path / "calibration.json").write_text("{}")
    updater.refit(args, calibration)
    output = capsys.readouterr().out
    assert "vendor/new: new -> 50.0" in output
    assert "paid/ref-1" not in output and "paid/ref-2" not in output


def test_unconnected_or_thinly_shared_benchmarks_are_rejected(tmp_path, calibration):
    def fit(models):
        scales, records = updater.read_measurements(write_measurements(tmp_path, models))
        return updater.fit_scores(records, scales, calibration, "2026-10-26")
    def result(model, **benchmarks):
        return {"model": model, "benchmarks": benchmarks, "sources": ["https://example.invalid/results"]}
    with pytest.raises(ValueError, match="(Delta|Gamma) Bench shares no chain"):
        fit(synthetic() + [result("vendor/f", **{"Gamma Bench": 40, "Delta Bench": 30}),
                           result("vendor/g", **{"Gamma Bench": 50, "Delta Bench": 20})])
    with pytest.raises(ValueError, match="Gamma Bench needs at least two models"):
        fit(synthetic() + [result("vendor/f", **{"Gamma Bench": 40, "SWE Verified": 50})])
    with pytest.raises(ValueError, match="to anchor the scale"):
        fit([measurement("vendor/a", "Alpha Bench", "Beta Bench"), measurement("vendor/b", "Alpha Bench", "Beta Bench")])


def test_conflicting_result_is_downweighted_but_kept(tmp_path, calibration):
    models = synthetic()
    models[1]["benchmarks"]["Beta Bench"] = 99.0
    scales, records = updater.read_measurements(write_measurements(tmp_path, models))
    scores, _ = updater.fit_scores(records, scales, calibration, "2026-10-26")
    assert scores["vendor/b"] != round(100 / (1 + 2.718281828459045 ** -ABILITIES["vendor/b"]), 1)
    assert 70 < scores["vendor/b"] < 90


def test_validation_predicts_each_result_of_multi_result_models(tmp_path, calibration):
    scales, records = updater.read_measurements(write_measurements(tmp_path, synthetic()))
    _, fitted = updater.fit_scores(records, scales, calibration, "2026-10-26")
    assert fitted["validation"]["count"] <= 9
    assert fitted["validation"]["maximum_absolute_error"] < 100


@pytest.mark.parametrize("change, message", [
    ({"reference": "yes"}, "reference must be true or false"),
    ({"benchmarks": {"SWE Verified": 120}}, "must be between"),
    ({"benchmarks": {}}, "at least one benchmark"),
    ({"sources": ["http://example.invalid"]}, "HTTPS source links"),
])
def test_invalid_measurements_are_rejected(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        updater.read_measurements(write_measurements(tmp_path, [measurement("vendor/a", "SWE Verified") | change]))


def test_custom_scale_is_applied_to_measurements(tmp_path, calibration):
    models = synthetic()
    models[1]["benchmarks"]["Beta Bench"] = measured("vendor/b", "Beta Bench", 10)
    models[2]["benchmarks"]["Beta Bench"] = measured("vendor/c", "Beta Bench", 10)
    models[3]["benchmarks"]["Beta Bench"] = measured("vendor/d", "Beta Bench", 10)
    path = write_measurements(tmp_path, models, scales={"Beta Bench": {"min_score": 1, "max_score": 10}})
    scales, records = updater.read_measurements(path)
    scores, fitted = updater.fit_scores(records, scales, calibration, "2026-10-26")
    assert fitted["benchmarks"]["Beta Bench"]["max_score"] == 10
    assert fitted["benchmarks"]["Beta Bench"]["offset"] == pytest.approx(OFFSETS["Beta Bench"], abs=1e-3)


def test_fit_writes_table_and_calibration_together(tmp_path, small_table):
    path = write_measurements(tmp_path, synthetic())
    target = tmp_path / "calibration.json"
    target.write_text((ROOT / "scripts/model-score-calibration.json").read_text())
    assert updater.main([str(path), "--table", str(small_table), "--calibration", str(target),
                         "--reviewed", "2026-10-26"]) == 0
    saved = json.loads(target.read_text())
    assert saved["reviewed"] == "2026-10-26" and saved["benchmarks"]["Alpha Bench"]["overlap_models"] == 3
    scores = runpy.run_path(str(small_table))["SCORES"]
    assert scores["vendor/a"] == 95.3 and scores["vendor/gap"] is None
    assert small_table.stat().st_mode & 0o777 == 0o640


def test_dry_run_changes_no_file(tmp_path, small_table):
    path = write_measurements(tmp_path, synthetic())
    target = tmp_path / "calibration.json"
    target.write_text((ROOT / "scripts/model-score-calibration.json").read_text())
    before = (small_table.read_bytes(), target.read_bytes())
    assert updater.main([str(path), "--table", str(small_table), "--calibration", str(target), "--dry-run"]) == 0
    assert (small_table.read_bytes(), target.read_bytes()) == before


def test_a_score_without_measurements_is_rejected_and_nothing_changes(tmp_path, small_table):
    small_table.write_text(small_table.read_text().replace('"vendor/gap": None', '"vendor/gap": None,\n    "vendor/old": 55.0'))
    target = tmp_path / "calibration.json"
    target.write_text((ROOT / "scripts/model-score-calibration.json").read_text())
    before = (small_table.read_bytes(), target.read_bytes())
    with pytest.raises(SystemExit) as error:
        updater.main([str(write_measurements(tmp_path, synthetic())), "--table", str(small_table), "--calibration", str(target)])
    assert error.value.code == 2
    assert (small_table.read_bytes(), target.read_bytes()) == before


@pytest.mark.parametrize("models", [
    [],
    [measurement("vendor/a", "SWE Verified"), measurement("vendor/a", "Alpha Bench")],
    [measurement("vendor/a", "SWE Verified", aliases=["vendor/b"]), measurement("vendor/b", "SWE Verified")],
    [measurement("vendor/a", "SWE Verified", sources=[])],
])
def test_invalid_measurements_leave_the_files_unchanged(tmp_path, small_table, models):
    target = tmp_path / "calibration.json"
    target.write_text((ROOT / "scripts/model-score-calibration.json").read_text())
    before = (small_table.read_bytes(), target.read_bytes())
    with pytest.raises(SystemExit):
        updater.main([str(write_measurements(tmp_path, models)), "--table", str(small_table), "--calibration", str(target)])
    assert (small_table.read_bytes(), target.read_bytes()) == before
