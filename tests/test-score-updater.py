"""Offline score maintenance preserves existing ratings and rejects bad evidence."""

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


def write_input(tmp_path, records):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(records))
    return path


def record(**changes):
    return {"model": "vendor/new-model", "benchmarks": {"SWE Verified": 77.0},
            "sources": ["https://example.invalid/published-model-card"], **changes}


def test_saved_offsets_reproduce_reviewed_examples(calibration):
    from slipagent.modelscores import model_score
    example = json.loads((ROOT / "scripts/model-score-example.json").read_text())
    assert updater.estimate(example["benchmarks"], calibration) == model_score("kilo", "reflection/beam-free")
    scales, records = updater.read_measurements(ROOT / "scripts/model-score-measurements.json")
    # A fitted score is each model's own Huber location at the fitted offsets.
    for record in records:
        if not record["reference"]:
            assert updater.estimate(record["benchmarks"], calibration) == model_score(
                "nvidia", record["model"]), record["model"]


def test_coding_judge_rating_uses_its_native_scale(calibration):
    from slipagent.modelscores import model_score
    assert updater.estimate({"MT-Bench coding": 5.2}, calibration) == model_score("nvidia", "zyphra/zamba2-7b-instruct")
    with pytest.raises(ValueError, match="between 1 and 10"):
        updater.estimate({"MT-Bench coding": 52}, calibration)
    with pytest.raises(ValueError, match="between 1 and 10"):
        updater.estimate({"MT-Bench coding": 0.5}, calibration)


@pytest.mark.parametrize("minimum, maximum", [(1, 1), (-1, 10), (0, 0), (0, float("inf")), (0, True)])
def test_invalid_benchmark_scale_is_rejected(tmp_path, calibration, minimum, maximum):
    calibration["benchmarks"]["MT-Bench coding"].update(min_score=minimum, max_score=maximum)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError):
        updater.load_calibration(path)


def test_new_models_use_fixed_calibration_and_preserve_existing_scores(tmp_path, table, capsys):
    before = runpy.run_path(str(table))
    data = write_input(tmp_path, [record(aliases=["host/new-model-free"]),
                                 record(model="vendor/another-model", benchmarks={"HumanEval": 81.1})])
    assert updater.main([str(data), "--table", str(table)]) == 0
    after = runpy.run_path(str(table))
    assert all(after["SCORES"][model] == score for model, score in before["SCORES"].items())
    assert after["model_score"]("openrouter", "vendor/new-model:free") == 77.0
    assert after["model_score"]("kilo", "host/new-model-free") == 77.0
    assert 0 < after["SCORES"]["vendor/another-model"] < 81.1
    assert set(after["SCORES"]) - set(before["SCORES"]) == {"vendor/new-model", "vendor/another-model"}
    assert table.stat().st_mode & 0o777 == 0o640
    assert "estimated coding score 77.0/100" in capsys.readouterr().out


def test_preview_does_not_modify_table_and_explicit_update_replaces_only_that_model(tmp_path, table):
    before = table.read_bytes()
    data = write_input(tmp_path, record(model="reflection/beam", benchmarks={"SWE Verified": 80}))
    assert updater.main([str(data), "--table", str(table), "--dry-run"]) == 0
    assert table.read_bytes() == before
    old_scores = runpy.run_path(str(table))["SCORES"]
    assert updater.main([str(data), "--table", str(table)]) == 0
    new_scores = runpy.run_path(str(table))["SCORES"]
    assert {name for name in old_scores if new_scores[name] != old_scores[name]} == {"reflection/beam"}
    assert new_scores["reflection/beam"] == 80.0


@pytest.mark.parametrize("evidence", [
    {}, None, [], {"Unknown benchmark": 75}, {"Visual coding": 90}, {"SWE Verified": -1},
    {"SWE Verified": 101}, {"SWE Verified": True}, {"SWE Verified": "80"},
    {"SWE Verified": float("nan")}, {"SWE Verified": float("inf")}, {"SWE Verified": 10**400},
])
def test_invalid_evidence_leaves_whole_batch_unchanged(tmp_path, table, evidence):
    before = table.read_bytes()
    data = write_input(tmp_path, [record(), record(model="vendor/second", benchmarks=evidence)])
    with pytest.raises(SystemExit) as error:
        updater.main([str(data), "--table", str(table)])
    assert error.value.code == 2
    assert table.read_bytes() == before


@pytest.mark.parametrize("records", [
    [record(), record()],
    record(aliases=["reflection/beam-free"]),
    record(aliases=["reflection/beam"]),
    record(model="reflection/beam-free"),
    [record(aliases=["vendor/second"]), record(model="vendor/second")],
    record(sources=[]),
    record(benchmarks={"SWE Verified": 75}, benchmark={"HumanEval": 99}),
])
def test_bad_identity_or_source_does_not_change_table(tmp_path, table, records):
    before = table.read_bytes()
    data = write_input(tmp_path, records)
    with pytest.raises(SystemExit):
        updater.main([str(data), "--table", str(table)])
    assert table.read_bytes() == before


def test_insufficient_shared_calibration_is_rejected(tmp_path, table, calibration):
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


def test_list_benchmarks_requires_no_model_input(capsys):
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


def test_reference_models_calibrate_a_benchmark_without_entering_the_table(tmp_path, calibration, table, capsys):
    models = synthetic() + [
        measurement("vendor/a", "SWE Verified", "Alpha Bench", "Gamma Bench", reference=True) | {"model": "paid/ref-1"},
        measurement("vendor/b", "SWE Verified", "Gamma Bench", reference=True) | {"model": "paid/ref-2"},
    ]
    models[-2]["benchmarks"] = {name: measured("vendor/a", name) for name in ("SWE Verified", "Alpha Bench", "Gamma Bench")}
    models[-1]["benchmarks"] = {name: measured("vendor/b", name) for name in ("SWE Verified", "Gamma Bench")}
    models.append({"model": "vendor/new", "benchmarks": {"Gamma Bench": measured("vendor/c", "Gamma Bench")},
                   "sources": ["https://example.invalid/results"]})
    path = write_measurements(tmp_path, models)
    args = updater.argparse.Namespace(fit=path, reviewed="2026-10-26", table=table, dry_run=True,
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


def test_fit_writes_table_and_calibration_together(tmp_path, calibration, table):
    path = write_measurements(tmp_path, synthetic())
    target = tmp_path / "calibration.json"
    target.write_text((ROOT / "scripts/model-score-calibration.json").read_text())
    assert updater.main(["--fit", str(path), "--table", str(table), "--calibration", str(target),
                         "--reviewed", "2026-10-26"]) == 0
    saved = json.loads(target.read_text())
    assert saved["reviewed"] == "2026-10-26" and saved["benchmarks"]["Alpha Bench"]["overlap_models"] == 3
    assert "vendor/a" in table.read_text()
    # An unmeasured entry keeps its reviewed score.
    assert '"google/gemini-2.5-flash": 44.2' in table.read_text()
