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
    assert updater.estimate({"FLTEval pass1": 28.9}, calibration) == model_score("mistral", "labs-leanstral-1-5-1")


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
    assert "SWE Verified\n" in output and "FLTEval pass1\n" in output
