"""Scientific aggregation guards using temporary result records only."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts" / "aggregate_results.py"
    spec = importlib.util.spec_from_file_location("aggregate_results", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_result(root, seed, value, *, mode="factorized", kind="planning", split="test", **updates):
    body = {"summary": {"o0_d0": {"success": {"mean": value}},
                        "o2_d2": {"success": {"mean": value}, "mse_h5": {"mean": value}},
                        "all": {"success": {"mean": value}}},
            "eligible_summary": {"o2_d2": {"success": {"mean": value}}}}
    record = {"status": "complete", "environment": "pusht", "model_mode": mode,
              "training_seed": None if mode == "frozen" else seed,
              "checkpoint_sha256": f"checkpoint-{mode}-{seed}", "data_manifest_sha256": "dataset-v1",
              "evaluator_sha256": "evaluator-v1", kind: body}
    record.update(updates)
    if kind == "planning":
        body["policy"] = "world_model"
        body["split"] = split
        body["planner"] = {**module().MAIN_PLANNER_BUDGET, **module().MAIN_PLANNER_IDENTITY}
        body["protocol"] = {key: body["planner"][key] for key in ("samples", "iterations", "elites", "horizon", "native_budget")}
        body["protocol"].update(policy="world_model", split=split, episodes_per_dynamics=64, goal_offset=5, seed=1701,
                               search_coordinates=body["planner"]["search_coordinates"],
                               evaluator_sha256=record["evaluator_sha256"],
                               run_identity={key: record[key] for key in ("checkpoint_sha256", "data_manifest_sha256")})
        body["signature"] = hashlib.sha256(json.dumps(body["protocol"], sort_keys=True).encode()).hexdigest()
    path = root / f"pusht_{mode}_s{seed}" / f"{kind}_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def aggregate(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["aggregate_results", "--results", str(tmp_path / "results"),
                                     "--output", str(output)])
    module().main()
    return json.loads((output / "primary_results.json").read_text())


def row(ledger, mode="factorized", metric="heldout_success"):
    return next(r for r in ledger["results"] if r["environment"] == "pusht" and r["mode"] == mode and r["metric"] == metric)


def test_three_seeds_required_missing_cells_stay_null_and_tex_missing(tmp_path, monkeypatch):
    for seed, value in ((0, .2), (1, .4)):
        fixture_result(tmp_path / "results", seed, value)
    ledger = aggregate(tmp_path, monkeypatch)
    result = row(ledger)
    assert result["status"] == "pending"
    assert result["available_seeds"] == [0, 1]
    assert result["required_seeds"] == [0, 1, 2]
    assert result["mean"] is None and result["training_seed_sd"] is None
    assert r"\missing" in (tmp_path / "output" / "primary_results.tex").read_text()
    assert row(ledger, metric="heldout_mse_h5")["mean"] is None


def test_all_three_seeds_use_correct_mean_and_sample_standard_deviation(tmp_path, monkeypatch):
    for seed, value in enumerate((.1, .2, .3)):
        fixture_result(tmp_path / "results", seed, value)
    result = row(aggregate(tmp_path, monkeypatch))
    assert result["status"] == "complete"
    assert result["mean"] == pytest.approx(20.0)
    assert result["training_seed_sd"] == pytest.approx(10.0)  # ddof=1, not population SD.


def test_frozen_uses_one_checkpoint_and_no_invented_training_sd(tmp_path, monkeypatch):
    fixture_result(tmp_path / "results", 0, .42, mode="frozen")
    result = row(aggregate(tmp_path, monkeypatch), mode="frozen")
    assert result["status"] == "complete" and result["mean"] == pytest.approx(42.0)
    assert result["training_seed_sd"] is None


def test_incomplete_evaluation_is_not_counted_as_a_completed_seed(tmp_path, monkeypatch):
    for seed in (0, 1):
        fixture_result(tmp_path / "results", seed, .2)
    fixture_result(tmp_path / "results", 2, .9, status="interrupted")
    assert row(aggregate(tmp_path, monkeypatch))["status"] == "pending"


@pytest.mark.parametrize("key,value,match", [
    ("data_manifest_sha256", "dataset-v2", "Incompatible data"),
    ("evaluator_sha256", "evaluator-v2", "Incompatible evaluator"),
])
def test_incompatible_versions_across_methods_are_rejected(tmp_path, monkeypatch, key, value, match):
    fixture_result(tmp_path / "results", 0, .1, mode="frozen")
    fixture_result(tmp_path / "results", 0, .2, **{key: value})
    with pytest.raises(ValueError, match=match):
        aggregate(tmp_path, monkeypatch)


def test_same_run_cannot_mix_different_checkpoint_identities(tmp_path, monkeypatch):
    fixture_result(tmp_path / "results", 0, .1)
    fixture_result(tmp_path / "results", 0, .2, kind="forecast", checkpoint_sha256="different-checkpoint")
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        aggregate(tmp_path, monkeypatch)


def test_identical_checkpoint_cannot_fake_distinct_training_seeds(tmp_path, monkeypatch):
    fixture_result(tmp_path / "results", 0, .1)
    fixture_result(tmp_path / "results", 1, .2, checkpoint_sha256="checkpoint-factorized-0")
    with pytest.raises(ValueError, match="Identical checkpoint reused"):
        aggregate(tmp_path, monkeypatch)


def test_explicit_training_seed_must_match_run_directory(tmp_path, monkeypatch):
    fixture_result(tmp_path / "results", 0, .1, training_seed=2)
    with pytest.raises(ValueError, match="Wrong training seed"):
        aggregate(tmp_path, monkeypatch)


@pytest.mark.parametrize("key,value", [("samples", 128), ("iterations", 5), ("elites", 16), ("horizon", 4), ("native_budget", 60)])
def test_main_tables_reject_a_different_planner_budget(tmp_path, monkeypatch, key, value):
    path = fixture_result(tmp_path / "results", 0, .1)
    record = json.loads(path.read_text())
    record["planning"]["planner"][key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incompatible main planner budget"):
        aggregate(tmp_path, monkeypatch)


def test_relabelling_planner_summary_cannot_hide_old_hashed_protocol(tmp_path, monkeypatch):
    path = fixture_result(tmp_path / "results", 0, .1)
    record = json.loads(path.read_text())
    # Summary claims300x30 while the actual serialized protocol says128x5.
    record["planning"]["protocol"].update(samples=128, iterations=5)
    record["planning"]["signature"] = hashlib.sha256(json.dumps(record["planning"]["protocol"], sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incompatible hashed planning protocol"):
        aggregate(tmp_path, monkeypatch)


def test_random_control_metadata_does_not_enter_primary_world_model_tables(tmp_path, monkeypatch):
    path = fixture_result(tmp_path / "results", 0, .9, mode="frozen")
    record = json.loads(path.read_text())
    record["planning"]["policy"] = "random"
    record["planning"]["planner"].update(samples=128, iterations=5, elites=16)
    path.write_text(json.dumps(record))
    ledger = aggregate(tmp_path, monkeypatch)
    assert row(ledger, mode="frozen")["status"] == "pending"
    assert ledger["sources"] == {}


def test_test_filename_cannot_contain_a_relabeled_extrapolation_protocol(tmp_path, monkeypatch):
    path = fixture_result(tmp_path / "results", 0, .1)
    record = json.loads(path.read_text())
    record["planning"]["split"] = "extrapolation"
    record["planning"]["protocol"]["split"] = "extrapolation"
    record["planning"]["signature"] = hashlib.sha256(json.dumps(record["planning"]["protocol"], sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Planning split metadata disagrees"):
        aggregate(tmp_path, monkeypatch)


def test_same_budget_different_search_space_is_rejected(tmp_path, monkeypatch):
    path = fixture_result(tmp_path / "results", 0, .1)
    record = json.loads(path.read_text())
    record["planning"]["planner"]["search_coordinates"] = "raw actions"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incompatible main planner identity"):
        aggregate(tmp_path, monkeypatch)
