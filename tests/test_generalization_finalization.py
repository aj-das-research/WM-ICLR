"""Completion and evidence gates for the separate generalization postprocessor."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generalization_finalizer", ROOT / "scripts/real_video_development/finalize_generalization.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_registry(tmp_path):
    rows = []
    for arm in module.ARMS:
        for seed in range(3):
            for mode in module.MODES:
                name = f"{arm}_{mode}_{seed}"
                path = tmp_path / (name + ".json")
                path.write_text(json.dumps({"mode": mode, "seed": seed, "epochs": 30, "lr": 1e-4,
                                            "output_dir": "runs/" + name}))
                rows.append({"name": name, "arm": arm, "seed": seed, "mode": mode,
                             "config": path.name, "config_sha256": module.sha(path)})
    return {"runs": rows, "dependencies": {}}


def test_missing_run_and_altered_matched_config_fail_before_paper_change(tmp_path):
    registry = fixture_registry(tmp_path)
    paper = tmp_path / "existing.tex"
    paper.write_text("Existing verified results")
    configs = module.check_registration(registry, tmp_path)
    with pytest.raises(ValueError, match="Incomplete campaign"):
        module.require_complete_files(registry, configs, tmp_path)
    assert paper.read_text() == "Existing verified results"
    record = registry["runs"][1]
    path = tmp_path / record["config"]
    value = json.loads(path.read_text())
    value["lr"] = 0.3
    path.write_text(json.dumps(value))
    record["config_sha256"] = module.sha(path)
    with pytest.raises(ValueError, match="unmatched training"):
        module.check_registration(registry, tmp_path)
    with pytest.raises(ValueError, match="exactly 36"):
        module.check_registration({**registry, "runs": registry["runs"][:-1]}, tmp_path)


def fixture_result():
    methods = ("model", "persistence", "constant_velocity", "reversed_future_actions")
    episodes = [{"episode_id": f"e{i}", "session_id": f"s{i}", "windows": 2,
                 "window_starts": [0, 5], "errors": {m: {"h5_standardized_mse": float(i+1)} for m in methods}} for i in range(2)]
    return {"scope": "validation development only", "arm": "slow", "mode": "factorized", "seed": 0,
            "horizon": 5, "registration_sha256": "registry", "checkpoint_sha256": "checkpoint",
            "result": {"episodes": episodes, "episode_count": 2, "session_count": 2, "window_count": 4,
                       "summary": {m: {"h5_standardized_mse": 1.5} for m in methods}}}


def checked(document):
    return module.check_evaluation(document, row={"arm": "slow", "mode": "factorized", "seed": 0},
        horizon=5, registry_hash="registry", checkpoint_hash="checkpoint",
        expected_population={f"e{i}": {"session_id": f"s{i}", "starts": [0, 5]} for i in range(2)})


def test_result_summary_is_recomputed_from_episode_evidence():
    value = fixture_result()
    assert len(checked(value)) == 2
    value["result"]["summary"]["model"]["h5_standardized_mse"] = 0.01
    with pytest.raises(ValueError, match="Episode-derived mean"):
        checked(value)


@pytest.mark.parametrize("mutation", ["split", "window", "checkpoint", "nonfinite", "duplicate"])
def test_wrong_population_or_checkpoint_cannot_be_published(mutation):
    value = fixture_result()
    if mutation == "split":
        value["scope"] = "test"
    elif mutation == "window":
        value["result"]["episodes"][0]["window_starts"] = [5, 10]
    elif mutation == "checkpoint":
        value["checkpoint_sha256"] = "other"
    elif mutation == "nonfinite":
        value["result"]["episodes"][0]["errors"]["model"]["h5_standardized_mse"] = float("nan")
    else:
        value["result"]["episodes"][0]["episode_id"] = "e1"
    with pytest.raises(ValueError):
        checked(value)


def test_gain_format_preserves_negative_and_inconclusive_values():
    assert module.gain_tex(-12.5) == "-12.50"
    assert module.gain_tex(0) == "+0.00"
    assert module.gain_tex(None) == "undefined"
    assert module.gain_tex(3.2) == r"\positivegain{+3.20}"


def test_all_tables_compile_at_exact_paper_width(tmp_path):
    # Synthetic layout fixture only, confined to pytest's temporary directory;
    # never written to the paper or represented as measured research results.
    records, aggregates = [], []
    for arm in module.ARMS:
        for mode in module.MODES:
            for seed in range(3):
                records.append({"arm": arm, "mode": mode, "seed": seed, "elapsed_seconds": 2500.3+seed,
                    "training": {"best_epoch": 30, "parameter_counts": {"total": 3763496, "trainable": 3468776}}})
        for horizon in (5, 10):
            methods = {mode: {"mean": 0.222333, "gain_percent": -1.239,
                       "paired_difference": {"mean_difference": 0.000421, "ci95": [-0.001234, 0.001999]}} for mode in module.MODES}
            methods["factorized"]["gain_percent"] = 3.123
            aggregates.append({"arm": arm, "horizon": horizon, "methods": methods})
    files = module.tables(records, aggregates)
    assert sum(value.count(r"\textbf{ShiftWM (ours)}") for value in files.values()) == 12
    review = module.proof(tmp_path, files)
    assert review["status"] == "passed" and review["overflow_or_reference_warnings"] == 0
