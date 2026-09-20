"""Synthetic-only completeness, matched-population and bootstrap tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reserved_recovery_finalization_tests", ROOT / "scripts/real_video_iws_reserved_recovery_v2/finalize.py")
final = importlib.util.module_from_spec(spec); spec.loader.exec_module(final)
ev = final.evaluation


def population(task):
    ids = [f"synthetic-{i:02d}" for i in range(10)]
    handles = [{"episode_id": eid, "start": s} for eid, n in zip(ids, [10, 30] + [20] * 8) for s in range(n)]
    return ev.population_contract(task, handles, ids)


def invocations(task):
    result = []
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
    for start in range(0, 200, 64):
        count = min(64, 200 - start)
        full = digest(f"{task}/{start}/prediction60")
        for h in (60, 15, 30, 45):
            row = {"first_handle": start, "count": count, "command_rows": h,
                   "prediction_shape": [count, h - 1, 6144], "initial_features_sha256": digest(f"{task}/{start}/input"),
                   "commands_sha256": digest(f"{task}/{start}/commands{h}"), "prediction_sha256": full if h == 60 else digest(f"{task}/{start}/prediction{h}")}
            if h != 60:
                row.update(full_prediction_sha256=full, full_output_index=h - 2, target_native_offset=h - 1,
                           maximum_absolute_difference=0., maximum_reference_absolute_value=1., maximum_tolerance_ratio=0.,
                           allclose=True, rtol=1e-5, atol=2e-5)
            result.append(row)
    return result


def fixture():
    populations = {t: population(t) for t in final.TASKS}
    rows = []
    for task in final.TASKS:
        pop = populations[task]
        idx = np.array([pop["episode_ids"].index(h["episode_id"]) for h in pop["handles"]], dtype=np.int64)
        starts = np.array([h["start"] for h in pop["handles"]], dtype=np.int64)
        for mode_i, mode in enumerate(ev.MODES):
            for seed in range(3):
                arrays = {"episode_index": idx.copy(), "window_start": starts.copy()}
                for mi, metric in enumerate(final.METRICS):
                    error = (.6 - .03 * mode_i + seed * .01 + idx / 1000) / (mi + 1)
                    arrays[metric] = np.repeat(error[:, None], 59, 1).astype(np.float64)
                    arrays["persistence_" + metric] = np.repeat(((1 + idx / 1000) / (mi + 1))[:, None], 59, 1).astype(np.float64)
                    arrays["prefix_" + metric] = arrays[metric][:, [13, 28, 43]].copy()
                summary = ev.aggregate_arrays(arrays, pop)
                receipt = {**summary, "invocations": invocations(task),
                           "normalization_sha256": {k: "a" * 64 for k in ("feature_mean", "feature_std", "command_mean", "command_std")}}
                rows.append({"task": task, "mode": mode, "seed": seed, "receipt": receipt, "arrays": arrays})
    return rows, populations


def test_complete36_grid_all5methods_and_two_weightings():
    rows, populations = fixture()
    assembled = final.assemble(rows, populations)
    report = final.numerical_report(*assembled, draws=40)
    metric = report["task_results"]["pusht"]["standardized_mse"]
    assert set(metric["equal_trajectory"]["mean_all59_offsets"]) == set(final.MODES)
    assert metric["equal_trajectory"]["horizon_means"]["60"]["autoregressive"] == pytest.approx(.6145)
    assert metric["equal_handle"]["horizon_means"]["60"]["autoregressive"] == pytest.approx(.61455)
    assert metric["h60_comparisons"]["bounded_vs_additive"]["relative_error_reduction_percent"] > 0
    assert metric["h60_comparisons"]["no_tanh_vs_bounded"]["status"] == "secondary_post_development_component"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "persistence", "normalization", "inputs", "population"])
def test_mismatched_or_incomplete_studies_fail(mutation):
    rows, populations = fixture()
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[-1] = rows[0]
    elif mutation == "persistence": rows[1]["arrays"]["persistence_standardized_mse"][0, 0] += .1
    elif mutation == "normalization": rows[1]["receipt"]["normalization_sha256"]["feature_std"] = "b" * 64
    elif mutation == "inputs": rows[1]["receipt"]["invocations"][0]["commands_sha256"] = "b" * 64
    elif mutation == "population": rows[1]["receipt"]["episodes"][0]["handles"] += 1
    with pytest.raises(ValueError): final.assemble(rows, populations)


def test_metadata_only_cache_compares_immutable_fields_not_access_state():
    pop = population("pusht")
    metadata = {"metadata_only": True, "episodes_loaded": [], "manifest_sha256": "a" * 64, "scope": "reserved"}
    evaluated = {**metadata, "metadata_only": False, "episodes_loaded": list(dict.fromkeys(h["episode_id"] for h in pop["handles"]))}
    final.validate_cache_audit(evaluated, metadata, pop)
    with pytest.raises(ValueError): final.validate_cache_audit(metadata, metadata, pop)
    evaluated["episodes_loaded"] = evaluated["episodes_loaded"][::-1]
    with pytest.raises(ValueError): final.validate_cache_audit(evaluated, metadata, pop)


def test_shared_seed_draws_and_macro_are_exact_paired_ratio_draws():
    rows, pops = fixture(); values, _, _, _ = final.assemble(rows, pops)
    # Distinct task/seed/trajectory effects make independent task seed draws detectable.
    for ti, task in enumerate(final.TASKS):
        for mi, mode in enumerate(final.MODES):
            values[task][mode] += (ti + 1) * np.arange(3)[:, None, None, None] * .04 * (mi + 1)
    draws = 53; actual = final.bootstrap(values, draws, 173)
    rng = np.random.default_rng(173); observed = []
    for _ in range(draws):
        seeds = rng.integers(0, 3, size=3); gains = []
        for task in final.TASKS:
            trajectories = rng.integers(0, 10, size=10)
            a = values[task]["bounded_spatial_mix"][seeds][:, trajectories, -1, 0].mean()
            b = values[task]["anchored_additive"][seeds][:, trajectories, -1, 0].mean()
            gains.append(100 * (b - a) / b)
        observed.append(np.mean(gains))
    np.testing.assert_allclose(actual["macro"]["standardized_mse"]["bounded_vs_additive"]["percentile95"], np.quantile(observed, [.025, .975]), rtol=1e-12)


def test_negative_and_zero_denominator_results_are_retained():
    rows, pops = fixture(); values, prefixes, handles, hp = final.assemble(rows, pops)
    for task in final.TASKS:
        values[task]["bounded_spatial_mix"][:] = 2
        values[task]["anchored_additive"][:] = 1
        values[task]["autoregressive"][:] = 0
    result = final.numerical_report(values, prefixes, handles, hp, draws=20)
    pairs = result["task_results"]["pusht"]["standardized_mse"]["h60_comparisons"]
    assert pairs["bounded_vs_additive"]["relative_error_reduction_percent"] == -100
    assert pairs["bounded_vs_additive"]["paired95"]["gain_percent"]["percentile95"] == [-100, -100]
    assert pairs["bounded_vs_autoregressive"]["relative_error_reduction_percent"] is None
    assert pairs["bounded_vs_autoregressive"]["paired95"]["gain_percent"] == {"percentile95": None, "undefined_draws": 20}


def test_partial_finalizer_does_not_emit_numerical_results(monkeypatch, tmp_path):
    registry = {"runs": [{"name": f"{t}-{m}-{s}", "task": t, "mode": m, "seed": s} for t in final.TASKS for m in ev.MODES for s in range(3)],
                "evaluation": {"device": "cpu", "dtype": "float32", "threads": 8, "batch_size": 64, "horizon": 60, "prefix_horizons": [15, 30, 45], "bootstrap_draws": 10000, "bootstrap_seed": 173}}
    monkeypatch.setattr(ev, "checked_registration", lambda *a: registry)
    monkeypatch.setattr(final, "collect", lambda *a: pytest.fail("Tried to finalize partial results"))
    result = final.finalize(tmp_path, if_ready=True)
    assert result["status"] == "pending" and result["completed_runs"] == 0
    assert not (tmp_path / ev.REPORT / "finalization.json").exists()


def test_receipt_reconstruction_rejects_aggregate_tamper(monkeypatch, tmp_path):
    rows, pops = fixture(); row = rows[0]; pop = pops[row["task"]]
    row.update(name="synthetic", package_kind="synthetic", selected_epoch=7, checkpoint_sha256="c" * 64)
    src = tmp_path / "scripts/real_video_iws_reserved_recovery_v2/evaluate.py"; src.parent.mkdir(parents=True); src.write_text("synthetic source")
    helper = tmp_path / "scripts/real_video_iws/evaluate.py"; helper.parent.mkdir(parents=True); helper.write_text("synthetic metric source")
    monkeypatch.setattr(ev, "__file__", str(src)); monkeypatch.setattr(ev.frozen_metrics, "__file__", str(helper))
    cache = SimpleNamespace(audit={"metadata_only": True, "episodes_loaded": [], "manifest_sha256": "d" * 64})
    path = tmp_path / "evaluations/synthetic.json"; path.parent.mkdir()
    ev.atomic_npz(path.with_suffix(".npz"), row["arrays"])
    receipt = {**row["receipt"], "schema": ev.SCHEMA, "status": "passed", "scope": ev.SCOPE,
               **{k: row[k] for k in ("name", "task", "mode", "seed", "package_kind", "selected_epoch")},
               "selected_checkpoint_sha256": row["checkpoint_sha256"], "completed_epochs": 30, "registration_sha256": "reg",
               "population": pop, "cache_audit": {**cache.audit, "metadata_only": False, "episodes_loaded": pop["episode_ids"]},
               "metrics": list(ev.METRICS), "offsets": list(range(1, 60)), "prefix_horizons": [15, 30, 45],
               "primary_aggregation": "equal_trajectory_then_equal_seed", "reserved_feature_episodes_read": 10,
               "reserved_raw_payloads_read": 0, "internal_development_payloads_read": 0, "selector_score_equality_required": False,
               "evaluator_sha256": ev.sha(src), "metric_helper_sha256": ev.sha(helper),
               "backend": {"device": "cpu", "dtype": "float32", "threads": 8, "interop_threads": 1, "batch_size": 64, "autocast": False, "tf32": False, "command_gru_dispatch": "one_native_row_per_call"},
               "window_ledger_path": str(path.with_suffix(".npz").relative_to(tmp_path)), "window_ledger_sha256": ev.sha(path.with_suffix(".npz")),
               "source_dependencies": {str(p.relative_to(tmp_path)): ev.sha(p) for p in (src, helper)}}
    path.write_text(json.dumps(receipt))
    final.validate_evaluation(tmp_path, path, row, pop, cache, "reg", {})
    receipt["equal_trajectory"]["standardized_mse"][0] += .1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="aggregate"):
        final.validate_evaluation(tmp_path, path, row, pop, cache, "reg", {})
