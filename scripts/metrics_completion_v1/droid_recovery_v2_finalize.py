#!/usr/bin/env python3
"""Aggregate only a complete, verified 34-row DROID complementary-metric campaign."""
import argparse
import copy
import fcntl
import json
import sys

import numpy as np

from droid_recovery_v2_common import (ROOT, REPORT, REG, METRICS, POLICY, atomic_json, module, now,
    read, relative, require, sha, validate_output, verify_registration)


def bootstrap_records(value, window_keys, seed=None):
    return {"mode": value["mode"], "seed": value["seed"] if seed is None else seed,
            "episodes": value["episodes"],
            "windows": [{"episode_id": e, "session_id": s, "window_start": w} for e, s, w in window_keys]}


def analyze(values, window_keys, draws=10000):
    """Use the frozen session x seed bootstrap; callers enforce full completion first."""
    learned = [v for v in values if v["mode"] != "persistence"]
    modes = sorted({v["mode"] for v in learned})
    require(len(learned) == 33 and len(modes) == 11, "Need all eleven learned arms x three seeds")
    for mode in modes:
        require(sorted(v["seed"] for v in learned if v["mode"] == mode) == [0, 1, 2],
                "Missing/duplicate learned seed: " + mode)
    persistence = [v for v in values if v["mode"] == "persistence"]
    require(len(persistence) == 1, "Need exactly one deterministic persistence evaluation")
    absolute = {}
    for mode in modes + ["persistence"]:
        group = [v for v in values if v["mode"] == mode]
        means = {key: np.mean([v["summary"][key] for v in group], axis=0).tolist() for key in METRICS}
        absolute[mode] = {"mean_by_horizon": means,
            "equal_horizon_mean_h1_to_h10": {key: float(np.mean(v)) for key, v in means.items()},
            "training_seeds": [] if mode == "persistence" else [0, 1, 2],
            "evaluation_runs": len(group)}
    # Replication here carries identical deterministic values across the matched
    # seed resamples. It is not three independent persistence runs/checkpoints.
    paired_rows = [bootstrap_records(v, window_keys) for v in learned]
    paired_rows += [bootstrap_records(persistence[0], window_keys, seed) for seed in (0, 1, 2)]
    helper = module("scripts/real_video_spatial/validate_ledger.py", "droid_completion_frozen_bootstrap")
    helper.METRICS = METRICS
    comparisons = []
    for reference in POLICY["bootstrap"]["references"]:
        for method in modes + ["persistence"]:
            if method == reference:
                continue
            comparison = helper.paired_intervals(paired_rows, first_mode=method, second_mode=reference,
                                                 draws=draws, bootstrap_seed=POLICY["bootstrap"]["seed"])
            comparison["scope"] = POLICY["scope"]
            comparison["multiplicity"] = POLICY["bootstrap"]["multiplicity"]
            for metric, rows in comparison["metrics"].items():
                for effect in rows:
                    base = effect["second_mean"]
                    effect["relative_error_reduction_percent"] = 100 * (base - effect["first_mean"]) / base if base > 0 else None
                    effect["relative_interval_available"] = False
            comparisons.append(comparison)
    return absolute, comparisons


def finalize():
    reg = verify_registration()
    missing = [r["name"] for r in reg["runs"]
               if not (REPORT / "evaluations" / (r["name"] + ".json")).exists()]
    if missing:
        print(json.dumps({"status": "incomplete", "missing_runs": missing,
                          "aggregation_performed": False}), flush=True)
        return 75
    with (REPORT / ".finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = REPORT / "completion.json"
        if complete.exists():
            marker = read(complete)
            require(marker.get("status") == "passed" and marker["finalization_sha256"] == sha(REPORT / "finalization.json")
                    and marker["registration_sha256"] == sha(REG), "Stale completion marker")
            # Still validate every row: a cached marker does not waive source checks.
            for row in reg["runs"]:
                validate_output(row, reg)
            print("Complete campaign already finalized and reverified", flush=True)
            return 0
        values, output_bindings = [], {}
        # No aggregate, rank or effect is computed until every row is verified.
        for row in reg["runs"]:
            value, arrays = validate_output(row, reg)
            values.append(value)
            for suffix in (".json", ".npz"):
                path = REPORT / "evaluations" / (row["name"] + suffix)
                output_bindings[relative(path)] = sha(path)
        absolute, effects = analyze(values, reg["window_keys"], POLICY["bootstrap"]["draws"])
        require(verify_registration() == reg, "Registration changed during finalization")
        for path, expected in output_bindings.items():
            require(sha(ROOT / path) == expected, "Evaluation changed during finalization")
        final = {"schema": "droid_complementary_metrics_recovery_v2_finalization", "status": "passed", "completed_utc": now(),
            "registration_sha256": sha(REG), "scope": POLICY["scope"], "policy": POLICY, "recovery": reg["recovery"],
            "completed_learned_runs": 33, "completed_persistence_runs": 1,
            "population": {"windows_per_run": 1631, "episodes": 141, "sessions": 59},
            "absolute_metrics": absolute, "paired_comparisons": effects,
            "per_run": [{"name": v["name"], "family": v["family"], "selected_epoch": v["selected_epoch"],
                         "prior_mse_parity": v["prior_mse_parity"], "roundtrip_checks": v["roundtrip_checks"]} for v in values],
            "source_dependencies": {**reg["source_dependencies"], **output_bindings, relative(REG): sha(REG)},
            "primary_identity": "transport remains bounded ShiftWM; unbounded_transport is the no-tanh ablation",
            "limits": ["Post-hoc secondary metrics on already revealed original validation, not fresh confirmation.",
                "Absolute paired intervals are unadjusted across metrics, horizons and comparator contrasts.",
                "Relative reductions are derived point effect sizes; no relative-effect CI was computed.",
                "No control success, RGB prediction quality, or superiority over a published leaderboard is measured.",
                "External v1 and raw v2 are different adapted recipes (30 versus 100 epochs), not isolated architecture effects.",
                "Persistence is evaluated once; identical copies in seed resampling do not create independent runs."]}
        atomic_json(final, REPORT / "finalization.json")
        atomic_json({"status": "passed", "completed_utc": now(), "registration_sha256": sha(REG),
                     "finalization_sha256": sha(REPORT / "finalization.json"), "completed_rows": 34}, complete)
        print(json.dumps({"status": "passed", "completed_rows": 34, "finalization": relative(REPORT / "finalization.json")}), flush=True)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(finalize())
