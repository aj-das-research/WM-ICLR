#!/usr/bin/env python3
"""Measure a necessary decoder reachability condition on training data only.

This is a model-independent diagnostic, not an evaluation of any checkpoint.
Shared-channel standardization makes the convex mixing range interpretable:
each output coordinate must lie between the initial channel minimum minus b
and maximum plus b, where b bounds innovation. Distances to this interval give
a lower bound on prediction MSE, not an attainable oracle predictor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws.windows import eligible_starts

spec = importlib.util.spec_from_file_location("envelope_campaign", ROOT / "scripts/real_video_iws/campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def interval_error(initial, target, bound):
    """[window, channel, patch] -> per-window squared distance/fraction outside."""
    if initial.shape != target.shape or initial.ndim != 3 or bound < 0:
        raise ValueError("Invalid channel/patch diagnostic inputs")
    lower = initial.min(-1, keepdims=True) - bound
    upper = initial.max(-1, keepdims=True) + bound
    distance = np.maximum(lower - target, 0) + np.maximum(target - upper, 0)
    return np.square(distance).mean((1, 2)), (distance > 0).mean((1, 2))


def self_check():
    initial = np.array([[[0., 2.]]])
    target = np.array([[[-2., 4.]]])
    error, outside = interval_error(initial, target, 1.)
    assert np.array_equal(error, [1.]) and np.array_equal(outside, [1.])
    error, outside = interval_error(initial, np.array([[[.5, 1.5]]]), 0.)
    assert np.array_equal(error, [0.]) and np.array_equal(outside, [0.])
    rng = np.random.default_rng(173)
    source = rng.normal(size=(7, 5, 16))
    transport = rng.uniform(size=(7, 16, 16))
    transport /= transport.sum(-1, keepdims=True)
    gate = rng.uniform(size=(7, 1, 16))
    prediction = ((1-gate)*source + gate*np.einsum('bij,bcj->bci', transport, source)
                  + np.tanh(rng.normal(size=source.shape)))
    error, outside = interval_error(source, prediction, 1.)
    assert np.array_equal(error, np.zeros(7)) and np.array_equal(outside, np.zeros(7))


def run(output):
    self_check()
    if output.exists():
        raise ValueError("Refusing to replace completed diagnostic")
    registration = campaign.check_registration()
    config = campaign.read(campaign.CONFIG)
    train = campaign.trainer()
    bound = float(config["model"]["innovation_bound"])
    bounds = sorted(set((0., bound, 2*bound, 4*bound)))
    offsets = (14, 29, 44, 59)
    result = {
        "schema": "shiftwm_iws_training_reachability_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scope": "Training-only necessary-condition diagnostic; no checkpoint prediction or model ranking.",
        "formula": "mean(square(max(min_patch(z0)-b-target,0)+max(target-max_patch(z0)-b,0)))",
        "interpretation": "Necessary coordinate envelope, a relaxation of shared convex weights. Its MSE floor is not an attainable oracle error and does not establish the cause of a performance gap.",
        "aggregation": "Equal training trajectory mean of equal eligible start means; strict H60 eligibility and stride5.",
        "official_validation_payloads_read": 0,
        "development_payloads_read": 0,
        "model_predictions_computed": 0,
        "training_registration_sha256": campaign.sha(campaign.REGISTRATION),
        "source_sha256": campaign.sha(__file__),
        "source_dependencies": registration["dependencies"],
        "configured_innovation_bound": bound,
        "counterfactual_bounds": bounds,
        "stored_offsets": list(offsets),
        "tasks": {},
    }
    for task in ("pusht", "bimanual_box", "bimanual_rope"):
        cache = train.open_cache(config, task)
        mean = np.asarray(cache.statistics["feature_mean"], dtype=np.float64)
        std = np.asarray(cache.statistics["feature_std"], dtype=np.float64)
        if not np.array_equal(std.reshape(384,16), np.repeat(std.reshape(384,16)[:,:1],16,axis=1)):
            raise ValueError("Expected common normalization for every patch in a channel")
        episodes = []
        for eid in cache.inventory.selected("internal_train"):
            receipt, arrays = cache.episode(eid, "internal_train")
            starts = np.asarray(eligible_starts(receipt["frames"], 60, 5), dtype=int)
            if len(starts) == 0:
                episodes.append({"episode_id": eid, "windows": 0, "status": "no_eligible_start"})
                continue
            features = (arrays["features"].astype(np.float64) - mean) / std
            initial = features[starts].reshape(-1,384,16)
            row = {"episode_id": eid, "windows": len(starts), "payload_sha256": receipt["payload_sha256"], "offsets": {}}
            for offset in offsets:
                target = features[starts+offset].reshape(-1,384,16)
                values = {"persistence_mse": float(np.square(target-initial).mean())}
                for b in bounds:
                    error, fraction = interval_error(initial, target, b)
                    values[str(b)] = {"mse_lower_bound": float(error.mean()), "fraction_coordinates_outside": float(fraction.mean())}
                row["offsets"][str(offset)] = values
            episodes.append(row)
        if [r["episode_id"] for r in episodes] != cache.inventory.partitions["internal_train"]:
            raise ValueError("Training population incomplete")
        eligible = [r for r in episodes if r["windows"]]
        summary = {}
        for offset in offsets:
            values = [r["offsets"][str(offset)] for r in eligible]
            summary[str(offset)] = {"persistence_mse": float(np.mean([r["persistence_mse"] for r in values])),
                                   **{str(b): {key: float(np.mean([r[str(b)][key] for r in values]))
                                              for key in ("mse_lower_bound", "fraction_coordinates_outside")}
                                      for b in bounds}}
        result["tasks"][task] = {"episodes": episodes, "eligible_trajectories": len(eligible),
                                 "windows": sum(r["windows"] for r in episodes), "summary_by_offset": summary}
        print(json.dumps({"task": task, "trajectories": len(episodes), "summary_h60": summary["59"]}), flush=True)
    campaign.check_registration()
    result["status"] = "passed_training_only_diagnostic"
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    campaign.atomic_json(result, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("Reachability analytical and random convex-combination checks passed")
    else:
        if args.output is None:
            parser.error("--output is required")
        run(args.output)
