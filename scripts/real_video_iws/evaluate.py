#!/usr/bin/env python3
"""Score every internal-development window after a complete IWS training run.

This evaluator cannot load official-validation data. It keeps primitive window
errors plus independently aggregated trajectory scores; RGB is never predicted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws.windows import IWSWindowDataset

spec = importlib.util.spec_from_file_location("iws_development_campaign", Path(__file__).with_name("campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
METRICS = ("standardized_mse", "standardized_mae", "raw_dinov2_l1", "feature_cosine_distance")


def feature_errors(prediction, target, feature_std, epsilon=1e-8):
    """Fixed FP32 coordinate errors; one value per window and future offset."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != len(feature_std):
        raise ValueError("Feature metric shapes differ")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite forecast/reference")
    if not torch.isfinite(feature_std).all() or (feature_std <= 0).any():
        raise ValueError("Feature scales must be finite and positive")
    difference = prediction.float() - target.float()
    normalized = difference / feature_std.float()
    # Epsilon applies to each vector norm independently, including zero vectors.
    p, t = prediction.float(), target.float()
    cosine = ((p / torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp_min(epsilon)) *
              (t / torch.linalg.vector_norm(t, dim=-1, keepdim=True).clamp_min(epsilon))).sum(-1).clamp(-1, 1)
    return {"standardized_mse": normalized.square().mean(-1),
            "standardized_mae": normalized.abs().mean(-1),
            "raw_dinov2_l1": difference.abs().mean(-1),
            "feature_cosine_distance": 1-cosine}


def aggregate_windows(arrays, audit):
    expected = [(r["episode_index"], start) for r in audit["records"]
                for start in range(0, r["frames"]-60, 5)]
    pairs = list(zip(arrays["episode_index"].tolist(), arrays["window_start"].tolist()))
    if pairs != expected or len(set(pairs)) != len(pairs):
        raise ValueError("Missing, reordered, duplicated or unregistered development windows")
    keys = list(METRICS) + ["persistence_" + key for key in METRICS]
    for key in keys:
        values = arrays[key]
        if values.shape != (len(expected), 59) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid complete-horizon primitive errors: " + key)
    episodes = []
    for row in audit["records"]:
        if row["windows"] == 0:
            continue
        selected = arrays["episode_index"] == row["episode_index"]
        if int(selected.sum()) != row["windows"]:
            raise ValueError("Trajectory window denominator differs")
        episodes.append({"episode_id": row["episode_id"], "windows": row["windows"],
                         **{key + "_by_offset": arrays[key][selected].mean(0).tolist() for key in keys}})
    if len(episodes) != audit["eligible_episodes"] or len(expected) != audit["windows"]:
        raise ValueError("Development population is incomplete")
    return episodes


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def evaluate(config_path, name, output, device="cuda"):
    registration = campaign.check_registration(config_path)
    config = campaign.read(config_path)
    row = next((r for r in registration["runs"] if r["name"] == name), None)
    if row is None:
        raise ValueError("Unknown registered run")
    train = campaign.trainer()
    directory = ROOT / row["output"]
    summary = train.validate_completed(directory)
    package_path, state = train.read_package(directory / "best")
    recipe = state["config"]["metadata"]["identity"]["scientific_config"]
    expected_recipe = {"task": row["task"], "mode": row["mode"], "seed": row["seed"],
                       "training": config["training"], "model": config["model"],
                       "task_config": config["tasks"][row["task"]], "study_config_sha256": campaign.sha(config_path)}
    if recipe != expected_recipe:
        raise ValueError("Selected checkpoint belongs to another registered recipe/task/arm/seed")
    if device == "cuda":
        train.allocated_device()
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = train.load_package(directory / "best", device)
    cache = train.open_cache(config, row["task"])
    dataset = IWSWindowDataset(cache, "internal_development", horizon=60, stride=5)
    if dataset.audit != state["config"]["metadata"]["identity"]["populations"]["val"]:
        raise ValueError("Development population changed since training")
    loader = DataLoader(dataset, batch_size=config["training"]["microbatch_size"], shuffle=False, num_workers=0)
    collected = {key: [] for key in ["episode_index", "window_start", *METRICS, *["persistence_"+k for k in METRICS]]}
    maximum_prefix_difference = 0.
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for batch in loader:
            initial, commands, targets = (batch[k].to(device) for k in ("initial_features", "commands", "targets"))
            prediction = model.predict(initial, commands)
            errors = feature_errors(prediction, targets, model.feature_std)
            # Each secondary endpoint receives only its declared command prefix.
            # The all-H60 curve is retained except these explicitly rescored cells.
            for horizon in (15, 30, 45):
                prefix = model.predict(initial, commands[:, :horizon])[:, -1:]
                gap = (prefix - prediction[:, horizon-2:horizon-1]).abs().max().item()
                maximum_prefix_difference = max(maximum_prefix_difference, gap)
                if not torch.allclose(prefix, prediction[:, horizon-2:horizon-1], rtol=1e-5, atol=2e-5):
                    raise ValueError("Causal prefix consistency failed beyond FP32 roundoff")
                values = feature_errors(prefix, targets[:, horizon-2:horizon-1], model.feature_std)
                for key in METRICS:
                    errors[key][:, horizon-2] = values[key][:, 0]
            persistence = initial[:, None].expand_as(targets)
            baseline = feature_errors(persistence, targets, model.feature_std)
            for key in METRICS:
                collected[key].append(errors[key].double().cpu().numpy())
                collected["persistence_"+key].append(baseline[key].double().cpu().numpy())
            for key in ("episode_index", "window_start"):
                collected[key].append(batch[key].numpy())
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    episodes = aggregate_windows(arrays, dataset.audit)
    h60 = float(np.mean([e["standardized_mse_by_offset"][-1] for e in episodes]))
    if not math.isclose(h60, summary["best_validation_mse"], rel_tol=2e-6, abs_tol=2e-7):
        raise ValueError("Reloaded selected endpoint does not reproduce training selection")
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to replace a completed evaluation")
    ledger = output.with_suffix(".npz")
    atomic_npz(ledger, arrays)
    value = {"schema": "shiftwm_iws_development_evaluation_v1", "status": "passed", "completed_utc": campaign.now(),
             "scope": "internal_development", "task": row["task"], "mode": row["mode"], "seed": row["seed"],
             "completed_epochs": 30, "selected_epoch": summary["best_epoch"], "selected_checkpoint_sha256": campaign.sha(package_path / "model.pt"),
             "registration_sha256": campaign.sha(campaign.REGISTRATION), "config_sha256": campaign.sha(config_path),
             "episodes": episodes, "total_windows": len(dataset), "eligible_trajectories": len(episodes),
             "horizons": [15, 30, 45, 60], "offsets": list(range(1, 60)), "population_audit": dataset.audit,
             "metrics": list(METRICS), "primary_aggregation": "equal_trajectory", "h60_standardized_mse": h60,
             "prefix_maximum_absolute_difference": maximum_prefix_difference,
             "cosine_formula": "1-clamp(dot(p/max(norm(p),1e-8),t/max(norm(t),1e-8)),-1,1)",
             "official_validation_payloads_read": 0, "window_ledger_path": str(ledger.resolve().relative_to(ROOT)),
             "window_ledger_sha256": campaign.sha(ledger), "evaluator_sha256": campaign.sha(__file__)}
    campaign.check_registration(config_path)
    campaign.atomic_json(value, output)
    return {"status": "passed", "run": name, "episodes": len(episodes), "windows": len(dataset), "H60_MSE": h60}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=campaign.CONFIG)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, args.name, args.output, args.device), sort_keys=True), flush=True)
