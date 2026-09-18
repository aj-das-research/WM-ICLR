#!/usr/bin/env python3
"""Audited held-out real-video forecasting and session-cluster uncertainty.

No output of this evaluator selects checkpoints. Exterior-camera-one/horizon-5
is primary; camera-two transfer and horizon-10 extrapolation are separate test
populations. Reversing future action blocks is an association diagnostic, not
an intervention on the real recorded scene.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as training
from shiftwm.real_video.data import RealVideoDataset, sha256

CAMERAS = ("exterior_image_1_left", "exterior_image_2_left")


def baseline_predictions(support, horizon):
    if support.ndim != 3 or support.shape[1] != 3 or horizon < 1:
        raise ValueError("Baselines require the same three observed support frames")
    last = support[:, -1:]
    steps = torch.arange(1, horizon + 1, device=support.device, dtype=support.dtype)[None, :, None]
    return {"persistence": last.expand(-1, horizon, -1),
            "constant_velocity": last + steps * (last - support[:, -2:-1])}


def reverse_future_actions(actions, history=3):
    """Reverse block order; preserve executed support and within-block commands."""
    if actions.ndim != 3 or actions.shape[1] < history:
        raise ValueError("Insufficient chronological action blocks")
    return torch.cat((actions[:, :history - 1], actions[:, history - 1:].flip(1)), 1)


def summarize_episode_errors(rows):
    if not rows or len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate evaluation episode population")
    methods = list(rows[0]["errors"])
    keys = list(rows[0]["errors"][methods[0]])
    return {method: {key: float(np.mean([row["errors"][method][key] for row in rows]))
                     for key in keys} for method in methods}


@torch.inference_mode()
def evaluate_dataset(model, dataset, device="cpu", batch_size=128):
    """Predictions read support and actions only; targets enter error scoring only."""
    device = torch.device(device)
    model.to(device).eval()
    horizon = dataset.horizon
    selected = (1, 3, 5) if horizon == 5 else (10,)
    if horizon not in (5, 10):
        raise ValueError("Only the registered primary and extrapolation horizons are supported")
    accumulators = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(1729))
    for batch in loader:
        features = batch["features"].to(device, dtype=torch.float32)
        actions = batch["actions"].to(device, dtype=torch.float32)
        if features.shape[1] != 3 + horizon or actions.shape[1] != 2 + horizon:
            raise ValueError("Prediction population has incorrect horizon/action alignment")
        support, target = features[:, :3], features[:, 3:]
        with torch.autocast(device_type=device.type, enabled=False):
            predictions = {"model": model.predict(support, actions[:, :2], actions[:, 2:]),
                           **baseline_predictions(support, horizon)}
            reversed_actions = reverse_future_actions(actions)
            predictions["reversed_future_actions"] = model.predict(
                support, reversed_actions[:, :2], reversed_actions[:, 2:])
            errors = {}
            for method, prediction in predictions.items():
                if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                    raise ValueError("Invalid predicted raw-feature trajectory")
                raw = (prediction.float() - target).square().mean(-1)
                standardized = ((prediction.float() - target) / model.feature_std).square().mean(-1)
                cosine = (1 - torch.nn.functional.cosine_similarity(prediction.float(), target, dim=-1)).clamp_min(0)
                errors[method] = {"mean_standardized_mse": standardized.mean(1),
                                  "mean_raw_mse": raw.mean(1), "mean_cosine_error": cosine.mean(1)}
                for h in selected:
                    errors[method][f"h{h}_standardized_mse"] = standardized[:, h - 1]
                    errors[method][f"h{h}_raw_mse"] = raw[:, h - 1]
                    errors[method][f"h{h}_cosine_error"] = cosine[:, h - 1]
        for i, episode_index in enumerate(batch["episode_index"].tolist()):
            episode = dataset.episodes[episode_index]
            episode_id = episode["episode_id"]
            if episode_id not in accumulators:
                accumulators[episode_id] = {"episode_id": episode_id, "session_id": episode["session_id"],
                                           "windows": 0, "window_starts": [],
                                           "errors": {method: {key: 0. for key in metrics} for method, metrics in errors.items()}}
            row = accumulators[episode_id]
            row["windows"] += 1
            row["window_starts"].append(int(batch["window_start"][i]))
            for method, metrics in errors.items():
                for key, values in metrics.items():
                    value = float(values[i])
                    if not np.isfinite(value) or value < 0:
                        raise ValueError("Nonfinite/negative evaluation error")
                    row["errors"][method][key] += value
    rows = []
    for episode_id in sorted(accumulators):
        row = accumulators[episode_id]
        if len(set(row["window_starts"])) != row["windows"]:
            raise ValueError("Duplicate evaluated windows")
        row["errors"] = {method: {key: value / row["windows"] for key, value in metrics.items()}
                         for method, metrics in row["errors"].items()}
        rows.append(row)
    return {"episodes": rows, "summary": summarize_episode_errors(rows),
            "episode_count": len(rows), "session_count": len({row["session_id"] for row in rows}),
            "window_count": sum(row["windows"] for row in rows)}


def evaluate(checkpoint, output, horizon=5, camera=CAMERAS[0], device="cuda", batch_size=128):
    checkpoint = Path(checkpoint).absolute()
    if checkpoint.name != "best":
        raise ValueError("Held-out evaluation accepts the validation-selected best package only")
    if horizon not in (5, 10) or camera not in CAMERAS:
        raise ValueError("Unregistered camera or horizon")
    config = json.loads((checkpoint.parent / "training_config.json").read_text())
    training.validate_completed(checkpoint.parent)
    manifest, _, live_identity = training.audit_inputs(config)
    model, state = training.load_package(checkpoint, device=device)
    if state["config"]["metadata"]["identity"] != live_identity:
        raise ValueError("Current cache/source/protocol differs from training identity")
    training.seed_everything(config["seed"])
    dataset = RealVideoDataset(config["cache_root"], "test", horizon=horizon,
                              stride=5, camera=camera, verify=True)
    result = evaluate_dataset(model, dataset, device=device, batch_size=batch_size)
    population = "primary" if horizon == 5 and camera == CAMERAS[0] else (
        "camera_transfer" if horizon == 5 else "horizon_extrapolation" if camera == CAMERAS[0]
        else "camera_and_horizon_transfer")
    result.update({"status": "completed", "dataset": "DROID real recordings", "split": "test",
                   "population": population, "camera": camera, "horizon": horizon,
                   "mode": config["mode"], "seed": config["seed"],
                   "checkpoint_epoch": state["epoch"], "training_identity": state["config"]["metadata"]["training_identity"],
                   "checkpoint_sha256": sha256(checkpoint / "model.pt"),
                   "checkpoint_path": str(checkpoint), "window_stride": 5,
                   "cache_manifest_sha256": sha256(Path(config["cache_root"]) / "manifest.json"),
                   "dataset_manifest_sha256": manifest["identity"]["dataset_manifest_sha256"],
                   "source_sha256": sha256(__file__), "validation_metric": training.SELECTION,
                   "precision": training.PRECISION,
                   "aggregation": "mean windows within each episode, then equally weight episodes",
                   "action_diagnostic": "reverse future action-block order; support and within-block commands unchanged; observational, not causal",
                   "uncertainty": "report session-cluster bootstrap crossed with three training seeds after complete matched aggregation",
                   "source_dependencies": live_identity["dependencies"],
                   "test_payloads": {str((Path(config["cache_root"]) / row["cameras"][camera]["file"]).resolve()):
                                     row["cameras"][camera]["sha256"] for row in dataset.episodes}})
    training.atomic_json(result, output)
    return result


def crossed_session_bootstrap(differences, sessions, draws=10000, seed=0):
    """Resample sessions and training seeds, preserving all episodes in a cluster.

    Point estimate is equal-episode/equal-seed mean. Sampled clusters retain all
    their episodes, so a larger recording session contributes more episodes;
    this is intentionally not a mean of session means. Values are absolute MSE
    differences (ours minus the named comparator), never percentage points.
    """
    differences = np.asarray(differences, dtype=np.float64)
    if (differences.ndim != 2 or differences.shape[0] != 3
            or differences.shape[1] != len(sessions) or not np.isfinite(differences).all()
            or draws < 1):
        raise ValueError("Bootstrap requires a complete three-seed, matched-episode matrix")
    groups = {session: np.flatnonzero(np.asarray(sessions) == session) for session in sorted(set(sessions))}
    if len(groups) < 2:
        raise ValueError("Session uncertainty requires at least two independent sessions")
    clusters = list(groups.values())
    rng = np.random.default_rng(seed)
    distribution = np.empty(draws)
    for index in range(draws):
        selected_seeds = rng.integers(0, 3, size=3)
        selected_clusters = rng.integers(0, len(clusters), size=len(clusters))
        selected_episodes = np.concatenate([clusters[i] for i in selected_clusters])
        distribution[index] = differences[np.ix_(selected_seeds, selected_episodes)].mean()
    return {"mean_difference": float(differences.mean()),
            "ci95": np.quantile(distribution, [.025, .975]).tolist(),
            "draws": draws, "bootstrap_seed": seed, "session_count": len(groups),
            "training_seed_count": 3, "episode_count": len(sessions),
            "unit": "standardized feature MSE", "direction": "negative favors first named method"}


def validate_result_checkpoint(record):
    """Bind a saved result to a still-valid complete run and selected weights."""
    checkpoint = Path(record["checkpoint_path"])
    if checkpoint.name != "best":
        raise ValueError("Result is not tied to the validation-selected checkpoint")
    summary = training.validate_completed(checkpoint.parent, record["training_identity"])
    if sha256(checkpoint / "model.pt") != record["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after evaluation")
    _, state = training.read_package(checkpoint)
    config = json.loads((checkpoint.parent / "training_config.json").read_text())
    _, _, live = training.audit_inputs(config)
    if (state["config"]["metadata"]["identity"] != live
            or state["epoch"] != record["checkpoint_epoch"]
            or config["mode"] != record["mode"] or config["seed"] != record["seed"]
            or record["source_dependencies"] != live["dependencies"]
            or record.get("window_stride") != 5
            or record.get("validation_metric") != training.SELECTION):
        raise ValueError("Result/run/source identity mismatch")


def aggregate(paths, output, draws=10000):
    records = []
    seen = set()
    for path in paths:
        record = json.loads(Path(path).read_text())
        if record.get("status") != "completed" or record.get("seed") not in (0, 1, 2):
            raise ValueError("Incomplete evaluation result")
        key = (record["mode"], record["seed"])
        if key in seen:
            raise ValueError("Duplicate method/seed result")
        seen.add(key)
        if record["summary"] != summarize_episode_errors(record["episodes"]):
            raise ValueError("Evaluation summary differs from episode records")
        validate_result_checkpoint(record)
        training.verify_sources(record["source_dependencies"])
        training.verify_sources(record["test_payloads"])
        records.append(record)
    expected = {(mode, seed) for mode in training.RealVideoWorldModel.MODES for seed in (0, 1, 2)}
    if seen != expected:
        raise ValueError("Aggregation requires all four registered methods and all three seeds")
    first = records[0]
    identity_keys = ("population", "camera", "horizon", "cache_manifest_sha256", "dataset_manifest_sha256",
                     "source_sha256", "aggregation", "precision")
    episode_keys = [(r["episode_id"], r["session_id"], r["window_starts"]) for r in first["episodes"]]
    for record in records:
        if any(record[key] != first[key] for key in identity_keys):
            raise ValueError("Unmatched evaluation population/protocol")
        if [(r["episode_id"], r["session_id"], r["window_starts"]) for r in record["episodes"]] != episode_keys:
            raise ValueError("Unmatched recording sessions/episode windows")
        for a, b in zip(first["episodes"], record["episodes"]):
            for baseline in ("persistence", "constant_velocity"):
                if a["errors"][baseline] != b["errors"][baseline]:
                    raise ValueError("Checkpoint-independent baseline errors differ across matched runs")
    lookup = {(r["mode"], r["seed"]): r for r in records}
    metrics = [key for key in first["summary"]["model"] if key.endswith("standardized_mse")]
    sessions = [row["session_id"] for row in first["episodes"]]
    comparisons = []
    for reference in ("framewise", "constant_dynamics", "action_free", "persistence", "constant_velocity"):
        for metric in metrics:
            differences = []
            for seed in (0, 1, 2):
                ours = lookup[("factorized", seed)]["episodes"]
                comparator = lookup[(reference, seed)]["episodes"] if reference in training.RealVideoWorldModel.MODES else ours
                comparator_key = "model" if reference in training.RealVideoWorldModel.MODES else reference
                differences.append([a["errors"]["model"][metric] - b["errors"][comparator_key][metric]
                                    for a, b in zip(ours, comparator)])
            comparisons.append({"method": "factorized", "reference": reference, "metric": metric,
                                **crossed_session_bootstrap(differences, sessions, draws=draws)})
    summaries = {}
    for mode in training.RealVideoWorldModel.MODES:
        summaries[mode] = {}
        for metric in first["summary"]["model"]:
            values = [lookup[(mode, seed)]["summary"]["model"][metric] for seed in (0, 1, 2)]
            summaries[mode][metric] = {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)),
                                      "per_seed": values}
    result = {"status": "completed", **{key: first[key] for key in identity_keys},
              "methods": summaries, "paired_comparisons": comparisons,
              "fixed_support_baselines": {name: first["summary"][name] for name in ("persistence", "constant_velocity")},
              "reversed_future_action_diagnostic": {
                  mode: {metric: {"per_seed": [lookup[(mode, seed)]["summary"]["reversed_future_actions"][metric]
                                               for seed in (0, 1, 2)],
                                  "mean": float(np.mean([lookup[(mode, seed)]["summary"]["reversed_future_actions"][metric]
                                                         for seed in (0, 1, 2)]))}
                         for metric in metrics} for mode in training.RealVideoWorldModel.MODES},
              "uncertainty": "paired session-cluster bootstrap crossed with training-seed resampling; no multiplicity adjustment",
              "sources": {str(Path(path).resolve()): sha256(path) for path in paths}}
    training.atomic_json(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, choices=(5, 10), default=5)
    parser.add_argument("--camera", choices=CAMERAS, default=CAMERAS[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--aggregate", nargs="+", metavar="RESULT_JSON")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.aggregate:
        if args.checkpoint:
            parser.error("Choose evaluation or complete aggregation")
        result = aggregate(args.aggregate, args.output, draws=args.bootstrap_draws)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for evaluation")
        result = evaluate(args.checkpoint, args.output, args.horizon, args.camera, args.device, args.batch_size)
    print(json.dumps({key: result[key] for key in ("status", "population")}, sort_keys=True))


if __name__ == "__main__":
    main()
