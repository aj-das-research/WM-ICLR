#!/usr/bin/env python3
"""Freeze, cache and evaluate the untouched fresh DROID confirmatory subset."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from fresh_evaluation_common import (ROOT, FRESH, DECODED, CACHE, OUTPUT, FREEZE,
    PROTOCOL, STATS, MODES, CAMERAS, atomic_json, sha256, scoped_adapter,
    validate_fresh_manifest, verify_freeze)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/real_video"))
import train as training
from evaluate import evaluate_dataset, crossed_session_bootstrap, summarize_episode_errors
from shiftwm.real_video import data as original_data, features as original_features
from shiftwm.real_video_development import ResidualCalibratedWorldModel, residual_scale


def register():
    if FREEZE.exists() or DECODED.exists() or CACHE.exists() or OUTPUT.exists():
        raise ValueError("Freeze must precede all fresh decoding/features/evaluation")
    metadata = json.loads((FRESH / "metadata_manifest.json").read_text())
    if metadata["images_decoded"] != 0 or metadata["model_outputs_computed"] != 0:
        raise ValueError("Fresh population was already opened")
    if metadata["retained_episodes"] != 65 or metadata["retained_sessions"] != 52:
        raise ValueError("Unexpected registered candidate population")
    calibration_campaign = ROOT / "runs/real_video_development/residual_calibration_v1/campaign.json"
    completed = json.loads(calibration_campaign.read_text())
    if completed["status"] != "completed" or completed["test_evaluated"] or len(completed["runs"]) != 12:
        raise ValueError("Require completed training-only fit and validation-only development")
    dependencies = training.source_files()
    paths = list(Path(__file__).parent.glob("fresh_evaluation*"))
    paths += [PROTOCOL, STATS, calibration_campaign,
              ROOT / "configs/real_video_development/residual_calibration_v1.json",
              ROOT / "reports/real_droid_residual_calibration_protocol.md",
              ROOT / "reports/real_droid_residual_calibration_results.json",
              ROOT / "scripts/real_video_development/calibrate_residual.py",
              ROOT / "src/shiftwm/real_video_development.py",
              ROOT / "scripts/real_video/prepare_droid.py",
              ROOT / "data/features/droid_selected_v1/manifest.json",
              ROOT / "data/features/droid_selected_v1/identity.json",
              ROOT / "data/real_video/droid_selected/processed/manifest.json",
              ROOT / "environments/real_video/requirements.lock.txt"]
    paths += [FRESH / name for name in ("registration.json", "inventory.json", "original_exclusions.json",
               "metadata_manifest.json", "metadata_audit.json", "raw/download_receipt.json")]
    paths += list((ROOT / "data/pretrained/dinov2-small").glob("*"))
    runs = []
    source_stats = json.loads(STATS.read_text())
    if source_stats["fit_split"] != "train":
        raise ValueError("Normalization must come from original training only")
    for mode in MODES:
        for seed in (0, 1, 2):
            name = f"droid_{mode}_s{seed}"
            checkpoint = ROOT / "runs/real_video" / name / "best"
            training.validate_completed(checkpoint.parent)
            calibration = ROOT / "configs/real_video_development/calibrations" / (name + ".json")
            package = json.loads(calibration.read_text())
            fit = package["fit"]
            if (package["mode"] != mode or package["seed"] != seed or fit["fit_split"] != "train"
                    or package["base_checkpoint_sha256"] != sha256(checkpoint / "model.pt")
                    or fit["scale"] != residual_scale(fit["displacement_energy"], fit["displacement_alignment"])):
                raise ValueError("Calibration/model/training identity mismatch")
            paths += [calibration, checkpoint / "model.pt", checkpoint / "config.json", checkpoint / "package_manifest.json",
                      checkpoint.parent / "training_config.json"]
            dependencies.update(json.loads((checkpoint / "config.json").read_text())["metadata"]["identity"]["dependencies"])
            runs.append({"mode": mode, "seed": seed, "checkpoint": str(checkpoint),
                         "checkpoint_sha256": package["base_checkpoint_sha256"],
                         "calibration": str(calibration), "scale": fit["scale"]})
    for path in paths:
        if path.is_file():
            # Preserve checkpoint pointer paths so a changed best symlink is
            # detected, even if its previous immutable generation still exists.
            dependencies[str(path.absolute())] = sha256(path)
    training.verify_sources(dependencies)
    freeze = {"status": "frozen_before_fresh_image_decoding", "created_utc": datetime.now(timezone.utc).isoformat(),
              "primary": {"camera": CAMERAS[0], "horizon": 5, "metric": "h5_standardized_mse",
                          "method": "calibrated/factorized", "reference": "calibrated/framewise"},
              "cameras": list(CAMERAS), "horizons": [5, 10], "variants": ["original", "calibrated"],
              "modes": list(MODES), "seeds": [0, 1, 2], "runs": runs, "window_stride": 5,
              "history_length": 3, "batch_size": 128, "feature_batch_size": 32,
              "bootstrap": {"draws": 10000, "seed": 20260919, "unit": "recording session crossed with training seed"},
              "point_estimate": "equal episodes after averaging windows, then equal training seeds",
              "precision": "BF16 frozen encoder, float32 pooling; FP32 prediction with autocast and TF32 disabled",
              "fresh_retained_episodes": 65, "fresh_retained_sessions": 52,
              "new_model_training": False, "fresh_model_selection": False, "dependencies": dependencies}
    atomic_json(freeze, FREEZE)
    print(json.dumps({"status": freeze["status"], "freeze_sha256": sha256(FREEZE),
                      "models": len(runs), "evaluations": 96, "primary": freeze["primary"]}), flush=True)


def copy_original_statistics(root):
    # Callback replacing only the feature extractor's train-statistics phase.
    # No original or fresh arrays are read and no new moments are estimated.
    root = Path(root)
    destination = root / "training_statistics.json"
    if destination.exists() and sha256(destination) != sha256(STATS):
        raise ValueError("Fresh cache normalization differs from original train-only statistics")
    shutil.copyfile(STATS, destination)
    return json.loads(STATS.read_text())


def cache():
    freeze = verify_freeze()
    if not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU")
    training.seed_everything(20260919)
    torch.set_num_threads(8)
    with scoped_adapter(original_features, "validate_manifest", validate_fresh_manifest), scoped_adapter(
            original_features, "training_statistics", copy_original_statistics):
        original_features.extract(DECODED, CACHE, ROOT / "data/pretrained/dinov2-small",
                                  batch_size=freeze["feature_batch_size"], device="cuda")
    validate_fresh_manifest(json.loads((CACHE / "manifest.json").read_text()))
    if sha256(CACHE / "training_statistics.json") != sha256(STATS):
        raise ValueError("Original normalization copy changed")
    verify_freeze()
    atomic_json({"status": "complete", "evaluation_freeze_sha256": sha256(FREEZE),
                 "fresh_feature_manifest_sha256": sha256(CACHE / "manifest.json"),
                 "original_training_statistics_sha256": sha256(STATS),
                 "training_statistics_refitted": False, "fresh_data_used_for_training": False,
                 "adapter": "test-only manifest validation and unchanged training-statistics copy; original pixel/feature computation unchanged"},
                CACHE / "fresh_cache_audit.json")


def fresh_dataset(camera, horizon):
    with scoped_adapter(original_data, "validate_manifest", validate_fresh_manifest):
        return original_data.RealVideoDataset(CACHE, "test", camera=camera, horizon=horizon, stride=5, verify=True)


def evaluate():
    freeze = verify_freeze()
    if not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU")
    training.seed_everything(20260919)
    torch.set_num_threads(8)
    audit = json.loads((CACHE / "fresh_cache_audit.json").read_text())
    if (audit["evaluation_freeze_sha256"] != sha256(FREEZE)
            or audit["fresh_feature_manifest_sha256"] != sha256(CACHE / "manifest.json")
            or sha256(CACHE / "training_statistics.json") != sha256(STATS)):
        raise ValueError("Fresh feature audit or original statistics changed")
    stats = json.loads(STATS.read_text())
    datasets = {(camera, horizon): fresh_dataset(camera, horizon) for camera in CAMERAS for horizon in (5, 10)}
    population_audit = {}
    for (camera, horizon), dataset in datasets.items():
        eligible_indices = {index for index, _ in dataset.windows}
        population_audit[f"{camera}/h{horizon}"] = {
            "retained_episodes": len(dataset.episodes), "eligible_episodes": len(eligible_indices),
            "eligible_sessions": len({dataset.episodes[index]["session_id"] for index in eligible_indices}),
            "windows": len(dataset.windows),
            "short_episode_ids": [row["episode_id"] for index, row in enumerate(dataset.episodes) if index not in eligible_indices]}
    atomic_json(population_audit, OUTPUT / "population_audit.json")
    completed = []
    for run in freeze["runs"]:
        if sha256(Path(run["checkpoint"]) / "model.pt") != run["checkpoint_sha256"]:
            raise ValueError("Frozen validation-selected checkpoint changed")
        base, state = training.load_package(run["checkpoint"], device="cuda")
        for field in ("feature_mean", "feature_std", "action_mean", "action_std"):
            torch.testing.assert_close(getattr(base, field).cpu(), torch.tensor(stats[field], dtype=torch.float32), rtol=0, atol=0)
        for variant in ("original", "calibrated"):
            model = base if variant == "original" else ResidualCalibratedWorldModel(base, run["scale"]).cuda().eval()
            for (camera, horizon), dataset in datasets.items():
                path = OUTPUT / variant / f"droid_{run['mode']}_s{run['seed']}" / camera / f"h{horizon}.json"
                identity = {"evaluation_freeze_sha256": sha256(FREEZE), "cache_manifest_sha256": sha256(CACHE / "manifest.json"),
                            "checkpoint_sha256": run["checkpoint_sha256"], "calibration_sha256": sha256(run["calibration"]),
                            "variant": variant, "mode": run["mode"], "seed": run["seed"], "camera": camera, "horizon": horizon}
                if path.exists():
                    result = json.loads(path.read_text())
                    if result.get("identity") != identity or result.get("status") != "completed":
                        raise ValueError("Existing evaluation belongs to another frozen identity")
                    if result["summary"] != summarize_episode_errors(result["episodes"]):
                        raise ValueError("Existing result summary differs from episode records")
                else:
                    result = evaluate_dataset(model, dataset, "cuda", batch_size=freeze["batch_size"])
                    result.update(status="completed", identity=identity, checkpoint_epoch=state["epoch"],
                                  scale=1.0 if variant == "original" else run["scale"],
                                  aggregation=freeze["point_estimate"], precision=freeze["precision"],
                                  input_contract="three support frames, two prior command blocks, future command blocks only; query frames used for error scoring only")
                    atomic_json(result, path)
                completed.append({"path": str(path), "sha256": sha256(path), **identity})
                atomic_json({"status": "running", "completed_evaluations": len(completed), "expected_evaluations": 96,
                             "evaluation_freeze_sha256": sha256(FREEZE)}, OUTPUT / "progress.json")
                print(json.dumps({"stage": "fresh_evaluation", "completed": len(completed), "total": 96,
                                  "mode": run["mode"], "seed": run["seed"], "variant": variant,
                                  "camera": camera, "horizon": horizon}), flush=True)
        del model, base
        torch.cuda.empty_cache()
    # Verify cached observations/actions again after all inference; no mutation
    # of fresh payloads or original checkpoints/statistics is permitted.
    for row in json.loads((CACHE / "manifest.json").read_text())["episodes"]:
        for record in row["cameras"].values():
            if sha256(CACHE / record["file"]) != record["sha256"]:
                raise ValueError("Fresh feature payload changed during evaluation")
    verify_freeze()
    atomic_json({"status": "completed", "evaluations": completed, "evaluation_freeze_sha256": sha256(FREEZE),
                 "completed_utc": datetime.now(timezone.utc).isoformat(),
                 "all_original_artifacts_and_fresh_metadata_verified_before_after": True,
                 "fresh_payloads_verified_before_after": True}, OUTPUT / "campaign.json")
    aggregate()


def aggregate():
    freeze = verify_freeze()
    campaign = json.loads((OUTPUT / "campaign.json").read_text())
    if campaign["status"] != "completed" or len(campaign["evaluations"]) != 96 or campaign["evaluation_freeze_sha256"] != sha256(FREEZE):
        raise ValueError("Require all 96 completed matched evaluations")
    lookup = {}
    for source in campaign["evaluations"]:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Saved evaluation changed")
        record = json.loads(Path(source["path"]).read_text())
        identity = record["identity"]
        key = (identity["camera"], identity["horizon"], identity["variant"], identity["mode"], identity["seed"])
        if key in lookup or record["status"] != "completed" or record["summary"] != summarize_episode_errors(record["episodes"]):
            raise ValueError("Duplicate, incomplete or inconsistent matched evaluation")
        lookup[key] = record
    populations = {}
    for camera in CAMERAS:
        for horizon in (5, 10):
            records = {(variant, mode, seed): lookup[(camera, horizon, variant, mode, seed)]
                       for variant in ("original", "calibrated") for mode in MODES for seed in (0, 1, 2)}
            first = records[("original", "framewise", 0)]
            episode_keys = [(r["episode_id"], r["session_id"], r["window_starts"]) for r in first["episodes"]]
            for record in records.values():
                if [(r["episode_id"], r["session_id"], r["window_starts"]) for r in record["episodes"]] != episode_keys:
                    raise ValueError("Unmatched fresh evaluation episodes/windows")
                for a, b in zip(first["episodes"], record["episodes"]):
                    for name in ("persistence", "constant_velocity"):
                        if a["errors"][name] != b["errors"][name]:
                            raise ValueError("Fixed-support baseline changed across methods")
            summaries = {}
            for variant in ("original", "calibrated"):
                for mode in MODES:
                    summaries[f"{variant}/{mode}"] = {
                        metric: {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)), "per_seed": values}
                        for metric in first["summary"]["model"]
                        for values in [[records[(variant, mode, seed)]["summary"]["model"][metric] for seed in (0, 1, 2)]]}
            pairs = []
            for variant in ("original", "calibrated"):
                for comparator in ("framewise", "constant_dynamics", "action_free", "persistence", "constant_velocity"):
                    pairs.append(((variant, "factorized"), (variant, comparator)))
            pairs.extend((("calibrated", mode), ("original", mode)) for mode in MODES)
            comparisons = []
            sessions = [row["session_id"] for row in first["episodes"]]
            for (left_variant, left_mode), (right_variant, right_mode) in pairs:
                for metric in (f"h{horizon}_standardized_mse", "mean_standardized_mse"):
                    differences, references = [], []
                    for seed in (0, 1, 2):
                        left = records[(left_variant, left_mode, seed)]["episodes"]
                        right = records[(right_variant, right_mode, seed)]["episodes"] if right_mode in MODES else left
                        comparator_key = "model" if right_mode in MODES else right_mode
                        differences.append([a["errors"]["model"][metric] - b["errors"][comparator_key][metric] for a, b in zip(left, right)])
                        references.append([row["errors"][comparator_key][metric] for row in right])
                    summary = crossed_session_bootstrap(
                        differences, sessions, draws=freeze["bootstrap"]["draws"], seed=freeze["bootstrap"]["seed"])
                    summary.update(method=f"{left_variant}/{left_mode}", reference=f"{right_variant}/{right_mode}" if right_mode in MODES else right_mode,
                                   metric=metric, relative_reduction_percent=-100 * float(np.mean(differences)) / float(np.mean(references)))
                    comparisons.append(summary)
            populations[f"{camera}/h{horizon}"] = {"camera": camera, "horizon": horizon,
                "episode_count": first["episode_count"], "session_count": first["session_count"], "window_count": first["window_count"],
                "methods": summaries, "paired_comparisons": comparisons,
                "fixed_support_baselines": {name: first["summary"][name] for name in ("persistence", "constant_velocity")},
                "reversed_future_action_diagnostic": {f"{variant}/{mode}": {
                    metric: float(np.mean([records[(variant, mode, seed)]["summary"]["reversed_future_actions"][metric] for seed in (0, 1, 2)]))
                    for metric in first["summary"]["model"]} for variant in ("original", "calibrated") for mode in MODES}}
    primary_population = populations[f"{CAMERAS[0]}/h5"]
    primary = next(row for row in primary_population["paired_comparisons"] if row["method"] == freeze["primary"]["method"]
                   and row["reference"] == freeze["primary"]["reference"] and row["metric"] == freeze["primary"]["metric"])
    result = {"status": "completed", "evaluation_freeze_sha256": sha256(FREEZE), "protocol_sha256": sha256(PROTOCOL),
              "campaign_sha256": sha256(OUTPUT / "campaign.json"), "metadata_manifest_sha256": sha256(FRESH / "metadata_manifest.json"),
              "fresh_feature_manifest_sha256": sha256(CACHE / "manifest.json"), "primary_comparison": primary,
              "primary_population": f"{CAMERAS[0]}/h5", "populations": populations,
              "uncertainty": "single primary 95% interval; all secondary intervals descriptive and unadjusted for multiplicity",
              "selection": "All methods/scalars frozen on original training/validation before decoding this subset; all eligible episodes retained",
              "limitations": "Small session-disjoint subset; not proven scene/object disjoint; observational feature forecasting, not physical control or novel calibration",
              "sources": {row["path"]: row["sha256"] for row in campaign["evaluations"]}}
    path = ROOT / "reports/real_droid_fresh_evaluation_results.json"
    atomic_json(result, path)
    lines = ["# Fresh real-DROID confirmatory comparison", "", "Completed all 96 evaluations; no new training or fresh-data selection.", "",
             f"Primary calibrated ShiftWM (ours) vs calibrated Framewise: {primary['relative_reduction_percent']:+.3f}% error reduction; "
             f"MSE difference {primary['mean_difference']:+.6f}, paired 95% CI [{primary['ci95'][0]:+.6f}, {primary['ci95'][1]:+.6f}].", "",
             "Negative MSE differences favor the first method. Positive reduction percentages favor ours. The primary endpoint is h5, not a mean across five frames.", ""]
    labels = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics", "factorized": "ShiftWM (ours)", "action_free": "Action-free"}
    for name, population in populations.items():
        metric = f"h{population['horizon']}_standardized_mse"
        lines.extend([f"## {name}", "", f"{population['episode_count']} eligible episodes, {population['session_count']} sessions, {population['window_count']} windows.", "",
                      "| Method | Original MSE | Calibrated MSE |", "|---|---:|---:|"])
        for mode in MODES:
            a, b = (population["methods"][f"{variant}/{mode}"][metric]["mean"] for variant in ("original", "calibrated"))
            lines.append(f"| {labels[mode]} | {a:.6f} | {b:.6f} |")
        for baseline in ("persistence", "constant_velocity"):
            lines.append(f"| {baseline.replace('_', ' ').title()} | {population['fixed_support_baselines'][baseline][metric]:.6f} | Same fixed baseline |")
        lines.extend(["", "| Calibrated ours versus | Error reduction | Paired MSE difference 95% CI |", "|---|---:|---:|"])
        for row in population["paired_comparisons"]:
            if row["method"] == "calibrated/factorized" and row["reference"] != "original/factorized" and row["metric"] == metric:
                lines.append(f"| {row['reference']} | {row['relative_reduction_percent']:+.3f}% | [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] |")
        lines.append("")
    lines += ["Secondary intervals are descriptive and have no multiplicity adjustment. All metrics, seed values, comparisons, action reversal diagnostics and source hashes are in the accompanying JSON.", "",
              result["limitations"], "", f"Evaluation freeze SHA256: `{sha256(FREEZE)}`", f"Result SHA256: `{sha256(path)}`", ""]
    (ROOT / "reports/real_droid_fresh_evaluation_results.md").write_text("\n".join(lines))
    verify_freeze()
    print(json.dumps({"status": "completed", "primary": primary, "result": str(path)}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("operation", choices=("register", "cache", "evaluate", "aggregate"))
    args = parser.parse_args()
    {"register": register, "cache": cache, "evaluate": evaluate, "aggregate": aggregate}[args.operation]()
