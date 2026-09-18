#!/usr/bin/env python3
"""Matched full-training generalization controls; original train/validation only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/real_video"))
import torch
import numpy as np
import train as training
import evaluate as evaluation
from shiftwm.real_video.data import RealVideoDataset

CONFIGS = ROOT / "configs/real_video_development/generalization_v1"
REGISTRY = CONFIGS / "registration.json"
PROTOCOL = "reports/real_droid_generalization_protocol.md"
MODES = ("framewise", "constant_dynamics", "factorized", "action_free")
ARMS = ("slow", "decay", "compact")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    if REGISTRY.exists():
        raise ValueError("Registered campaign already exists; preserve its identity")
    CONFIGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ARMS:
        for seed in (0, 1, 2):
            for mode in MODES:
                name = f"{arm}_{mode}_s{seed}"
                config = json.loads((ROOT / f"configs/real_video/droid_{mode}_s{seed}.json").read_text())
                config.update(output_dir=f"runs/real_video_development/generalization_v1/{name}",
                              protocol_path=PROTOCOL)
                if arm == "slow":
                    config.update(lr=3e-5, min_lr=3e-7)
                elif arm == "decay":
                    config["weight_decay"] = 0.1
                else:
                    config["model_config"] = {"hidden_dim": 96, "depth": 2, "context_dim": 16, "context_hidden": 64}
                path = CONFIGS / (name + ".json")
                if path.exists():
                    raise ValueError("New configuration path already exists")
                training.atomic_json(config, path)
                rows.append({"name": name, "arm": arm, "seed": seed, "mode": mode,
                             "config": str(path.relative_to(ROOT)), "config_sha256": sha(path)})
    dependencies = {str(Path(p).relative_to(ROOT)): d for p, d in training.source_files().items()}
    for relative in (PROTOCOL, str(Path(__file__).relative_to(ROOT)),
                     "scripts/real_video_development/generalization_train.slurm",
                     "data/features/droid_selected_v1/manifest.json",
                     "data/features/droid_selected_v1/training_statistics.json",
                     "data/real_video/droid_selected/processed/manifest.json",
                     "data/real_video/droid_selected/processed/data_audit.json"):
        dependencies[relative] = sha(ROOT / relative)
    registration = {"status": "registered_before_training", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": "original training and validation only", "expected_runs": 36,
                    "expected_validation_evaluations": 72, "dependencies": dependencies, "runs": rows}
    training.atomic_json(registration, REGISTRY)
    print(json.dumps({"status": registration["status"], "runs": len(rows), "registration_sha256": sha(REGISTRY)}))


def verify(registration):
    for relative, expected in registration["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Registered dependency changed: " + relative)
    for row in registration["runs"]:
        if sha(ROOT / row["config"]) != row["config_sha256"]:
            raise ValueError("Registered configuration changed")


def run_group(index):
    if not 0 <= index < 9:
        raise ValueError("Nine arm/seed job groups are registered")
    registration = json.loads(REGISTRY.read_text())
    verify(registration)
    arm, seed = ARMS[index // 3], index % 3
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    datasets = {h: RealVideoDataset(ROOT / "data/features/droid_selected_v1", "val", horizon=h,
                                   stride=5, verify=True) for h in (5, 10)}
    for row in registration["runs"]:
        if (row["arm"], row["seed"]) != (arm, seed):
            continue
        verify(registration)
        config = json.loads((ROOT / row["config"]).read_text())
        config["resume_if_present"] = True
        began = time.monotonic()
        summary = training.train(config)
        checkpoint = ROOT / config["output_dir"] / "best"
        model, package = training.load_package(checkpoint, "cuda")
        outputs = {}
        for horizon, dataset in datasets.items():
            result = evaluation.evaluate_dataset(model, dataset, "cuda", 128)
            output = ROOT / config["output_dir"] / f"validation_h{horizon}.json"
            training.atomic_json({"scope": "validation development only", "arm": arm,
                                  "mode": row["mode"], "seed": seed, "horizon": horizon,
                                  "registration_sha256": sha(REGISTRY), "checkpoint_sha256": sha(checkpoint / "model.pt"),
                                  "result": result}, output)
            outputs[str(horizon)] = str(output.relative_to(ROOT))
        # Strict CPU reload equality on one original-validation causal input.
        cpu_model, _ = training.load_package(checkpoint, "cpu")
        sample = datasets[10][0]
        support = sample["features"][None, :3].float()
        actions = sample["actions"][None].float()
        with torch.inference_mode():
            model.cpu().eval()
            expected = model.predict(support, actions[:, :2], actions[:, 2:])
            actual = cpu_model.predict(support, actions[:, :2], actions[:, 2:])
        if not torch.equal(expected, actual):
            raise ValueError("Offline CPU reload parity failed")
        verify(registration)
        training.atomic_json({"status": "completed", "training": summary, "evaluations": outputs,
                              "offline_cpu_reload_max_error": 0.0, "elapsed_seconds": time.monotonic() - began},
                             ROOT / config["output_dir"] / "development_receipt.json")
        print(json.dumps({"run": row["name"], "status": "completed", "best_epoch": summary["best_epoch"]}), flush=True)


def summarize():
    registration = json.loads(REGISTRY.read_text())
    verify(registration)
    records, missing = [], []
    for row in registration["runs"]:
        config = json.loads((ROOT / row["config"]).read_text())
        directory = ROOT / config["output_dir"]
        receipt = directory / "development_receipt.json"
        if not receipt.exists():
            missing.append(row["name"])
            continue
        training.validate_completed(directory)
        record = {**row, "receipt": json.loads(receipt.read_text()), "validation": {}}
        for h in (5, 10):
            path = directory / f"validation_h{h}.json"
            result = json.loads(path.read_text())
            if result["registration_sha256"] != sha(REGISTRY) or result["checkpoint_sha256"] != sha(directory / "best/model.pt"):
                raise ValueError("Evaluation identity mismatch")
            record["validation"][str(h)] = result
        records.append(record)
    aggregate = []
    for arm in ARMS:
        subset = [r for r in records if r["arm"] == arm]
        if len(subset) != 12:
            continue
        for horizon in (5, 10):
            metric = f"h{horizon}_standardized_mse"
            method_means = {mode: float(np.mean([r["validation"][str(horizon)]["result"]["summary"]["model"][metric]
                                                 for r in subset if r["mode"] == mode])) for mode in MODES}
            matrices = {}
            reference_ids, reference_sessions = None, None
            for mode in MODES:
                matrices[mode] = []
                for r in sorted((r for r in subset if r["mode"] == mode), key=lambda r: r["seed"]):
                    episodes = sorted(r["validation"][str(horizon)]["result"]["episodes"], key=lambda e: e["episode_id"])
                    ids, sessions = [e["episode_id"] for e in episodes], [e["session_id"] for e in episodes]
                    if reference_ids is None:
                        reference_ids, reference_sessions = ids, sessions
                    if ids != reference_ids or sessions != reference_sessions:
                        raise ValueError("Unmatched validation populations")
                    matrices[mode].append([e["errors"]["model"][metric] for e in episodes])
            paired = evaluation.crossed_session_bootstrap(np.array(matrices["factorized"]) - np.array(matrices["framewise"]),
                                                         reference_sessions, draws=10000, seed=5197000 + horizon)
            aggregate.append({"arm": arm, "horizon": horizon, "means": method_means, "paired_ours_minus_framewise": paired,
                              "ours_reduction_vs_framewise_percent": 100 * (method_means["framewise"] - method_means["factorized"]) / method_means["framewise"]})
    # Per-episode observations stay in immutable run outputs; the public report
    # carries source hashes, complete aggregates and paired uncertainty.
    for record in records:
        for horizon, result in record["validation"].items():
            source = ROOT / json.loads((ROOT / record["config"]).read_text())["output_dir"] / f"validation_h{horizon}.json"
            result["source_sha256"] = sha(source)
            result["source"] = str(source.relative_to(ROOT))
            result["result"].pop("episodes")
    report = {"status": "completed" if not missing else "in_progress", "completed_runs": len(records),
              "expected_runs": 36, "missing": missing, "scope": "validation development only",
              "registration_sha256": sha(REGISTRY), "runs": records, "aggregate": aggregate}
    path = ROOT / "reports/real_droid_generalization_results.json"
    training.atomic_json(report, path)
    lines = ["# Matched real-DROID generalization development", "", f"Status: {report['status']}; {len(records)}/36 complete 30-epoch runs.",
             "Original training/validation only. Neither original nor fresh test data select these revisions.", "",
             "| Arm | Horizon | Framewise | Constant dynamics | ShiftWM (ours) | Action-free | Ours vs Framewise | Paired difference interval |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in aggregate:
        means, pair = row["means"], row["paired_ours_minus_framewise"]
        lines.append(f"| {row['arm']} | {row['horizon']} | " + " | ".join(f"{means[m]:.6f}" for m in MODES) +
                     f" | {row['ours_reduction_vs_framewise_percent']:+.3f}% | [{pair['ci95'][0]:+.6f}, {pair['ci95'][1]:+.6f}] |")
    lines += ["", "All paired intervals are exploratory validation intervals, unadjusted for multiple comparisons. Negative MSE differences favor ours; positive percentage reductions favor ours. All completed arms are retained.", ""]
    path.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--register", action="store_true")
    actions.add_argument("--group", type=int)
    actions.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.register:
        register()
    elif args.group is not None:
        run_group(args.group)
    else:
        summarize()


if __name__ == "__main__":
    main()
