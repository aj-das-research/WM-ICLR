#!/usr/bin/env python3
"""Measure full training action distributions without inspecting final test data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/evidence/training_action_distribution.json"))
    args = parser.parse_args()
    report = {"scope": "Every physical training transition, once; appearance replicas and overlapping windows are not duplicated.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "environments": {}}
    for environment, directory in (("pusht", "pusht_relative"), ("reacher", "reacher")):
        root = Path("data/world") / directory
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        stats_path = Path("data/upstream") / environment / "action_stats.json"
        upstream = json.loads(stats_path.read_text())["action"]
        selected = [e for e in manifest["episodes"] if e["split"] == "train"]
        blocks, per_dynamics = [], {}
        for episode in selected:
            with np.load(root / episode["file"], allow_pickle=False) as archive:
                grouped = archive["actions"]
                expected = (episode["steps"], manifest["model_action_dimension"])
                if grouped.shape != expected or not np.isfinite(grouped).all():
                    raise ValueError(f"Invalid training actions: {episode['trajectory_id']}")
                native = grouped.reshape(-1, manifest["native_action_dimension"]).astype(np.float64)
            blocks.append(native)
            per_dynamics.setdefault(episode["dynamics_id"], []).append(native)

        def summarize(parts):
            actions = np.concatenate(parts)
            std = actions.std(axis=0, ddof=1)
            return {"native_actions": len(actions), "mean": actions.mean(axis=0).tolist(),
                    "sample_std": std.tolist(), "minimum": actions.min(axis=0).tolist(),
                    "maximum": actions.max(axis=0).tolist(),
                    "quantile_probabilities": [.01, .05, .5, .95, .99],
                    "quantiles_per_coordinate": np.quantile(actions, [.01, .05, .5, .95, .99], axis=0).tolist(),
                    "std_ratio_to_upstream": (std / np.asarray(upstream["std"])).tolist(),
                    "fraction_coordinates_abs_above_half": float((np.abs(actions) > .5).mean())}

        report["environments"][environment] = {
            "training_episodes": len(selected), "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "normalization_source_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "upstream_action_statistics": upstream, "all": summarize(blocks),
            "by_dynamics": {str(key): summarize(value) for key, value in per_dynamics.items()},
            "interpretation": "Descriptive action-distribution comparison; no normalization changes or causal performance attribution."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    for name, row in report["environments"].items():
        print(name, row["training_episodes"], row["all"]["native_actions"], row["all"]["sample_std"], flush=True)


if __name__ == "__main__":
    main()
