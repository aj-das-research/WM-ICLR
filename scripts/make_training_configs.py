#!/usr/bin/env python3
"""Materialize the prespecified, identical-budget three-seed comparison."""
import json
from pathlib import Path

root = Path("configs/world")
root.mkdir(parents=True, exist_ok=True)
tasks = []
for seed in (0, 1, 2):
    for environment in ("reacher", "pusht"):
        for mode in ("factorized", "plain", "single"):
            name = f"{environment}_{mode}_s{seed}"
            data_name = "pusht_relative" if environment == "pusht" else environment
            config = {
                "seed": seed, "device": "cuda", "cpu_threads": 4,
                "pretrained_dir": f"data/pretrained/{environment}",
                "action_stats": f"data/upstream/{environment}/action_stats.json",
                "data_root": f"data/world/{data_name}",
                "dataset_kwargs": {"feature_cache": f"data/features/{data_name}",
                                   "preload_features": True},
                "output_dir": f"runs/world/{name}",
                "sequence_length": 8, "stride": 1,
                "epochs": 30, "batch_size": 128, "num_workers": 2,
                "lr": 5e-5, "min_lr": 1e-6, "weight_decay": 1e-3,
                "grad_clip": 1.0, "bf16": True, "log_every": 50,
                "model": {"mode": mode, "context_dim": 32, "context_hidden": 128,
                          "history_length": 3, "freeze_visual": True,
                          "train_predictor": True, "alignment_weight": 1.0,
                          "dynamics_consistency_weight": 0.1,
                          "observation_consistency_weight": 0.01},
            }
            path = root / f"{name}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            tasks.append({"id": name, "config": str(path)})
(root / "training_campaign.json").write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")
print(f"Wrote {len(tasks)} complete-training configurations")
