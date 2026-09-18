#!/usr/bin/env python3
"""Package unchanged released LeWM weights for matched ShiftWM evaluation.

This is a format conversion, not new training. Official action statistics are
mandatory. Environment selection allows export before the other archive is ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from shiftwm.checkpoint import save_package
from shiftwm.model import ModelConfig, ShiftWorldModel
from shiftwm.train import read_action_stats
from shiftwm.upstream import load_base


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--env", choices=["pusht", "reacher", "all"], default="all")
    parser.add_argument("--pretrained-root", type=Path, default=Path("data/pretrained"))
    parser.add_argument("--stats-root", type=Path, default=Path("data/upstream"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/world"))
    parser.add_argument("--action-stats", type=Path, help="Explicit stats override; requires one --env")
    args = parser.parse_args()
    if args.action_stats and args.env == "all":
        parser.error("--action-stats requires a single --env")
    envs = ["pusht", "reacher"] if args.env == "all" else [args.env]
    stats_files = {env: args.action_stats or args.stats_root / env / "action_stats.json" for env in envs}
    # Fail before creating packages if any required source is absent.
    for path in stats_files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Official action statistics not ready: {path}")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    for env in envs:
        stats_file = stats_files[env]
        base, config, provenance = load_base(args.pretrained_root / env)
        means, stds = read_action_stats(stats_file, config["action_encoder"]["input_dim"])
        provenance["action_stats"] = str(stats_file.resolve())
        provenance["action_stats_sha256"] = hashlib.sha256(stats_file.read_bytes()).hexdigest()
        provenance["action_interface"] = "relative pusher control, target=position+100*action" if env == "pusht" else "normalized torque with upstream action_repeat=2"
        provenance["upstream_revision"] = subprocess.check_output(
            ["git", "-C", "external/le-wm", "rev-parse", "HEAD"], text=True).strip()
        provenance["implementation_hashes"] = {
            name: hashlib.sha256(Path("src/shiftwm", name).read_bytes()).hexdigest()
            for name in ("model.py", "upstream.py", "checkpoint.py")}
        model = ShiftWorldModel(base, config, ModelConfig(mode="frozen"), means, stds, provenance)
        destination = args.output_root / f"{env}_frozen_s0" / "best"
        metadata = {"environment": env, "training_status": "released_upstream_weights_not_finetuned",
                    "conversion": "strict_state_dict_load_without_parameter_updates",
                    "action_grouping": "five chronological two-dimensional native controls",
                    "preprocessing": "RGB[0,1], ImageNet normalization, bilinear antialiased resize224",
                    "evaluation_status": "not_evaluated_by_export", "seed": 0}
        save_package(model, destination, epoch=0, step=0, metadata=metadata)
        card = Path("artifacts/model_cards/frozen_lewm.md").read_text()
        (destination / "MODEL_CARD.md").write_text(card.replace("{{ENVIRONMENT}}", env))
        manifest = {"environment": env, "directory": str(destination.resolve()),
                    "model_sha256": hashlib.sha256((destination / "model.pt").read_bytes()).hexdigest(),
                    "upstream_weights_sha256": provenance["weights_sha256"],
                    "action_stats_sha256": provenance["action_stats_sha256"], **metadata}
        (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
