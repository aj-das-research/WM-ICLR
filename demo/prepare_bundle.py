#!/usr/bin/env python3
"""Create a portable, provenance-pinned demo using real held-out clips.

By default copies two PushT clips and one Reacher clip (first sorted test IDs
at dynamics condition2), plus actual frozen checkpoints. No synthetic frames.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    source = project / "src/shiftwm"
    package = output / "runtime/shiftwm"
    package.mkdir(parents=True, exist_ok=True)
    files = ["__init__.py", "model.py", "checkpoint.py", "upstream.py"]
    files += [str(p.relative_to(source)) for p in (source / "vendor").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    runtime_hashes = {}
    for name in files:
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
        runtime_hashes[name] = sha(destination)
    samples = []
    (output / "samples").mkdir(exist_ok=True)
    for environment, relative, count in [("pusht", "pusht_relative", 2), ("reacher", "reacher", 1)]:
        data = project / "data/world" / relative
        manifest_path = data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if environment == "pusht" and manifest.get("action_interface") != "relative":
            raise ValueError("Refusing obsolete absolute-action PushT data")
        episodes = sorted((ep for ep in manifest["episodes"] if ep["split"] == "test" and ep["dynamics_id"] == 2),
                          key=lambda ep: ep["trajectory_id"])[:count]
        for episode in episodes:
            original = data / episode["file"]
            if sha(original) != episode["sha256"]:
                raise ValueError(f"Source episode checksum mismatch: {original}")
            with np.load(original, allow_pickle=False) as archive:
                # H=3 support observations + K=5 evaluated future observations.
                images = archive["images"][:8].copy()
                actions = archive["actions"][:7].copy()
            if images.shape[0] != 8 or actions.shape != (7, 10):
                raise ValueError("Unexpected action/frame sequence")
            name = f"{environment}_{episode['trajectory_id']}.npz"
            target = output / "samples" / name
            np.savez_compressed(target, images=images, actions=actions)
            samples.append({"id": f"{environment}/{episode['trajectory_id']}", "file": f"samples/{name}",
                            "sha256": sha(target), "source_episode_sha256": sha(original),
                            "source_manifest_sha256": sha(manifest_path), "environment": environment,
                            "trajectory_id": episode["trajectory_id"], "split": "test", "start": 0,
                            "dynamics_id": 2, "dynamics": manifest["dynamics_factors"]["2"],
                            "action_units": manifest["action_units"], "action_block": 5,
                            "appearances": manifest["appearance_factors"],
                            "selection_rule": "First sorted test trajectory IDs at dynamics2; first8 frames; no performance selection."})
    (output / "samples/manifest.json").write_text(json.dumps({"samples": samples}, indent=2) + "\n")
    checkpoints = []
    if not args.skip_checkpoints:
        for environment in ("pusht", "reacher"):
            origin = project / f"runs/world/{environment}_frozen_s0/best"
            if not (origin / "model.pt").exists():
                continue
            config = json.loads((origin / "config.json").read_text())
            if config["model_config"]["mode"] != "frozen":
                raise ValueError("Only frozen upstream exports bundled by default")
            destination = output / "checkpoints" / f"{environment}_frozen"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("model.pt", "config.json", "README.md"):
                if (origin / name).is_file():
                    shutil.copy2(origin / name, destination / name)
            descriptor = {"environment": environment, "mode": "frozen", "eligibility": "verified_frozen_export",
                          "model_sha256": sha(destination / "model.pt"), "config_sha256": sha(destination / "config.json"),
                          "upstream_provenance": config["provenance"]}
            (destination / "origin.json").write_text(json.dumps(descriptor, indent=2) + "\n")
            checkpoints.append({"directory": str(destination.relative_to(output)), **descriptor})
        # A prespecified seed/mode, never selected for test-clip performance.
        for environment in ("pusht", "reacher"):
            origin = project / f"artifacts/releases/{environment}_factorized_s0"
            if not (origin / "release_manifest.json").is_file():
                continue
            release = json.loads((origin / "release_manifest.json").read_text())
            if release.get("status") != "ready" or release.get("verification", {}).get("status") != "passed":
                continue
            destination = output / "checkpoints" / origin.name
            destination.mkdir(parents=True, exist_ok=True)
            for name, expected in release["package_files"].items():
                if sha(origin / name) != expected:
                    raise ValueError(f"Release checksum mismatch: {origin / name}")
                shutil.copy2(origin / name, destination / name)
            shutil.copy2(origin / "release_manifest.json", destination / "release_manifest.json")
            checkpoints.append({"directory": str(destination.relative_to(output)), "environment": environment,
                                "mode": "factorized", "eligibility": "completed_training_verified_release",
                                "model_sha256": sha(destination / "model.pt"),
                                "config_sha256": sha(destination / "config.json")})
    (output / "bundle_manifest.json").write_text(json.dumps({"runtime_hashes": runtime_hashes,
                    "checkpoints": checkpoints, "sample_manifest_sha256": sha(output / "samples/manifest.json"),
                    "source_model_card": "MODEL_CARD.md", "publication_status": "local_only"}, indent=2) + "\n")
    print(json.dumps({"samples": len(samples), "bundled_checkpoints": len(checkpoints), "output": str(output)}))


if __name__ == "__main__":
    main()
