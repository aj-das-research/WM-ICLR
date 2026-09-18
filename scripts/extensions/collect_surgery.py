#!/usr/bin/env python3
"""Reproducible, whole-episode LapGym data with separate privileged audits."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from shiftwm.extensions.surgery import SurgeryAdapter, SurgeryConfig, UPSTREAM_COMMIT, exploration_actions

GAINS = (1.0, 0.75, 1.25)
TRAIN_COMBINATIONS = ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(task):
    root = Path(task["root"])
    uid = f"{task['split']}-s{task['seed']}-d{task['dynamics_id']}"
    file = Path("episodes") / f"{uid}.npz"
    sidecar = (root / file).with_suffix(".json")
    task_hash = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    if sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record["task_sha256"] != task_hash or record["sha256"] != digest(root / file):
            raise RuntimeError(f"Existing incompatible/corrupt episode {uid}")
        return record
    config = SurgeryConfig(seed=task["seed"], actuator_gain=GAINS[task["dynamics_id"]], image_size=task["image_size"])
    commands = exploration_actions(task["seed"], task["native_calls"])
    begin = time.perf_counter()
    env = SurgeryAdapter(config)
    images = [env.observe()]
    initial = env.diagnostics()
    diagnostics = [initial]
    for command in commands:
        image, diag = env.step(command)
        images.append(image)
        diagnostics.append(diag)
    env.close()
    images = np.asarray(images, dtype=np.uint8)
    commands = np.asarray(commands, dtype=np.float32)
    states = np.array([d["tissue_target_xyz_m"] for d in diagnostics])
    if not np.isfinite(states).all() or np.ptp(images.astype(np.float32)) == 0:
        raise RuntimeError(f"Invalid state or constant blank rendering: {uid}")
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    (root / "audit").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / file, images=images[::5], actions=commands.reshape(-1, 10))
    audit_file = Path("audit") / f"{uid}.npz"
    np.savez_compressed(root / audit_file,
                        images_native=images, actions_commanded=commands,
                        actions_executed=np.array([d["executed_action"] for d in diagnostics[1:]], np.float32),
                        gripper_delta_xyz_m=np.array([d["gripper_delta_xyz_m"] for d in diagnostics[1:]]),
                        tissue_target_xyz_m=states,
                        gripper_poses=np.array([d["gripper_pose"] for d in diagnostics]),
                        goal_xyz_m=np.array([d["goal_xyz_m"] for d in diagnostics]),
                        distance_m=np.array([d["distance_m"] for d in diagnostics]),
                        success=np.array([d["success"] for d in diagnostics]),
                        valid_action=np.array([d["valid_action"] for d in diagnostics[1:]]),
                        stable_deformation=np.array([d["stable_deformation"] for d in diagnostics[1:]]))
    record = {"trajectory_id": uid, "split": task["split"], "seed": task["seed"],
              "dynamics_id": task["dynamics_id"], "steps": len(commands) // 5,
              "file": str(file), "sha256": digest(root / file),
              "audit_file": str(audit_file), "audit_sha256": digest(root / audit_file),
              "task_sha256": task_hash, "config": asdict(config),
              "native_calls": len(commands), "wall_seconds": time.perf_counter() - begin,
              "invalid_action_calls": sum(not d["valid_action"] for d in diagnostics[1:]),
              "unstable_calls": sum(not d["stable_deformation"] for d in diagnostics[1:]),
              "native_goal_sampling": True, "reference_image_goal": False}
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"episode": uid, "seconds": record["wall_seconds"]}), flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/extensions/surgery_v1"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--native-calls", type=int, default=200)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--train-per-gain", type=int, default=64)
    p.add_argument("--val-per-gain", type=int, default=16)
    p.add_argument("--development-per-gain", type=int, default=16)
    p.add_argument("--test-per-gain", type=int, default=32)
    args = p.parse_args()
    if args.native_calls % 5:
        p.error("native calls must be divisible by five")
    counts = {split: getattr(args, f"{split}_per_gain") for split in ("train", "val", "development", "test")}
    source_hash = {str(path): digest(path) for path in [Path(__file__), Path("src/shiftwm/extensions/surgery.py")]}
    tasks = []
    for split_index, (split, count) in enumerate(counts.items()):
        for dyn in range(3):
            for index in range(count):
                # Identical seed/exploration across gains is paired only within a split.
                tasks.append({"root": str(args.output), "split": split,
                              "seed": 4200000 + split_index * 100000 + index,
                              "dynamics_id": dyn, "native_calls": args.native_calls,
                              "image_size": args.image_size, "source_hashes": source_hash})
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        episodes = list(pool.map(collect, tasks))
    manifest = {"schema_version": 1, "env": "surgery", "task": "LapGym TissueManipulationEnv",
                "upstream_commit": UPSTREAM_COMMIT, "upstream_license": "MIT",
                "source_hashes": source_hash, "train_combinations": TRAIN_COMBINATIONS,
                "dynamics": {str(i): {"actuator_gain": gain} for i, gain in enumerate(GAINS)},
                "action_block": 5, "native_action_dim": 2, "image_size": args.image_size,
                "model_inputs": ["images", "actions"], "privileged_audits_are_not_model_inputs": True,
                "episodes": episodes, "wall_seconds": time.perf_counter() - start,
                "num_episodes": len(episodes), "native_calls": sum(e["native_calls"] for e in episodes),
                "goal_protocol": "native_random_targets_for_forecasting; no_reference_image_goals_in_this_dataset"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "episodes": len(episodes), "wall_seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
