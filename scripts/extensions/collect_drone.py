#!/usr/bin/env python3
"""Collect canonical RGB/action drone trajectories with separate privileged audits."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from shiftwm.extensions.drone import DroneConfig, VisualDroneEnv, DYNAMICS_GAINS, UPSTREAM_COMMIT

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def collect(task):
    output = Path(task["output"])
    uid = f"drone-{task['split']}-s{task['seed']}-d{task['dynamics_id']}"
    model_path = output / "episodes" / f"{uid}.npz"
    audit_path = output / "audit" / f"{uid}.npz"
    sidecar = output / "episodes" / f"{uid}.json"
    digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    if sidecar.exists():
        previous = json.loads(sidecar.read_text())
        if previous["config_sha256"] != digest:
            raise ValueError(f"Incompatible existing episode {uid}")
        if previous["sha256"] != sha256(model_path) or previous["audit_sha256"] != sha256(audit_path):
            raise ValueError(f"Existing episode hash mismatch {uid}")
        return previous
    start = time.perf_counter()
    config = DroneConfig(image_size=task["image_size"], max_steps=task["native_steps"])
    env = VisualDroneEnv(task["dynamics_id"], config)
    rng = np.random.default_rng(task["seed"] + 9000000)
    image, _ = env.reset(seed=task["seed"])
    images, states = [image], [env.simulator_state()]
    actions, executed, rpm = [], [], []
    target = rng.uniform(-0.28, 0.28, 2)
    max_speed, max_altitude_error = 0.0, 0.0
    try:
        for step in range(task["native_steps"]):
            # The collector may read state; learned training/evaluation inputs
            # contain only RGB and commanded actions. No goal state is an input.
            if step % 25 == 0:
                target = rng.uniform(-0.28, 0.28, 2)
            if step >= task["native_steps"] - 20:
                action = np.zeros(2, dtype=np.float32)
            else:
                delta = target - env.simulator_state()[:2]
                action = np.clip(4 * delta + rng.normal(0, 0.04, 2), -1, 1).astype(np.float32)
            image, _, terminated, _, info = env.step(action)
            if terminated:
                raise RuntimeError(f"Collector left valid task region: {uid}, step={step}, {info}")
            images.append(image)
            states.append(env.simulator_state())
            actions.append(action)
            executed.append(env.last_executed_action.copy())
            rpm.append(env.last_rpm.copy())
            max_speed = max(max_speed, info["speed_m_s"])
            max_altitude_error = max(max_altitude_error, info["altitude_error_m"])
        native_rgb = np.asarray(images, dtype=np.uint8)
        native_actions = np.asarray(actions, dtype=np.float32)
        states = np.asarray(states, dtype=np.float64)
        executed = np.asarray(executed, dtype=np.float32)
        rpm = np.asarray(rpm, dtype=np.float32)
        if not all(np.isfinite(v).all() for v in (states, executed, rpm, native_actions)):
            raise FloatingPointError(uid)
        block = task["action_block"]
        grouped_actions = native_actions.reshape(-1, block * 2)
        grouped_images = native_rgb[::block]
        assert len(grouped_images) == len(grouped_actions) + 1
        write_npz(model_path, images=grouped_images, actions=grouped_actions)
        write_npz(audit_path, images=native_rgb, actions=native_actions,
                  executed_actions=executed, low_level_rpm=rpm,
                  simulator_states=states, goal_image=native_rgb[-1], goal_state=states[-1])
        env.set_goal(states[-1])
        goal_metrics = env.metrics()
        if not goal_metrics["success"]:
            raise ValueError(f"Recorded goal not settled/reachable under task criteria: {uid}, {goal_metrics}")
        metadata = {"trajectory_id": uid, "split": task["split"], "seed": task["seed"],
                    "dynamics_id": task["dynamics_id"], "steps": len(grouped_actions),
                    "native_steps": task["native_steps"], "action_block": block,
                    "file": str(model_path.relative_to(output)), "sha256": sha256(model_path),
                    "audit_file": str(audit_path.relative_to(output)), "audit_sha256": sha256(audit_path),
                    "config_sha256": digest, "protocol": env.protocol(),
                    "collection_seconds": time.perf_counter() - start,
                    "max_speed_m_s": max_speed, "max_altitude_error_m": max_altitude_error,
                    "goal_native_index": task["native_steps"], "goal_source": "actual recorded settled endpoint",
                    "goal_displacement_m": float(np.linalg.norm(states[-1, :2] - states[0, :2])),
                    "terminal_goal_metrics": goal_metrics}
        write_json(sidecar, metadata)
        return metadata
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-gain", type=int, default=64)
    parser.add_argument("--val-per-gain", type=int, default=16)
    parser.add_argument("--development-per-gain", type=int, default=16)
    parser.add_argument("--test-per-gain", type=int, default=32)
    parser.add_argument("--native-steps", type=int, default=200)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=53000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.native_steps < 30 or args.native_steps % args.action_block:
        raise ValueError("Require at least30nativecalls and exact action blocks")
    upstream = subprocess.check_output(["git", "-C", str(ROOT / "external/gym-pybullet-drones"),
                                       "rev-parse", "HEAD"], text=True).strip()
    if upstream != UPSTREAM_COMMIT:
        raise ValueError("Upstream source commit changed")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in
                     [Path(__file__).resolve(), ROOT / "src/shiftwm/extensions/drone.py"]}
    tasks = []
    counts = [("train", args.train_per_gain), ("val", args.val_per_gain),
              ("development", args.development_per_gain), ("test", args.test_per_gain)]
    for split_index, (split, count) in enumerate(counts):
        if count < 0 or count >= 100000:
            raise ValueError("Invalid per-split count")
        for index in range(count):
            for gain in range(3):
                tasks.append({"output": str(args.output.resolve()), "split": split,
                              "seed": args.seed + split_index * 1000000 + index,
                              "dynamics_id": gain, "native_steps": args.native_steps,
                              "action_block": args.action_block, "image_size": args.image_size,
                              "source_hashes": source_hashes, "upstream_commit": upstream})
    start = time.perf_counter()
    episodes = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                                               mp_context=multiprocessing.get_context("spawn")) as pool:
        for episode in pool.map(collect, tasks):
            episodes.append(episode)
            if len(episodes) % 12 == 0 or len(episodes) == len(tasks):
                print(json.dumps({"completed": len(episodes), "total": len(tasks),
                                  "elapsed_seconds": time.perf_counter() - start}), flush=True)
    for relative, digest in source_hashes.items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError("Collector source changed during generation")
    manifest = {"schema_version": 1, "environment": "drone", "action_interface": "planar_velocity",
                "status": "collected", "upstream_commit": upstream,
                "source_hashes": source_hashes, "dynamics": DYNAMICS_GAINS,
                "action_block": args.action_block, "image_size": args.image_size,
                "train_combinations": [[0, 0], [1, 0], [2, 0], [0, 1], [2, 1], [0, 2], [1, 2]],
                "episodes": episodes, "collection_seconds": time.perf_counter() - start,
                "model_fields": ["images", "actions"], "privileged_fields": "separate audit files",
                "split_unit": "whole physical episode; seed families disjoint",
                "evaluation_status": "fresh test set collected; no models evaluated"}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "episodes": len(episodes),
                      "seconds": manifest["collection_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
