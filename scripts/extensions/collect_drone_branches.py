#!/usr/bin/env python3
"""Registered TRAIN-only counterfactual branches; no model fitting or evaluation.

Register before collection. Each future starts from a fresh simulator followed
by all ten original support commands; every support RGB and native state must
match the original audit exactly. Physical values are written only to audit/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = 10
FUTURE = 25
BLOCK = 5
CANDIDATE_SEED = 20260919
GAINS = {0: 1.0, 1: 0.75, 2: 1.25}
MODEL_FIELDS = {"support_images", "support_actions", "candidate_actions", "terminal_images",
                "future_images", "future_image_mask", "executed_lengths", "executed_mask", "valid_mask"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def candidate_library():
    """Byte-identical to the previously fixed action-ranking diagnostic library."""
    labels, commands = ["zero"], [np.zeros((FUTURE, 2), np.float32)]
    axes = [("x+", [1., 0.]), ("x-", [-1., 0.]), ("y+", [0., 1.]), ("y-", [0., -1.])]
    for name, vector in axes:
        labels.append("hold_" + name)
        commands.append(np.tile(vector, (FUTURE, 1)).astype(np.float32))
    for x, y in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        labels.append(f"diagonal_{x:+d}_{y:+d}")
        commands.append(np.tile(np.array([x, y], np.float32) / np.sqrt(2), (FUTURE, 1)).astype(np.float32))
    for name, vector in axes:
        pulse = np.zeros((FUTURE, 2), np.float32)
        pulse[:BLOCK] = vector
        labels.append("pulse5_" + name)
        commands.append(pulse)
    rng = np.random.default_rng(CANDIDATE_SEED)
    for index in range(19):
        labels.append(f"random_blocks_{index:02d}")
        commands.append(np.repeat(rng.uniform(-1, 1, (5, 2)).astype(np.float32), BLOCK, axis=0))
    actions = np.stack(commands)
    if actions.shape != (32, 25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
        raise ValueError("Malformed fixed candidate library")
    return labels, actions


def select_contexts(manifest):
    """First32 distinct TRAIN seeds, all three original gains; no goal filtering."""
    if manifest.get("environment") != "drone":
        raise ValueError("Source must be the canonical drone dataset")
    rows = manifest["episodes"]
    train = [row for row in rows if row["split"] == "train"]
    seeds = sorted({row["seed"] for row in train})[:32]
    if len(seeds) != 32:
        raise ValueError("Exactly32 distinct training seeds are required")
    if set(seeds) & {row["seed"] for row in rows if row["split"] != "train"}:
        raise ValueError("Training seeds overlap another split")
    lookup = {}
    for row in train:
        if row["seed"] in seeds:
            key = row["seed"], row["dynamics_id"]
            if key in lookup:
                raise ValueError("Duplicate seed/gain source trajectory")
            lookup[key] = row
    selected = []
    for seed in seeds:
        for dynamics_id in GAINS:
            row = lookup.get((seed, dynamics_id))
            if row is None:
                raise ValueError("Selected training seed lacks one of three gains")
            if row.get("action_block") != BLOCK or row.get("native_steps", 0) < SUPPORT:
                raise ValueError("Source action grouping or support length changed")
            if row["protocol"]["response_gain"] != GAINS[dynamics_id]:
                raise ValueError("Source dynamics label/gain mismatch")
            selected.append(row)
    return selected


def runtime_identity():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("pybullet", "gymnasium", "scipy")},
            "renderer": "PyBullet ER_TINY_RENDERER CPU", "platform": platform.system()}


def verify_sources(registration):
    for relative, digest in registration["source_hashes"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Registered source changed: {relative}")
    if runtime_identity() != registration["runtime"]:
        raise ValueError("Registered simulator runtime changed")
    _, candidates = candidate_library()
    if array_sha256(candidates) != registration["candidate_commands_sha256"]:
        raise ValueError("Registered candidate bytes changed")


def load_support(data, row):
    """Load and verify only original training support and separate scoring goal."""
    if row["split"] != "train":
        raise ValueError("Training branch collector cannot consume another split")
    data = Path(data)
    for name, hash_name in (("file", "sha256"), ("audit_file", "audit_sha256")):
        if sha256(data / row[name]) != row[hash_name]:
            raise ValueError(f"Source payload hash mismatch: {row[name]}")
    with np.load(data / row["audit_file"], allow_pickle=False) as audit:
        images = audit["images"][:SUPPORT + 1].copy()
        actions = audit["actions"][:SUPPORT].copy()
        states = audit["simulator_states"][:SUPPORT + 1].copy()
        goal_state = audit["goal_state"].copy()
    with np.load(data / row["file"], allow_pickle=False) as main:
        if set(main.files) != {"images", "actions"}:
            raise ValueError("Unexpected privileged source model fields")
        if not np.array_equal(main["images"][:3], images[::BLOCK]):
            raise ValueError("Native/grouped support RGB differs")
        if not np.array_equal(main["actions"][:2], actions.reshape(2, 10)):
            raise ValueError("Native/grouped commanded support differs")
    if images.dtype != np.uint8 or images.shape != (11, 128, 128, 3):
        raise ValueError("Expected11 native128RGB support images")
    if actions.dtype != np.float32 or actions.shape != (10, 2) or np.abs(actions).max() > 1:
        raise ValueError("Expected10 valid float32 commanded actions")
    if states.shape != (11, 20) or goal_state.shape != (20,):
        raise ValueError("Malformed separate physical audit")
    if not all(np.isfinite(value).all() for value in (actions, states, goal_state)):
        raise ValueError("Nonfinite source data")
    return {"images": images, "actions": actions, "states": states, "goal_state": goal_state}


def register(data, output):
    """Hash sources and all96 source contexts before any future is simulated."""
    from shiftwm.extensions.drone import DroneConfig, UPSTREAM_COMMIT
    data, output = Path(data).resolve(), Path(output).resolve()
    registration_path = output / "registration.json"
    if registration_path.exists():
        existing = json.loads(registration_path.read_text())
        if existing["data_root"] != str(data):
            raise ValueError("Existing registration has a different source root")
        verify_sources(existing)
        if sha256(data / "manifest.json") != existing["data_manifest_sha256"]:
            raise ValueError("Registered source dataset manifest changed")
        if sha256(output / "candidates.npz") != existing["candidates_file_sha256"]:
            raise ValueError("Registered candidate file changed")
        for row in existing["contexts"]:
            load_support(data, row)
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refuse nonempty unregistered output directory")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = select_contexts(manifest)
    labels, candidates = candidate_library()
    # This copy is fixed to an existing development diagnostic's command design.
    # No development image/state/outcome is consumed by this training collector.
    diagnostic = ROOT / "reports/evidence/drone_action_ranking/registration.json"
    diagnostic_registration = json.loads(diagnostic.read_text())
    if diagnostic_registration["candidate_labels"] != labels or not np.array_equal(
            np.array(diagnostic_registration["candidate_commands"], np.float32), candidates):
        raise ValueError("Candidates differ from the previously fixed diagnostic")
    upstream = ROOT / "external/gym-pybullet-drones"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT or manifest["upstream_commit"] != commit:
        raise ValueError("Upstream commit differs from source collection")
    sources = [Path(__file__).resolve(), ROOT / "src/shiftwm/extensions/drone.py",
               ROOT / "environments/drone/requirements.lock.txt",
               ROOT / "reports/drone_branched_training_protocol.md",
               ROOT / "tests/test_drone_branch_collection.py",
               ROOT / "scripts/extensions/collect_drone_branches.slurm"]
    # Hash all simulator package source/assets, not merely the git commit name.
    sources += sorted(path for path in (upstream / "gym_pybullet_drones").rglob("*")
                      if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    rows = []
    for row in selected:
        support = load_support(data, row)
        copied = {key: row[key] for key in ("trajectory_id", "split", "seed", "dynamics_id",
                  "steps", "native_steps", "action_block", "file", "sha256", "audit_file",
                  "audit_sha256", "config_sha256")}
        copied["support_commands_sha256"] = array_sha256(support["actions"])
        copied["support_rgb_sha256"] = array_sha256(support["images"])
        copied["support_states_sha256"] = array_sha256(support["states"])
        rows.append(copied)
    registration = {
        "schema_version": 1, "scope": "train_only_counterfactual_supervision_no_model_or_benchmark",
        "created_unix": time.time(), "data_root": str(data), "data_manifest_sha256": sha256(manifest_path),
        "split": "train", "selection": "first32 sorted training seeds, all gains0/1/2, no outcome filtering",
        "contexts": rows, "context_count": 96, "candidate_count": 32, "branch_count": 3072,
        "candidate_seed": CANDIDATE_SEED, "candidate_labels": labels,
        "candidate_commands_sha256": array_sha256(candidates), "candidate_commands": candidates.tolist(),
        "candidate_library_origin_sha256": sha256(diagnostic),
        "support_native_calls": SUPPORT, "future_native_calls": FUTURE, "native_budget": SUPPORT + FUTURE,
        "action_block": BLOCK, "image_size": 128, "command_bounds": [-1.0, 1.0],
        "gains": GAINS, "simulator_config": asdict(DroneConfig(image_size=128, max_steps=SUPPORT + FUTURE)),
        "upstream_commit": commit, "runtime": runtime_identity(),
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "replay_contract": "exact RGB and20Dstate at reset and every support command, each candidate fresh reset",
        "stopping": "ignore goal success; stop on crash/workspace escape or35total calls",
        "unsafe_contract": "retain every branch, actual terminal RGB, prefix length/mask, unsafe reasons in audit",
        "valid_mask_definition": "all25future calls executed without crash/workspace escape",
        "model_fields": sorted(MODEL_FIELDS), "privileged_fields": "separate audit NPZ and JSON only",
        "split_unit": "original training seed family, all gains/branches stay together",
        "future_use": "five future boundary RGB frames and terminal RGB are targets, never context input; no model/loss change registered",
        "future_image_mask_definition": "actual observed five-call boundaries; zero missing targets with false mask, safety separate",
        "appearance": "canonical RGB only; any later appearance transform belongs to a separately specified loader",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_npz(output / "candidates.npz", actions=candidates, grouped_actions=candidates.reshape(32, 5, 10))
    registration["candidates_file_sha256"] = sha256(output / "candidates.npz")
    atomic_json(registration_path, registration)
    return registration


def assert_reference(image, state, support, index):
    if not np.array_equal(image, support["images"][index]):
        raise ValueError(f"Exact support RGB replay failed at native index{index}")
    if not np.array_equal(state, support["states"][index]):
        raise ValueError(f"Exact support physical replay failed at native index{index}")


def simulate_context(env, support, candidates):
    """Collect one context. Seed equality alone never establishes a shared branch."""
    count = len(candidates)
    frames, lengths, valid, outcomes = [], [], [], []
    future_images = np.zeros((count, FUTURE // BLOCK, 128, 128, 3), np.uint8)
    future_image_mask = np.zeros((count, FUTURE // BLOCK), bool)
    state_history = np.full((count, FUTURE + 1, 20), np.nan, np.float64)
    gained_commands = np.full((count, FUTURE, 2), np.nan, np.float32)
    rpm = np.full((count, FUTURE, 12, 4), np.nan, np.float32)
    masks = np.zeros((count, FUTURE), bool)
    for candidate_index, commands in enumerate(candidates):
        image, _ = env.reset(seed=support["seed"])
        env.set_goal(support["goal_state"])
        assert_reference(image, env.simulator_state(), support, 0)
        for index, command in enumerate(support["actions"], 1):
            image, _, terminated, truncated, info = env.step(command)
            assert_reference(image, env.simulator_state(), support, index)
            if terminated or truncated or info["crash"] or info["workspace_escape"]:
                raise ValueError("Registered observed support became unsafe or exhausted")
        state_history[candidate_index, 0] = env.simulator_state()
        metrics = []
        reason = "full_horizon"
        for index, command in enumerate(commands):
            image, _, terminated, truncated, info = env.step(command)
            state = env.simulator_state()
            if not np.isfinite(state).all() or not np.isfinite(env.last_rpm).all():
                raise FloatingPointError("Nonfinite simulator output; collection is not complete")
            state_history[candidate_index, index + 1] = state
            gained_commands[candidate_index, index] = env.last_executed_action
            rpm[candidate_index, index] = env.last_rpm
            masks[candidate_index, index] = True
            metrics.append(info)
            if (index + 1) % BLOCK == 0:
                future_images[candidate_index, index // BLOCK] = image
                future_image_mask[candidate_index, index // BLOCK] = True
            if terminated or info["crash"] or info["workspace_escape"]:
                reason = "crash_and_workspace_escape" if info["crash"] and info["workspace_escape"] else (
                    "crash" if info["crash"] else "workspace_escape" if info["workspace_escape"] else "other_termination")
                break
            if truncated:
                reason = "full_horizon" if index + 1 == FUTURE else "unexpected_early_budget"
                break
        length = len(metrics)
        full_safe = length == FUTURE and reason == "full_horizon"
        frames.append(image.copy())
        lengths.append(length)
        valid.append(full_safe)
        outcomes.append({"candidate_index": candidate_index, "executed_length": length,
                         "reason": reason, "valid": full_safe, "metrics_per_native_step": metrics})
    main = {"support_images": support["images"][::BLOCK].copy(),
            "support_actions": support["actions"].reshape(2, 10).copy(),
            "candidate_actions": candidates.reshape(count, 5, 10).copy(),
            "terminal_images": np.stack(frames).astype(np.uint8),
            "future_images": future_images, "future_image_mask": future_image_mask,
            "executed_lengths": np.asarray(lengths, np.int16),
            "executed_mask": masks, "valid_mask": np.asarray(valid, bool)}
    audit = {"support_native_images": support["images"], "support_native_states": support["states"],
             "goal_state": support["goal_state"], "native_states": state_history,
             "gain_scaled_commands": gained_commands, "low_level_rpm": rpm}
    validate_model_arrays(main, count)
    return main, audit, outcomes


def validate_model_arrays(main, count=32):
    expected = {"support_images": ((3, 128, 128, 3), np.uint8),
                "support_actions": ((2, 10), np.float32),
                "candidate_actions": ((count, 5, 10), np.float32),
                "terminal_images": ((count, 128, 128, 3), np.uint8),
                "future_images": ((count, 5, 128, 128, 3), np.uint8),
                "future_image_mask": ((count, 5), np.bool_),
                "executed_lengths": ((count,), np.int16),
                "executed_mask": ((count, 25), np.bool_), "valid_mask": ((count,), np.bool_)}
    if set(main) != MODEL_FIELDS:
        raise ValueError("Model shard fields changed or privileged fields leaked")
    for key, (shape, dtype) in expected.items():
        if main[key].shape != shape or main[key].dtype != dtype or not np.isfinite(main[key]).all():
            raise ValueError(f"Malformed model field {key}")
    lengths = main["executed_lengths"]
    if np.any(lengths < 1) or np.any(lengths > FUTURE):
        raise ValueError("Invalid executed length")
    if not np.array_equal(main["executed_mask"], np.arange(FUTURE)[None] < lengths[:, None]):
        raise ValueError("Executed mask must be exactly the actual prefix")
    expected_frames = np.arange(BLOCK, FUTURE + 1, BLOCK)[None] <= lengths[:, None]
    if not np.array_equal(main["future_image_mask"], expected_frames):
        raise ValueError("Future image mask must represent executed block boundaries")
    if np.any(main["future_images"][~expected_frames] != 0):
        raise ValueError("Unavailable future targets must be zero with a false mask")
    boundary_stops = lengths % BLOCK == 0
    for index in np.flatnonzero(boundary_stops):
        if not np.array_equal(main["terminal_images"][index], main["future_images"][index, lengths[index] // BLOCK - 1]):
            raise ValueError("Terminal and boundary target RGB differ")
    if np.any(main["valid_mask"] & (lengths != FUTURE)):
        raise ValueError("Partial branch marked valid")
    if any(np.abs(main[key]).max() > 1 for key in ("support_actions", "candidate_actions")):
        raise ValueError("Commands outside[-1,1]")


def collect_context(task):
    from shiftwm.extensions.drone import DroneConfig, VisualDroneEnv
    output, row = Path(task["output"]), task["row"]
    registration = json.loads((output / "registration.json").read_text())
    verify_sources(registration)
    if sha256(output / "registration.json") != task["registration_sha256"]:
        raise ValueError("Registration changed during collection")
    uid = row["trajectory_id"]
    main_path = output / "contexts" / f"{uid}.npz"
    audit_path = output / "audit" / f"{uid}.npz"
    audit_json = output / "audit" / f"{uid}.json"
    sidecar = output / "contexts" / f"{uid}.json"
    support = load_support(registration["data_root"], row)
    support["seed"] = row["seed"]
    if sidecar.exists():
        previous = json.loads(sidecar.read_text())
        if previous["registration_sha256"] != task["registration_sha256"]:
            raise ValueError("Existing context belongs to another registration")
        for key, path in (("sha256", main_path), ("audit_sha256", audit_path), ("audit_json_sha256", audit_json)):
            if sha256(path) != previous[key]:
                raise ValueError(f"Existing context payload changed: {path}")
        with np.load(main_path, allow_pickle=False) as archive:
            validate_model_arrays({key: archive[key] for key in archive.files})
        return previous
    start = time.perf_counter()
    _, candidates = candidate_library()
    env = VisualDroneEnv(row["dynamics_id"], DroneConfig(**registration["simulator_config"]))
    try:
        main, audit, outcomes = simulate_context(env, support, candidates)
    finally:
        env.close()
    verify_sources(registration)
    atomic_npz(main_path, **main)
    atomic_npz(audit_path, **audit)
    atomic_json(audit_json, {"trajectory_id": uid, "seed": row["seed"], "dynamics_id": row["dynamics_id"],
                           "gain": GAINS[row["dynamics_id"]], "outcomes": outcomes,
                           "support_replay_checks": {"branches": 32, "native_states_per_branch": 11,
                                                     "native_rgb_per_branch": 11, "all_exact": True},
                           "padding": "NaN only after executed prefix in privileged audit; no synthetic states"})
    metadata = {"trajectory_id": uid, "source_trajectory_id": uid, "split": "train", "seed": row["seed"],
                "dynamics_id": row["dynamics_id"], "registration_sha256": task["registration_sha256"],
                "file": str(main_path.relative_to(output)), "sha256": sha256(main_path),
                "audit_file": str(audit_path.relative_to(output)), "audit_sha256": sha256(audit_path),
                "audit_json": str(audit_json.relative_to(output)), "audit_json_sha256": sha256(audit_json),
                "candidate_count": 32, "valid_candidates": int(main["valid_mask"].sum()),
                "future_native_calls": int(main["executed_lengths"].sum()),
                "collection_seconds": time.perf_counter() - start}
    atomic_json(sidecar, metadata)
    return metadata


def collect(output, workers):
    output = Path(output).resolve()
    registration = json.loads((output / "registration.json").read_text())
    verify_sources(registration)
    if not 1 <= workers <= 4:
        raise ValueError("Registered collection permits1–4 CPU workers")
    if sha256(output / "candidates.npz") != registration["candidates_file_sha256"]:
        raise ValueError("Registered candidate file changed")
    source_manifest = Path(registration["data_root"]) / "manifest.json"
    if sha256(source_manifest) != registration["data_manifest_sha256"]:
        raise ValueError("Source dataset manifest changed")
    digest = sha256(output / "registration.json")
    tasks = [{"output": str(output), "row": row, "registration_sha256": digest}
             for row in registration["contexts"]]
    start, results = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for row in pool.map(collect_context, tasks):
            results.append(row)
            print(json.dumps({"completed_contexts": len(results), "total_contexts": 96,
                              "elapsed_seconds": time.perf_counter() - start}), flush=True)
    verify_sources(registration)
    if len(results) != 96 or len({row["trajectory_id"] for row in results}) != 96:
        raise ValueError("Incomplete or duplicated contexts")
    manifest = {"schema_version": 1, "environment": "drone", "status": "collected",
                "scope": registration["scope"], "split": "train", "registration_sha256": digest,
                "model_fields": sorted(MODEL_FIELDS), "contexts": results,
                "context_count": 96, "branch_count": 3072,
                "valid_branches": sum(row["valid_candidates"] for row in results),
                "retained_unsafe_or_short_branches": 3072 - sum(row["valid_candidates"] for row in results),
                "collection_seconds": time.perf_counter() - start,
                "workers": workers, "model_training_status": "no model trained by collector"}
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "manifest.json"), "seconds": manifest["collection_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "collect"))
    parser.add_argument("--data", type=Path, default=ROOT / "data/extensions/drone_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/extensions/drone_branches_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if args.mode == "register":
        registration = register(args.data, args.output)
        print(json.dumps({"status": "registered_without_simulation", "contexts": len(registration["contexts"]),
                          "registration": str(args.output / "registration.json"),
                          "sha256": sha256(args.output / "registration.json")}), flush=True)
    else:
        collect(args.output, args.workers)


if __name__ == "__main__":
    main()
