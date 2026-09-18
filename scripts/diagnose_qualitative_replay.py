#!/usr/bin/env python3
"""Exactly replay archived development actions; no planning or model inference.

All outcomes discordant between ShiftWM and Framewise are selected before replay,
including every baseline-only success. Every archived image and terminal metric
must reproduce before reconstructed intermediate physical states become usable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shiftwm.data import pixels_to_tensor
from shiftwm.evaluate import atomic_json, state_error_diagnostics
from shiftwm.generate import (make_env, render_env, restore_simulator_state,
                             set_task_goal, simulator_state, task_distance_success,
                             task_state)


def local_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gallery = local_module("replay_gallery_contract", "paper/scripts/render_qualitative.py")
goal_diagnostic = local_module("replay_renderer_contract", "scripts/diagnose_goal_calibration.py")
sha = gallery.sha
MODES = gallery.MODES


def criterion(environment, current, goal):
    """Exact upstream success operands, not the mixed-unit state distance."""
    current, goal = np.asarray(current), np.asarray(goal)
    if environment == "pusht":
        angle = abs(float(goal[4] - current[4])) % (2 * np.pi)
        values = np.array([np.linalg.norm(goal[:4] - current[:4]),
                           min(angle, 2 * np.pi - angle)], dtype=np.float64)
        thresholds = np.array([20., np.pi / 9])
        names = ["combined_agent_and_block_position_error_px", "block_angle_error_rad"]
    elif environment == "reacher":
        values = np.abs(current - goal).astype(np.float64)
        if values.shape != (2,):
            raise ValueError("Expected the original two-joint Reacher")
        thresholds = np.full(2, .05)
        names = ["raw_absolute_joint0_error_rad", "raw_absolute_joint1_error_rad"]
    else:
        raise ValueError(environment)
    return values, {"names": names, "thresholds": thresholds.tolist(),
                    "strict_comparison": "each_error_less_than_threshold",
                    "position_units": "native_512_pixel_coordinates" if environment == "pusht" else None,
                    "angle_convention": "T_full_rotation_symmetry" if environment == "pusht" else "raw_unwrapped"}


def shifted_uint8(pixels, appearance):
    # Exactly the archive conversion; inference should use the unquantized tensor.
    return (pixels_to_tensor(pixels, appearance).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def require_pixels(fresh, archived, label):
    if (fresh.dtype != np.uint8 or fresh.shape != archived.shape
            or not np.array_equal(fresh, archived)):
        raise ValueError(f"Exact pixel replay failed: {label}")


def require_metric(actual, expected, label):
    if isinstance(expected, (int, bool)):
        matched = actual == expected
    else:
        matched = math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-8)
    if not matched:
        raise ValueError(f"Endpoint/support mismatch {label}: replay={actual}, archived={expected}")


def replay_method(manifest, episode, stored, method, environment, appearance):
    """Replay exact float32 actions, retaining states after every native step.

    Environment helpers are shared with evaluate.py. Rendering occurs at the same
    original support/planning boundaries. Additional state reads do not advance
    physics. No full action block is synthesized after an early terminal prefix.
    """
    row = method["record"]
    if not row["policy_eligible"] or row["success_during_context"]:
        raise ValueError("Discordant examples must be policy-eligible")
    if row["goal_index"] != 7 or manifest["action_block"] != 5 or manifest["image_size"] != 224:
        raise ValueError("Replay requires the unchanged H=3, block=5, goal=7 protocol")
    expected_times = gallery.frame_steps(row, method["frames"])
    interface = manifest.get("action_interface", "absolute" if manifest.get("action_units") == "absolute target pixels" else "relative")
    env = make_env(environment, episode["dynamics_id"], episode["seed"], 224,
                   action_interface=interface)
    try:
        restore_simulator_state(env, environment, stored["simulator_states"][0])
        goal = stored["task_states"][7].copy()
        set_task_goal(env, environment, goal)
        if environment == "pusht":
            shape = env.shapes[int(env.variation_space["block"]["shape"].value)]
            if shape != "T":
                raise ValueError("Physical diagnostic requires the archived T shape")
        elif not math.isclose(float(env.env.task.qpos_threshold), .05, rel_tol=0, abs_tol=0):
            raise ValueError("Changed upstream joint tolerance")
        goal_env = make_env(environment, episode["dynamics_id"], episode["seed"], 224,
                            action_interface=interface)
        try:
            set_task_goal(goal_env, environment, goal)
            restore_simulator_state(goal_env, environment, stored["simulator_states"][7])
            goal_pixels = np.array(render_env(goal_env, environment, 224), copy=True)
        finally:
            goal_env.close()
        shifted_goal = shifted_uint8(goal_pixels, appearance)
        require_pixels(shifted_goal, method["goal"], "goal")
        # Also verify canonical historical targets, not only their quantization.
        require_pixels(goal_pixels, stored["images"][7], "canonical dataset goal")

        states, simstates, successes, errors = [], [], [], []
        actions, canonical_frames, shifted_frames, frame_times = [], [], [], []
        first_success = None
        distances = []

        def observe_state():
            current = task_state(env, environment).copy()
            values, metadata = criterion(environment, current, goal)
            distance, hit = task_distance_success(env, environment, goal)
            if bool(np.all(values < metadata["thresholds"])) != bool(hit):
                raise ValueError("Diagnostic criterion differs from upstream success")
            states.append(current)
            simstates.append(simulator_state(env, environment).copy())
            errors.append(values)
            successes.append(hit)
            distances.append(distance)

        def observe_pixels(archived=None, label=""):
            pixels = np.array(render_env(env, environment, 224), copy=True)
            shifted = shifted_uint8(pixels, appearance)
            if archived is not None:
                require_pixels(shifted, archived, label)
            canonical_frames.append(pixels)
            shifted_frames.append(shifted)
            frame_times.append(len(actions))

        def advance(action):
            nonlocal first_success
            if first_success is not None:
                raise ValueError("Saved actions continue after first upstream success")
            native = np.asarray(action, dtype=np.float32)
            if native.shape != (2,) or not np.isfinite(native).all():
                raise ValueError("Invalid native action")
            env.step(native)
            actions.append(native.copy())
            observe_state()
            if successes[-1]:
                first_success = len(actions)

        observe_state()
        observe_pixels()
        require_pixels(canonical_frames[0], stored["images"][0], "canonical support t0")
        for support_block in range(2):
            for action in stored["actions"][support_block].reshape(5, 2):
                advance(action)
                if first_success is not None:
                    raise ValueError("Archived policy-eligible task succeeded during replayed support")
            observe_pixels()
            require_pixels(canonical_frames[-1], stored["images"][support_block + 1],
                           f"canonical support t{len(actions)}")
        require_pixels(shifted_frames[-1], method["frames"][0], "post-support t10")
        initial_errors = state_error_diagnostics(environment, states[-1], goal)
        require_metric(distances[-1], row["initial_distance_after_context"], "initial_distance_after_context")
        for key, value in initial_errors.items():
            require_metric(value, row["initial_" + key], "initial_" + key)

        for block_index, raw_block in enumerate(row["executed_action_blocks"]):
            if len(actions) + 5 > 50 or first_success is not None:
                raise ValueError("Saved block violates the original planning stop rule")
            block = np.asarray(raw_block, dtype=np.float32).reshape(-1, 2)
            # The JSON values came from float32.tolist(), so roundtrip is exact.
            if not np.array_equal(block.astype(np.float64).reshape(-1), np.asarray(raw_block, dtype=np.float64)):
                raise ValueError("Saved policy actions do not round-trip through their original float32 dtype")
            for action in block:
                advance(action)
            if len(block) < 5 and first_success != len(actions):
                raise ValueError("Partial action prefix did not end in success")
            observe_pixels(method["frames"][block_index + 1], f"policy block {block_index}, t{len(actions)}")
        if frame_times[2:] != expected_times:
            raise ValueError("Replayed frame times differ from archived native times")
        if first_success is None and len(actions) + 5 <= 50:
            raise ValueError("Saved failed policy stopped before exhausting its budget")
        actual = {"success": int(first_success is not None), "final_success": int(successes[-1]),
                  "success_during_context": 0, "policy_eligible": 1,
                  "native_steps": len(actions), "steps_to_success_or_budget": first_success or 50,
                  "final_distance": distances[-1], "num_replans": len(row["executed_action_blocks"]),
                  **state_error_diagnostics(environment, states[-1], goal)}
        if environment == "pusht":
            actual["requires_manipulation"] = int(initial_errors["block_translation_error_px"] >= 20
                                                  or initial_errors["block_angle_error_rad"] >= np.pi / 9)
        for key, value in actual.items():
            require_metric(value, row[key], key)
        _, metadata = criterion(environment, states[-1], goal)
        error_array = np.asarray(errors, dtype=np.float64)
        arrays = {"native_times": np.asarray(frame_times, dtype=np.int64),
                  "canonical_frames": np.asarray(canonical_frames, dtype=np.uint8),
                  "shifted_frames": np.asarray(shifted_frames, dtype=np.uint8),
                  "goal_canonical_frame": goal_pixels, "goal_shifted_frame": shifted_goal,
                  "trace_native_times": np.arange(len(actions) + 1, dtype=np.int64),
                  "task_states": np.asarray(states, dtype=np.float64),
                  "simulator_states": np.asarray(simstates, dtype=np.float64),
                  "goal_task_state": np.asarray(goal, dtype=np.float64),
                  "executed_native_actions": np.asarray(actions, dtype=np.float32),
                  "success_flags": np.asarray(successes, dtype=np.bool_),
                  "criterion_errors": error_array,
                  "criterion_margin": np.max(error_array / metadata["thresholds"], axis=1)}
        validation = {"status": "exact", "claim_eligible": True,
                      "all_archived_frames_pixel_identical": True,
                      "archived_frame_count": len(method["frames"]),
                      "goal_pixel_identical": True, "canonical_support_and_goal_pixel_identical": True,
                      "endpoint_and_support_metrics_match": True, "stop_rule_matches": True,
                      "first_success_native_step": first_success,
                      "metric_relative_tolerance": 1e-10, "metric_absolute_tolerance": 1e-8}
        return arrays, {"validation": validation, "criterion": metadata,
                        "renderer": goal_diagnostic.existing_renderer_metadata(env, environment)}
    finally:
        env.close()


def verify_sources(sources, root=ROOT):
    for name, digest in sources.items():
        if sha(root / name) != digest:
            raise ValueError(f"Source changed during exact replay: {name}")


def source_metadata(sources):
    names = ["scripts/diagnose_qualitative_replay.py", "scripts/diagnose_goal_calibration.py"]
    upstream = ROOT / "external/stable-worldmodel/stable_worldmodel/envs"
    names += [str(path.relative_to(ROOT)) for path in upstream.rglob("*")
              if path.is_file() and path.suffix in {".py", ".xml"}]
    result = dict(sources)
    result.update({name: sha(ROOT / name) for name in names})
    return result


def run(output, environments=("pusht", "reacher")):
    examples, source_hashes = gallery.collect(ROOT)
    selected = [item for item in examples if item["environment"] in environments
                and item["category"] in {"ours_only", "baseline_only"}]
    if not selected:
        raise ValueError("No discordant development episodes")
    source_hashes = source_metadata(source_hashes)
    manifest_cache = {}
    for environment in environments:
        data = ROOT / "data/world" / ("pusht_relative" if environment == "pusht" else "reacher")
        manifest = json.loads((data / "manifest.json").read_text())
        manifest_cache[environment] = (data, manifest)
    for example in selected:
        data, manifest = manifest_cache[example["environment"]]
        episode = next(ep for ep in manifest["episodes"] if ep["trajectory_id"] == example["trajectory_id"])
        path = data / episode["file"]
        if sha(path) != episode["sha256"]:
            raise ValueError("Dataset episode digest differs from manifest")
        source_hashes[str(path.relative_to(ROOT))] = episode["sha256"]
    packages = {}
    for name in ("numpy", "torch", "mujoco", "dm-control", "pygame", "pymunk"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_found"
    result = {"schema_version": 1, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "selection": "all_policy_eligible_ours_only_and_Framewise_only_development_tasks_all_three_methods",
              "expected_records": len(selected) * len(MODES), "sources": source_hashes,
              "environments": list(environments), "records": [], "failures": [],
              "packages": packages, "render_environment": {key: os.getenv(key) for key in goal_diagnostic.RENDER_ENVIRONMENT_KEYS},
              "limitations": ["Reconstructed intermediate simulator states; validated at every archived image boundary and terminal metric.",
                              "Post-hoc diagnostic of saved actions, not a newly evaluated policy or causal mechanism test.",
                              "No latent predictions, CEM candidates/costs, or context vectors were archived."]}
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "results/qualitative_diagnostics"):
        raise ValueError("Replay output must remain in results/qualitative_diagnostics")
    with gallery.contract.contract.exclusive_output_lock(output.parent):
        if output.exists():
            previous = json.loads(output.read_text())
            if previous["sources"] != source_hashes or previous["environments"] != list(environments):
                raise ValueError("Existing replay has different source identity or selection")
            if previous["status"] == "complete":
                verify_sources(source_hashes)
                for record in previous["records"]:
                    if sha(ROOT / record["npz"]) != record["npz_sha256"]:
                        raise ValueError("Completed replay artifact changed")
                return previous
        atomic_json(result, output)
        for example in selected:
            data, manifest = manifest_cache[example["environment"]]
            episode = next(ep for ep in manifest["episodes"] if ep["trajectory_id"] == example["trajectory_id"])
            with np.load(data / episode["file"], allow_pickle=False) as archive:
                stored = {key: archive[key] for key in archive.files}
            for mode in MODES:
                header = {key: example[key] for key in ("environment", "trajectory_id", "seed", "observation_id", "category")}
                header.update(behavior_mode=mode, original_record=example["methods"][mode]["record"])
                try:
                    arrays, details = replay_method(manifest, episode, stored, example["methods"][mode],
                                                    example["environment"], example["observation_id"])
                    path = output.parent / "episodes" / example["environment"] / f"{example['trajectory_id']}-o{example['observation_id']}" / f"{mode}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".npz.partial")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    temporary.replace(path)
                    result["records"].append({**header, **details, "npz": str(path.relative_to(ROOT)), "npz_sha256": sha(path)})
                    print(json.dumps({"replay": len(result["records"]), "expected": result["expected_records"],
                                      **{key: value for key, value in header.items() if key != "original_record"}}, default=str), flush=True)
                except Exception as error:
                    result["failures"].append({**header, "error": f"{type(error).__name__}: {error}",
                                               "validation": {"status": "failed", "claim_eligible": False}})
                    print(json.dumps(result["failures"][-1]), flush=True)
                atomic_json(result, output)
        verify_sources(source_hashes)
        result["status"] = "complete" if not result["failures"] and len(result["records"]) == result["expected_records"] else "failed"
        atomic_json(result, output)
        if result["status"] != "complete":
            raise RuntimeError(f"Exact replay failed for {len(result['failures'])} records; do not render diagnostic claims")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/qualitative_diagnostics/replay_diagnostics.json")
    parser.add_argument("--environment", choices=("all", "pusht", "reacher"), default="all")
    args = parser.parse_args()
    environments = ("pusht", "reacher") if args.environment == "all" else (args.environment,)
    run(args.output, environments)


if __name__ == "__main__":
    main()
