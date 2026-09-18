#!/usr/bin/env python3
"""Development-only goal calibration under exactly the planner's observed support.

No candidate search, simulator future-action execution, training, or checkpoint
selection occurs. Zero actions are a model-input diagnostic, not a measured
simulator counterfactual. Canonical renders are diagnostic targets only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from shiftwm.checkpoint import load_package
from shiftwm.data import pixels_to_tensor, split_combinations, validate_manifest
from shiftwm.evaluate import atomic_json, grouped_summary, state_error_diagnostics
from shiftwm.generate import (make_env, render_env, restore_simulator_state,
                             set_task_goal, task_distance_success, task_state)
import shiftwm.evaluate as evaluation_source
import shiftwm.generate as generation_source
import shiftwm.model as model_source


EPISODES = 32
GOAL_OFFSET = 5
ALLOWED_MODES = {"frozen", "single", "factorized", "framewise"}
RENDER_ENVIRONMENT_KEYS = ("MUJOCO_GL", "EGL_PLATFORM", "LIBGL_ALWAYS_SOFTWARE",
                           "MUJOCO_EGL_DEVICE_ID", "PYOPENGL_PLATFORM",
                           "CUDA_VISIBLE_DEVICES", "SDL_VIDEODRIVER")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def existing_renderer_metadata(env, environment):
    """Inspect an existing render context only; never instantiate a GL context."""
    result = {"status": "unknown", "renderer": None, "vendor": None}
    if environment != "reacher":
        return {**result, "status": "not_applicable", "backend": "pygame"}
    try:
        physics = getattr(getattr(env, "env", None), "physics", None)
        # dm_control's public contexts property creates contexts lazily. Reading
        # the cached private field avoids that side effect on this metadata path.
        contexts = getattr(physics, "_contexts", None)
        context = getattr(contexts, "gl", None)
        if context is None:
            return {**result, "reason": "no_existing_GL_context"}
        from OpenGL import GL
        with context.make_current():
            values = [GL.glGetString(GL.GL_RENDERER), GL.glGetString(GL.GL_VENDOR)]
        result.update(renderer=values[0].decode(errors="replace") if values[0] else None,
                      vendor=values[1].decode(errors="replace") if values[1] else None)
        result["status"] = "available" if all(values) else "unknown"
        return result
    except Exception as error:
        return {**result, "reason": type(error).__name__ + ": " + str(error)}


def pixel_difference(fresh, archived):
    fresh, archived = np.asarray(fresh), np.asarray(archived)
    if fresh.shape != archived.shape or fresh.ndim not in (3, 4) or fresh.shape[-1] != 3:
        raise ValueError("Pixel comparison requires matched HWC or THWC RGB shapes")
    if fresh.dtype != np.uint8 or archived.dtype != np.uint8:
        raise ValueError("Pixel comparison requires raw canonical uint8 renders")
    difference = np.abs(fresh.astype(np.int16) - archived.astype(np.int16))
    return {"mean_absolute_channel_difference_0_255": float(difference.mean()),
            "maximum_absolute_channel_difference_0_255": int(difference.max()),
            "fraction_changed_channel_values": float(np.mean(difference != 0)),
            "fraction_changed_spatial_pixels": float(np.mean(np.any(difference != 0, axis=-1)))}


def archived_render_comparison(acquired, stored):
    """Diagnostic only: these saved images never replace policy inputs."""
    if not acquired["policy_eligible"]:
        return {"status": "excluded_support_success"}
    images = stored["images"]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Recorded images must be THWC uint8 RGB")
    if acquired["support_native_steps"] != 10 or len(acquired["support_pixels"]) != 3:
        raise ValueError("Archived support comparison requires complete aligned support")
    archived_support, archived_goal = images[:3], images[acquired["goal_index"]]
    return {"status": "measured", "support_grouped_frame_indices": [0, 1, 2],
            "goal_grouped_frame_index": acquired["goal_index"],
            "canonical_pixel_units": "uint8_0_255_before_any_appearance_transform",
            "support": pixel_difference(acquired["support_pixels"], archived_support),
            "support_by_frame": [pixel_difference(acquired["support_pixels"][i], archived_support[i]) for i in range(3)],
            "goal": pixel_difference(acquired["goal_pixels"], archived_goal),
            "archived_support_pixels_sha256": array_sha256(archived_support),
            "archived_goal_pixels_sha256": array_sha256(archived_goal)}


def selected_episodes(manifest, count=EPISODES):
    """Mirror evaluate_planning ordering/filter, but prohibit other splits."""
    validate_manifest(manifest)
    combinations = split_combinations("development")
    counts, selected = defaultdict(int), []
    for episode in sorted(manifest["episodes"], key=lambda x: (x["seed"], x["dynamics_id"])):
        dynamics = episode["dynamics_id"]
        if episode["split"] != "development" or counts[dynamics] >= count:
            continue
        if not any(d == dynamics for _, d in combinations):
            continue
        selected.append(episode)
        counts[dynamics] += 1
    if any(counts[d] != count for _, d in combinations):
        raise ValueError(f"Expected exactly {count} available development episodes per selected dynamics")
    return selected


@torch.inference_mode()
def acquire_planning_support(model, manifest, episode, stored, appearance):
    """Literal support/goal setup from evaluate.py:224–268; never advance future.

    Additional canonical images and simulator states are retained for diagnostics
    but never passed to context inference. A support success stops at the same
    native interaction as the evaluator and is excluded from rollout diagnostics.
    """
    environment = manifest["environment"]
    action_interface = manifest.get("action_interface", "absolute" if manifest.get("action_units") == "absolute target pixels" else "relative")
    image_size, action_block = manifest["image_size"], manifest["action_block"]
    history = model.config.history_length
    if history != 3 or action_block != 5 or model.action_dim != 10:
        raise ValueError("Diagnostic is locked to the current H=3, five-by-two action protocol")
    goal_index = history - 1 + GOAL_OFFSET
    if episode["steps"] < goal_index:
        raise ValueError("Recorded goal/future actions are unavailable")
    device = next(model.parameters()).device
    env = make_env(environment, episode["dynamics_id"], episode["seed"], image_size,
                   action_interface=action_interface)
    try:
        restore_simulator_state(env, environment, stored["simulator_states"][0])
        goal_state = stored["task_states"][goal_index]
        set_task_goal(env, environment, goal_state)
        goal_env = make_env(environment, episode["dynamics_id"], episode["seed"], image_size,
                            action_interface=action_interface)
        try:
            set_task_goal(goal_env, environment, goal_state)
            restore_simulator_state(goal_env, environment, stored["simulator_states"][goal_index])
            goal_pixels = np.array(render_env(goal_env, environment, image_size), copy=True)
            goal_renderer = existing_renderer_metadata(goal_env, environment)
        finally:
            goal_env.close()
        goal_image = pixels_to_tensor(goal_pixels, appearance)[None].to(device)
        features, past, support_pixels, support_states, executed_support = [], [], [], [], []

        def observe():
            pixels = np.array(render_env(env, environment, image_size), copy=True)
            support_pixels.append(pixels)
            support_states.append(task_state(env, environment).copy())
            features.append(model.encode_images(pixels_to_tensor(pixels, appearance)[None].to(device)))

        observe()
        native_used, success, first_success = 0, False, None
        for t in range(history - 1):
            block = stored["actions"][t].reshape(action_block, 2)
            for native_action in block:
                env.step(native_action)
                executed_support.append(native_action.copy())
                native_used += 1
                _, hit = task_distance_success(env, environment, goal_state)
                if hit and first_success is None:
                    first_success = native_used
                success = success or hit
                if hit:
                    break
            # Match evaluator exactly; a partial block is never used for rollout
            # because success cases are skipped below.
            past.append(torch.tensor(stored["actions"][t], device=device)[None])
            observe()
            if success:
                break
        distance, _ = task_distance_success(env, environment, goal_state)
        return {
            "history_features": torch.stack(features, 1),
            "past_actions": torch.stack(past, 1),
            "support_pixels": np.stack(support_pixels),
            "goal_pixels": goal_pixels, "goal_image": goal_image,
            "support_states": np.stack(support_states),
            "goal_state": goal_state.copy(),
            "executed_support_actions": np.asarray(executed_support),
            "success_during_context": int(success), "policy_eligible": int(not success),
            "support_native_steps": native_used, "first_support_success_step": first_success,
            "goal_index": goal_index, "initial_distance_after_context": distance,
            "initial_errors": state_error_diagnostics(environment, task_state(env, environment), goal_state),
            "renderer": {"goal": goal_renderer, "support": existing_renderer_metadata(env, environment)},
        }
    finally:
        env.close()


def mse(left, right):
    value = float((left.float() - right.float()).square().mean())
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite diagnostic metric")
    return value


@torch.inference_mode()
def measure_episode(model, acquired, stored, appearance):
    """Canonical targets and future action sequences cannot enter infer_context."""
    if not acquired["policy_eligible"]:
        return {"diagnostic_status": "excluded_support_success"}
    device = next(model.parameters()).device
    history, past = acquired["history_features"], acquired["past_actions"]
    # The context is fixed before constructing any privileged diagnostic target.
    contexts = model.infer_context(history, past)
    corrected_support = model.correct_observations(history, contexts[0])
    corrected_goal = model.goal_embedding(acquired["goal_image"], contexts[0])
    raw_goal = model.encode_images(acquired["goal_image"])
    canonical_support_pixels = pixels_to_tensor(acquired["support_pixels"], 0)[None].to(device)
    canonical_goal_pixels = pixels_to_tensor(acquired["goal_pixels"], 0)[None].to(device)
    canonical_support = model.encode_images(canonical_support_pixels, reference=True)
    canonical_goal = model.encode_images(canonical_goal_pixels, reference=True)
    boundary = model.config.history_length - 1
    future = torch.tensor(stored["actions"][boundary:acquired["goal_index"]], device=device)[None]
    if future.shape != (1, GOAL_OFFSET, model.action_dim):
        raise ValueError("Future recorded action slice must end exactly at the goal")
    recorded_prediction = model.rollout_features(history, past, future, contexts=contexts)
    zero_prediction = model.rollout_features(history, past, torch.zeros_like(future), contexts=contexts)
    recorded_terminal, zero_terminal = recorded_prediction[:, -1], zero_prediction[:, -1]
    result = {
        "diagnostic_status": "measured",
        "goal_calibration_mse": mse(corrected_goal, canonical_goal),
        "raw_goal_calibration_mse": mse(raw_goal, canonical_goal),
        "support_calibration_mse": mse(corrected_support, canonical_support),
        "raw_support_calibration_mse": mse(history, canonical_support),
        "recorded_prediction_to_canonical_goal_mse": mse(recorded_terminal, canonical_goal),
        "zero_prediction_to_canonical_goal_mse": mse(zero_terminal, canonical_goal),
        "recorded_prediction_to_adapted_goal_mse": mse(recorded_terminal, corrected_goal),
        "zero_prediction_to_adapted_goal_mse": mse(zero_terminal, corrected_goal),
        "terminal_action_sensitivity_mse": mse(recorded_terminal, zero_terminal),
        "observed_final_support_to_canonical_goal_mse": mse(corrected_support[:, -1], canonical_goal),
        "observed_final_support_to_adapted_goal_mse": mse(corrected_support[:, -1], corrected_goal),
        "recorded_future_actions_sha256": array_sha256(future.cpu().numpy()),
        "context_observation_sha256": array_sha256(contexts[0].cpu().numpy()),
        "context_dynamics_sha256": array_sha256(contexts[1].cpu().numpy()),
    }
    result["support_frame_calibration_mse"] = [mse(corrected_support[:, t], canonical_support[:, t]) for t in range(history.shape[1])]
    return result


def planning_agreement(record, planning_result, identity):
    """Compare to completed planning without reading policy outcomes as inputs."""
    if planning_result.get("status") != "complete":
        raise ValueError("Planning comparison must be complete")
    for key in ("checkpoint_sha256", "data_manifest_sha256", "evaluator_sha256"):
        if planning_result.get(key) != identity[key]:
            raise ValueError(f"Planning comparison {key} differs")
    planning = planning_result["planning"]
    if planning["split"] != "development" or planning["protocol"]["goal_offset"] != GOAL_OFFSET:
        raise ValueError("Planning comparison split or goal offset differs")
    rows = {(x["trajectory_id"], x["observation_id"]): x for x in planning["records"]}
    expected = {(x["trajectory_id"], x["observation_id"]) for x in record["records"]}
    if len(rows) != len(planning["records"]) or set(rows) != expected:
        raise ValueError("Planning comparison selected keys differ or contain duplicates")
    maximum_distance_difference = 0.0
    for row in record["records"]:
        prior = rows[(row["trajectory_id"], row["observation_id"])]
        for key in ("goal_index", "success_during_context", "policy_eligible"):
            if row[key] != prior[key]:
                raise ValueError(f"Planning setup {key} differs for {row['trajectory_id']}")
        difference = abs(row["initial_distance_after_context"] - prior["initial_distance_after_context"])
        maximum_distance_difference = max(maximum_distance_difference, difference)
        if not math.isclose(row["initial_distance_after_context"], prior["initial_distance_after_context"], rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(f"Planning support state differs for {row['trajectory_id']}")
        for key, value in row["initial_errors"].items():
            if not math.isclose(value, prior["initial_" + key], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"Planning support {key} differs for {row['trajectory_id']}")
    return {"status": "passed", "compared_records": len(rows),
            "maximum_initial_distance_difference": maximum_distance_difference,
            "scope": "selected keys, goal indices, support-success mask, support-state errors; pixel history fingerprints are additionally recorded here"}


def verify_completed_source(checkpoint):
    checkpoint = Path(checkpoint)
    model_file = checkpoint if checkpoint.is_file() else checkpoint / "model.pt"
    package = model_file.parent
    config = json.loads((package / "config.json").read_text())
    mode = config["model_config"]["mode"]
    if mode not in ALLOWED_MODES or package.name != "best":
        raise ValueError("Use a frozen/single/factorized/framewise run's best package")
    if mode == "frozen":
        export = json.loads((package / "export_manifest.json").read_text())
        if export["training_status"] != "released_upstream_weights_not_finetuned":
            raise ValueError("Expected explicit unchanged-upstream frozen export")
    else:
        summary = json.loads((package.parent / "training_summary.json").read_text())
        run_config = json.loads((package.parent / "run_config.json").read_text())
        if summary.get("status") != "completed" or summary["completed_epochs"] != run_config["epochs"]:
            raise ValueError("Diagnostic requires fully completed training")
    return model_file, config


def run(args):
    model_file, config = verify_completed_source(args.checkpoint)
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    selected = selected_episodes(manifest)
    identity = {
        "checkpoint_sha256": sha256(model_file), "config_sha256": sha256(model_file.parent / "config.json"),
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "evaluator_sha256": sha256(evaluation_source.__file__),
        "diagnostic_source_sha256": sha256(__file__),
        "model_source_sha256": sha256(model_source.__file__),
        "generation_source_sha256": sha256(generation_source.__file__),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    torch.set_num_threads(args.cpu_threads)
    model, state = load_package(model_file, device=args.device)
    model.requires_grad_(False)
    if state["config"] != config:
        raise ValueError("Embedded and external checkpoint configs differ")
    environment = manifest["environment"]
    if not model.provenance.get("download", {}).get("repo", "").endswith("lewm-" + environment):
        raise ValueError("Checkpoint provenance does not match environment")
    started, records = time.monotonic(), []
    for episode in selected:
        episode_sha256 = sha256(data / episode["file"])
        if episode.get("sha256") and episode_sha256 != episode["sha256"]:
            raise ValueError(f"Episode differs from its manifest checksum: {episode['file']}")
        with np.load(data / episode["file"], allow_pickle=False) as archive:
            stored = {key: archive[key] for key in ("simulator_states", "task_states", "actions", "images")}
        for appearance, dynamics in split_combinations("development"):
            if dynamics != episode["dynamics_id"]:
                continue
            acquired = acquire_planning_support(model, manifest, episode, stored, appearance)
            row = {"trajectory_id": episode["trajectory_id"], "seed": episode["seed"],
                   "observation_id": appearance, "dynamics_id": dynamics,
                   "episode_file": episode["file"], "source_episode_sha256": episode_sha256}
            for key in ("success_during_context", "policy_eligible", "support_native_steps", "first_support_success_step",
                        "goal_index", "initial_distance_after_context", "initial_errors"):
                row[key] = acquired[key]
            for key in ("support_pixels", "goal_pixels", "support_states", "goal_state", "executed_support_actions"):
                row[key + "_sha256"] = array_sha256(acquired[key])
            row.update(measure_episode(model, acquired, stored, appearance))
            row["archived_render_comparison"] = archived_render_comparison(acquired, stored)
            row["renderer"] = acquired["renderer"]
            records.append(row)
            print(json.dumps({"event": "goal_calibration_episode", "completed": len(records),
                              "trajectory_id": row["trajectory_id"], "status": row["diagnostic_status"]}), flush=True)
    measured = [row for row in records if row["diagnostic_status"] == "measured"]
    metrics = [key for key, value in measured[0].items()
               if key.endswith("_mse") and isinstance(value, (int, float))] if measured else []
    # One row per selected seed/condition; explicit seed means remain valid if
    # the development protocol later contains multiple appearances per seed.
    by_seed = defaultdict(list)
    for row in measured:
        by_seed[row["seed"]].append(row)
    seed_means = [{"seed": seed, "records": len(rows), **{
        metric: sum(row[metric] for row in rows) / len(rows) for metric in metrics}}
        for seed, rows in sorted(by_seed.items())]
    record = {
        "status": "complete", "kind": "development_goal_calibration_diagnostic", "split": "development",
        "environment": environment, "checkpoint": str(model_file.resolve()), "model_mode": model.config.mode,
        "checkpoint_epoch": state["epoch"], "created_utc": datetime.now(timezone.utc).isoformat(), **identity,
        "protocol": {"episodes_per_dynamics": EPISODES, "goal_offset": GOAL_OFFSET, "history_frames": 3,
                     "action_block": 5, "support_native_budget": 10,
                     "selection": "sort manifest episodes by (seed,dynamics_id); development family; first32 per development dynamics; split_combinations('development')",
                     "future_actions": "recorded actions[2:7] versus all-zero raw controls; model rollouts only",
                     "excluded": "support successes: planner never invokes context/control for these cases",
                     "claim_scope": "feature calibration and model input sensitivity, not simulator counterfactual accuracy or planning success"},
        "selected_keys": [[row["trajectory_id"], row["observation_id"]] for row in records],
        "counts": {"selected": len(records), "measured": len(measured), "support_success": len(records) - len(measured)},
        "summary": grouped_summary(measured, metrics, "seed") if measured else {},
        "seed_means": seed_means, "records": records, "elapsed_seconds": time.monotonic() - started,
        "execution_context": {"device": args.device, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                              "render_environment": {key: os.environ.get(key) for key in RENDER_ENVIRONMENT_KEYS},
                              "main_efficiency_claim_eligible": False},
    }
    if args.planning_result:
        comparison = json.loads(Path(args.planning_result).read_text())
        record["planning_setup_agreement"] = planning_agreement(record, comparison, identity)
        record["planning_setup_agreement"]["source_sha256"] = sha256(args.planning_result)
        record["planning_setup_agreement"]["source"] = str(Path(args.planning_result).resolve())
    else:
        record["planning_setup_agreement"] = {"status": "not_compared", "reason": "--planning-result not supplied"}
    atomic_json(record, output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planning-result", type=Path, help="Optional completed same-checkpoint development planning JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", default=4, type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
