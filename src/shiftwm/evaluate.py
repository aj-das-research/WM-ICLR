"""Fixed-coordinate forecasts and real closed-loop goal-image planning.

The upstream stable-worldmodel CEM solver is reused unchanged. The learned
policy receives shifted RGB observations and executed action histories only.
Simulator states are used solely to initialize paired episodes and score them.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_package
from .data import TrajectoryDataset, pixels_to_tensor, split_combinations, split_family
from .generate import (make_env, native_action_bounds, render_env, restore_simulator_state,
                       set_task_goal, task_distance_success, task_state)


def atomic_json(value: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def state_error_diagnostics(environment: str, current: np.ndarray, goal: np.ndarray) -> dict:
    """Interpretable physical-unit errors; simulator state is scoring-only."""
    if environment == "pusht":
        angle = abs(float((current[4] - goal[4] + np.pi) % (2 * np.pi) - np.pi))
        return {"block_translation_error_px": float(np.linalg.norm(current[2:4] - goal[2:4])),
                "block_angle_error_rad": angle,
                "agent_position_error_px": float(np.linalg.norm(current[:2] - goal[:2]))}
    difference = (current - goal + np.pi) % (2 * np.pi) - np.pi
    return {"wrapped_joint_error_rad": float(np.linalg.norm(difference))}


def clustered_interval(records: list[dict], metric: str, cluster_key: str,
                       seed: int = 1701, repetitions: int = 2000) -> dict:
    """Percentile bootstrap over independent trajectory/initial-state clusters."""
    groups = defaultdict(list)
    for row in records:
        groups[str(row[cluster_key])].append(float(row[metric]))
    keys = sorted(groups)
    if not keys:
        raise ValueError("Cannot summarize empty records")
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        selected = rng.integers(len(keys), size=len(keys))
        estimates.append(sums[selected].sum() / counts[selected].sum())
    return {"mean": float(sums.sum() / counts.sum()),
            "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "clusters": len(keys), "observations": int(counts.sum()),
            "interval_method": "trajectory-cluster percentile bootstrap"}


def grouped_summary(records: list[dict], metrics: list[str], cluster_key: str) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[f"o{row['observation_id']}_d{row['dynamics_id']}"].append(row)
    groups["all"] = records
    return {condition: {metric: clustered_interval(rows, metric, cluster_key)
                        for metric in metrics} for condition, rows in groups.items()}


@torch.inference_mode()
def evaluate_forecasts(model, data: str | Path, split: str, feature_cache=None,
                       batch_size: int = 128, num_workers: int = 4,
                       sequence_length: int = 8, stride: int = 5) -> dict:
    device = next(model.parameters()).device
    dataset = TrajectoryDataset(data, split=split, sequence_length=sequence_length,
                                stride=stride, feature_cache=feature_cache)
    if feature_cache:
        cache = json.loads((Path(feature_cache) / "manifest.json").read_text())
        if cache["encoder_weights_sha256"] != model.provenance["weights_sha256"]:
            raise ValueError("Feature encoder and model pretrained encoder differ")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=device.type == "cuda")
    history = model.config.history_length
    horizons = [h for h in (1, 3, 5) if h <= sequence_length - history]
    records = []
    start = time.time()
    for batch in loader:
        tensors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        features = model._features(tensors)
        targets = model._features(tensors, "reference_")
        actions = tensors["actions"]
        predicted = model.rollout_features(features[:, :history], actions[:, :history - 1],
                                            actions[:, history - 1:])
        zero_action_prediction = model.rollout_features(features[:, :history], actions[:, :history - 1],
                                                        torch.zeros_like(actions[:, history - 1:]))
        target = targets[:, history:]
        observation_context, _ = model.infer_context(features[:, :history], actions[:, :history - 1])
        persistence = model.correct_observations(features[:, :history], observation_context)[:, -1:]
        per_horizon = (predicted - target).square().mean(-1).cpu().numpy()
        persistence_error = (persistence - target).square().mean(-1).cpu().numpy()
        zero_action_error = (zero_action_prediction - target).square().mean(-1).cpu().numpy()
        action_effect = (zero_action_prediction - predicted).square().mean(-1).cpu().numpy()
        for i in range(len(features)):
            row = {"trajectory_id": batch["trajectory_id"][i], "start": int(batch["start"][i]),
                   "seed": int(batch["seed"][i]),
                   "observation_id": int(batch["observation_id"][i]), "dynamics_id": int(batch["dynamics_id"][i])}
            for h in horizons:
                row[f"mse_h{h}"] = float(per_horizon[i, h - 1])
                row[f"persistence_mse_h{h}"] = float(persistence_error[i, h - 1])
                row[f"zero_action_mse_h{h}"] = float(zero_action_error[i, h - 1])
                row[f"action_effect_mse_h{h}"] = float(action_effect[i, h - 1])
            records.append(row)
    metrics = [name for h in horizons for name in (f"mse_h{h}", f"persistence_mse_h{h}",
                                                  f"zero_action_mse_h{h}", f"action_effect_mse_h{h}")]
    return {"kind": "fixed_reference_forecasting", "split": split,
            "elapsed_seconds": time.time() - start, "window_stride": stride,
            "sequence_length": sequence_length, "summary": grouped_summary(records, metrics, "seed"),
            "records": records}


class LatentGoalCost(nn.Module):
    """Adapt our raw-action rollout API to the unchanged upstream CEM interface."""
    def __init__(self, model, environment: str, action_interface: str = "relative"):
        super().__init__()
        self.model = model
        low, high = native_action_bounds(environment, action_interface)
        self.register_buffer("native_low", torch.tensor(np.tile(low, 5)))
        self.register_buffer("native_high", torch.tensor(np.tile(high, 5)))

    def to_native(self, standardized: torch.Tensor) -> torch.Tensor:
        # CEM's unit Gaussian lives in the checkpoint's standardized action
        # coordinates. Clipping z-scores would incorrectly truncate ordinary
        # actions beyond one training standard deviation. Bound native controls.
        native = self.model.action_mean + self.model.action_std * standardized
        return native.maximum(self.native_low).minimum(self.native_high)

    def get_cost(self, info_dict: dict, actions: torch.Tensor) -> torch.Tensor:
        batch, candidates, horizon, dimension = actions.shape
        flatten = lambda key: info_dict[key].flatten(0, 1)
        rollout = self.model.rollout_features(flatten("history_features"), flatten("past_actions"),
                    self.to_native(actions).reshape(batch * candidates, horizon, dimension),
                    contexts=(flatten("observation_context"), flatten("dynamics_context")))
        cost = (rollout[:, -1] - flatten("goal_features")).square().mean(-1)
        return cost.reshape(batch, candidates)


@torch.inference_mode()
def evaluate_planning(model, data: str | Path, split: str = "test", episodes_per_dynamics: int = 64,
                      horizon: int = 5, samples: int = 128, iterations: int = 5, elites: int = 16,
                      native_budget: int = 50, goal_offset: int = 5, seed: int = 1701,
                      progress_path: str | Path | None = None, save_video: bool = False,
                      policy: str = "world_model", max_runtime_seconds: int | None = None,
                      run_identity: dict | None = None) -> dict:
    from gymnasium.spaces import Box
    from stable_worldmodel.planning.solver.cem import CEMSolver

    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    environment = manifest["environment"]
    action_interface = manifest.get("action_interface", "absolute" if manifest.get("action_units") == "absolute target pixels" else "relative")
    image_size, action_block = manifest["image_size"], manifest["action_block"]
    if action_block != 5 or model.action_dim != 10:
        raise ValueError("Planner expects official five-step, two-action interface")
    if elites > samples or elites < 2:
        raise ValueError("CEM elites must be between2 and samples")
    if policy not in {"world_model", "random", "replay_oracle"}:
        raise ValueError(f"Unsupported policy {policy}")
    history = model.config.history_length
    if native_budget <= (history - 1) * action_block:
        raise ValueError("Budget must include both history acquisition and actual planning")
    device = next(model.parameters()).device
    cost = LatentGoalCost(model, environment, action_interface).to(device)
    combinations = split_combinations(split)
    selected, per_dynamics = [], defaultdict(int)
    for ep in sorted(manifest["episodes"], key=lambda x: (x["seed"], x["dynamics_id"])):
        if ep["split"] != split_family(split) or per_dynamics[ep["dynamics_id"]] >= episodes_per_dynamics:
            continue
        if not any(d == ep["dynamics_id"] for _, d in combinations):
            continue
        if ep["steps"] < history - 1 + goal_offset:
            raise ValueError("Goal offset exceeds recorded trajectory")
        selected.append(ep)
        per_dynamics[ep["dynamics_id"]] += 1
    if not selected:
        raise ValueError("No planning episodes")
    protocol = {"run_identity": run_identity, "split": split, "episodes_per_dynamics": episodes_per_dynamics,
                "horizon": horizon, "samples": samples, "iterations": iterations, "elites": elites,
                "native_budget": native_budget, "goal_offset": goal_offset, "seed": seed, "policy": policy,
                "search_coordinates": "checkpoint_training_action_z_scores; native actions clipped to environment bounds",
                "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    records = []
    if progress_path and Path(progress_path).exists():
        progress = json.loads(Path(progress_path).read_text())
        if progress.get("signature") != signature:
            raise ValueError("Planning progress belongs to different weights, data, code, or protocol; use a new output path")
        records = progress["records"]
    done = {(row["trajectory_id"], row["observation_id"]) for row in records}
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for ep in selected:
        with np.load(data / ep["file"], allow_pickle=False) as archive:
            stored = {key: archive[key] for key in archive.files}
        goal_index = history - 1 + goal_offset
        for appearance, dynamics in combinations:
            if dynamics != ep["dynamics_id"]:
                continue
            if (ep["trajectory_id"], appearance) in done:
                continue
            if max_runtime_seconds is not None and time.time() - started > max_runtime_seconds:
                return {"status": "interrupted", "kind": "closed_loop_goal_image_planning", "split": split,
                        "policy": policy, "signature": signature, "protocol": protocol, "records": records}
            env = make_env(environment, dynamics, ep["seed"], image_size, action_interface=action_interface)
            frames, planned_blocks = [], []
            episode_start = time.time()
            try:
                restore_simulator_state(env, environment, stored["simulator_states"][0])
                goal_state = stored["task_states"][goal_index]
                set_task_goal(env, environment, goal_state)
                # Render the desired state under the same task-goal and visual
                # settings. A separate environment avoids perturbing contact
                # caches in the controlled episode while constructing its goal.
                goal_env = make_env(environment, dynamics, ep["seed"], image_size, action_interface=action_interface)
                try:
                    set_task_goal(goal_env, environment, goal_state)
                    restore_simulator_state(goal_env, environment, stored["simulator_states"][goal_index])
                    goal_pixels = render_env(goal_env, environment, image_size)
                finally:
                    goal_env.close()
                goal_image = pixels_to_tensor(goal_pixels, appearance)[None].to(device)
                # Acquire the same chronological context for every method; all
                # native interactions count toward native_budget, including this.
                features, past = [], []
                current_frame = render_env(env, environment, image_size)
                features.append(model.encode_images(pixels_to_tensor(current_frame, appearance)[None].to(device)))
                native_used = 0
                success = False
                first_success = None
                for t in range(history - 1):
                    block = stored["actions"][t].reshape(action_block, 2)
                    for native_action in block:
                        env.step(native_action)
                        native_used += 1
                        _, hit = task_distance_success(env, environment, goal_state)
                        if hit and first_success is None:
                            first_success = native_used
                        success = success or hit
                        if hit:
                            break
                    past.append(torch.tensor(stored["actions"][t], device=device)[None])
                    current_frame = render_env(env, environment, image_size)
                    features.append(model.encode_images(pixels_to_tensor(current_frame, appearance)[None].to(device)))
                    if success:
                        break
                initial_distance, _ = task_distance_success(env, environment, goal_state)
                initial_errors = state_error_diagnostics(environment, task_state(env, environment), goal_state)
                success_during_context = success
                if save_video:
                    frames.append((pixels_to_tensor(current_frame, appearance).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                solver = CEMSolver(cost=cost, batch_size=1, num_samples=samples, n_steps=iterations,
                                   topk=elites, device=device, seed=seed + ep["seed"])
                solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32), n_envs=1,
                                 config=SimpleNamespace(horizon=horizon, action_block=action_block))
                warm_start = None
                solve_times = []
                action_rng = np.random.default_rng(seed + ep["seed"])
                while native_used + action_block <= native_budget and not success:
                    if policy == "world_model":
                        history_features = torch.stack(features[-history:], 1)
                        past_actions = torch.stack(past[-history + 1:], 1)
                        observation_context, dynamics_context = model.infer_context(history_features, past_actions)
                        goal_features = model.goal_embedding(goal_image, observation_context)
                        info = {"history_features": history_features, "past_actions": past_actions,
                                "observation_context": observation_context, "dynamics_context": dynamics_context,
                                "goal_features": goal_features}
                        before_solve = time.time()
                        with contextlib.redirect_stdout(io.StringIO()):
                            solution = solver.solve(info, init_action=warm_start)
                        solve_times.append(time.time() - before_solve)
                        normalized = solution["actions"].to(device)
                        block = cost.to_native(normalized[:, 0]).reshape(action_block, 2).cpu().numpy()
                        warm_start = normalized[:, 1:]
                    elif policy == "random":
                        lo, hi = native_action_bounds(environment, action_interface)
                        block = action_rng.uniform(lo, hi, (action_block, 2)).astype(np.float32)
                    else:
                        # Explicitly privileged, offline reachability diagnostic.
                        # These future recorded actions are never used by WM policy.
                        replay_index = min(native_used // action_block, len(stored["actions"]) - 1)
                        block = stored["actions"][replay_index].reshape(action_block, 2)
                    executed = []
                    for native_action in block:
                        env.step(native_action)
                        native_used += 1
                        executed.extend(native_action.tolist())
                        _, hit = task_distance_success(env, environment, goal_state)
                        if hit and first_success is None:
                            first_success = native_used
                        success = success or hit
                        if hit:
                            break
                    planned_blocks.append(executed)
                    past.append(torch.tensor(executed, device=device)[None])
                    current_frame = render_env(env, environment, image_size)
                    features.append(model.encode_images(pixels_to_tensor(current_frame, appearance)[None].to(device)))
                    if save_video:
                        frames.append((pixels_to_tensor(current_frame, appearance).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                final_distance, final_success = task_distance_success(env, environment, goal_state)
                final_errors = state_error_diagnostics(environment, task_state(env, environment), goal_state)
                row = {"trajectory_id": ep["trajectory_id"], "seed": ep["seed"],
                       "observation_id": appearance, "dynamics_id": dynamics,
                       "success": int(success), "final_success": int(final_success),
                       "success_during_context": int(success_during_context),
                       "policy_eligible": int(not success_during_context),
                       "native_steps": native_used, "steps_to_success_or_budget": first_success or native_budget,
                       "initial_distance_after_context": initial_distance, "final_distance": final_distance,
                       "distance_definition": "upstream full-state Euclidean distance; mixed units" if environment == "pusht" else "joint-angle Euclidean norm",
                       "elapsed_seconds": time.time() - episode_start,
                       "mean_solve_seconds": float(np.mean(solve_times)) if solve_times else 0.0,
                       "total_solve_seconds": float(sum(solve_times)),
                       "num_replans": len(solve_times), "executed_action_blocks": planned_blocks,
                       "goal_index": goal_index, "policy": policy}
                row.update(final_errors)
                row.update({"initial_" + key: value for key, value in initial_errors.items()})
                if environment == "pusht":
                    # Prespecified upstream success tolerances define the task
                    # stratum, not outcomes or model-specific predictions.
                    row["requires_manipulation"] = int(initial_errors["block_translation_error_px"] >= 20.0
                                                       or initial_errors["block_angle_error_rad"] >= np.pi / 9)
                records.append(row)
                print(json.dumps({"planning_episode": len(records), **{k: row[k] for k in (
                    "trajectory_id", "observation_id", "dynamics_id", "success", "elapsed_seconds")}}), flush=True)
                if save_video and progress_path:
                    video = Path(progress_path).parent / "videos" / f"{ep['trajectory_id']}-o{appearance}.npz"
                    video.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(video, frames=np.asarray(frames),
                                        goal_image=(pixels_to_tensor(goal_pixels, appearance).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                if progress_path:
                    atomic_json({"status": "running", "signature": signature, "protocol": protocol,
                                 "records": records}, progress_path)
            finally:
                env.close()
    eligible = [row for row in records if row["policy_eligible"]]
    total_replans = sum(row["num_replans"] for row in records)
    total_solve_seconds = sum(row["total_solve_seconds"] for row in records)
    diagnostic_metrics = ["block_translation_error_px", "block_angle_error_rad", "agent_position_error_px"] if environment == "pusht" else ["wrapped_joint_error_rad"]
    manipulation = [row for row in eligible if row.get("requires_manipulation", 0)]
    return {"status": "complete", "kind": "closed_loop_goal_image_planning", "split": split, "policy": policy,
            "signature": signature, "protocol": protocol,
            "planner": {"implementation": "stable_worldmodel.planning.solver.cem.CEMSolver",
                        "samples": samples, "iterations": iterations, "elites": elites,
                        "horizon": horizon, "receding_horizon": 1, "action_block": action_block,
                        "search_coordinates": "checkpoint_training_action_z_scores; native actions clipped to environment bounds",
                        "native_budget": native_budget, "history_steps_charged": (history - 1) * action_block,
                        "goal_offset_from_end_of_history": goal_offset},
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in records),
            "attempt_wall_seconds": time.time() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "aggregate_costs": {"total_replans": total_replans, "total_solve_seconds": total_solve_seconds,
                                "seconds_per_replan": total_solve_seconds / total_replans if total_replans else None},
            "summary": grouped_summary(records, ["success", "final_success", "success_during_context", "steps_to_success_or_budget",
                                                   "final_distance", "elapsed_seconds", "mean_solve_seconds", *diagnostic_metrics], "seed"),
            "eligible_summary": grouped_summary(eligible, ["success", "final_success", "steps_to_success_or_budget"], "seed") if eligible else {},
            "manipulation_summary": grouped_summary(manipulation, ["success", "final_success", "steps_to_success_or_budget"], "seed") if manipulation else {},
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained package directory or model.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--kind", choices=["forecast", "planning", "both"], default="both")
    parser.add_argument("--split", choices=["val", "development", "test", "extrapolation"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=64, help="Planning episodes per physical condition")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--native-budget", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--policy", choices=["world_model", "random", "replay_oracle"], default="world_model")
    parser.add_argument("--max-runtime-seconds", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, state = load_package(args.checkpoint, device=args.device)
    model.eval().requires_grad_(False)
    package_file = args.checkpoint if args.checkpoint.is_file() else args.checkpoint / "model.pt"
    result = {"status": "complete", "checkpoint": str(package_file.resolve()),
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": json.loads((args.data / "manifest.json").read_text())["environment"],
              "checkpoint_sha256": hashlib.sha256(package_file.read_bytes()).hexdigest(),
              "model_mode": model.config.mode, "checkpoint_epoch": state["epoch"],
              "training_seed": None if model.config.mode == "frozen" else state.get("config", {}).get("metadata", {}).get("config", {}).get("seed"),
              "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()}
    if args.kind in {"forecast", "both"}:
        result["forecast"] = evaluate_forecasts(model, args.data, args.split, args.feature_cache,
                                                args.batch_size, args.workers)
        atomic_json({**result, "status": "forecast_complete" if args.kind == "both" else "complete"}, args.output)
    if args.kind in {"planning", "both"}:
        result["planning"] = evaluate_planning(model, args.data, args.split, args.episodes, args.horizon,
                                                args.samples, args.iterations, args.elites, args.native_budget,
                                                args.goal_offset, args.seed, args.output.with_suffix(".progress.json"),
                                                args.save_video, args.policy, args.max_runtime_seconds,
                                                {"checkpoint_sha256": result["checkpoint_sha256"],
                                                 "data_manifest_sha256": result["data_manifest_sha256"]})
        if result["planning"]["status"] == "interrupted":
            result["status"] = "interrupted"
    atomic_json(result, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
