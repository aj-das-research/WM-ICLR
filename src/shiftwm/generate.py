"""Collect real simulator trajectories for compositional world-model adaptation.

Uses unchanged upstream stable-worldmodel environments. PushT uses relative
target displacements (100 pixels per action unit), as confirmed from official
checkpoint training-data action statistics. Reacher
uses upstream action_repeat=2. Both group five native controls per observation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from .data import APPEARANCES, TRAIN_COMBINATIONS, validate_manifest


DYNAMICS = {
    "pusht": {
        0: {"space_damping": 0.0},
        1: {"space_damping": 0.15},
        2: {"space_damping": 0.65},
        3: {"space_damping": 0.95},
    },
    "reacher": {
        0: {"arm_density": 1000.0, "finger_density": 1000.0},
        1: {"arm_density": 650.0, "finger_density": 650.0},
        2: {"arm_density": 1350.0, "finger_density": 1350.0},
        3: {"arm_density": 1500.0, "finger_density": 1500.0},
    },
}


def make_env(name: str, dynamics_id: int, seed: int, image_size: int = 224,
             action_interface: str = "relative"):
    """Return reset real simulator, with actual dynamics interventions."""
    if name == "pusht":
        from stable_worldmodel.envs.pusht.env import PushT
        if action_interface not in {"relative", "absolute"}:
            raise ValueError(action_interface)
        env = PushT(relative=action_interface == "relative", resolution=image_size,
                    damping=DYNAMICS[name][dynamics_id]["space_damping"])
        env.reset(seed=seed)
        return env
    if name == "reacher":
        from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper
        env = ReacherDMControlWrapper(task="qpos_match", seed=seed)
        physics = DYNAMICS[name][dynamics_id]
        env.reset(seed=seed, options={"variation_values": {
            "agent.arm_density": np.array([physics["arm_density"]], dtype=np.float32),
            "agent.finger_density": np.array([physics["finger_density"]], dtype=np.float32),
        }})
        return env
    raise ValueError(name)


def render_env(env, name: str, image_size: int) -> np.ndarray:
    if name == "reacher":
        return env.render(width=image_size, height=image_size)
    return env.render()


def simulator_state(env, name: str) -> np.ndarray:
    """State for exact simulator restoration; unavailable to model/policy."""
    if name == "pusht":
        return np.array([*env.agent.position, *env.agent.velocity,
                         *env.block.position, *env.block.velocity,
                         env.block.angle, env.block.angular_velocity], dtype=np.float64)
    return np.concatenate([env.env.physics.data.qpos.copy(), env.env.physics.data.qvel.copy()])


def restore_simulator_state(env, name: str, state: np.ndarray) -> None:
    if name == "pusht":
        env.agent.position = tuple(state[:2])
        env.agent.velocity = tuple(state[2:4])
        env.block.angle = state[8]
        # Pymunk's position is relative to the body's center of gravity; changing
        # angle after position translates an asymmetric T. Restore angle first.
        env.block.position = tuple(state[4:6])
        env.block.velocity = tuple(state[6:8])
        env.block.angular_velocity = state[9]
        env.space.reindex_shapes_for_body(env.agent)
        env.space.reindex_shapes_for_body(env.block)
    else:
        nq = env.env.physics.model.nq
        env.set_state(state[:nq], state[nq:])


def task_state(env, name: str) -> np.ndarray:
    return env._get_obs().copy() if name == "pusht" else env.env.physics.data.qpos.copy()


def set_task_goal(env, name: str, goal: np.ndarray) -> None:
    if name == "pusht":
        env._set_goal_state(goal.copy())
    else:
        env.set_target_qpos(goal.copy())


def task_distance_success(env, name: str, goal: np.ndarray) -> tuple[float, bool]:
    current = task_state(env, name)
    if name == "pusht":
        success, distance = env.eval_state(goal, current)
        return float(distance), bool(success)
    # Match upstream qpos_match exactly; no looser bespoke tolerance.
    difference = np.abs(current - goal)
    return float(np.linalg.norm(difference)), bool(np.all(difference < env.env.task.qpos_threshold))


def native_action_bounds(name: str, action_interface: str = "relative") -> tuple[np.ndarray, np.ndarray]:
    if name == "pusht" and action_interface == "absolute":
        return np.zeros(2, np.float32), np.full(2, 512.0, np.float32)
    return np.full(2, -1.0, np.float32), np.ones(2, np.float32)


def collection_action(env, name: str, rng: np.random.Generator) -> np.ndarray:
    if name == "pusht":
        # Match upstream WeakPolicy exactly, including clipping the relative
        # displacement after constraining the target near the block.
        agent = np.asarray(env.agent.position)
        block = np.asarray(env.block.position)
        target = agent + rng.uniform(-100.0, 100.0, size=2)
        target = np.clip(target, block - 100.0, block + 100.0)
        if env.relative:
            return np.clip((target - agent) / env.action_scale, -1, 1).astype(np.float32)
        return np.clip(target, 0, 512).astype(np.float32)
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def collect_episode(task: dict) -> dict:
    root = Path(task["output"])
    uid = f"{task['split']}-s{task['seed']}-d{task['dynamics_id']}"
    relative_path = Path("episodes") / f"{uid}.npz"
    path = root / relative_path
    sidecar = path.with_suffix(".json")
    config_digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    if path.exists() and sidecar.exists():
        prior = json.loads(sidecar.read_text())
        if prior.get("config_sha256") == config_digest:
            return prior
        raise RuntimeError(f"Refusing incompatible overwrite: {path}")
    env = make_env(task["env"], task["dynamics_id"], task["seed"], task["image_size"],
                   action_interface=task.get("action_interface", "relative"))
    rng = np.random.default_rng(task["seed"] + 7_000_000)
    images = [render_env(env, task["env"], task["image_size"])]
    states = [simulator_state(env, task["env"])]
    task_states = [task_state(env, task["env"])]
    actions, contacts = [], []
    try:
        for _ in range(task["steps"]):
            group, contact_count = [], 0
            for _ in range(task["action_block"]):
                action = collection_action(env, task["env"], rng)
                _, _, _, _, info = env.step(action)
                group.extend(action.tolist())
                contact_count += int(info.get("n_contacts", 0))
            images.append(render_env(env, task["env"], task["image_size"]))
            states.append(simulator_state(env, task["env"]))
            task_states.append(task_state(env, task["env"]))
            actions.append(group)
            contacts.append(contact_count)
        arrays = {"images": np.asarray(images, dtype=np.uint8),
                  "actions": np.asarray(actions, dtype=np.float32),
                  "simulator_states": np.asarray(states, dtype=np.float64),
                  "task_states": np.asarray(task_states, dtype=np.float64),
                  "contacts": np.asarray(contacts, dtype=np.int32)}
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise FloatingPointError(f"Nonfinite simulator data {uid}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.partial")
        with temporary.open("wb") as f:
            np.savez_compressed(f, **arrays)
        temporary.replace(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {"trajectory_id": uid, "split": task["split"], "seed": task["seed"],
                    "dynamics_id": task["dynamics_id"], "steps": task["steps"],
                    "file": str(relative_path), "sha256": sha,
                    "config_sha256": config_digest,
                    "contact_steps": int(np.count_nonzero(contacts)),
                    "state_motion_mean": float(np.linalg.norm(np.diff(arrays["task_states"], axis=0), axis=-1).mean())}
        sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["pusht", "reacher"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=256, help="Per dynamics condition")
    parser.add_argument("--val-episodes", type=int, default=32)
    parser.add_argument("--development-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64, help="Grouped observation transitions per episode")
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action_block != 5:
        raise ValueError("Released LeWM checkpoints require action_block=5")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for split_id, (split, count) in enumerate([
        ("train", args.train_episodes), ("val", args.val_episodes),
        ("development", args.development_episodes), ("test", args.test_episodes)
    ]):
        if count < 0 or count >= 100000:
            raise ValueError("Episode counts must be in [0,100000)")
        for seed_idx in range(count):
            # Sharing initial RNG across dynamics is intentional for paired
            # comparisons, but there is zero seed overlap between split families.
            seed = args.seed + split_id * 1_000_000 + seed_idx
            for dynamics_id in (range(4) if split == "test" else range(3)):
                tasks.append({"env": args.env, "output": str(args.output.resolve()),
                              "split": split, "seed": seed, "dynamics_id": dynamics_id,
                              "steps": args.steps, "action_block": args.action_block,
                              "image_size": args.image_size, "action_interface": "relative" if args.env == "pusht" else "torque"})
    start = time.time()
    episodes = []
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                 mp_context=multiprocessing.get_context("spawn"))
    with executor:
        for ep in executor.map(collect_episode, tasks):
            episodes.append(ep)
            if len(episodes) % 10 == 0:
                print(json.dumps({"completed": len(episodes), "total": len(tasks),
                                  "elapsed_seconds": time.time() - start}), flush=True)
    revision = subprocess.run(["git", "-C", "external/stable-worldmodel", "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    manifest = {"schema_version": 1, "environment": args.env, "upstream_revision": revision,
                "image_size": args.image_size, "action_block": args.action_block,
                "native_action_dimension": 2, "model_action_dimension": 10,
                "action_interface": "relative" if args.env == "pusht" else "torque",
                "action_units": "relative target displacement / 100 pixels" if args.env == "pusht" else "normalized torque",
                "appearance_factors": APPEARANCES, "dynamics_factors": DYNAMICS[args.env],
                "train_combinations": TRAIN_COMBINATIONS,
                "development_combinations": [[1, 1]], "heldout_test_combinations": [[2, 2]],
                "elapsed_seconds": time.time() - start, "episodes": episodes}
    validate_manifest(manifest)
    temporary = args.output / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.output / "manifest.json")
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "episodes": len(episodes),
                      "native_transitions": len(episodes) * args.steps * args.action_block}), flush=True)


if __name__ == "__main__":
    main()
