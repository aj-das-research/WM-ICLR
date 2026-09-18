#!/usr/bin/env python3
"""Measure real SOFA transitions and verify deterministic reachable-image goals."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from shiftwm.extensions.surgery import SurgeryAdapter, SurgeryConfig, exploration_actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("reports/surgery/profile"))
    p.add_argument("--calls", type=int, default=30)
    p.add_argument("--image-size", type=int, default=128)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = SurgeryConfig(image_size=args.image_size)
    commands = exploration_actions(config.seed, args.calls)
    start = time.perf_counter()
    env = SurgeryAdapter(config)
    init_seconds = time.perf_counter() - start
    initial = env.diagnostics()
    frames, states = [env.observe()], [initial["tissue_target_xyz_m"]]
    step_start = time.perf_counter()
    for command in commands:
        image, diagnostic = env.step(command)
        frames.append(image)
        states.append(diagnostic["tissue_target_xyz_m"])
    step_seconds = time.perf_counter() - step_start
    target = states[-1].copy()
    env.close()
    # A new simulator and identical reset seed test the actual release contract.
    env = SurgeryAdapter(config)
    second_initial = env.diagnostics()
    replay = [env.set_reachable_goal(target)]
    replay_states = [second_initial["tissue_target_xyz_m"]]
    diagnostics = []
    for command in commands:
        image, diagnostic = env.step(command)
        replay.append(image)
        replay_states.append(diagnostic["tissue_target_xyz_m"])
        diagnostics.append(diagnostic)
    goal_image = replay[-1].copy()
    final = env.diagnostics()
    env.close()
    state_error = float(np.max(np.abs(np.asarray(states) - replay_states)))
    # Third replay has the SAME visible goal as pass 2: compare RGB byte-for-byte.
    env = SurgeryAdapter(config)
    third = [env.set_reachable_goal(target)]
    for command in commands:
        third.append(env.step(command)[0])
    env.close()
    rgb_equal = bool(np.array_equal(replay, third))
    Image.fromarray(replay[0]).save(args.output / "initial.png")
    Image.fromarray(goal_image).save(args.output / "reachable_goal.png")
    np.savez_compressed(args.output / "replay.npz", images=np.array(replay),
                        goal_image=goal_image, actions_commanded=commands,
                        tissue_target_xyz_m=np.array(replay_states))
    result = {
        "config": asdict(config), "calls": args.calls,
        "initialization_seconds": init_seconds, "step_seconds": step_seconds,
        "native_calls_per_second": args.calls / step_seconds,
        "replay_state_max_abs_error_m": state_error,
        "same_goal_rgb_replay_exact": rgb_equal,
        "reference_native_distance_m": final["distance_m"],
        "reference_native_success": bool(final["success"]),
        "initial_reference_distance_m": float(np.linalg.norm((target - initial["tissue_target_xyz_m"])[[0,2]])),
        "invalid_action_calls": sum(not d["valid_action"] for d in diagnostics),
        "unstable_calls": sum(not d["stable_deformation"] for d in diagnostics),
        "finite_states": bool(np.isfinite(replay_states).all()),
        "goal_protocol": "reachable_endpoint_from_recorded_actions; differs_from_native_random_goal_sampler",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if state_error > 1e-8 or not rgb_equal or not final["success"] or not result["finite_states"]:
        raise SystemExit("Replay/goal contract failed")


if __name__ == "__main__":
    main()
