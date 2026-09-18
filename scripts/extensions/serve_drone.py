#!/usr/bin/env python3
"""JSON-lines drone physics RPC; launch with the isolated drone Python runtime.

Only protocol responses reach stdout. Each response image is base64-encoded
contiguous uint8 RGB. Native state is used internally for PID/scoring only.
"""
from __future__ import annotations

import base64
import json
import os
import sys

# Preserve the protocol pipe, then redirect Python and C-level simulator output
# to stderr before importing PyBullet or any upstream modules.
PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

import numpy as np

from shiftwm.extensions.drone import DroneConfig, VisualDroneEnv, UPSTREAM_COMMIT


def encode_image(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return {"shape": list(image.shape), "dtype": "uint8",
            "data": base64.b64encode(image.tobytes()).decode("ascii")}


def main():
    env = None
    try:
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
                op = request.get("op")
                if op == "reset":
                    size = int(request.get("image_size", 128))
                    max_steps = int(request.get("max_steps", 200))
                    if size > 512 or max_steps > 10000:
                        raise ValueError("Image size or max_steps exceeds bounded RPC limit")
                    config = DroneConfig(image_size=size, max_steps=max_steps)
                    new_env = VisualDroneEnv(int(request.get("dynamics_id", 0)), config)
                    goal = request.get("goal_state")
                    if goal is not None:
                        goal = np.asarray(goal, dtype=np.float64)
                        if goal.shape != (20,) or not np.isfinite(goal).all():
                            raise ValueError("goal_state must be a finite 20-vector")
                    if env is not None:
                        env.close()
                    env = new_env
                    image, _ = env.reset(seed=int(request["seed"]))
                    if goal is not None:
                        env.set_goal(goal)
                    response = {"ok": True, "image": encode_image(image), "metrics": env.metrics(),
                                "upstream_commit": UPSTREAM_COMMIT, "native_steps": 0}
                elif op == "step":
                    if env is None:
                        raise ValueError("reset must precede step")
                    actions = np.asarray(request["actions"], dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[1:] != (2,) or not 1 <= len(actions) <= 5:
                        raise ValueError("actions must have shape [1..5,2]")
                    if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                        raise ValueError("All actions must be finite and within [-1,1]")
                    stop_success = request.get("stop_on_success", True)
                    if not isinstance(stop_success, bool):
                        raise ValueError("stop_on_success must be boolean")
                    executed, metrics = [], []
                    terminated, truncated = False, env.steps >= env.config.max_steps
                    image = env.render()
                    if not truncated:
                        for action in actions:
                            image, _, terminated, truncated, info = env.step(action)
                            executed.append(action.tolist())
                            metrics.append(info)
                            if terminated or truncated or (stop_success and info["success"]):
                                break
                    response = {"ok": True, "image": encode_image(image),
                                "executed_commands": executed, "metrics_per_step": metrics,
                                "metrics": env.metrics(), "terminated": bool(terminated),
                                "truncated": bool(truncated), "native_steps": env.steps}
                elif op == "close":
                    if env is not None:
                        env.close()
                        env = None
                    response = {"ok": True, "closed": True}
                else:
                    raise ValueError("Unknown op; expected reset, step, or close")
            except Exception as error:
                response = {"ok": False, "error": {"type": type(error).__name__, "message": str(error)}}
            if isinstance(request, dict) and "id" in request:
                response["id"] = request["id"]
            PROTOCOL.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
            if response.get('closed'):
                break
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
