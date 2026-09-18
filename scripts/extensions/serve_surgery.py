#!/usr/bin/env python3
"""JSON-line surgery RPC; SOFA/Python 3.10 is isolated from the Torch process.

Launch: environments/surgery/run_python.sh scripts/extensions/serve_surgery.py
All native and Python simulator stdout is sent to stderr before import. Only
protocol replies use the original stdout descriptor. One request, one reply.
"""
from __future__ import annotations
import base64
import json
import os
import sys
import traceback

protocol_stream = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

import numpy as np
from shiftwm.extensions.surgery import SurgeryAdapter, SurgeryConfig

GAINS = (1.0, 0.75, 1.25)


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def packed_image(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return {"shape": list(image.shape), "dtype": "uint8",
            "data": base64.b64encode(image.tobytes()).decode("ascii")}


def reply(payload):
    protocol_stream.write(json.dumps(serializable(payload), allow_nan=False) + "\n")


def main():
    env = None
    steps = 0
    max_steps = 200
    terminal = False
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request.get("op", request.get("command"))
                if op == "close":
                    if env is not None:
                        env.close()
                        env = None
                    reply({"ok": True, "closed": True})
                    break
                if op == "reset":
                    dyn = int(request["dynamics_id"])
                    if dyn not in range(3):
                        raise ValueError("dynamics_id must be 0, 1, or 2")
                    max_steps = int(request.get("max_steps", 200))
                    if max_steps < 1:
                        raise ValueError("max_steps must be positive")
                    if env is not None:
                        env.close()
                    env = SurgeryAdapter(SurgeryConfig(seed=int(request["seed"]),
                                         actuator_gain=GAINS[dyn],
                                         image_size=int(request.get("image_size", 128))))
                    if "goal_state" in request:
                        image = env.set_reachable_goal(np.asarray(request["goal_state"], np.float64))
                    else:
                        image = env.observe()
                    metrics = env.diagnostics()
                    metrics.update({"steps": 0, "eligible_initial": bool(metrics["distance_m"] >= 0.004 and not metrics["success"])})
                    from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
                    renderer = {"vendor": glGetString(GL_VENDOR).decode(),
                                "renderer": glGetString(GL_RENDERER).decode(),
                                "version": glGetString(GL_VERSION).decode(),
                                "egl_vendor_file": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
                                "host": os.uname().nodename}
                    print(json.dumps({"surgery_renderer": renderer}), file=sys.stderr, flush=True)
                    steps, terminal = 0, False
                    reply({"ok": True, "image": packed_image(image), "metrics": metrics,
                           "terminated": False, "truncated": False, "renderer": renderer})
                elif op == "step":
                    if env is None:
                        raise ValueError("Reset before stepping")
                    if terminal:
                        raise ValueError("Episode has stopped; reset before stepping")
                    commands = np.asarray(request["actions"], dtype=np.float32)
                    if commands.ndim != 2 or commands.shape[1] != 2 or not 1 <= len(commands) <= 5:
                        raise ValueError("Expected 1..5 commanded 2D actions")
                    if not np.isfinite(commands).all() or np.any(np.abs(commands) > 1):
                        raise ValueError("Commands must be finite in [-1,1]")
                    stop_on_success = bool(request.get("stop_on_success", True))
                    current = env.diagnostics()
                    current["steps"] = steps
                    if stop_on_success and current["success"]:
                        terminal = True
                        reply({"ok": True, "image": packed_image(env.observe()),
                               "executed_commands": [], "metrics_per_step": [],
                               "metrics": current, "terminated": True, "truncated": False,
                               "stop_reason": "already_successful"})
                        continue
                    if env.env._initial_env_state["distance_ttp_idp"] == 0:
                        raise ValueError("Cannot advance an initially exact-goal episode: upstream motion-efficiency divides by its zero initial distance")
                    executed_commands, metrics_per_step = [], []
                    terminated, truncated = False, False
                    stop_reason = None
                    for command in commands:
                        image, metric = env.step(command)
                        steps += 1
                        metric["steps"] = steps
                        executed_commands.append(command.copy())
                        metrics_per_step.append(metric)
                        if not metric["valid_action"]:
                            terminated, stop_reason = True, "invalid_action"
                        elif not metric["stable_deformation"]:
                            terminated, stop_reason = True, "unstable_deformation"
                        elif stop_on_success and metric["success"]:
                            terminated, stop_reason = True, "success"
                        if steps >= max_steps and not terminated:
                            truncated, stop_reason = True, "budget"
                        if terminated or truncated:
                            break
                    terminal = terminated or truncated
                    reply({"ok": True, "image": packed_image(image),
                           "executed_commands": executed_commands,
                           "metrics_per_step": metrics_per_step, "metrics": metrics_per_step[-1],
                           "terminated": terminated, "truncated": truncated,
                           "stop_reason": stop_reason})
                else:
                    raise ValueError(f"Unknown op {op!r}")
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                reply({"ok": False, "error": type(error).__name__, "message": str(error)})
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
