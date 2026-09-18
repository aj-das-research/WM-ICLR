import base64
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from shiftwm.extensions.drone import VisualDroneEnv

ROOT = Path(__file__).resolve().parents[2]


def test_rpc_stdout_is_json_and_reset_step_limits_are_deterministic():
    source = VisualDroneEnv()
    source.reset(seed=23)
    initial_state = source.simulator_state().tolist()
    source.close()
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts/extensions/serve_drone.py")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
    def request(value):
        process.stdin.write(json.dumps(value) + "\n")
        process.stdin.flush()
        answer = json.loads(process.stdout.readline())
        if "id" in value:
            assert answer["id"] == value["id"]
        return answer
    try:
        initial = request({"op": "reset", "seed": 23, "image_size": 64, "max_steps": 3, "id": "r1"})
        assert initial["ok"]
        image = initial["image"]
        assert image["shape"] == [64, 64, 3] and image["dtype"] == "uint8"
        assert len(base64.b64decode(image["data"])) == 64 * 64 * 3
        invalid = request({"op": "step", "actions": [[2, 0]]})
        assert not invalid["ok"]
        stepped = request({"op": "step", "actions": [[.3, .2]] * 5})
        assert stepped["ok"] and stepped["native_steps"] == 3
        assert len(stepped["executed_commands"]) == len(stepped["metrics_per_step"]) == 3
        assert stepped["truncated"]
        exhausted = request({"op": "step", "actions": [[0, 0]]})
        assert exhausted["ok"] and not exhausted["executed_commands"] and exhausted["truncated"]
        repeated = request({"op": "reset", "seed": 23, "image_size": 64, "max_steps": 3})
        assert repeated["image"] == initial["image"]
        ready = request({"op": "reset", "seed": 23, "goal_state": initial_state})
        assert ready["metrics"]["success"]
        reached = request({"op": "step", "actions": [[0, 0]] * 5, "stop_on_success": True})
        assert reached["metrics"]["success"] and len(reached["executed_commands"]) == 1
        assert request({"op": "close"})["closed"]
        assert process.stdout.readline() == ""
        assert process.wait(timeout=20) == 0
    finally:
        process.stdin.close()
        assert process.wait(timeout=20) == 0
