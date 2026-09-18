#!/usr/bin/env python3
"""Exercise the real isolated simulator protocol, budget, and goal semantics."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    directory = Path("reports/surgery/rpc")
    directory.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    with (directory / "simulator.stderr.log").open("w") as log:
        proc = subprocess.Popen(["environments/surgery/run_python.sh", "scripts/extensions/serve_surgery.py"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        def request(data):
            proc.stdin.write(json.dumps(data) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f"RPC exited early with {proc.poll()}")
            result = json.loads(line)
            return result
        checks = {}
        first = request({"op": "reset", "seed": 4100000, "dynamics_id": 0,
                         "image_size": 128, "max_steps": 2})
        assert first["ok"]
        image = first["image"]
        checks["rgb_payload_valid"] = image["shape"] == [128,128,3] and len(base64.b64decode(image["data"])) == 128*128*3
        checks["initial_eligibility_computed"] = first["metrics"]["eligible_initial"] == (first["metrics"]["distance_m"] >= 0.004)
        step = request({"op": "step", "actions": [[0.1,0.1]] * 5, "stop_on_success": False})
        checks["budget_partial_block"] = step["ok"] and step["truncated"] and len(step["executed_commands"]) == 2 and step["metrics"]["steps"] == 2
        repeated = request({"op": "step", "actions": [[0,0]]})
        checks["reject_after_budget"] = not repeated["ok"]
        second = request({"op": "reset", "seed": 4100000, "dynamics_id": 0,
                          "image_size": 128, "max_steps": 200,
                          "goal_state": first["metrics"]["tissue_target_xyz_m"]})
        checks["real_goal_assignment"] = second["ok"] and second["metrics"]["success"] and not second["metrics"]["eligible_initial"]
        success = request({"op": "step", "actions": [[0,0]] * 5})
        checks["already_solved_uses_zero_actions"] = success["ok"] and success["terminated"] and success["stop_reason"] == "already_successful" and len(success["executed_commands"]) == 0
        invalid = request({"op": "reset", "seed": 4100000, "dynamics_id": 8})
        checks["reject_invalid_domain"] = not invalid["ok"]
        final = request({"op": "close"})
        checks["graceful_close"] = final["ok"] and final["closed"]
        proc.stdin.close()
        extra_stdout = proc.stdout.read()
        code = proc.wait(timeout=30)
        checks["protocol_stdout_clean"] = extra_stdout == ""
        checks["exit_zero"] = code == 0
    report = {"checks": checks, "passed": all(checks.values()), "wall_seconds": time.perf_counter() - begin,
              "launcher": ["environments/surgery/run_python.sh", "scripts/extensions/serve_surgery.py"]}
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
