#!/usr/bin/env python3
"""Audit every collected drone episode against native physics records."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    started = time.time()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    counts, seeds, seen = Counter(), {}, set()
    displacements, bytes_total, state_count = [], 0, 0
    for ep in manifest["episodes"]:
        uid = ep["trajectory_id"]
        assert uid not in seen, uid
        seen.add(uid)
        family = seeds.setdefault(ep["seed"], ep["split"])
        assert family == ep["split"], (ep["seed"], family, ep["split"])
        counts[ep["split"]] += 1
        model_path, audit_path = root / ep["file"], root / ep["audit_file"]
        assert digest(model_path) == ep["sha256"], uid
        assert digest(audit_path) == ep["audit_sha256"], uid
        bytes_total += model_path.stat().st_size + audit_path.stat().st_size
        with np.load(model_path, allow_pickle=False) as model, np.load(audit_path, allow_pickle=False) as audit:
            assert set(model.files) == {"images", "actions"}, uid
            block = ep["action_block"]
            native_steps = ep["native_steps"]
            assert model["images"].dtype == np.uint8 and model["actions"].dtype == np.float32
            assert model["actions"].shape == (ep["steps"], 2 * block)
            np.testing.assert_array_equal(model["images"], audit["images"][::block], err_msg=uid)
            np.testing.assert_array_equal(model["actions"], audit["actions"].reshape(-1, 2 * block), err_msg=uid)
            np.testing.assert_array_equal(audit["goal_image"], audit["images"][-1], err_msg=uid)
            np.testing.assert_array_equal(audit["goal_state"], audit["simulator_states"][-1], err_msg=uid)
            np.testing.assert_allclose(audit["executed_actions"],
                                       ep["protocol"]["response_gain"] * audit["actions"], err_msg=uid)
            states = audit["simulator_states"]
            assert states.shape == (native_steps + 1, 20)
            assert np.isfinite(states).all() and np.isfinite(audit["low_level_rpm"]).all()
            assert audit["low_level_rpm"].shape == (native_steps, 12, 4)
            assert np.max(np.abs(audit["actions"])) <= 1
            config = ep["protocol"]["config"]
            assert np.linalg.norm(states[-1, 10:13]) < config["speed_tolerance_m_s"]
            assert abs(states[-1, 2] - config["altitude_m"]) < config["altitude_tolerance_m"]
            assert np.max(np.abs(states[:, :2])) <= config["workspace_radius_m"] + 0.08
            state_count += len(states)
            displacements.append(float(np.linalg.norm(states[-1, :2] - states[0, :2])))
    return {"status": "passed", "dataset": str(root), "manifest_sha256": digest(manifest_path),
            "episodes": len(seen), "split_counts": dict(counts), "native_states_checked": state_count,
            "model_and_audit_bytes": bytes_total,
            "initial_goal_distance_m_quantiles": dict(zip(["min", "q25", "median", "q75", "max"],
                                                           np.quantile(displacements, [0, .25, .5, .75, 1]).tolist())),
            "initial_goal_distance_below_4cm_count": int(np.sum(np.asarray(displacements) < .04)),
            "initial_goal_distance_at_least_12cm_count": int(np.sum(np.asarray(displacements) >= .12)),
            "checks": ["whole-episode seed-family disjointness", "all model and audit SHA256",
                       "grouped RGB and action identity", "model files contain only RGB and commanded actions",
                       "executed gain/action identity", "finite native states and RPM records",
                       "goal RGB/state are actual settled reachable endpoints", "workspace bounds"],
            "seconds": time.time() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/extensions/drone_v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/evidence/drone_data_validation.json"))
    args = parser.parse_args()
    result = audit(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
