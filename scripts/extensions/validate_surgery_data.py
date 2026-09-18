#!/usr/bin/env python3
"""Audit the complete extension dataset without using any model outcomes."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/extensions/surgery_v1"))
    p.add_argument("--output", type=Path, default=Path("reports/surgery/dataset_validation.json"))
    args = p.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    checks = Counter()
    rows, failures, seed_splits = [], [], {}
    pairs = defaultdict(dict)
    for ep in manifest["episodes"]:
        try:
            family = seed_splits.setdefault(ep["seed"], ep["split"])
            assert family == ep["split"], "seed crosses split"
            path, audit_path = args.dataset / ep["file"], args.dataset / ep["audit_file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == ep["sha256"], "episode hash"
            assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == ep["audit_sha256"], "audit hash"
            with np.load(path, allow_pickle=False) as data, np.load(audit_path, allow_pickle=False) as audit:
                images, actions = data["images"], data["actions"]
                native, commands = audit["images_native"], audit["actions_commanded"]
                assert set(data.files) == {"images", "actions"}, "privileged fields in model file"
                assert images.shape == (41,128,128,3) and images.dtype == np.uint8, "RGB shape/type"
                assert actions.shape == (40,10) and actions.dtype == np.float32, "actions shape/type"
                assert np.array_equal(images, native[::5]), "grouped frame mismatch"
                assert np.array_equal(actions, commands.reshape(-1,10)), "grouped command mismatch"
                gain = manifest["dynamics"][str(ep["dynamics_id"])]["actuator_gain"]
                assert np.array_equal(audit["actions_executed"], np.clip(commands*gain,-1,1).astype(np.float32)), "gain contract"
                tissue, goals = audit["tissue_target_xyz_m"], audit["goal_xyz_m"]
                assert all(np.isfinite(audit[k]).all() for k in audit.files), "non-finite audit"
                distance = np.linalg.norm((tissue-goals)[:,[0,2]], axis=1)
                assert np.max(np.abs(distance-audit["distance_m"])) < 1e-12, "distance contract"
                assert np.array_equal(distance<=0.002,audit["success"]), "native success contract"
                assert np.any(images[0] != images[-1]), "no rendered endpoint motion"
                pairs[(ep["split"],ep["seed"])][ep["dynamics_id"]] = (commands.copy(),tissue[0].copy(),tissue[-1].copy())
                initial_distance = float(np.linalg.norm((tissue[-1]-tissue[0])[[0,2]]))
                rows.append({"trajectory_id": ep["trajectory_id"], "split": ep["split"],
                             "invalid_calls": int((~audit["valid_action"]).sum()),
                             "unstable_calls": int((~audit["stable_deformation"]).sum()),
                             "endpoint_separation_m": initial_distance,
                             "eligible_endpoint_goal": bool(initial_distance>=0.004 and audit["valid_action"].all() and audit["stable_deformation"].all())})
            checks["episodes_validated"] += 1
        except Exception as error:
            failures.append({"episode":ep["trajectory_id"],"error":repr(error)})
    gain_distances = []
    for key, domains in pairs.items():
        try:
            assert set(domains) == {0,1,2}, "missing matched gain"
            assert all(np.array_equal(domains[0][0],domains[d][0]) for d in (1,2)), "unpaired commands"
            assert all(np.array_equal(domains[0][1],domains[d][1]) for d in (1,2)), "different initial tissue state"
            gain_distances.extend(float(np.linalg.norm(domains[0][2]-domains[d][2]))*1000 for d in (1,2))
            checks["matched_gain_groups"] += 1
        except Exception as error:
            failures.append({"pair":str(key),"error":repr(error)})
    report = {"status":"passed" if not failures else "failed", "checks":dict(checks), "failures":failures,
              "episodes":len(manifest["episodes"]),"splits":dict(Counter(ep["split"] for ep in manifest["episodes"])),
              "native_calls":manifest["native_calls"], "invalid_action_calls":sum(r["invalid_calls"] for r in rows),
              "unstable_calls":sum(r["unstable_calls"] for r in rows),
              "eligible_goal_counts":dict(Counter(r["split"] for r in rows if r["eligible_endpoint_goal"])),
              "gain_endpoint_difference_mm":{"median":float(np.median(gain_distances)),"min":float(np.min(gain_distances)),"max":float(np.max(gain_distances))},
              "source_manifest_sha256":hashlib.sha256((args.dataset/"manifest.json").read_bytes()).hexdigest(),
              "per_episode":rows,
              "initialization_logs":"SOFA emitted topology/compliance warnings and errors; retained logs. This audit verifies finite data, action/metric contracts, replay separately, and actual gain response; it does not establish clinical fidelity."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="per_episode"},indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
