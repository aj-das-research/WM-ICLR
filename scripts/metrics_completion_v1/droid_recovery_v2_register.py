#!/usr/bin/env python3
"""Preserve all v1 evidence and register the full 34-run operational recovery."""
import argparse
import copy
from pathlib import Path

import droid_common as original
from droid_recovery_v2_common import (ROOT, REPORT, REG, POLICY, atomic_json, local, now, read,
                                     relative, require, sha, verify_registration)


def register():
    require(not REG.exists(), "Recovery registration already exists")
    old = original.verify_registration()
    sources = dict(old["source_dependencies"])
    sources[relative(original.REG)] = sha(original.REG)
    paths = sorted(Path(__file__).parent.glob("droid_recovery_v2_*"))
    paths += [REPORT / "protocol.md"]
    prior = original.REPORT
    paths += sorted((prior / "evaluations").glob("*.json"))
    paths += sorted((prior / "evaluations").glob("*.npz"))
    paths += sorted(prior.glob("slurm-203085_*.log")) + sorted(prior.glob("slurm-203087_*.log"))
    paths += [prior / "finalizer-203089.log"]
    for p in paths:
        if p.is_file():
            sources[relative(p)] = sha(p)
    passed, missing = [], []
    for row in old["runs"]:
        p = prior / "evaluations" / (row["name"] + ".json")
        if p.exists():
            require(read(p).get("status") == "passed", "Non-passing prior marker")
            passed.append(row["name"])
        else:
            missing.append(row["name"])
    require(len(passed) == 22 and len(missing) == 12 and not (prior / "completion.json").exists()
            and not (prior / "finalization.json").exists(), "Original failure/completion state differs")
    require("aggregation_performed\": false" in (prior / "finalizer-203089.log").read_text(),
            "Original finalizer did not explicitly retain incomplete status")
    require(len(list(prior.glob("slurm-203085_*.log"))) == 21 and len(list(prior.glob("slurm-203087_*.log"))) == 13,
            "Incomplete original job-log preservation")
    recovery = {"original_registration": relative(original.REG), "original_registration_sha256": sha(original.REG),
        "original_passed_names": passed, "original_failed_names": missing,
        "all_original_outputs_and_job_logs_bound": True, "original_aggregate_available": False,
        "reuse_any_original_metric_row": False,
        "change": POLICY["roundtrip_contract"],
        "diagnostic_timing": "operational revision after v1 exact-GPU-reload failures; before any v2 scoring"}
    result = {**copy.deepcopy(old), "schema": "droid_complementary_metrics_recovery_v2_registration",
        "created_utc": now(), "policy": POLICY, "source_dependencies": sources, "recovery": recovery,
        "new_model_inference_performed_before_registration": False}
    for p, expected in sources.items():
        require(sha(local(p)) == expected, "Source changed during recovery registration: " + p)
    atomic_json(result, REG)
    verify_registration()
    print("Recovery registered " + str(REG) + " SHA256=" + sha(REG), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "verify"))
    args = parser.parse_args()
    if args.command == "register":
        register()
    else:
        verify_registration(); print("Recovery source closure verified")
