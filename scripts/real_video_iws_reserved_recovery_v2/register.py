#!/usr/bin/env python3
"""Freeze a reviewed numerical correction, explicitly after initial data access."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws_reserved import protocol as original
from shiftwm.real_video_iws_reserved_recovery import protocol


def prepare(diagnostic_path):
    baseline = original.checked_registration(ROOT)
    proof_path = Path(diagnostic_path).resolve()
    if not proof_path.is_relative_to(ROOT):
        raise ValueError("Diagnostic receipt must be inside the workspace")
    proof = original.read_json(proof_path)
    if (proof.get("schema") != "reserved_prefix_input_only_diagnostic_20260920_v2"
            or proof.get("status") != "completed" or proof.get("threads") != 8
            or proof.get("batch_size") != 64
            or proof.get("model_load_context") != "outside_inference_mode"
            or proof.get("loader") != "frozen_evaluator_load_selected"
            or proof.get("future_targets_scored_or_materialized") is not False
            or proof.get("feature_errors_called") is not False
            or proof.get("tolerance_changed") is not False
            or proof.get("scientific_evaluation_or_checkpoint_files_modified") is not False
            or proof.get("registration_sha256") != original.sha(ROOT / original.REGISTRATION_PATH)):
        raise ValueError("Complete input-only diagnosis with unchanged science is required")
    names = {f"bimanual_box_autoregressive_s{s}" for s in (1, 2)}
    models = proof.get("models", [])
    if (len(models) != 2 or {m["run"] for m in models} != names
            or any(m["state_dict_before_sha256"] != m["state_dict_after_sha256"]
                   or m.get("state_keys_unchanged") is not True for m in models)):
        raise ValueError("Both failed checkpoints must retain their exact state")
    selected = {r["name"]: r["checkpoint_sha256"] for r in baseline["runs"]}
    if any(model["checkpoint_sha256"] != selected[model["run"]] for model in models):
        raise ValueError("Diagnostic checkpoint differs from the frozen selection")
    loader_path = ROOT / "scripts/real_video_iws_reserved_v1/evaluate.py"
    if proof["source_sha256"].get("original_evaluator_loader") != original.sha(loader_path):
        raise ValueError("Diagnostic did not use the exact frozen evaluation loader")
    rows = proof.get("results", [])
    if len(rows) != 10:
        raise ValueError("Require training fixture and all four reserved input batches for both models")
    for name in names:
        matched = [row for row in rows if row["run"] == name]
        if len(matched) != 5 or sorted(row["first_handle"] for row in matched if "first_handle" in row) != [0, 64, 128, 192]:
            raise ValueError("Incomplete fixed diagnostic input population")
    for row in rows:
        if set(row["prefixes"]) != {"15", "30", "45"}:
            raise ValueError("Every registered prefix must be diagnosed")
        if any(prefix["rowwise_all_prefix_outputs"]["bitwise_equal"] is not True
               or prefix["rowwise_gru"]["bitwise_equal"] is not True for prefix in row["prefixes"].values()):
            raise ValueError("Candidate does not establish exact numerical prefix consistency")
    dependencies = dict(baseline["dependencies"])
    def bind(path):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Recovery source leaves workspace")
        name = str(path.relative_to(ROOT))
        if any((name + "/").startswith(p) for p in original.reserved_payload_prefixes(baseline)):
            raise ValueError("Recovery registration must not open reserved raw/cache payloads")
        dependencies[name] = original.sha(path)
    fixed = [ROOT / original.REGISTRATION_PATH, ROOT / original.REVIEW_PATH, proof_path,
             ROOT / "reports/real_video_iws_reserved_v1/execution_incident.json",
             ROOT / "reports/real_video_iws_reserved_diagnostics_20260920/source_review_v2.json",
             ROOT / "scripts/real_video_iws_reserved_diagnostics_20260920/run_v2.slurm",
             ROOT / "scripts/real_video_iws_reserved_diagnostics_20260920/test_probe_v2.py",
             ROOT / "artifacts/releases/iws_single_observation_local_v1/manifest.json",
             ROOT / "artifacts/releases/iws_single_observation_local_v1/fixtures/bimanual_box.npz",
             ROOT / "scripts/real_video_iws_reserved_diagnostics_20260920/probe_v2.py"]
    for path in fixed:
        bind(path)
    for folder in ("src/shiftwm/real_video_iws_reserved_recovery", "scripts/real_video_iws_reserved_recovery_v2"):
        for path in sorted((ROOT / folder).glob("*")):
            if path.is_file() and path.suffix in (".py", ".slurm"):
                bind(path)
    for path in sorted((ROOT / "tests").glob("test_iws_reserved_recovery*.py")):
        bind(path)
    for key, path in (("probe", fixed[-1]), ("prefix_backend", ROOT / "src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py")):
        if proof["source_sha256"][key] != original.sha(path):
            raise ValueError("Runtime source differs from tested diagnostic: " + key)
    recovery = {"command_gru_dispatch": "one_native_row_per_call",
                "same_weights": True, "same_normalization": True, "same_tolerance": True,
                "same_metrics": True, "same_examples": True, "same_comparators": True,
                "all36_rerun": True, "original_results_retained": True,
                "future_targets_used_to_choose_backend": False, "accuracy_used_to_choose_backend": False,
                "initial_reserved_access_precedes_this_revision": True}
    value = {"schema": protocol.SCHEMA, "status": protocol.STATE, "created_utc": original.now(),
             "scope": "Common numerical backend recovery of the original fixed reserved evaluation",
             **{key: copy.deepcopy(baseline[key]) for key in ("runs", "tasks", "evaluation", "extraction", "qualitative_selection", "expected_runs")},
             "cache_registration_path": str(original.REGISTRATION_PATH),
             "original_registration_sha256": original.sha(ROOT / original.REGISTRATION_PATH),
             "diagnostic_path": str(proof_path.relative_to(ROOT)), "diagnostic_sha256": original.sha(proof_path),
             "numerical_recovery": recovery, "dependencies": dict(sorted(dependencies.items()))}
    return protocol.validate_recovery(value, baseline)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("candidate", "freeze", "check"))
    parser.add_argument("--diagnostic", type=Path)
    args = parser.parse_args()
    candidate = ROOT / "reports/real_video_iws_reserved_recovery_v2/registration_candidate.json"
    if args.command == "check":
        value = protocol.checked_registration(ROOT)
        print(json.dumps({"status": "recovery_review_passed", "runs": len(value["runs"])}))
        return
    if args.command == "candidate":
        if args.diagnostic is None:
            raise ValueError("A completed, input-only diagnostic is required")
        value = prepare(args.diagnostic)
        if candidate.exists():
            value["created_utc"] = original.read_json(candidate)["created_utc"]
        original.immutable_json(value, candidate)
        print(json.dumps({"status": "candidate_created", "sha256": original.sha(candidate)}))
        return
    expected = original.read_json(candidate)
    actual = prepare(ROOT / expected["diagnostic_path"])
    actual["created_utc"] = expected["created_utc"]
    if actual != expected:
        raise ValueError("Recovery candidate changed before freeze")
    original.immutable_json(expected, ROOT / protocol.REGISTRATION_PATH)
    protocol.checked_registration(ROOT, require_review=False)
    print(json.dumps({"status": "frozen_awaiting_independent_review", "sha256": original.sha(ROOT / protocol.REGISTRATION_PATH)}))


if __name__ == "__main__":
    main()
