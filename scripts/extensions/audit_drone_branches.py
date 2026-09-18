#!/usr/bin/env python3
"""Independent read-only audit of registered drone training branch supervision."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REGISTRATION = "b93d847a90077e8e8d496dc89fa93fa4f86f2dd3bf745fdc6c1968a52e09482e"
MAIN_SCHEMA = {
    "support_images": ((3, 128, 128, 3), "uint8"),
    "support_actions": ((2, 10), "float32"),
    "candidate_actions": ((32, 5, 10), "float32"),
    "future_images": ((32, 5, 128, 128, 3), "uint8"),
    "future_image_mask": ((32, 5), "bool"),
    "terminal_images": ((32, 128, 128, 3), "uint8"),
    "executed_lengths": ((32,), "int16"),
    "executed_mask": ((32, 25), "bool"),
    "valid_mask": ((32,), "bool"),
}
AUDIT_FIELDS = {"support_native_images", "support_native_states", "goal_state", "native_states",
                "gain_scaled_commands", "low_level_rpm"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def same(actual, expected, description):
    require(np.array_equal(actual, expected), description)


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def audit_context(output, metadata, source, commands, registration_hash, config):
    uid = source["trajectory_id"]
    require(metadata["trajectory_id"] == metadata["source_trajectory_id"] == uid, "Source trajectory identity changed")
    for key in ("split", "seed", "dynamics_id"):
        require(metadata[key] == source[key], f"Context metadata mismatch: {key}")
    require(metadata["split"] == "train", "Nontraining context")
    require(metadata["registration_sha256"] == registration_hash, "Context registration changed")
    for filekey, hashkey in (("file", "sha256"), ("audit_file", "audit_sha256"), ("audit_json", "audit_json_sha256")):
        require(sha(output / metadata[filekey]) == metadata[hashkey], f"Shard checksum mismatch {uid}/{filekey}")
    main, audit = npz(output / metadata["file"]), npz(output / metadata["audit_file"])
    privileged = json.loads((output / metadata["audit_json"]).read_text())
    require(set(main) == set(MAIN_SCHEMA), "Unexpected model fields or privileged leakage")
    require(set(audit) == AUDIT_FIELDS, "Unexpected audit fields")
    for key, (shape, dtype) in MAIN_SCHEMA.items():
        require(main[key].shape == shape and str(main[key].dtype) == dtype, f"Model shape/dtype mismatch: {key}")
        require(np.isfinite(main[key]).all(), f"Nonfinite model field: {key}")
    same(main["candidate_actions"], commands.reshape(32, 5, 10), "Future commands differ from registered candidates")
    require(np.abs(main["support_actions"]).max() <= 1, "Support actions outside command bounds")
    source_root = Path(config["data_root"])
    for filekey, hashkey in (("file", "sha256"), ("audit_file", "audit_sha256")):
        require(sha(source_root / source[filekey]) == source[hashkey], f"Original payload checksum changed: {uid}")
    original = npz(source_root / source["audit_file"])
    original_main = npz(source_root / source["file"])
    same(audit["support_native_images"], original["images"][:11], "Original native support RGB changed")
    same(audit["support_native_states"], original["simulator_states"][:11], "Original physical support changed")
    same(audit["goal_state"], original["goal_state"], "Audit scoring goal changed")
    same(main["support_images"], original["images"][[0, 5, 10]], "Model support RGB differs from native source")
    same(main["support_images"], original_main["images"][:3], "Grouped source support RGB differs")
    same(main["support_actions"], original["actions"][:10].reshape(2, 10), "Support commanded actions changed")
    same(main["support_actions"], original_main["actions"][:2], "Grouped source support commands differ")
    for sourcekey, array in (("support_rgb_sha256", original["images"][:11]),
                             ("support_commands_sha256", original["actions"][:10]),
                             ("support_states_sha256", original["simulator_states"][:11])):
        require(hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == source[sourcekey], "Registered support checksum changed")
    for key in ("trajectory_id", "seed", "dynamics_id"):
        require(privileged[key] == source[key], f"Audit metadata mismatch: {key}")
    gain = {0: 1.0, 1: .75, 2: 1.25}[source["dynamics_id"]]
    require(privileged["gain"] == gain, "Audit gain changed")
    require(privileged["support_replay_checks"] == {"branches": 32, "native_states_per_branch": 11,
            "native_rgb_per_branch": 11, "all_exact": True}, "Support replay receipt incomplete")
    require(audit["native_states"].shape == (32, 26, 20) and audit["native_states"].dtype == np.float64, "Native state shape/dtype")
    require(audit["gain_scaled_commands"].shape == (32, 25, 2) and audit["gain_scaled_commands"].dtype == np.float32, "Executed request shape/dtype")
    require(audit["low_level_rpm"].shape == (32, 25, 12, 4) and audit["low_level_rpm"].dtype == np.float32, "RPM shape/dtype")
    lengths = main["executed_lengths"]
    require(np.all((lengths >= 1) & (lengths <= 25)), "Invalid execution length")
    same(main["executed_mask"], np.arange(25)[None] < lengths[:, None], "Executed action prefix mask incorrect")
    image_mask = np.arange(5, 26, 5)[None] <= lengths[:, None]
    same(main["future_image_mask"], image_mask, "Future frame availability differs from executed boundaries")
    require(np.all(main["future_images"][~image_mask] == 0), "Missing RGB target was populated")
    require(len(privileged["outcomes"]) == 32, "Missing or extra branch outcomes")
    reason_counts, invalid = Counter(), []
    max_metric_error = 0.
    for index, row in enumerate(privileged["outcomes"]):
        length = int(lengths[index])
        require(row["candidate_index"] == index and row["executed_length"] == length, "Candidate audit index/count changed")
        states = audit["native_states"][index]
        same(states[0], original["simulator_states"][10], "Candidate did not branch from actual common physical support")
        for array, limit in ((states, length + 1), (audit["gain_scaled_commands"][index], length),
                             (audit["low_level_rpm"][index], length)):
            require(np.isfinite(array[:limit]).all(), "Nonfinite executed physical prefix")
            require(np.isnan(array[limit:]).all(), "Unexecuted physical suffix fabricated or padding changed")
        same(audit["gain_scaled_commands"][index, :length], (commands[index, :length] * gain).astype(np.float32),
             "Executed gain-scaled request is not the candidate's exact prefix")
        same(audit["low_level_rpm"][index, :length, -1], states[1:length + 1, 16:20].astype(np.float32),
             "Recorded final control RPM differs from simulator state")
        metrics = row["metrics_per_native_step"]
        require(len(metrics) == length, "Native metric count differs from execution")
        failure = False
        for native_index, (state, metric) in enumerate(zip(states[1:length + 1], metrics)):
            distance = float(np.linalg.norm(state[:2] - audit["goal_state"][:2]))
            speed = float(np.linalg.norm(state[10:13]))
            altitude = float(abs(state[2] - config["simulator_config"]["altitude_m"]))
            escape = bool(np.max(np.abs(state[:2])) > config["simulator_config"]["workspace_radius_m"] + .08)
            crash = bool(state[2] < .2 or np.max(np.abs(state[7:9])) > .8)
            success = bool(distance < config["simulator_config"]["position_tolerance_m"]
                           and speed < config["simulator_config"]["speed_tolerance_m_s"]
                           and altitude < config["simulator_config"]["altitude_tolerance_m"] and not escape and not crash)
            for key, calculated in (("goal_distance_m", distance), ("speed_m_s", speed), ("altitude_error_m", altitude)):
                error = abs(metric[key] - calculated)
                max_metric_error = max(max_metric_error, error)
                require(error <= 1e-12, f"Physical metric recomputation failed: {key}")
            for key, calculated in (("workspace_escape", escape), ("crash", crash), ("success", success)):
                require(metric[key] == calculated, f"Physical flag recomputation failed: {key}")
            if crash or escape:
                require(native_index == length - 1, "Branch continued after native failure")
                failure = True
        expected_valid = length == 25 and not failure
        require(bool(main["valid_mask"][index]) == expected_valid == row["valid"], "Branch validity inconsistent with native safety")
        if failure:
            last = metrics[-1]
            expected_reason = "crash_and_workspace_escape" if last["crash"] and last["workspace_escape"] else (
                "crash" if last["crash"] else "workspace_escape")
        else:
            expected_reason = "full_horizon" if length == 25 else "unexpected_early_budget"
        require(row["reason"] == expected_reason, "Stop reason inconsistent with native outcome")
        reason_counts[expected_reason] += 1
        if not expected_valid:
            invalid.append({"candidate_index": index, "reason": row["reason"], "executed_length": length})
        if length % 5 == 0:
            same(main["terminal_images"][index], main["future_images"][index, length // 5 - 1], "Terminal/boundary RGB mismatch")
    require(metadata["candidate_count"] == 32, "Metadata candidate count")
    require(metadata["valid_candidates"] == int(main["valid_mask"].sum()), "Metadata valid count")
    require(metadata["future_native_calls"] == int(lengths.sum()), "Metadata action count")
    return {"trajectory_id": uid, "seed": source["seed"], "dynamics_id": source["dynamics_id"],
            "branches": 32, "valid": int(main["valid_mask"].sum()), "invalid": invalid,
            "future_native_calls": int(lengths.sum()), "future_rgb_targets": int(image_mask.sum()),
            "stop_reasons": dict(reason_counts), "max_recomputed_metric_error": max_metric_error,
            "collection_seconds": metadata["collection_seconds"]}


def audit(output, partial=False):
    start = time.perf_counter()
    output = Path(output)
    registration = json.loads((output / "registration.json").read_text())
    registration_hash = sha(output / "registration.json")
    require(registration_hash == EXPECTED_REGISTRATION, "Registration differs from the frozen pre-collection record")
    for relative, digest in registration["source_hashes"].items():
        require(sha(ROOT / relative) == digest, f"Frozen source changed: {relative}")
    source_root = Path(registration["data_root"])
    require(sha(source_root / "manifest.json") == registration["data_manifest_sha256"], "Source manifest changed")
    original = json.loads((source_root / "manifest.json").read_text())
    original_lookup = {row["trajectory_id"]: row for row in original["episodes"]}
    rows = registration["contexts"]
    expected_pairs = {(seed, gain) for seed in range(53000, 53032) for gain in range(3)}
    require(len(rows) == 96 and {(row["seed"], row["dynamics_id"]) for row in rows} == expected_pairs, "Training context selection changed")
    forbidden_seeds = {row["seed"] for row in original["episodes"] if row["split"] != "train"}
    require(not forbidden_seeds & {row["seed"] for row in rows}, "Cross-split seed leakage")
    require(sha(output / "candidates.npz") == registration["candidates_file_sha256"], "Candidate file changed")
    candidate_file = npz(output / "candidates.npz")
    commands = np.array(registration["candidate_commands"], np.float32)
    same(candidate_file["actions"], commands, "Candidate registry/file mismatch")
    same(candidate_file["grouped_actions"], commands.reshape(32, 5, 10), "Grouped candidate layout changed")
    require(hashlib.sha256(commands.tobytes()).hexdigest() == registration["candidate_commands_sha256"], "Candidate command bytes changed")
    require(np.isfinite(commands).all() and np.abs(commands).max() <= 1, "Candidate action bounds changed")
    require(len({row["trajectory_id"] for row in rows}) == 96, "Duplicated source trajectory")
    results, reason_counts = [], Counter()
    for source in rows:
        require(source["split"] == "train", "Registered nontraining source")
        original_row = original_lookup[source["trajectory_id"]]
        for key in ("seed", "dynamics_id", "split", "file", "sha256", "audit_file", "audit_sha256"):
            require(original_row[key] == source[key], f"Registered source identity differs: {key}")
        sidecar = output / "contexts" / (source["trajectory_id"] + ".json")
        if partial and not sidecar.exists():
            continue
        metadata = json.loads(sidecar.read_text())
        result = audit_context(output, metadata, source, commands, registration_hash, registration)
        results.append(result)
        reason_counts.update(result["stop_reasons"])
    manifest = None
    if not partial:
        manifest = json.loads((output / "manifest.json").read_text())
        require(manifest["status"] == "collected" and manifest["registration_sha256"] == registration_hash, "Final manifest identity/status")
        require(len(results) == 96 and manifest["context_count"] == 96 and manifest["branch_count"] == 3072, "Incomplete final dataset")
        require({row["trajectory_id"] for row in manifest["contexts"]} == {row["trajectory_id"] for row in rows}, "Final context IDs differ")
        for row in manifest["contexts"]:
            require(row == json.loads((output / "contexts" / (row["trajectory_id"] + ".json")).read_text()), "Manifest/sidecar metadata differ")
        require(manifest["valid_branches"] == sum(row["valid"] for row in results), "Manifest valid total differs")
        require(manifest["retained_unsafe_or_short_branches"] == sum(len(row["invalid"]) for row in results), "Manifest invalid total differs")
        for folder, suffix, expected in (("contexts", ".npz", 96), ("contexts", ".json", 96), ("audit", ".npz", 96), ("audit", ".json", 96)):
            require(len(list((output / folder).glob("*" + suffix))) == expected, f"Unexpected number of {folder}/{suffix} shards")
        require(not list(output.rglob("*.partial.*")), "Leftover partial artifacts")
    gains = {}
    for gain in range(3):
        selected = [row for row in results if row["dynamics_id"] == gain]
        gains[str(gain)] = {"contexts": len(selected), "branches": len(selected) * 32,
                           "valid": sum(row["valid"] for row in selected),
                           "invalid": sum(len(row["invalid"]) for row in selected)}
    return {"status": "partial_audit_passed" if partial else "complete_audit_passed", "dataset": str(output),
            "registration_sha256": registration_hash, "manifest_sha256": None if manifest is None else sha(output / "manifest.json"),
            "auditor_sha256": sha(Path(__file__)), "split": "train", "contexts": len(results),
            "branches": len(results) * 32, "valid_branches": sum(row["valid"] for row in results),
            "retained_unsafe_or_short_branches": sum(len(row["invalid"]) for row in results),
            "future_native_calls": sum(row["future_native_calls"] for row in results),
            "support_native_calls_replayed": len(results) * 32 * 10,
            "observed_future_rgb_targets": sum(row["future_rgb_targets"] for row in results),
            "stop_reasons": dict(reason_counts), "by_gain": gains, "all_hashes_verified": True,
            "all_48_frozen_source_hashes_verified": len(registration["source_hashes"]) == 48,
            "all_source_episode_and_audit_payloads_verified": True, "all_model_fields_allowlisted": True,
            "all_action_prefixes_and_availability_masks_verified": True,
            "all_recorded_native_metrics_independently_recomputed": True,
            "max_recomputed_metric_error": max((row["max_recomputed_metric_error"] for row in results), default=0.),
            "collection_seconds": None if manifest is None else manifest["collection_seconds"],
            "audit_seconds": time.perf_counter() - start, "context_results": results,
            "interpretation": "Training supervision data only; no model trained and no performance claim"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/extensions/drone_branches_v1")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/evidence/drone_branched_data_audit.json")
    args = parser.parse_args()
    result = audit(args.data, args.partial)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "context_results"}, indent=2))


if __name__ == "__main__":
    main()
