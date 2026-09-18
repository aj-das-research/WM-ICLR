"""Corruption checks for the independent read-only branch-data auditor."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("drone_branch_auditor", ROOT / "scripts/extensions/audit_drone_branches.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


@pytest.fixture
def copied_context(tmp_path):
    dataset = ROOT / "data/extensions/drone_branches_v1"
    if not (dataset / "registration.json").exists() or not (dataset / "contexts").exists():
        pytest.skip("Registered branch dataset is not available locally")
    registration = json.loads((dataset / "registration.json").read_text())
    lookup = {row["trajectory_id"]: row for row in registration["contexts"]}
    candidates = np.array(registration["candidate_commands"], np.float32)
    for file in sorted((dataset / "contexts").glob("*.json")):
        metadata = json.loads(file.read_text())
        main = auditor.npz(dataset / metadata["file"])
        # Exercise missing-boundary padding and an actual retained unsafe branch.
        if np.any(~main["future_image_mask"]):
            break
    else:
        pytest.skip("Needs a completed context with an early stopped branch")
    for key in ("file", "audit_file", "audit_json"):
        destination = tmp_path / metadata[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset / metadata[key], destination)
    return tmp_path, metadata, lookup[metadata["trajectory_id"]], candidates, registration


def run_audit(fixture):
    output, metadata, source, candidates, registration = fixture
    return auditor.audit_context(output, metadata, source, candidates, auditor.EXPECTED_REGISTRATION, registration)


def update_npz(fixture, file_key, mutate):
    output, metadata, *_ = fixture
    path = output / metadata[file_key]
    arrays = auditor.npz(path)
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    metadata["sha256" if file_key == "file" else "audit_sha256"] = auditor.sha(path)


def test_actual_context_passes_independent_audit(copied_context):
    result = run_audit(copied_context)
    assert result["branches"] == 32 and result["invalid"]
    assert result["max_recomputed_metric_error"] == 0.


@pytest.mark.parametrize("mutation,pattern", [
    ("hidden", "privileged leakage"),
    ("candidate", "registered candidates"),
    ("prefix", "prefix mask"),
    ("future_mask", "availability"),
    ("future_padding", "Missing RGB"),
    ("source_support", "native source"),
    ("validity", "validity inconsistent"),
])
def test_semantic_model_corruption_rejected_even_with_recomputed_file_hash(copied_context, mutation, pattern):
    def mutate(arrays):
        invalid = int(np.flatnonzero(~arrays["valid_mask"])[0])
        absent = np.argwhere(~arrays["future_image_mask"])[0]
        if mutation == "hidden":
            arrays["physical_state"] = np.zeros(20)
        elif mutation == "candidate":
            arrays["candidate_actions"][0, 0, 0] = .01
        elif mutation == "prefix":
            arrays["executed_mask"][invalid, -1] = True
        elif mutation == "future_mask":
            arrays["future_image_mask"][tuple(absent)] = True
        elif mutation == "future_padding":
            arrays["future_images"][tuple(absent)] = 99
        elif mutation == "source_support":
            arrays["support_images"][0, 0, 0, 0] ^= 1
        elif mutation == "validity":
            arrays["valid_mask"][invalid] = True
    update_npz(copied_context, "file", mutate)
    with pytest.raises(ValueError, match=pattern):
        run_audit(copied_context)


@pytest.mark.parametrize("mutation,pattern", [
    ("common_state", "actual common physical support"),
    ("gain_prefix", "candidate's exact prefix"),
    ("rpm", "final control RPM"),
])
def test_semantic_physical_corruption_rejected_even_with_recomputed_file_hash(copied_context, mutation, pattern):
    def mutate(arrays):
        if mutation == "common_state":
            arrays["native_states"][0, 0, 0] += .000001
        elif mutation == "gain_prefix":
            arrays["gain_scaled_commands"][0, 0, 0] += .01
        else:
            arrays["low_level_rpm"][0, 0, -1, 0] += 10
    update_npz(copied_context, "audit_file", mutate)
    with pytest.raises(ValueError, match=pattern):
        run_audit(copied_context)


def test_native_success_speed_requirement_is_independently_recomputed(copied_context):
    output, metadata, *_ = copied_context
    path = output / metadata["audit_json"]
    payload = json.loads(path.read_text())
    payload["outcomes"][0]["metrics_per_native_step"][0]["speed_m_s"] += .01
    path.write_text(json.dumps(payload))
    metadata["audit_json_sha256"] = auditor.sha(path)
    with pytest.raises(ValueError, match="speed_m_s"):
        run_audit(copied_context)


def test_unrebound_payload_corruption_fails_checksum(copied_context):
    output, metadata, *_ = copied_context
    (output / metadata["file"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_audit(copied_context)
