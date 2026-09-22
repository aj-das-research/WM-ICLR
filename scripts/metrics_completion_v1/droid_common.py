"""Post-hoc DROID metric completion; immutable inputs and fail-closed ledgers."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
REPORT = ROOT / "reports/metrics_completion_v1/droid"
REG = REPORT / "registration.json"
METRICS = ("standardized_mse", "standardized_mae", "raw_dinov2_l1", "feature_cosine_distance")
POLICY = {
    "scope": "posthoc_secondary_original_validation_development_only",
    "target": "raw frozen DINOv2-small native 4x4 features, 6144 coordinates",
    "observed_frames": 3, "query_horizons": list(range(1, 11)), "window_stride": 5,
    "expected_windows": 1631, "expected_episodes": 141, "expected_sessions": 59,
    "learned_runs": 33, "internal_runs": 21, "external_runs": 12,
    "persistence_evaluations": 1, "training_seeds": [0, 1, 2],
    "aggregation": "mean windows within episode; equal episodes; equal training seeds",
    "prediction_precision": "float32; autocast disabled; CUDA matmul/cudnn TF32 disabled",
    "metric_precision": "FP32 errors; FP64 stored values and aggregation",
    "formulas": {
        "standardized_mse": "mean_d(((prediction_d-target_d)/train_std_d)^2)",
        "standardized_mae": "mean_d(abs(prediction_d-target_d)/train_std_d)",
        "raw_dinov2_l1": "mean_d(abs(prediction_d-target_d))",
        "feature_cosine_distance": "1-clamp(dot(p/max(norm(p),1e-8),t/max(norm(t),1e-8)),-1,1)",
    },
    "prior_mse_tolerance": {"rtol": 2e-5, "atol": 2e-6},
    "arithmetic_tolerance": {"rtol": 1e-12, "atol": 1e-12},
    "bootstrap": {"draws": 10000, "seed": 173,
        "units": "recording sessions crossed with training seeds; paired across methods",
        "interval": "95% percentile interval for method-minus-reference absolute error",
        "references": ["autoregressive", "unbounded_transport"],
        "multiplicity": "secondary exploratory intervals, unadjusted; correlated horizons and metrics"},
    "restrictions": "No training, reselection, test access, RGB decoding, control-success or SOTA inference.",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def local(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    require(p.resolve().is_relative_to(ROOT.resolve()), "Source escapes workspace")
    return p


def relative(path):
    return str(Path(path).absolute().relative_to(ROOT))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, local(path))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def atomic_json(value, path, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), "Refusing to replace existing artifact: " + str(path))
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)  # Exclusive creation, including concurrent finalizers.
    finally:
        Path(tmp).unlink(missing_ok=True)


def atomic_npz(arrays, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        with np.load(tmp, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays), "NPZ round-trip keys differ")
            for key in arrays:
                np.testing.assert_array_equal(saved[key], arrays[key])
        os.link(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def population(manifest, original_manifest):
    helper = module("scripts/real_video_spatial/validate_ledger.py", "droid_completion_population")
    episodes, windows = helper.expected_population(manifest, original_manifest)
    require(len(windows) == POLICY["expected_windows"] and len(episodes) == POLICY["expected_episodes"],
            "Registered validation population differs")
    sessions = {r["session_id"] for r in episodes.values()}
    require(len(sessions) == POLICY["expected_sessions"], "Session population differs")
    return sorted(windows)


def validate_arrays(arrays, expected_keys):
    require(set(arrays) == {"episode_id", "session_id", "window_start", *METRICS}, "Unexpected metric ledger keys")
    n = len(expected_keys)
    require(all(arrays[k].shape == (n,) for k in ("episode_id", "session_id", "window_start")), "Invalid identifier shapes")
    require(np.issubdtype(arrays["window_start"].dtype, np.integer), "Window starts must be integers")
    keys = list(zip(arrays["episode_id"].tolist(), arrays["session_id"].tolist(), arrays["window_start"].tolist()))
    require(keys == list(map(tuple, expected_keys)) and len(set(keys)) == n, "Missing, duplicate or reordered windows")
    for key in METRICS:
        values = arrays[key]
        require(values.shape == (n, 10) and np.issubdtype(values.dtype, np.floating), "Wrong metric shape/type: " + key)
        require(np.isfinite(values).all() and (values >= 0).all(), "Nonfinite/negative metric: " + key)
        if key == "feature_cosine_distance":
            require((values <= 2).all(), "Cosine distance outside [0,2]")
    by_episode = defaultdict(list)
    for index, (eid, session, start) in enumerate(keys):
        by_episode[(eid, session)].append(index)
    episodes = []
    for (eid, session), indices in sorted(by_episode.items()):
        episodes.append({"episode_id": eid, "session_id": session, "windows": len(indices),
            **{key: arrays[key][indices].mean(0, dtype=np.float64).tolist() for key in METRICS}})
    summary = {key: np.mean([e[key] for e in episodes], axis=0, dtype=np.float64).tolist() for key in METRICS}
    return episodes, summary


def prior_mse_check(arrays, episodes, summary, prior, persistence=False):
    field = "native_persistence_mse" if persistence else "native_mse"
    old = {(w["episode_id"], w["session_id"], w["window_start"]): w[field] for w in prior["windows"]}
    keys = list(zip(arrays["episode_id"].tolist(), arrays["session_id"].tolist(), arrays["window_start"].tolist()))
    require(len(old) == len(prior["windows"]) and set(old) == set(keys), "Prior MSE window population differs")
    expected = np.asarray([old[k] for k in keys], dtype=np.float64)
    np.testing.assert_allclose(arrays["standardized_mse"], expected, **POLICY["prior_mse_tolerance"])
    old_episodes = {e["episode_id"]: e for e in prior["episodes"]}
    require(len(old_episodes) == len(episodes), "Prior episode population differs")
    for e in episodes:
        ref = old_episodes[e["episode_id"]]
        require(e["session_id"] == ref["session_id"] and e["windows"] == ref["windows"], "Prior episode denominator differs")
        np.testing.assert_allclose(e["standardized_mse"], ref[field], **POLICY["prior_mse_tolerance"])
    np.testing.assert_allclose(summary["standardized_mse"], prior["summary"][field], **POLICY["prior_mse_tolerance"])
    return {"status": "passed", "windows_checked": len(keys), "endpoints_per_window": 10,
        "max_window_absolute_error": float(np.abs(arrays["standardized_mse"] - expected).max()),
        "tolerance": POLICY["prior_mse_tolerance"], "prior_field": field}


def verify_registration():
    reg = read(REG)
    require(reg.get("status") == "registered" and reg.get("policy") == POLICY, "Registration policy differs")
    rows = reg["runs"]
    require(len(rows) == 34 and [r["index"] for r in rows] == list(range(34)), "Incomplete registered run roster")
    require(len({r["name"] for r in rows}) == 34 and sum(r["family"] == "persistence" for r in rows) == 1,
            "Duplicate run or persistence roster")
    require(len([r for r in rows if r["family"] in ("internal", "component")]) == 21,
            "Internal roster incomplete")
    expected_modes = {
        "autoregressive", "anchored_additive", "transport", "context_off", "action_free",
        "bounded_additive", "unbounded_transport", "official_one_step_shifted", "matched_recursive_h10",
        "official_raw_one_step", "official_raw_recursive_h10",
    }
    learned = [r for r in rows if r["family"] != "persistence"]
    require({(r["mode"], r["seed"]) for r in learned} == {(m, s) for m in expected_modes for s in (0, 1, 2)},
            "Named method/seed roster differs")
    for path, digest in reg["source_dependencies"].items():
        require(sha(local(path)) == digest, "Registered dependency changed: " + path)
    return reg


def validate_output(row, reg):
    path = REPORT / "evaluations" / (row["name"] + ".json")
    value = read(path)
    require(value.get("status") == "passed" and value.get("name") == row["name"]
            and value.get("mode") == row["mode"] and value.get("seed") == row["seed"]
            and value.get("family") == row["family"] and value.get("scope") == POLICY["scope"]
            and value.get("selected_epoch") == row.get("selected_epoch")
            and value.get("registration_sha256") == sha(REG)
            and value.get("selected_checkpoint_sha256") == row.get("checkpoint_sha256"), "Result header/checkpoint differs")
    ledger = REPORT / "evaluations" / (row["name"] + ".npz")
    require(value.get("ledger_path") == relative(ledger) and value.get("ledger_sha256") == sha(ledger), "Result ledger binding differs")
    with np.load(ledger, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    episodes, summary = validate_arrays(arrays, reg["window_keys"])
    require(value["episodes"] == episodes and value["summary"] == summary, "Saved aggregates differ from primitive errors")
    parity = prior_mse_check(arrays, episodes, summary, read(local(row["prior_ledger"])), row["family"] == "persistence")
    require(value.get("prior_mse_parity") == parity and value.get("roundtrip_checks", {}).get("npz") == "exact",
            "MSE parity/round-trip receipt differs")
    if row["family"] != "persistence":
        require(value.get("roundtrip_checks", {}).get("checkpoint_prediction") == "exact",
                "Checkpoint reload round-trip absent")
    return value, arrays
