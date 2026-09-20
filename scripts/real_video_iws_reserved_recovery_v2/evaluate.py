#!/usr/bin/env python3
"""Frozen-roster CPU evaluation of exact reserved IWS handles.

Payload access is possible only through the reviewed reserved cache. Targets
remain in the scorer; predict receives one observation and commands only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
REGISTRATION = Path("configs/real_video_iws_reserved_recovery_v2/registration.json")
REPORT = Path("reports/real_video_iws_reserved_recovery_v2")
TASKS = {"pusht": 4, "bimanual_box": 14, "bimanual_rope": 8}
MODES = ("autoregressive", "anchored_additive", "bounded_spatial_mix", "unbounded_spatial_mix")
PREFIXES = (15, 30, 45)
SCOPE = "reserved_upstream_validation"
SCHEMA = "shiftwm_iws_reserved_recovery_evaluation_v2"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


# Reuse the exact existing FP32 feature metric and atomic NPZ writer.
frozen_metrics = module(ROOT / "scripts/real_video_iws/evaluate.py", "_reserved_frozen_metrics")
feature_errors = frozen_metrics.feature_errors
METRICS = tuple(frozen_metrics.METRICS)
atomic_npz = frozen_metrics.atomic_npz


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def local(root, path):
    root = Path(root).resolve()
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Reserved evidence path escapes the workspace")
    return path


def checked_registration(root, registration_path=None):
    from shiftwm.real_video_iws_reserved_recovery.protocol import checked_registration as check
    return check(root=root, registration_path=registration_path)


def check_grid(registry):
    rows = registry["runs"]
    expected = {(t, m, s) for t in TASKS for m in MODES for s in range(3)}
    if len(rows) != 36 or {(r["task"], r["mode"], r["seed"]) for r in rows} != expected or len({r["name"] for r in rows}) != 36:
        raise ValueError("Reserved study requires all36 distinct selected predictors")
    expected_evaluation = {"device": "cpu", "dtype": "float32", "threads": 8, "batch_size": 64,
                           "horizon": 60, "prefix_horizons": [15, 30, 45], "bootstrap_draws": 10000, "bootstrap_seed": 173}
    if any(registry["evaluation"].get(k) != v for k, v in expected_evaluation.items()):
        raise ValueError("Unexpected registered CPU evaluation protocol")


def population_contract(task, handles, episode_ids):
    if task not in TASKS or len(handles) != 200 or episode_ids != sorted(set(episode_ids)) or len(episode_ids) != 10:
        raise ValueError("Expected200 exact handles and10 distinct trajectories per task")
    pairs = []
    for h in handles:
        if set(h) != {"episode_id", "start"} or h["episode_id"] not in episode_ids or type(h["start"]) is not int or h["start"] < 0:
            raise ValueError("Invalid reserved handle")
        pairs.append((h["episode_id"], h["start"]))
    if len(set(pairs)) != 200 or {p[0] for p in pairs} != set(episode_ids):
        raise ValueError("Missing or duplicate reserved handles")
    return {"task": task, "scope": SCOPE, "total_handles": 200, "eligible_trajectories": 10,
            "handles": handles, "episode_ids": episode_ids,
            "records": [{"episode_id": eid, "episode_index": index, "handles": sum(h["episode_id"] == eid for h in handles)}
                        for index, eid in enumerate(episode_ids)],
            "command_rows": 60, "target_offsets": list(range(1, 60)), "strict_eligibility": "start+60<frames"}


def metric_keys():
    return (*METRICS, *("persistence_" + m for m in METRICS), *("prefix_" + m for m in METRICS))


def aggregate_arrays(arrays, population):
    """Check original order, then independently expose both population weights."""
    if set(arrays) != {"episode_index", "window_start", *metric_keys()}:
        raise ValueError("Unexpected primitive metric ledger fields")
    n = population["total_handles"]
    for key in ("episode_index", "window_start"):
        if arrays[key].dtype != np.int64 or arrays[key].shape != (n,):
            raise ValueError("Invalid primitive integer handle indices")
    ids = population["episode_ids"]
    expected = [(ids.index(h["episode_id"]), h["start"]) for h in population["handles"]]
    if list(zip(arrays["episode_index"].tolist(), arrays["window_start"].tolist())) != expected:
        raise ValueError("Missing, duplicated or reordered reserved handles")
    for key in metric_keys():
        width = 3 if key.startswith("prefix_") else 59
        value = arrays[key]
        if value.dtype != np.float64 or value.shape != (n, width) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("Invalid primitive metric: " + key)
        if key.endswith("feature_cosine_distance") and (value > 2).any():
            raise ValueError("Cosine distance exceeds its fixed range")
    episodes = []
    for row in population["records"]:
        select = arrays["episode_index"] == row["episode_index"]
        if int(select.sum()) != row["handles"] or not select.any():
            raise ValueError("Trajectory denominator differs")
        episodes.append({"episode_id": row["episode_id"], "handles": row["handles"],
                         **{key + "_by_offset": arrays[key][select].mean(0).tolist() for key in metric_keys()}})
    equal_handle = {key: arrays[key].mean(0).tolist() for key in metric_keys()}
    equal_trajectory = {key: np.array([e[key + "_by_offset"] for e in episodes]).mean(0).tolist() for key in metric_keys()}
    return {"episodes": episodes, "equal_handle": equal_handle, "equal_trajectory": equal_trajectory}


def make_batch(cache, handles, task, episode_ids, episode_store):
    initial, commands, targets, indices, starts = [], [], [], [], []
    for handle in handles:
        eid, start = handle["episode_id"], handle["start"]
        if eid not in episode_store:
            episode = cache.episode(eid)
            z, a = episode["features"], episode["commands"]
            n = len(z)
            if (z.shape != (n, 6144) or a.shape != (n, TASKS[task]) or z.dtype != np.float32 or a.dtype != np.float32
                    or not np.isfinite(z).all() or not np.isfinite(a).all()
                    or episode["frame_indices"].dtype != np.int64 or episode["command_row_indices"].dtype != np.int64
                    or not np.array_equal(episode["frame_indices"], np.arange(n, dtype=np.int64))
                    or not np.array_equal(episode["command_row_indices"], np.arange(n, dtype=np.int64))):
                raise ValueError("Reserved native indexing/shape differs")
            episode_store[eid] = episode
        episode = episode_store[eid]
        z, a = episode["features"], episode["commands"]
        n = len(z)
        if start + 60 >= n:
            raise ValueError("Reserved strict handle eligibility differs")
        initial.append(z[start]); commands.append(a[start:start + 60]); targets.append(z[start + 1:start + 60])
        indices.append(episode_ids.index(eid)); starts.append(start)
    return {"initial_features": torch.from_numpy(np.stack(initial)), "commands": torch.from_numpy(np.stack(commands)),
            "targets": torch.from_numpy(np.stack(targets)), "episode_index": np.array(indices, dtype=np.int64),
            "window_start": np.array(starts, dtype=np.int64)}


def score_batch(model, batch, first_handle):
    initial, commands, targets = (batch[k] for k in ("initial_features", "commands", "targets"))
    if initial.device.type != "cpu" or initial.dtype != torch.float32 or commands.dtype != torch.float32:
        raise ValueError("Reserved inference requires CPU FP32")
    prediction = model.predict(initial, commands)
    if prediction.shape != targets.shape or prediction.dtype != torch.float32:
        raise ValueError("Invalid full-H60 prediction")
    result = {key: value.double().numpy() for key, value in feature_errors(prediction, targets, model.feature_std).items()}
    baseline = feature_errors(initial[:, None].expand_as(targets), targets, model.feature_std)
    result.update({"persistence_" + key: value.double().numpy() for key, value in baseline.items()})
    invocations = []
    prefix_values = {key: [] for key in METRICS}
    common = {"first_handle": first_handle, "count": len(initial), "initial_features_sha256": array_sha(initial.numpy())}
    full_sha = array_sha(prediction.numpy())
    invocations.append({**common, "command_rows": 60, "prediction_shape": list(prediction.shape),
                        "commands_sha256": array_sha(commands.numpy()), "prediction_sha256": full_sha})
    for horizon in PREFIXES:
        prefix = model.predict(initial, commands[:, :horizon])
        if prefix.shape != (len(initial), horizon - 1, initial.shape[-1]) or prefix.dtype != torch.float32 or not torch.isfinite(prefix).all():
            raise ValueError("Invalid separate-prefix forecast")
        endpoint = prefix[:, -1:]
        comparison = prediction[:, horizon - 2:horizon - 1]
        difference = float((endpoint - comparison).abs().max())
        tolerance_ratio = float(((endpoint - comparison).abs() / (2e-5 + 1e-5 * comparison.abs())).max())
        passed = torch.allclose(endpoint, comparison, rtol=1e-5, atol=2e-5)
        if not passed:
            raise ValueError("CPU causal prefix consistency exceeds registered FP32 tolerance")
        for key, value in feature_errors(endpoint, targets[:, horizon - 2:horizon - 1], model.feature_std).items():
            prefix_values[key].append(value[:, 0].double().numpy())
        invocations.append({**common, "command_rows": horizon, "prediction_shape": list(prefix.shape),
                            "commands_sha256": array_sha(commands[:, :horizon].numpy()), "prediction_sha256": array_sha(prefix.numpy()),
                            "full_prediction_sha256": full_sha, "full_output_index": horizon - 2,
                            "target_native_offset": horizon - 1, "maximum_absolute_difference": difference,
                            "maximum_reference_absolute_value": float(comparison.abs().max()),
                            "maximum_tolerance_ratio": tolerance_ratio,
                            "allclose": True, "rtol": 1e-5, "atol": 2e-5})
    result.update({"prefix_" + key: np.stack(values, axis=1) for key, values in prefix_values.items()})
    return result, invocations


def validate_invocations(invocations, n=200, batch_size=64):
    expected = [(start, min(batch_size, n - start), horizon) for start in range(0, n, batch_size) for horizon in (60, *PREFIXES)]
    if [(v.get("first_handle"), v.get("count"), v.get("command_rows")) for v in invocations] != expected:
        raise ValueError("Missing/reordered explicit prefix invocations")
    groups = {}
    for value in invocations:
        start, count, horizon = value["first_handle"], value["count"], value["command_rows"]
        if value.get("prediction_shape") != [count, horizon - 1, 6144]:
            raise ValueError("Invocation output shape differs")
        for key in ("initial_features_sha256", "commands_sha256", "prediction_sha256"):
            digest = value.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Unbound prediction invocation")
        if horizon == 60:
            groups[start] = value
        else:
            full = groups[start]
            gap = value.get("maximum_absolute_difference")
            reference = value.get("maximum_reference_absolute_value")
            ratio = value.get("maximum_tolerance_ratio")
            if (value.get("initial_features_sha256") != full["initial_features_sha256"]
                    or value.get("full_prediction_sha256") != full["prediction_sha256"]
                    or value.get("full_output_index") != horizon - 2 or value.get("target_native_offset") != horizon - 1
                    or value.get("allclose") is not True or value.get("rtol") != 1e-5 or value.get("atol") != 2e-5
                    or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (gap, reference, ratio))
                    or ratio > 1 or gap > 2e-5 + 1e-5 * reference + 1e-12):
                raise ValueError("Invalid causal-prefix invocation provenance")


def normalization_hashes(root, task_row):
    path = local(root, task_row["training_statistics_path"])
    if sha(path) != task_row["training_statistics_sha256"]:
        raise ValueError("Frozen training-statistics source changed")
    statistics = json.loads(path.read_text())
    return {key: array_sha(np.asarray(statistics[key], dtype=np.float32))
            for key in ("feature_mean", "feature_std", "command_mean", "command_std")}


def load_selected(root, row):
    if row["mode"] == "unbounded_spatial_mix":
        from shiftwm.real_video_iws_unbounded import training as trainer
    else:
        from shiftwm.real_video_iws import training as trainer
    package = local(root, row["package_path"])
    expected = {"model.pt": row["checkpoint_sha256"], "config.json": row["config_sha256"],
                "package_manifest.json": row["package_manifest_sha256"]}
    if any(sha(package / name) != digest for name, digest in expected.items()):
        raise ValueError("Selected model package changed after registration")
    model, state = trainer.load_package(package, "cpu")
    recipe = state["config"]["metadata"]["identity"]["scientific_config"]
    if (state["epoch"] != row["selected_epoch"] or state["config"]["package_kind"] != row["package_kind"]
            or any(recipe[key] != row[key] for key in ("task", "mode", "seed"))):
        raise ValueError("Selected task/arm/seed/epoch/kind differs")
    verify = trainer.engine.verify_model_contract if hasattr(trainer, "engine") else trainer.verify_model_contract
    verify(state["config"], state["config"]["metadata"]["identity"], model)
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.state_dict().values()):
        raise ValueError("Selected predictor is not FP32")
    from shiftwm.real_video_iws_reserved_recovery.prefix_backend import install_rowwise_gru
    install_rowwise_gru(model)
    return model.eval(), state


def write_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError("Refusing to replace completed reserved evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def evaluate(name, root=ROOT, registration_path=None):
    root = Path(root).resolve()
    registration_path = local(root, registration_path or REGISTRATION)
    registry = checked_registration(root, registration_path); check_grid(registry)
    row = next((row for row in registry["runs"] if row["name"] == name), None)
    if row is None:
        raise ValueError("Unknown reserved run")
    output = root / REPORT / "evaluations" / (name + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists() or output.with_suffix(".npz").exists():
            raise ValueError("Reserved evaluation already exists; never replace or silently retry")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, state = load_selected(root, row)
        expected_norms = normalization_hashes(root, registry["tasks"][row["task"]])
        norms = {k: array_sha(getattr(model, k).numpy()) for k in expected_norms}
        if norms != expected_norms:
            raise ValueError("Executed model statistics differ from registered training statistics")
        from shiftwm.real_video_iws_reserved.cache import ReservedFeatureCache
        cache = ReservedFeatureCache(root, row["task"], registration_path=registry["cache_registration_path"])
        population = population_contract(row["task"], cache.handles, cache.episode_ids)
        chunks = {key: [] for key in ("episode_index", "window_start", *metric_keys())}
        calls, store = [], {}
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            for start in range(0, 200, 64):
                batch = make_batch(cache, cache.handles[start:start + 64], row["task"], cache.episode_ids, store)
                metrics, invocations = score_batch(model, batch, start)
                calls.extend(invocations)
                for key in chunks:
                    chunks[key].append(batch[key] if key in ("episode_index", "window_start") else metrics[key])
        arrays = {key: np.concatenate(values) for key, values in chunks.items()}
        summaries = aggregate_arrays(arrays, population); validate_invocations(calls)
        metadata_paths = {filename: local(root, registry["tasks"][row["task"]]["cache_root"]) / filename
                          for filename in ("manifest.json", "identity.json", "episode_index.json")}
        sources = {str(path.relative_to(root)): sha(path) for path in metadata_paths.values()}
        for path in (Path(__file__), ROOT / "scripts/real_video_iws/evaluate.py", registration_path):
            sources[str(path.resolve().relative_to(root))] = sha(path)
        if checked_registration(root, registration_path) != registry:
            raise ValueError("Reserved registration changed during inference")
        ledger = output.with_suffix(".npz")
        atomic_npz(ledger, arrays)
        result = {"schema": SCHEMA, "status": "passed", "scope": SCOPE, "name": name,
                  **{k: row[k] for k in ("task", "mode", "seed", "package_kind", "selected_epoch")},
                  "selected_checkpoint_sha256": row["checkpoint_sha256"], "completed_epochs": 30,
                  "registration_sha256": sha(registration_path), "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "population": population, "cache_audit": cache.audit, "source_dependencies": sources,
                  "normalization_sha256": norms,
                  "metrics": list(METRICS), "offsets": list(range(1, 60)), "prefix_horizons": list(PREFIXES),
                  "primary_aggregation": "equal_trajectory_then_equal_seed", **summaries,
                  "backend": {"device": "cpu", "dtype": "float32", "threads": 8, "interop_threads": 1, "batch_size": 64,
                              "autocast": False, "tf32": False, "command_gru_dispatch": "one_native_row_per_call", "torch": torch.__version__, "numpy": np.__version__},
                  "invocations": calls, "reserved_feature_episodes_read": len(store), "reserved_raw_payloads_read": 0,
                  "internal_development_payloads_read": 0, "selector_score_equality_required": False,
                  "numerical_recovery": registry["numerical_recovery"],
                  "window_ledger_path": str(ledger.relative_to(root)), "window_ledger_sha256": sha(ledger),
                  "evaluator_sha256": sha(__file__), "metric_helper_sha256": sha(frozen_metrics.__file__),
                  "prefix_policy": "Full-H60 curves are recomputed under the common row-wise backend; H15/30/45 use separately scored command-prefix calls."}
        write_json_new(output, result)
        return {"status": "passed", "name": name, "handles": 200, "trajectories": 10, "evaluation_path": str(output.relative_to(root))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.name, args.root, args.registration), sort_keys=True), flush=True)
