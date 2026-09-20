#!/usr/bin/env python3
"""Finalize the complete36 reserved evaluations without opening dataset payloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_reserved_finalizer_evaluation", Path(__file__).with_name("evaluate.py"))
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)
TASKS, MODES, METRICS = tuple(evaluation.TASKS), (*evaluation.MODES, "persistence"), evaluation.METRICS
COMPARISONS = {
    "bounded_vs_additive": ("bounded_spatial_mix", "anchored_additive", "primary_H60_standardized_mse"),
    "bounded_vs_autoregressive": ("bounded_spatial_mix", "autoregressive", "secondary"),
    "bounded_vs_persistence": ("bounded_spatial_mix", "persistence", "secondary"),
    "no_tanh_vs_bounded": ("unbounded_spatial_mix", "bounded_spatial_mix", "secondary_post_development_component"),
    "no_tanh_vs_autoregressive": ("unbounded_spatial_mix", "autoregressive", "secondary_post_development_component"),
}
SCHEMA = "shiftwm_iws_reserved_recovery_finalization_v2"


def bind(root, path, sources, expected=None):
    path = evaluation.local(root, path)
    digest = evaluation.sha(path)
    key = str(path.relative_to(Path(root).resolve()))
    if expected is not None and digest != expected or key in sources and sources[key] != digest:
        raise ValueError("Reserved evidence changed: " + key)
    sources[key] = digest
    return digest


def validate_cache_audit(actual, metadata, population):
    dynamic = {"metadata_only", "episodes_loaded"}
    expected_order = list(dict.fromkeys(h["episode_id"] for h in population["handles"]))
    if (actual.get("metadata_only") is not False or actual.get("episodes_loaded") != expected_order
            or metadata.get("metadata_only") is not True or metadata.get("episodes_loaded") != []
            or {k: v for k, v in actual.items() if k not in dynamic} != {k: v for k, v in metadata.items() if k not in dynamic}):
        raise ValueError("Evaluator cache identity or exact payload access ledger differs")


def validate_evaluation(root, path, row, population, cache, registry_sha, sources):
    """Reconstruct every trajectory/handle metric from its primitive ledger."""
    path = evaluation.local(root, path)
    bind(root, path, sources)
    receipt = json.loads(path.read_text())
    required = {"schema": evaluation.SCHEMA, "status": "passed", "scope": evaluation.SCOPE,
                "name": row["name"], "task": row["task"], "mode": row["mode"], "seed": row["seed"],
                "package_kind": row["package_kind"], "selected_epoch": row["selected_epoch"],
                "selected_checkpoint_sha256": row["checkpoint_sha256"], "completed_epochs": 30,
                "registration_sha256": registry_sha, "population": population,
                "metrics": list(METRICS), "offsets": list(range(1, 60)), "prefix_horizons": [15, 30, 45],
                "primary_aggregation": "equal_trajectory_then_equal_seed", "reserved_feature_episodes_read": 10,
                "reserved_raw_payloads_read": 0, "internal_development_payloads_read": 0,
                "selector_score_equality_required": False,
                "evaluator_sha256": evaluation.sha(evaluation.__file__),
                "metric_helper_sha256": evaluation.sha(evaluation.frozen_metrics.__file__)}
    if any(receipt.get(k) != v for k, v in required.items()):
        raise ValueError("Reserved evaluation identity/population/scope mismatch: " + row["name"])
    validate_cache_audit(receipt.get("cache_audit", {}), cache.audit, population)
    backend = {"device": "cpu", "dtype": "float32", "threads": 8, "interop_threads": 1,
               "batch_size": 64, "autocast": False, "tf32": False, "command_gru_dispatch": "one_native_row_per_call"}
    if any(receipt.get("backend", {}).get(k) != v for k, v in backend.items()):
        raise ValueError("Reserved evaluation backend differs")
    evaluation.validate_invocations(receipt["invocations"])
    ledger = evaluation.local(root, receipt["window_ledger_path"])
    if ledger != path.with_suffix(".npz"):
        raise ValueError("Metric ledger must be the receipt's immutable sibling")
    bind(root, ledger, sources, receipt["window_ledger_sha256"])
    with np.load(ledger, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    reconstructed = evaluation.aggregate_arrays(arrays, population)
    if any(receipt.get(k) != v for k, v in reconstructed.items()):
        raise ValueError("Stored aggregate differs from independently reconstructed primitives")
    mandatory = {str(Path(evaluation.__file__).resolve().relative_to(root)),
                 "scripts/real_video_iws/evaluate.py"}
    dependencies = receipt.get("source_dependencies", {})
    if not mandatory <= set(dependencies):
        raise ValueError("Evaluation lacks source dependencies")
    for name, digest in dependencies.items():
        bind(root, name, sources, digest)
    norms = receipt.get("normalization_sha256", {})
    if set(norms) != {"feature_mean", "feature_std", "command_mean", "command_std"} or any(not isinstance(v, str) or len(v) != 64 for v in norms.values()):
        raise ValueError("Missing fixed-normalization identity")
    return receipt, arrays


def assemble(rows, populations):
    """All36 learned ledgers; persistence must match at every primitive cell."""
    expected = {(t, m, s) for t in TASKS for m in evaluation.MODES for s in range(3)}
    if len(rows) != 36 or {(r["task"], r["mode"], r["seed"]) for r in rows} != expected:
        raise ValueError("Complete36-run reserved grid required")
    values = {t: {m: np.empty((3, 10, 59, len(METRICS)), dtype=np.float64) for m in MODES} for t in TASKS}
    prefixes = {t: {m: np.empty((3, 10, 3, len(METRICS)), dtype=np.float64) for m in MODES} for t in TASKS}
    handles = {t: {m: np.empty((3, 200, 59, len(METRICS)), dtype=np.float64) for m in MODES} for t in TASKS}
    handle_prefixes = {t: {m: np.empty((3, 200, 3, len(METRICS)), dtype=np.float64) for m in MODES} for t in TASKS}
    reference, norms, inputs = {}, {}, {}
    for row in rows:
        task, mode, seed = row["task"], row["mode"], row["seed"]
        receipt, arrays = row["receipt"], row["arrays"]
        baseline = np.stack([arrays["persistence_" + metric] for metric in METRICS], axis=-1)
        if task in reference and not np.array_equal(reference[task], baseline):
            raise ValueError("Per-handle persistence differs across matched arms/seeds")
        reference[task] = baseline
        if task in norms and norms[task] != receipt["normalization_sha256"]:
            raise ValueError("Training normalization differs across methods/seeds")
        norms[task] = receipt["normalization_sha256"]
        invocation_inputs = [{key: v[key] for key in ("first_handle", "count", "command_rows", "initial_features_sha256", "commands_sha256")} for v in receipt["invocations"]]
        if task in inputs and inputs[task] != invocation_inputs:
            raise ValueError("Matched models received different observed inputs or command prefixes")
        inputs[task] = invocation_inputs
        expected_episodes = [(r["episode_id"], r["handles"]) for r in populations[task]["records"]]
        if [(e["episode_id"], e["handles"]) for e in receipt["episodes"]] != expected_episodes:
            raise ValueError("Paired trajectory identities differ")
        for mi, metric in enumerate(METRICS):
            values[task][mode][seed, :, :, mi] = [e[metric + "_by_offset"] for e in receipt["episodes"]]
            values[task]["persistence"][seed, :, :, mi] = [e["persistence_" + metric + "_by_offset"] for e in receipt["episodes"]]
            prefixes[task][mode][seed, :, :, mi] = [e["prefix_" + metric + "_by_offset"] for e in receipt["episodes"]]
            prefixes[task]["persistence"][seed, :, :, mi] = values[task]["persistence"][seed, :, :, mi][:, [13, 28, 43]]
            handles[task][mode][seed, :, :, mi] = arrays[metric]
            handles[task]["persistence"][seed, :, :, mi] = arrays["persistence_" + metric]
            handle_prefixes[task][mode][seed, :, :, mi] = arrays["prefix_" + metric]
            handle_prefixes[task]["persistence"][seed, :, :, mi] = arrays["persistence_" + metric][:, [13, 28, 43]]
    return values, prefixes, handles, handle_prefixes


def bootstrap(values, draws=10000, seed=173):
    """Paired seed x trajectory draws; shared seed indices across the3 tasks.

    Same percentile construction as the frozen development helper, extended to
    the predeclared bounded primary and no-tanh secondary comparison pairs.
    """
    rng = np.random.default_rng(seed)
    pairs = list(COMPARISONS.values())
    sampled = {t: {"difference": np.empty((draws, len(pairs), len(METRICS))),
                   "gain_percent": np.empty((draws, len(pairs), len(METRICS)))} for t in TASKS}
    macro = np.empty((draws, len(pairs), len(METRICS)))
    stacked = {t: np.stack([values[t][m][:, :, -1, :] for m in MODES], axis=2) for t in TASKS}
    for i in range(draws):
        seeds = rng.integers(0, 3, size=3)
        task_gains = []
        for task in TASKS:
            n = stacked[task].shape[1]
            trajectories = rng.integers(0, n, size=n)
            means = stacked[task][seeds][:, trajectories].mean(axis=(0, 1))
            for ci, (method, comparator, _) in enumerate(pairs):
                a, b = means[MODES.index(method)], means[MODES.index(comparator)]
                difference = a - b
                sampled[task]["difference"][i, ci] = difference
                sampled[task]["gain_percent"][i, ci] = np.divide(-100 * difference, b, out=np.full(len(METRICS), np.nan), where=b != 0)
            task_gains.append(sampled[task]["gain_percent"][i])
        macro[i] = np.mean(task_gains, axis=0)
    def summary(array):
        return {"percentile95": np.quantile(array, [.025, .975]).tolist() if np.isfinite(array).all() else None,
                "undefined_draws": int((~np.isfinite(array)).sum())}
    return {"tasks": {task: {metric: {name: {kind: summary(sampled[task][kind][:, ci, mi])
                                            for kind in ("difference", "gain_percent")}
                                    for ci, name in enumerate(COMPARISONS)} for mi, metric in enumerate(METRICS)} for task in TASKS},
            "macro": {metric: {name: summary(macro[:, ci, mi]) for ci, name in enumerate(COMPARISONS)} for mi, metric in enumerate(METRICS)},
            "protocol": {"draws": draws, "seed": seed, "method": "paired seed-by-trajectory percentile bootstrap",
                         "shared_seed_draws_across_tasks": True, "trajectory_draws": "10 independently drawn within each task, shared across methods and metrics",
                         "interval_scope": "H60 only; primary standardized MSE bounded versus additive; other metrics/comparisons secondary and unadjusted"}}


def numerical_report(values, prefixes, handles, handle_prefixes, draws=10000, seed=173):
    uncertainty = bootstrap(values, draws, seed)
    results = {}
    for task in TASKS:
        results[task] = {}
        for mi, metric in enumerate(METRICS):
            summaries = {}
            for weighting, curves, short in (("equal_trajectory", values, prefixes), ("equal_handle", handles, handle_prefixes)):
                mean = {m: curves[task][m][:, :, :, mi].mean(axis=(0, 1)) for m in MODES}
                summaries[weighting] = {"mean_all59_offsets": {m: a.tolist() for m, a in mean.items()},
                    "horizon_means": {str(h): {m: float(mean[m][-1]) if h == 60 else float(short[task][m][:, :, j, mi].mean()) for m in MODES}
                                      for j, h in enumerate((15, 30, 45, 60))},
                    "per_seed_horizon_means": {str(h): {m: (curves[task][m][:, :, -1, mi] if h == 60 else short[task][m][:, :, j, mi]).mean(axis=1).tolist() for m in MODES}
                                               for j, h in enumerate((15, 30, 45, 60))}}
            effects = {}
            h60 = summaries["equal_trajectory"]["horizon_means"]["60"]
            for name, (method, comparator, status) in COMPARISONS.items():
                a, b = h60[method], h60[comparator]
                effects[name] = {"method": method, "comparator": comparator, "status": status if metric == "standardized_mse" else "secondary_metric",
                                 "method_mean": a, "comparator_mean": b, "method_minus_comparator": a - b,
                                 "relative_error_reduction_percent": None if b == 0 else 100 * (b - a) / b,
                                 "paired95": uncertainty["tasks"][task][metric][name]}
            results[task][metric] = {**summaries, "h60_comparisons": effects}
    macro = {}
    for metric in METRICS:
        macro[metric] = {}
        for name in COMPARISONS:
            gains = [results[t][metric]["h60_comparisons"][name]["relative_error_reduction_percent"] for t in TASKS]
            macro[metric][name] = {"equal_task_relative_error_reduction_percent": None if any(g is None for g in gains) else float(np.mean(gains)),
                                   "paired95": uncertainty["macro"][metric][name]}
    return {"task_results": results, "macro_h60": macro, "bootstrap": uncertainty["protocol"],
            "primary": "H60 standardized MSE bounded spatial mixing versus anchored additive; equal trajectory, equal seed, equal task macro of task-relative reductions",
            "scope": "Reserved upstream-validation feature forecasting; no RGB, physical-control, external-SOTA or independent-session claim",
            "prefix_policy": "All59 curves use fullH60 calls; H15/30/45 summaries use separate command-prefix calls."}


def collect(root, registration_path, registry):
    root = Path(root).resolve()
    sources = {}
    for name, digest in registry["dependencies"].items():
        bind(root, name, sources, digest)
    registry_sha = bind(root, registration_path, sources)
    for path in (Path(__file__), Path(evaluation.__file__), ROOT / "scripts/real_video_iws/evaluate.py",
                 ROOT / "tests/test_iws_reserved_recovery_evaluation.py", ROOT / "tests/test_iws_reserved_recovery_finalization.py"):
        bind(root, path, sources)
    from shiftwm.real_video_iws_reserved.cache import ReservedFeatureCache
    caches, populations = {}, {}
    for task in TASKS:
        cache = ReservedFeatureCache.metadata_only(root, task, registration_path=registry["cache_registration_path"])
        caches[task] = cache
        populations[task] = evaluation.population_contract(task, cache.handles, cache.episode_ids)
        for name in ("manifest.json", "identity.json", "inputs.json", "episode_index.json"):
            bind(root, evaluation.local(root, registry["tasks"][task]["cache_root"]) / name, sources)
        for name, digest in cache.audit["source_dependencies"].items():
            if name.endswith(".json"):
                bind(root, name, sources, digest)
    rows, identities = [], []
    for row in registry["runs"]:
        package = evaluation.local(root, row["package_path"])
        for name, field in (("model.pt", "checkpoint_sha256"), ("config.json", "config_sha256"), ("package_manifest.json", "package_manifest_sha256")):
            bind(root, package / name, sources, row[field])
        path = root / evaluation.REPORT / "evaluations" / (row["name"] + ".json")
        receipt, arrays = validate_evaluation(root, path, row, populations[row["task"]], caches[row["task"]], registry_sha, sources)
        if receipt.get("numerical_recovery") != registry["numerical_recovery"]:
            raise ValueError("Evaluation numerical recovery scope differs")
        if receipt["normalization_sha256"] != evaluation.normalization_hashes(root, registry["tasks"][row["task"]]):
            raise ValueError("Evaluation normalization differs from registered training statistics")
        rows.append({**row, "receipt": receipt, "arrays": arrays})
        identities.append({**{k: row[k] for k in ("name", "task", "mode", "seed", "package_kind", "selected_epoch", "checkpoint_sha256")},
                           "evaluation_path": str(path.relative_to(root)), "evaluation_sha256": evaluation.sha(path),
                           "window_ledger_path": receipt["window_ledger_path"], "window_ledger_sha256": receipt["window_ledger_sha256"]})
    arrays = assemble(rows, populations)
    result = numerical_report(*arrays, draws=registry["evaluation"]["bootstrap_draws"], seed=registry["evaluation"]["bootstrap_seed"])
    if evaluation.checked_registration(root, registration_path) != registry:
        raise ValueError("Registration changed during finalization")
    for name, digest in sources.items():
        bind(root, name, sources, digest)
    return {"schema": SCHEMA, "status": "passed", "scope": evaluation.SCOPE, "expected_runs": 36, "completed_runs": 36,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "registration_sha256": registry_sha,
            "numerical_recovery": registry["numerical_recovery"],
            "source_dependencies": sources, "per_run": identities, "populations": populations, "results": result,
            "cache_payload_sha256_from_metadata": {task: {name: digest for name, digest in cache.audit["source_dependencies"].items() if name.endswith(".npz")}
                                                    for task, cache in caches.items()},
            "total_unique_handles": 600, "total_unique_trajectories": 30, "persistence_cases": 3,
            "reserved_raw_or_feature_payloads_read_by_finalizer": 0, "model_inference_performed_by_finalizer": False,
            "paper_ingestion": "Only after complete36 validation and independent numerical/pixel review"}


def verify_existing(root, path, value, registry, registration_path):
    if (value.get("schema") != SCHEMA or value.get("status") != "passed" or value.get("scope") != evaluation.SCOPE
            or value.get("expected_runs") != 36 or value.get("completed_runs") != 36
            or value.get("registration_sha256") != evaluation.sha(registration_path)):
        raise ValueError("Invalid existing reserved finalization")
    # Reconstruct primitives and uncertainty again; timestamps alone may differ.
    rebuilt = collect(root, registration_path, registry)
    if {k: v for k, v in rebuilt.items() if k != "completed_utc"} != {k: v for k, v in value.items() if k != "completed_utc"}:
        raise ValueError("Existing finalization differs from complete reconstruction")


def finalize(root=ROOT, registration_path=None, if_ready=False):
    root = Path(root).resolve()
    registration_path = evaluation.local(root, registration_path or evaluation.REGISTRATION)
    registry = evaluation.checked_registration(root, registration_path); evaluation.check_grid(registry)
    report = root / evaluation.REPORT
    report.mkdir(parents=True, exist_ok=True)
    with (report / ".finalization.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = report / "finalization.json"
        if path.exists():
            value = json.loads(path.read_text()); verify_existing(root, path, value, registry, registration_path)
            return {"status": "already_complete_verified", "completed_runs": 36, "finalization_sha256": evaluation.sha(path)}
        missing = [r["name"] for r in registry["runs"] if not (report / "evaluations" / (r["name"] + ".json")).is_file()]
        if missing:
            if not if_ready:
                raise ValueError("All36 complete reserved evaluations are required")
            return {"status": "pending", "completed_runs": 36 - len(missing), "expected_runs": 36, "missing_runs": missing, "outputs_written": False}
        value = collect(root, registration_path, registry)
        evaluation.write_json_new(path, value)
        return {"status": "passed", "completed_runs": 36, "finalization_sha256": evaluation.sha(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    print(json.dumps(finalize(args.root, args.registration, args.if_ready), sort_keys=True), flush=True)
