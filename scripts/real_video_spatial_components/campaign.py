#!/usr/bin/env python3
"""Freeze six component runs and six revealed controls; retain all outcomes."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video.data import sha256
from shiftwm.real_video_spatial_components.model import MODES, SCHEMA, PACKAGE_KIND

CONFIG = ROOT / "configs/real_video_spatial_components/v1"
REPORT = ROOT / "reports/real_video_spatial_components"
REG = CONFIG / "registration.json"
PROTOCOL = ROOT / "reports/real_video_development/spatial_components_protocol.md"
OLD_REG = ROOT / "configs/real_video_spatial/v1/registration.json"
OLD_REG_SHA = "ae99340f7a3761ece066ee7376f4bd119b61a92503948c142196c40a1a76e337"
CONTROLS = ("anchored_additive", "transport")
ALL_MODES = ("anchored_additive", "bounded_additive", "unbounded_transport", "transport")


def module(name, old=False):
    path = ROOT / "scripts" / ("real_video_spatial" if old else "real_video_spatial_components") / (name + ".py")
    spec = importlib.util.spec_from_file_location("component_campaign_private_" + ("old_" if old else "") + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def expected_config(mode, seed):
    if mode not in MODES or type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Unknown component mode/seed")
    original = json.loads((ROOT / "configs/real_video_spatial/v1" / f"anchored_additive_s{seed}.json").read_text())
    return {**original, "mode": mode, "component_schema": SCHEMA,
            "protocol_path": str(PROTOCOL.relative_to(ROOT)),
            "output_dir": f"runs/real_video_spatial_components/v1/{mode}_s{seed}"}


def original_gate():
    if sha256(OLD_REG) != OLD_REG_SHA:
        raise ValueError("Frozen original spatial registration differs")
    old = module("campaign", old=True)
    registry = old.verify()
    final = json.loads((ROOT / "reports/real_video_spatial/finalization.json").read_text())
    if (final.get("status") != "passed" or final.get("registration_sha256") != OLD_REG_SHA
            or final.get("completed_models") != 15 or final.get("epochs_per_model") != 30
            or len(final.get("offline_cpu_parity", [])) != 15
            or any(p.get("status") != "passed" or p.get("max_abs_error") != 0 for p in final["offline_cpu_parity"])):
        raise ValueError("Original completed-control campaign is not finalized")
    gate = json.loads((ROOT / "reports/real_video_spatial/cache_and_budget_gate.json").read_text())
    if gate.get("status") != "passed" or gate.get("registration_sha256") != OLD_REG_SHA:
        raise ValueError("Original cache/resource gate differs")
    for key, path in (("cache_manifest_sha256", "data/features/droid_spatial_v1/manifest.json"),
                      ("statistics_sha256", "data/features/droid_spatial_v1/training_statistics.json"),
                      ("throughput_sha256", "reports/real_video_spatial/throughput.json")):
        if sha256(ROOT / path) != gate[key]:
            raise ValueError("Original cache/resource identity changed")
    return registry, final


def control_rows(old_registry):
    old_train, validator = module("train", old=True), module("validate_ledger", old=True)
    rows, paths = [], set()
    for row in old_registry["runs"]:
        if row["mode"] not in CONTROLS:
            continue
        config = json.loads((ROOT / row["config"]).read_text())
        recipe = expected_config("bounded_additive", row["seed"])
        for key in set(config) | set(recipe):
            if key not in ("mode", "output_dir", "protocol_path", "component_schema") and config.get(key) != recipe.get(key):
                raise ValueError("Existing control does not match the full component recipe: " + key)
        run = ROOT / config["output_dir"]
        summary = old_train.validate_completed(run)
        require_executed_recipe(old_train, run, config)
        _, selected = old_train.read_package(run / "best")
        evaluation = ROOT / "reports/real_video_spatial" / (row["name"] + "_validation.json")
        result = json.loads(evaluation.read_text())
        validator.validate_ledger(result, config, ROOT, selected)
        rows.append({**row, "validation": str(evaluation.relative_to(ROOT)), "selected_epoch": summary["best_epoch"],
                     "checkpoint_sha256": sha256(run / "best/model.pt"), "revealed_before_registration": True})
        paths.update((ROOT / row["config"], evaluation))
        paths.update(run / p for p in ("training_config.json", "training_summary.json", "metrics.jsonl",
                                      "best/model.pt", "best/config.json", "best/package_manifest.json",
                                      "last/model.pt", "last/config.json", "last/package_manifest.json"))
    if {(r["mode"], r["seed"]) for r in rows} != {(m, s) for m in CONTROLS for s in (0, 1, 2)} or len(rows) != 6:
        raise ValueError("Need six unique frozen controls")
    return rows, paths


def require_executed_recipe(trainer, run, registered_config):
    executed = json.loads((Path(run) / "training_config.json").read_text())
    if trainer.base.scientific_config(executed) != trainer.base.scientific_config(registered_config):
        raise ValueError("Executed scientific training recipe differs from the registered configuration")


def register():
    if REG.exists():
        raise ValueError("Component study is already registered; immutable")
    old_registry, _ = original_gate()
    controls, paths = control_rows(old_registry)
    train = module("train")
    runs = []
    for mode in MODES:
        for seed in (0, 1, 2):
            name = f"{mode}_s{seed}"
            config = expected_config(mode, seed)
            if (ROOT / config["output_dir"]).exists():
                raise ValueError("Registration must precede every new training output")
            path = CONFIG / (name + ".json")
            train.atomic_json(config, path)
            runs.append({"name": name, "mode": mode, "seed": seed, "config": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    paths.update(Path(p) for p in train.source_files())
    paths.update(ROOT / p for p in (
        "configs/real_video_spatial/v1/registration.json", "reports/real_video_spatial/finalization.json",
        "reports/real_video_spatial/cache_and_budget_gate.json", "reports/real_video_spatial/throughput.json",
        "data/features/droid_spatial_v1/manifest.json", "data/features/droid_spatial_v1/training_statistics.json",
        "data/features/droid_selected_v1/manifest.json", "data/features/droid_selected_v1/training_statistics.json",
        "data/real_video/droid_selected/processed/manifest.json", "data/real_video/droid_selected/processed/data_audit.json",
        "data/pretrained/dinov2-small/provenance.json"))
    throughput = json.loads((ROOT / "reports/real_video_spatial/throughput.json").read_text())
    # Planning estimate only, inherited measurements on the two corresponding
    # controls, not a newly measured speed claim for the component arms.
    seconds = {r["mode"]: r["estimated_seconds_per_30_epoch_run_without_io"] for r in throughput["rows"]}
    projected_hours = 3 * sum(seconds[m] for m in CONTROLS) / 3600
    if not np.isfinite(projected_hours) or projected_hours > 6:
        raise ValueError("Six-run proxy projection exceeds six GPU-hours")
    registry = {"status": "registered", "created_utc": now(), "component_schema": SCHEMA, "package_kind": PACKAGE_KIND,
        "scope": "follow-up original train/validation development after seeing existing spatial controls",
        "expected_new_runs": 6, "expected_control_runs": 6, "expected_epochs_per_run": 30,
        "runs": runs, "controls": controls, "old_registration_sha256": OLD_REG_SHA,
        "dependencies": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)},
        "analysis": {"primary_endpoint": "native_mse at horizon10", "secondary_endpoints": "native/original_2x2 endpoint errors at horizons5,10",
                     "edge_contrasts": 16, "interaction_contrasts": 4, "bootstrap_draws": 10000, "bootstrap_seed": 173,
                     "adjustment": "none; exploratory development, no confirmatory inference"},
        "resource_policy": {"partition": "ws-ia", "known_working_nodes": ["ws-l1-002", "ws-l5-004"],
                            "max_concurrent_gpus": 2, "cpu_threads_per_gpu": 8,
                            "proxy_six_run_gpu_hours_without_io": projected_hours,
                            "proxy_scope": "existing matching-control throughput; new arms not timed yet",
                            "maximum_new_full_runs": 6, "allocation_hours": 7+50/60,
                            "epoch_boundary_requeue_seconds": 6*3600}}
    train.atomic_json(registry, REG)
    verify()
    print(json.dumps({"status": "registered", "sha256": sha256(REG), "new_runs": 6, "controls": 6}))


def verify():
    registry = json.loads(REG.read_text())
    if (registry.get("status") != "registered" or registry.get("component_schema") != SCHEMA
            or registry.get("package_kind") != PACKAGE_KIND or registry.get("expected_new_runs") != 6
            or registry.get("expected_control_runs") != 6 or registry.get("expected_epochs_per_run") != 30
            or registry.get("old_registration_sha256") != OLD_REG_SHA or sha256(OLD_REG) != OLD_REG_SHA):
        raise ValueError("Component registration header differs")
    for path, digest in registry["dependencies"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Frozen component dependency changed: " + path)
    for key, modes in (("runs", MODES), ("controls", CONTROLS)):
        rows = registry[key]
        if (len(rows) != 6 or {(r["mode"], r["seed"]) for r in rows} != {(m, s) for m in modes for s in (0, 1, 2)}
                or len({r["name"] for r in rows}) != 6 or len({r["config"] for r in rows}) != 6):
            raise ValueError("Registration must contain unique complete mode/seed grids")
        for row in rows:
            if row["name"] != f"{row['mode']}_s{row['seed']}" or sha256(ROOT / row["config"]) != row["sha256"]:
                raise ValueError("Component registered name/config identity differs")
            config = json.loads((ROOT / row["config"]).read_text())
            if config["mode"] != row["mode"] or config["seed"] != row["seed"]:
                raise ValueError("Component config mode/seed differs")
            if key == "runs" and config != expected_config(row["mode"], row["seed"]):
                raise ValueError("Full matched component recipe differs")
    return registry


def run(name):
    registry = verify()
    row = next((r for r in registry["runs"] if r["name"] == name), None)
    if row is None:
        raise ValueError("Unknown registered run")
    if os.environ.get("SLURM_JOB_PARTITION") != "ws-ia" or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("Production component training requires an allocated ws-ia Slurm job")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one scheduler-visible GPU required; do not override visibility")
    train = module("train")
    config = json.loads((ROOT / row["config"]).read_text())
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    train.atomic_json({"utc": now(), "job_id": os.environ["SLURM_JOB_ID"],
                      "partition": os.environ["SLURM_JOB_PARTITION"], "node": os.environ.get("SLURMD_NODENAME"),
                      "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "gpu": torch.cuda.get_device_name(0), "gpu_free_bytes_before_training": free_bytes,
                      "gpu_total_bytes": total_bytes, "minimum_free_bytes": 6*1024**3,
                      "registration_sha256": sha256(REG)},
                     REPORT / (name + "_allocation.json"))
    if free_bytes < 6*1024**3:
        raise RuntimeError("Allocated GPU has under6GiB free; preserve allocation audit, do not change batch or GPU visibility")
    config.update(resume_if_present=True, max_runtime_seconds=6*3600)
    result = train.train(config)
    if result["status"] != "completed":
        if int(os.environ.get("SLURM_RESTART_COUNT", "0")) >= 5:
            raise RuntimeError("Repeated allocation limit reached; checkpoint retained, no partial result reported")
        train.atomic_json({"status": "epoch_checkpointed_requeue_requested", "job_id": os.environ["SLURM_JOB_ID"],
                          "completed_epochs": result["completed_epochs"], "utc": now()}, REPORT / (name + "_continuation.json"))
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=True)
        return
    verify()
    module("evaluate").evaluate(config, REPORT / (name + "_validation.json"))
    verify()
    train.atomic_json({"status": "completed", "name": name, "epochs": 30, "utc": now(),
                      "evaluation_sha256": sha256(REPORT / (name + "_validation.json")), "registration_sha256": sha256(REG)},
                     REPORT / (name + "_completed.json"))


def collect(registry):
    rows = []
    evaluator, train = module("evaluate"), module("train")
    old_train, old_validator = module("train", old=True), module("validate_ledger", old=True)
    for kind in ("controls", "runs"):
        for row in registry[kind]:
            config = json.loads((ROOT / row["config"]).read_text())
            t = old_train if kind == "controls" else train
            summary = t.validate_completed(ROOT / config["output_dir"])
            require_executed_recipe(t, ROOT / config["output_dir"], config)
            _, selected = t.read_package(ROOT / config["output_dir"] / "best")
            path = ROOT / row["validation"] if kind == "controls" else REPORT / (row["name"] + "_validation.json")
            result = json.loads(path.read_text())
            verified = old_validator.validate_ledger(result, config, ROOT, selected) if kind == "controls" else evaluator.validate(result, config, selected)
            if summary["completed_epochs"] != 30:
                raise ValueError("No incomplete runs can enter component finalization")
            result["summary"] = verified
            rows.append((row, config, summary, result, path))
    return rows


def finalize():
    registry = verify()
    train = module("train")
    if (REPORT / "finalization.json").exists():
        raise ValueError("Component finalization already exists; immutable")
    rows = collect(registry)
    results = [r[3] for r in rows]
    analysis = module("analysis").contrasts(results)
    if len(analysis["reported_effects"]) != 20:
        raise ValueError("Missing predeclared component or interaction contrasts")
    aggregate = {mode: {metric: np.mean([r["summary"][metric] for r in results if r["mode"] == mode], axis=0).tolist()
                        for metric in results[0]["summary"]} for mode in ALL_MODES}
    destination = ROOT / "artifacts/releases/spatial_components_v1"
    if destination.exists():
        raise ValueError("Component inference export already exists")
    temporary = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
    parity = []
    try:
        for row, config, summary, result, path in rows:
            if row["mode"] in MODES:
                proof = module("inference").export_and_check(config, temporary / "models" / row["name"])
                parity.append({"name": row["name"], **proof})
        verify()
        evidence = {str(path.relative_to(ROOT)): sha256(path) for _, _, _, _, path in rows}
        for row, config, _, _, _ in rows:
            for file in ("best/model.pt", "best/config.json", "best/package_manifest.json", "training_summary.json", "metrics.jsonl"):
                path = ROOT / config["output_dir"] / file
                evidence[str(path.relative_to(ROOT))] = sha256(path)
        final = {"status": "passed", "completed_utc": now(), "registration_sha256": sha256(REG), "package_kind": PACKAGE_KIND,
                 "scope": registry["scope"], "completed_new_models": 6, "completed_frozen_controls": 6, "epochs_per_model": 30,
                 "aggregate": aggregate, **analysis, "offline_cpu_parity": parity, "evidence_sha256": evidence,
                 "runs": [{"name": row["name"], "mode": row["mode"], "seed": row["seed"],
                           "selected_epoch": summary["best_epoch"], "parameter_counts": summary["parameter_counts"],
                           "validation": str(path.relative_to(ROOT)), "revealed_control": row["mode"] in CONTROLS}
                          for row, config, summary, result, path in rows],
                 "limitations": "Follow-up after revealed controls; original validation only. Mixing includes learned gate, identity bias and ~1.84% more active parameters. Features, not RGB or physical control. No fresh-test or SOTA claim."}
        train.atomic_json(final, temporary / "results.json")
        for path, digest in evidence.items():
            if sha256(ROOT / path) != digest:
                raise ValueError("Completed evidence changed during finalization")
        verify()
        os.replace(temporary, destination)
        train.atomic_json(final, REPORT / "finalization.json")
        lines = ["# Spatial component follow-up", "", final["limitations"], "",
                 "All six new runs and six previously revealed controls completed 30 epochs. All six new inference packages passed exact relocated CPU parity.", "",
                 "| Mode | Native h5 | Native h10 | Original 2x2 h5 | Original 2x2 h10 |", "|---|---:|---:|---:|---:|"]
        for mode, values in aggregate.items():
            lines.append(f"| {mode} | {values['native_mse'][4]:.6f} | {values['native_mse'][9]:.6f} | {values['original_2x2_mse'][4]:.6f} | {values['original_2x2_mse'][9]:.6f} |")
        lines.extend(["", "Endpoint errors: average windows within episode, then equal episodes and training seeds. Selection remains window-weighted mean across all ten query steps.", "",
                      "| Contrast | Metric | Horizon | Signed difference [paired 95% interval] |", "|---|---|---:|---|"])
        for effect in analysis["reported_effects"]:
            value = effect.get("method_minus_comparator", effect.get("difference_of_differences"))
            lo, hi = effect["paired_95_percent_interval"]
            lines.append(f"| {effect['contrast']} | {effect['metric']} | {effect['horizon']} | {value:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
        lines.extend(["", "Negative edge differences favor the first named component arm. Negative interaction means bounding reduces error more with the mixing package. Ten thousand paired recording-session and seed bootstrap draws (seed 173); intervals are exploratory and unadjusted. No run, epoch, horizon or failed comparison was filtered by outcome."])
        (REPORT / "results.md").write_text("\n".join(lines) + "\n")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"status": "passed", "new_models": 6, "controls": 6, "contrasts": 20}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "verify", "run", "finalize"))
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.command == "register": register()
    elif args.command == "verify": print(json.dumps({"status": "passed", "runs": len(verify()["runs"])}))
    elif args.command == "run": run(args.name)
    else: finalize()
