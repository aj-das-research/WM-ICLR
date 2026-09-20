#!/usr/bin/env python3
"""One reviewed synthetic CUDA profile, never a benchmark or training run."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

import torch
import adapter

ROOT = adapter.ROOT
HERE = Path(__file__).resolve().parent
REPORT = ROOT / "reports/external_dinowm_profile_v1"
REGISTRATION = REPORT / "registration.json"
REVIEW = REPORT / "source_review.json"
SPEC = HERE / "profile_spec.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dependency_paths():
    local = ["adapter.py", "profile_cuda.py", "profile_spec.json", "test_contract.py", "run.slurm", "README.md"]
    upstream = ["models/vit.py", "models/proprio.py", "models/visual_world_model.py", "models/dino.py",
                "conf/train.yaml", "conf/predictor/vit.yaml", "conf/action_encoder/proprio.yaml", "LICENSE", "train.py"]
    return ([HERE / p for p in local] + [adapter.UPSTREAM / p for p in upstream]
            + [ROOT / "reports/real_video_spatial/throughput.json",
               ROOT / "reports/real_video_development/spatial_protocol.md",
               ROOT / "configs/real_video_spatial/v1/transport_s0.json"])


def upstream_identity():
    revision = subprocess.check_output(["git", "-C", str(adapter.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != adapter.REVISION:
        raise ValueError("Unexpected upstream revision")
    subprocess.run(["git", "-C", str(adapter.UPSTREAM), "diff", "--quiet", "HEAD", "--"], check=True)
    return revision


def prepare_registration():
    spec = read(SPEC)
    upstream_identity()
    source = read(ROOT / "reports/real_video_spatial/throughput.json")
    if any((r["train_windows"], r["validation_windows"]) != (18660, 1631) for r in source["rows"]):
        raise ValueError("Existing train/development window-count receipt differs")
    value = {"schema": "external_dinowm_synthetic_profile_registration_v1",
             "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "upstream_revision": adapter.REVISION, "spec": spec,
             "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in dependency_paths()},
             "data_access": "Synthetic tensors only; existing report/config metadata supplies window counts. No feature, video, checkpoint, test or reserved payloads.",
             "scope": "Resource and implementation feasibility only; no benchmark accuracy or full-training authorization."}
    write_new(REGISTRATION, value)
    print(json.dumps({"registration": str(REGISTRATION.relative_to(ROOT)), "sha256": sha(REGISTRATION)}))


def verify_reviewed_registration():
    # Reject absent/stale review before importing/loading upstream predictor or
    # constructing CUDA tensors. Review itself contains no model/data payload.
    registration = read(REGISTRATION)
    review = read(REVIEW)
    if (review.get("status") != "passed" or review.get("registration_sha256") != sha(REGISTRATION)
            or review.get("source_sha256") != registration["source_sha256"]):
        raise ValueError("Missing or stale independent source review")
    if registration["spec"] != read(SPEC):
        raise ValueError("Profile specification changed")
    for relative, expected in registration["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Registered dependency changed: " + relative)
    upstream_identity()
    return registration


def estimate(train_seconds, validation_seconds, spec):
    training_batches = math.ceil(spec["train_windows_for_projection"] / spec["batch_size"])
    validation_batches = math.ceil(spec["validation_windows_for_projection"] / spec["batch_size"])
    epoch = training_batches * train_seconds + validation_batches * validation_seconds
    run_hours = epoch * spec["proposed_epochs"] / 3600
    return {"training_batches_per_epoch": training_batches,
            "validation_batches_per_epoch": validation_batches,
            "estimated_30epoch_hours_one_seed_without_io": run_hours,
            "estimated_three_seed_gpu_hours_without_io": 3 * run_hours,
            "ideal_three_equal_gpu_wall_hours_without_io": run_hours,
            "limitations": "Projection from synthetic batches and recorded window counts; excludes data-loader/transfer, checkpoints, extraction, final evaluation, contention and queue. Last partial batches charged as full. Not an end-to-end runtime measurement."}


def profile_objective(objective, batch, spec):
    torch.manual_seed(spec["seed"])
    torch.cuda.manual_seed_all(spec["seed"])
    model = adapter.synthetic_model().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    count = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    training_times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(spec["warmup_steps"] + spec["timed_training_steps"]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(batch, objective=objective)
        if not torch.isfinite(result["loss"]):
            raise ValueError("Nonfinite synthetic loss; no accuracy value will be reported")
        result["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= spec["warmup_steps"]:
            training_times.append(elapsed)
        del result, norm
    train_allocated = torch.cuda.max_memory_allocated()
    train_reserved = torch.cuda.max_memory_reserved()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    validation_times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for step in range(1 + spec["timed_validation_steps"]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            # Common full H10 forecast in FP32 for either training objective.
            predictions = model.predict_normalized(batch["features"][:, :3], batch["actions"][:, :2], batch["actions"][:, 2:])
            if not torch.isfinite(predictions).all():
                raise ValueError("Nonfinite synthetic validation forecast")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step:
                validation_times.append(elapsed)
            del predictions
    row = {"objective": objective, "status": "measured", "batch_size": spec["batch_size"],
           "parameters_total": count, "parameters_trainable": active,
           "target_grid_indices": [1, 2, 3] if objective == adapter.OBJECTIVES[0] else list(range(3, 13)),
           "target_offsets_from_last_observed_grid": [-1, 0, 1] if objective == adapter.OBJECTIVES[0] else list(range(1, 11)),
           "supervised_feature_elements_per_window": 3 * 6144 if objective == adapter.OBJECTIVES[0] else 10 * 6144,
           "transformer_calls_per_training_window": 1 if objective == adapter.OBJECTIVES[0] else 10,
           "training_seconds_raw": training_times,
           "training_seconds_median": statistics.median(training_times),
           "training_seconds_mean": statistics.mean(training_times),
           "h10_fp32_forward_seconds_raw": validation_times,
           "h10_fp32_forward_seconds_median": statistics.median(validation_times),
           "training_peak_cuda_allocated_bytes": train_allocated,
           "training_peak_cuda_reserved_bytes": train_reserved,
           "validation_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
           "validation_peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
           "validation_memory_scope": "Training process with AdamW optimizer state resident; allocator reservation includes earlier training caches. Not standalone inference memory.",
           "disposable_synthetic_optimizer_steps": spec["warmup_steps"] + spec["timed_training_steps"],
           "checkpoints_written": False, "benchmark_accuracy_measured": False,
           "training_precision": "BF16 autocast, FP32 parameters and loss", "validation_precision": "FP32, no autocast or TF32"}
    row["projection"] = estimate(statistics.median(training_times), statistics.median(validation_times), spec)
    del model, optimizer
    torch.cuda.empty_cache()
    return row


def run():
    registration = verify_reviewed_registration()
    spec = registration["spec"]
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires one allocated visible CUDA device and a Slurm job ID")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Registered BF16 execution is not supported; no precision fallback")
    output = REPORT / ("job_" + job) / "profile.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(spec["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    batch = {k: v.cuda() for k, v in adapter.synthetic_batch(spec["batch_size"], spec["seed"]).items()}
    result = {"schema": "external_dinowm_synthetic_profile_result_v1", "status": "running",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "registration_sha256": sha(REGISTRATION), "source_review_sha256": sha(REVIEW),
              "source_sha256": registration["source_sha256"], "spec": spec,
              "rows": [], "execution": {"slurm_job_id": job, "host": platform.node(),
                 "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                 "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                 "cpu_threads": torch.get_num_threads(), "tf32": False},
              "reserved_test_and_training_payload_reads": False,
              "normalization": "Synthetic zero mean/unit scale; no statistics fitted or dataset normalization arrays loaded.",
              "accuracy_claim": "None; random synthetic features/actions and random initial weights. No benchmark or trained checkpoint is produced."}
    try:
        for objective in adapter.OBJECTIVES:
            print("Profiling " + objective, flush=True)
            result["rows"].append(profile_objective(objective, batch, spec))
        verify_reviewed_registration()
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result["cpu_process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        write_new(output, result)
        print(json.dumps({"output": str(output.relative_to(ROOT)), "status": result["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-registration", action="store_true")
    args = parser.parse_args()
    prepare_registration() if args.prepare_registration else run()
