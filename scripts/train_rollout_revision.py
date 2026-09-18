#!/usr/bin/env python3
"""Isolated 2x2 objective/context study over frozen completed Framewise donors.

The resumable loop is adapted from train_dynamics_revision.py, with both
objectives selected by the same FP32 recursive validation error. Original
campaign sources and checkpoints remain immutable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from shiftwm.checkpoint import load_package, resume_training, save_package
from shiftwm.dynamics_revision import (completed_framewise_source, file_sha256, validate_donor_state)
from shiftwm.rollout_revision import RolloutRevision, load_rollout_package
import shiftwm.rollout_revision as revision_source
from shiftwm.train import make_dataset, read_action_stats, to_device


OPERATIONAL = {"resume", "max_runtime_seconds", "device", "cpu_threads", "num_workers", "log_every"}


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def training_lock(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".training.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def training_identity(config, model):
    value = {"config": {k: v for k, v in config.items() if k not in OPERATIONAL},
             "provenance": model.provenance, "revision": model.export_config()["revision_config"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def initialize(config):
    if int(config.get("seed", -1)) != 0 or int(config.get("epochs", -1)) != 30:
        raise ValueError("This bounded revision uses seed0 and all 30 training epochs")
    if "model" in config:
        raise ValueError("Remove the old model config; use the isolated revision config")
    revision = config.get("revision", {})
    if revision.get("objective") not in {"teacher_forced", "recursive"} or revision.get("context_mode") not in {"inferred", "constant"}:
        raise ValueError("Declare one of the four locked objective/context conditions")
    if revision.get("history_length", 3) != 3 or config.get("sequence_length") != 8:
        raise ValueError("This study requires support 0:3 and all five query targets 3:8")
    source = completed_framewise_source(config["donor_checkpoint"])
    donor, state = load_package(config["donor_checkpoint"], device="cpu")
    validate_donor_state(state, source)
    old = source["run_config"]
    for key in ("sequence_length", "stride", "batch_size", "lr", "min_lr", "weight_decay",
                "grad_clip", "bf16"):
        if config.get(key) != old.get(key):
            raise ValueError(f"Revision must retain the donor training setting: {key}")
    if set(config.get("dataset_kwargs", {})) - {"feature_cache", "preload_features"}:
        raise ValueError("Revision requires complete data; no dataset subsampling/extra overrides")
    provenance = deepcopy(donor.provenance)
    data_path = Path(config["data_root"]) / "manifest.json"
    cache_path = Path(config["dataset_kwargs"]["feature_cache"]) / "manifest.json"
    for name, path in (("data_manifest", data_path), ("cache_manifest", cache_path),
                       ("action_stats", Path(config["action_stats"]))):
        if file_sha256(path) != donor.provenance.get(name + "_sha256"):
            raise ValueError(f"Revision {name} differs from the completed donor")
    cache = json.loads(cache_path.read_text())
    if cache.get("encoder_weights_sha256") != donor.provenance["weights_sha256"]:
        raise ValueError("Feature cache is in different visual coordinates")
    means, stds = read_action_stats(config["action_stats"], donor.action_dim)
    if not torch.equal(torch.tensor(means, dtype=torch.float32), donor.action_mean) or not torch.equal(
            torch.tensor(stds, dtype=torch.float32), donor.action_std):
        raise ValueError("Loaded action buffers differ from official statistics")
    manifest = json.loads(data_path.read_text())
    environment = manifest["environment"]
    if environment not in {"pusht", "reacher"} or manifest["action_block"] != 5:
        raise ValueError("Revision is limited to established PushT/Reacher action blocks")
    if environment == "pusht" and manifest.get("action_interface") != "relative":
        raise ValueError("PushT revision requires existing relative controls")
    expected_id = f"{environment}_{revision['objective']}_{revision['context_mode']}_s0"
    expected_output = Path(__file__).resolve().parents[1] / "runs/rollout_revision" / expected_id
    if Path(config["output_dir"]).resolve() != expected_output.resolve():
        raise ValueError("Use the isolated environment/objective/context training namespace")
    if provenance["download"]["repo"] != f"quentinll/lewm-{environment}":
        raise ValueError("Wrong pretrained family for dataset")
    provenance["revision_donor"] = {"path": source["path"], "hashes": source["hashes"],
                                    "epoch": state["epoch"], "best_metric": source["best_metric"]}
    provenance["revision_implementation_hashes"] = {
        name: file_sha256(Path(revision_source.__file__).with_name(name))
        for name in ("rollout_revision.py", "dynamics_revision.py", "model.py", "checkpoint.py", "upstream.py", "data.py", "train.py")}
    provenance["revision_implementation_hashes"]["train_rollout_revision.py"] = file_sha256(__file__)
    provenance["revision_environment"] = environment
    return RolloutRevision(donor, config.get("revision"), provenance)


def validate_completed(output, identity):
    output = Path(output)
    summary = json.loads((output / "training_summary.json").read_text())
    if summary.get("status") != "completed" or summary.get("completed_epochs") != 30:
        raise ValueError("Revision is not a completed 30-epoch run")
    if summary.get("training_identity") != identity:
        raise ValueError("Completed revision uses another training identity")
    if summary.get("validation_metric") != "recursive_mse_all_query_steps":
        raise ValueError("Completed revision did not use the common recursive validation criterion")
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if sorted(row["epoch"] for row in rows) != list(range(1, 31)):
        raise ValueError("Expected complete unique revision epoch history")
    values = [float(row["val"]["prediction_loss"]) for row in rows]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite revision validation metrics")
    best = min(values)
    if not math.isclose(best, summary["best_validation_prediction_loss"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Revision summary differs from validation metrics")
    for name in ("best", "last"):
        package = output / name
        state = torch.load(package / "model.pt", weights_only=True, map_location="cpu")
        if state["config"] != json.loads((package / "config.json").read_text()):
            raise ValueError("Revision embedded/external package configs differ")
        if state["config"]["metadata"]["training_identity"] != identity:
            raise ValueError("Revision package has a different training identity")
        metadata = state["config"]["metadata"]
        if (metadata.get("selection") != "minimum_validation_recursive_mse_targets_H_to_T-1_over_completed_training_epochs"
                or metadata.get("validation_protocol") != "recursive_from_observed_support_all_query_steps"
                or metadata.get("validation_precision") != "float32"):
            raise ValueError("Revision package has a different validation selection protocol")
        revision = state["config"]["revision_config"]
        if (metadata.get("training_objective") != revision.get("objective")
                or metadata.get("context_mode") != revision.get("context_mode")):
            raise ValueError("Revision package metadata disagrees with its objective/context condition")
        if not math.isclose(state["best_metric"], best, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Revision package has a different best metric")
        if name == "last" and (state["epoch"] != 30 or state["step"] != summary["step"]):
            raise ValueError("Revision last checkpoint does not complete training")
        if name == "best" and not any(row["epoch"] == state["epoch"] and math.isclose(
                row["val"]["prediction_loss"], best, rel_tol=1e-10, abs_tol=1e-12) for row in rows):
            raise ValueError("Revision best package is not validation-best")
    # Strict portable reconstruction also checks state keys, shapes and format.
    load_rollout_package(output / "best", device="cpu")
    return summary


def fit(model, config, train_set, val_set):
    """Full epoch loop, reusing campaign sampler/checkpoint/resume primitives."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cuda"))
    model.to(device)
    identity = training_identity(config, model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad for p in model.donor.parameters()):
        raise ValueError("All donor parameters must stay frozen")
    if not parameters or not len(train_set) or not len(val_set):
        raise ValueError("Trainable revision and complete nonempty train/validation splits required")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_kwargs = dict(batch_size=config["batch_size"], num_workers=config.get("num_workers", 2),
                         pin_memory=device.type == "cuda", persistent_workers=False)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=config["min_lr"])
    metadata = {"config": config, "training_identity": identity,
                "train_windows": len(train_set), "val_windows": len(val_set),
                "trainable_parameters": sum(p.numel() for p in parameters),
                "frozen_donor_parameters": sum(p.numel() for p in model.donor.parameters()),
                "total_parameters_including_reference": sum(p.numel() for p in model.parameters()),
                "target_coordinates": "immutable_pretrained_encoder_on_canonical_training_renders",
                "context_protocol": "support_0_to_H-1_query_H_onwards",
                "selection": "minimum_validation_recursive_mse_targets_H_to_T-1_over_completed_training_epochs",
                "validation_protocol": "recursive_from_observed_support_all_query_steps",
                "validation_precision": "float32",
                "training_objective": model.config.objective,
                "context_mode": model.config.context_mode,
                "constant_control_scope": "same architecture and nominal trainable parameters; zero transition inputs remove episode information, not equal effective function capacity",
                "interpretation": "posthoc_development_revision; extra training, not a matched training-budget claim"}
    start_epoch, step, best, progress = 0, 0, float("inf"), {}
    if config.get("resume"):
        saved = torch.load(Path(config["resume"]) / "model.pt", weights_only=True, map_location="cpu")
        if saved["config"]["metadata"]["training_identity"] != identity:
            raise ValueError("Resume sources, architecture or scientific training settings changed")
        state = resume_training(config["resume"], model, optimizer, scheduler, generator)
        start_epoch, step, best = state["epoch"], state["step"], state["best_metric"]
        progress = state.get("progress", {})
        # A crash between epoch metrics and package writes can leave a dangling
        # metric row; discard only rows beyond the authoritative checkpoint.
        metric_file = output / "metrics.jsonl"
        if metric_file.exists():
            kept = [json.loads(line) for line in metric_file.read_text().splitlines() if line.strip()]
            kept = [row for row in kept if row["epoch"] <= start_epoch]
            metric_file.write_text("".join(json.dumps(row) + "\n" for row in kept))
    elif (output / "metrics.jsonl").exists() or (output / "last/model.pt").exists():
        raise ValueError("Existing training output requires explicit resume")
    atomic_json(config, output / "run_config.json")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported() and config.get("bf16", True)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    started, stopping = time.monotonic(), [False]
    old_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        old_handlers[signum] = signal.signal(signum, lambda *args: stopping.__setitem__(0, True))
    print(json.dumps({"event": "start", **metadata}), flush=True)
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            assert not any(module.training for module in model.donor.modules())
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_generator = generator.get_state().clone()
            skip = progress.get("batches_completed", 0) if epoch == start_epoch else 0
            sums = progress.get("sums", {}).copy() if skip else {}
            count = progress.get("count", 0) if skip else 0
            for index, raw in enumerate(train_loader):
                if index < skip:
                    continue
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    outputs = model(batch)
                if not torch.isfinite(outputs["loss"]):
                    raise FloatingPointError("Nonfinite revision training loss")
                outputs["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.get("grad_clip", 1.),
                                                          error_if_nonfinite=True)
                optimizer.step()
                n = len(batch["actions"])
                count += n; step += 1
                for key, value in outputs.items():
                    if key.endswith("loss") or key.endswith("context_std"):
                        sums[key] = sums.get(key, 0.) + float(value.detach()) * n
                if step % config.get("log_every", 50) == 0:
                    print(json.dumps({"event": "step", "epoch": epoch + 1, "step": step,
                                      "loss": float(outputs["loss"].detach()),
                                      "grad_norm": float(grad_norm)}), flush=True)
                if stopping[0] or (config.get("max_runtime_seconds") and
                                   time.monotonic() - started > config["max_runtime_seconds"]):
                    sampler = torch.Generator(); sampler.set_state(epoch_generator)
                    save_package(model, output / "last", optimizer=optimizer, scheduler=scheduler,
                                 epoch=epoch, step=step, best_metric=best, metadata=metadata,
                                 loader_generator=sampler,
                                 progress={"batches_completed": index + 1, "sums": sums, "count": count})
                    result = {"status": "interrupted_checkpoint_saved", "completed_epochs": epoch,
                              "step": step, "batches_completed_in_epoch": index + 1,
                              "training_identity": identity}
                    atomic_json(result, output / "training_summary.json")
                    return result
            model.eval()
            val_sums, val_count = {}, 0
            with torch.inference_mode():
                for raw in val_loader:
                    batch = to_device(raw, device)
                    # Common deployment-matched criterion for BOTH training objectives.
                    # Keep validation in FP32, matching planning inference precision.
                    outputs = model.recursive_validation(batch)
                    n = len(batch["actions"]); val_count += n
                    for key, value in outputs.items():
                        if key.endswith("loss") or key.endswith("context_std"):
                            val_sums[key] = val_sums.get(key, 0.) + float(value) * n
            scheduler.step()
            metrics = {"epoch": epoch + 1, "step": step,
                       "train": {k: v / count for k, v in sums.items()},
                       "val": {k: v / val_count for k, v in val_sums.items()},
                       "learning_rate": optimizer.param_groups[0]["lr"],
                       "elapsed_seconds": time.monotonic() - started,
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                       "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
            if not all(math.isfinite(v) for scope in ("train", "val") for v in metrics[scope].values()):
                raise FloatingPointError("Nonfinite revision epoch metrics")
            improved = metrics["val"]["prediction_loss"] < best
            best = min(best, metrics["val"]["prediction_loss"])
            kwargs = dict(optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1, step=step,
                          best_metric=best, metadata=metadata, loader_generator=generator)
            # Save improved best before last so a durable last never advertises
            # a best metric for which no best model was written.
            if improved:
                save_package(model, output / "best", **kwargs)
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            save_package(model, output / "last", **kwargs)
            print(json.dumps({"event": "epoch", **metrics}), flush=True)
        summary = {"status": "completed", "completed_epochs": epochs, "step": step,
                   "best_validation_prediction_loss": best, "training_identity": identity,
                   "validation_metric": "recursive_mse_all_query_steps",
                   "elapsed_seconds": time.monotonic() - started}
        atomic_json(summary, output / "training_summary.json")
        return summary
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def train(config, resume_if_present=False):
    seed = int(config.get("seed", 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 4))
    if str(config.get("device", "cuda")).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    with training_lock(config["output_dir"]):
        model = initialize(config)
        output = Path(config["output_dir"])
        identity = training_identity(config, model)
        summary_file = output / "training_summary.json"
        if summary_file.exists() and json.loads(summary_file.read_text()).get("status") == "completed":
            summary = validate_completed(output, identity)
            print(json.dumps({"event": "reuse_completed", **summary}), flush=True)
            return summary
        if resume_if_present and (output / "last/model.pt").exists():
            config = {**config, "resume": str(output / "last")}
        return fit(model, config, make_dataset(config, "train"), make_dataset(config, "val"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-if-present", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_runtime_seconds is not None:
        if not math.isfinite(args.max_runtime_seconds) or args.max_runtime_seconds <= 0:
            parser.error("max-runtime-seconds must be finite and positive")
        config["max_runtime_seconds"] = args.max_runtime_seconds
    result = train(config, args.resume_if_present)
    return 0 if result["status"] == "completed" else 75


if __name__ == "__main__":
    raise SystemExit(main())
