"""Complete, resumable extension training with one common recursive criterion.

Usage: python -m shiftwm.extensions.train --config path.json --resume-if-present
The new namespace leaves all previously frozen campaigns and loaders unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import argparse
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

from shiftwm.data import validate_manifest
from shiftwm.train import make_dataset, read_action_stats, to_device
from .checkpoint import (atomic_json, file_sha256, load_package, read_package,
                         resume_training, save_package)
from .model import ARCHITECTURES, MODES, build_model


OPERATIONAL = {"resume", "max_runtime_seconds", "device", "cpu_threads", "num_workers", "log_every"}
SELECTION = "minimum_validation_recursive_mse_targets_H_to_T-1_over_completed_training_epochs"
VALIDATION_PROTOCOL = "recursive_from_observed_support_all_query_steps"


def scientific_config(config):
    return {key: value for key, value in config.items() if key not in OPERATIONAL}


def training_identity(config, model):
    value = {"config": scientific_config(config), "package_config": model.package_config}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def training_lock(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".training.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_config(config):
    if type(config.get("epochs")) is not int or config["epochs"] != 30:
        raise ValueError("Extension training requires all 30 epochs")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("Declare a nonnegative integer training seed")
    if config.get("architecture") not in ARCHITECTURES or config.get("mode") not in MODES:
        raise ValueError("Declare a supported extension predictor and method")
    if config.get("sequence_length") != 8:
        raise ValueError("Extension training uses eight-frame windows and five query targets")
    if set(config.get("dataset_kwargs", {})) - {"feature_cache", "preload_features", "cache_size"}:
        raise ValueError("Use full training/validation splits; unsupported dataset override")
    if not config.get("dataset_kwargs", {}).get("feature_cache"):
        raise ValueError("Extension training requires a frozen visual feature cache")
    for key in ("batch_size", "stride"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "min_lr", "weight_decay", "grad_clip"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["lr"] <= 0 or config["grad_clip"] <= 0 or config["min_lr"] > config["lr"]:
        raise ValueError("Invalid optimizer learning-rate/clipping settings")


def initialize(config):
    validate_config(config)
    data_root = Path(config["data_root"])
    cache_root = Path(config["dataset_kwargs"]["feature_cache"])
    pretrained = Path(config["pretrained_dir"])
    manifest_path, cache_path = data_root / "manifest.json", cache_root / "manifest.json"
    manifest, cache = json.loads(manifest_path.read_text()), json.loads(cache_path.read_text())
    validate_manifest(manifest)
    if cache.get("dataset_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Feature cache differs from the declared dataset manifest")
    base_config = json.loads((pretrained / "config.json").read_text())
    dimension = int(base_config["action_encoder"]["input_dim"])
    means, stds = read_action_stats(config["action_stats"], dimension)
    if (len(means) != dimension or len(stds) != dimension
            or not all(math.isfinite(v) for v in means + stds) or any(v <= 0 for v in stds)):
        raise ValueError("Action normalization requires finite means and positive standard deviations")
    model = build_model(pretrained, config["architecture"], config["mode"], means, stds, config["seed"])
    if model.history_length != 3 or model.action_dim != dimension:
        raise ValueError("Extension model must use H=3 and the declared action dimension")
    if cache.get("encoder_weights_sha256") != model.provenance["weights_sha256"]:
        raise ValueError("Feature cache uses a different pretrained visual encoder")
    # The consumed features/actions are pinned, not merely their manifest names.
    # Raw RGB payloads are not consumed during cached training; their declared
    # source hashes remain transitively pinned by the dataset manifest.
    feature_hashes = {}
    for episode in manifest["episodes"]:
        if episode["split"] in {"train", "val"}:
            relative = Path(episode["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Feature payload path escapes the cache")
            feature_hashes[str(relative)] = file_sha256(cache_root / relative)
    if not feature_hashes:
        raise ValueError("Empty train/validation feature identity")
    module_root = Path(__file__).resolve().parent
    scientific_sources = [module_root / name for name in ("model.py", "train.py", "checkpoint.py")]
    scientific_sources += [module_root.parent / name for name in
                           ("model.py", "train.py", "data.py", "checkpoint.py", "upstream.py")]
    vendor = module_root.parent / "vendor/lewm"
    scientific_sources += [vendor / name for name in ("NOTICE.json", "module.py", "jepa.py")]
    model.provenance.update({
        "data_manifest_sha256": file_sha256(manifest_path),
        "cache_manifest_sha256": file_sha256(cache_path),
        "action_stats_sha256": file_sha256(config["action_stats"]),
        "pretrained_config_sha256": file_sha256(pretrained / "config.json"),
        "extension_training_feature_hashes": feature_hashes,
        "extension_source_hashes": {str(p.relative_to(module_root.parent)): file_sha256(p)
                                    for p in scientific_sources},
        "extension_environment": manifest["environment"],
        "extension_source_scope": "full train/val cached features and actions; dataset manifest pins raw-source hashes",
    })
    if config.get("protocol_path"):
        model.provenance["extension_protocol_sha256"] = file_sha256(config["protocol_path"])
    return model


def metric_rows(output):
    path = Path(output) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def validate_completed(output, identity):
    output = Path(output)
    summary = json.loads((output / "training_summary.json").read_text())
    run = json.loads((output / "run_config.json").read_text())
    validate_config(run)
    if (summary.get("status") != "completed" or summary.get("completed_epochs") != 30
            or summary.get("training_identity") != identity
            or summary.get("validation_metric") != "recursive_mse_all_query_steps"):
        raise ValueError("Extension is not a completed run under this training identity/criterion")
    rows = metric_rows(output)
    if any(type(r.get("epoch")) is not int for r in rows) or [r["epoch"] for r in rows] != list(range(1, 31)):
        raise ValueError("Expected all 30 unique, chronological metric epochs")
    values = [float(r["val"]["prediction_loss"]) for r in rows]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Nonfinite or negative recursive validation metrics")
    best = min(values)
    if not math.isclose(best, summary["best_validation_prediction_loss"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Training summary differs from complete validation history")
    if summary.get("step") != rows[-1].get("step"):
        raise ValueError("Summary step differs from the final completed epoch")
    for name in ("best", "last"):
        _, state = read_package(output / name, require_training=True)
        config, metadata = state["config"], state["config"]["metadata"]
        if (metadata.get("training_identity") != identity
                or scientific_config(metadata.get("config", {})) != scientific_config(run)
                or metadata.get("selection") != SELECTION
                or metadata.get("validation_protocol") != VALIDATION_PROTOCOL
                or metadata.get("validation_precision") != "float32"
                or config.get("architecture") != run["architecture"] or config.get("mode") != run["mode"]
                or config.get("objective") != "recursive_mse_all_query_steps"):
            raise ValueError("Extension package differs from its declared model/training protocol")
        identity_config = {key: value for key, value in config.items()
                           if key not in {"format_version", "package_kind", "metadata"}}
        expected_identity = hashlib.sha256(json.dumps(
            {"config": scientific_config(run), "package_config": identity_config},
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        if expected_identity != identity:
            raise ValueError("Extension model configuration and training identity disagree")
        if not math.isclose(state["best_metric"], best, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Package best metric differs from complete validation history")
        if name == "last" and (state["epoch"] != 30 or state["step"] != summary["step"] or state["progress"]):
            raise ValueError("Last package does not finish all 30 epochs")
        if name == "best" and not any(r["epoch"] == state["epoch"] and math.isclose(
                r["val"]["prediction_loss"], best, rel_tol=1e-10, abs_tol=1e-12) for r in rows):
            raise ValueError("Best package is not selected by recursive validation")
    load_package(output / "best", device="cpu")
    return summary


def _accumulate(sums, outputs, n):
    for key, value in outputs.items():
        if key.endswith("loss") or key.endswith("context_std"):
            number = float(value.detach())
            if not math.isfinite(number):
                raise FloatingPointError(f"Nonfinite metric: {key}")
            sums[key] = sums.get(key, 0.) + number * n


def fit(model, config, train_set, val_set):
    validate_config(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cuda"))
    model.to(device).float()
    identity = training_identity(config, model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or not len(train_set) or not len(val_set):
        raise ValueError("Require trainable parameters and complete nonempty train/validation splits")
    generator = torch.Generator().manual_seed(config["seed"])
    loader_kwargs = dict(batch_size=config["batch_size"], num_workers=config.get("num_workers", 2),
                         pin_memory=device.type == "cuda", persistent_workers=False)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=config["min_lr"])
    metadata = {"config": config, "training_identity": identity,
                "train_windows": len(train_set), "val_windows": len(val_set),
                "trainable_parameters": sum(p.numel() for p in parameters),
                "total_parameters_including_reference": sum(p.numel() for p in model.parameters()),
                "selection": SELECTION, "validation_protocol": VALIDATION_PROTOCOL,
                "validation_precision": "float32", "training_objective": "recursive_mse_all_query_steps",
                "targets": [3, 4, 5, 6, 7],
                "context_information_control": "constant_dynamics retains observation conditioning; dynamics network receives zeros; equal nominal parameters do not imply equal effective capacity"}
    start_epoch, step, best, progress = 0, 0, float("inf"), {}
    if config.get("resume"):
        _, saved = read_package(config["resume"], require_training=True)
        if saved["config"]["metadata"].get("training_identity") != identity:
            raise ValueError("Resume sources, model or scientific training settings changed")
        state = resume_training(config["resume"], model, optimizer, scheduler, generator)
        start_epoch, step, best, progress = state["epoch"], state["step"], state["best_metric"], state["progress"]
        kept = [row for row in metric_rows(output) if row["epoch"] <= start_epoch]
        if [row["epoch"] for row in kept] != list(range(1, start_epoch + 1)):
            raise ValueError("Resume lacks the authoritative completed epoch history")
        (output / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept))
    elif any((output / name).exists() for name in ("metrics.jsonl", "last/model.pt", "best/model.pt")):
        raise ValueError("Existing output requires explicit resume")
    atomic_json(config, output / "run_config.json")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported() and config.get("bf16", True)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    started, stopping, old_handlers = time.monotonic(), [False], {}
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        old_handlers[signum] = signal.signal(signum, lambda *args: stopping.__setitem__(0, True))
    print(json.dumps({"event": "start", **metadata}), flush=True)
    try:
        for epoch in range(start_epoch, 30):
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_generator = generator.get_state().clone()
            skip = progress.get("batches_completed", 0) if epoch == start_epoch else 0
            if type(skip) is not int or not 0 <= skip <= len(train_loader):
                raise ValueError("Invalid within-epoch resume batch count")
            sums, count = (progress.get("sums", {}).copy(), progress.get("count", 0)) if skip else ({}, 0)
            for index, raw in enumerate(train_loader):
                if index < skip:
                    continue
                batch = to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    outputs = model(batch)
                if not torch.isfinite(outputs["loss"]):
                    raise FloatingPointError("Nonfinite extension training loss")
                outputs["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                n = len(batch["actions"])
                count += n
                step += 1
                _accumulate(sums, outputs, n)
                if step % config.get("log_every", 50) == 0:
                    print(json.dumps({"event": "step", "epoch": epoch + 1, "step": step,
                                      "loss": float(outputs["loss"].detach()), "grad_norm": float(grad_norm)}), flush=True)
                if stopping[0] or (config.get("max_runtime_seconds") and
                                   time.monotonic() - started > config["max_runtime_seconds"]):
                    sampler = torch.Generator()
                    sampler.set_state(epoch_generator)
                    save_package(model, output / "last", optimizer=optimizer, scheduler=scheduler,
                                 epoch=epoch, step=step, best_metric=best, metadata=metadata,
                                 loader_generator=sampler,
                                 progress={"batches_completed": index + 1, "sums": sums, "count": count})
                    result = {"status": "interrupted_checkpoint_saved", "completed_epochs": epoch,
                              "step": step, "batches_completed_in_epoch": index + 1, "training_identity": identity}
                    atomic_json(result, output / "training_summary.json")
                    return result
            if count != len(train_set):
                raise ValueError("Training epoch did not cover the complete split")
            model.eval()
            val_sums, val_count = {}, 0
            with torch.inference_mode(), torch.autocast(device.type, enabled=False):
                for raw in val_loader:
                    batch = to_device(raw, device)
                    outputs = model.recursive_validation(batch)
                    n = len(batch["actions"])
                    val_count += n
                    _accumulate(val_sums, outputs, n)
            if val_count != len(val_set):
                raise ValueError("Validation did not cover the complete split")
            scheduler.step()
            metrics = {"epoch": epoch + 1, "step": step,
                       "train": {k: v / count for k, v in sums.items()},
                       "val": {k: v / val_count for k, v in val_sums.items()},
                       "learning_rate": optimizer.param_groups[0]["lr"],
                       "elapsed_seconds": time.monotonic() - started,
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                       "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
            prediction = metrics["val"]["prediction_loss"]
            if prediction < 0:
                raise ValueError("Recursive validation MSE cannot be negative")
            improved, best = prediction < best, min(best, prediction)
            kwargs = dict(optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1, step=step,
                          best_metric=best, metadata=metadata, loader_generator=generator)
            if improved:
                save_package(model, output / "best", **kwargs)
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                handle.flush()
            save_package(model, output / "last", **kwargs)
            print(json.dumps({"event": "epoch", **metrics}), flush=True)
        summary = {"status": "completed", "completed_epochs": 30, "step": step,
                   "best_validation_prediction_loss": best, "training_identity": identity,
                   "validation_metric": "recursive_mse_all_query_steps",
                   "elapsed_seconds": time.monotonic() - started}
        atomic_json(summary, output / "training_summary.json")
        return summary
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def train(config, resume_if_present=False):
    validate_config(config)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 4))
    if str(config.get("device", "cuda")).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    with training_lock(config["output_dir"]):
        model = initialize(config)
        output = Path(config["output_dir"])
        identity = training_identity(config, model)
        summary_path = output / "training_summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "completed":
            summary = validate_completed(output, identity)
            print(json.dumps({"event": "reuse_completed", **summary}), flush=True)
            return summary
        if resume_if_present and (output / "last/model.pt").exists():
            config = {**config, "resume": str(output / "last")}
        return fit(model, config, make_dataset(config, "train"), make_dataset(config, "val"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, type=Path)
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
