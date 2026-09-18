"""Complete epoch-based training with held-out validation and portable checkpoints.

Usage: python -m shiftwm.train --config configs/world/pusht_factorized.json
Each epoch traverses the complete training split. No short-run/smoke defaults.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import resume_training, save_package
from .model import ModelConfig, ShiftWorldModel
from .upstream import load_base


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def make_dataset(config, split):
    from .data import TrajectoryDataset
    return TrajectoryDataset(config["data_root"], split=split,
                             sequence_length=config.get("sequence_length", 8),
                             stride=config.get("stride", 1),
                             **config.get("dataset_kwargs", {}))


def read_action_stats(path, dimension):
    stats = json.loads(Path(path).read_text())
    stats = stats.get("action", stats)
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    if mean.size != dimension:
        if dimension % mean.size != 0:
            raise ValueError("Action statistics not divisible into grouped action dimension")
        mean, std = np.tile(mean, dimension // mean.size), np.tile(std, dimension // std.size)
    return mean.tolist(), std.tolist()


def train(config):
    seed = int(config.get("seed", 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(config.get("cpu_threads", 4))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    base, base_config, provenance = load_base(config["pretrained_dir"])
    stats_path = config["action_stats"]
    means, stds = read_action_stats(stats_path, base_config["action_encoder"]["input_dim"])
    provenance["action_stats"] = str(Path(stats_path).resolve())
    provenance["action_stats_sha256"] = hashlib.sha256(Path(stats_path).read_bytes()).hexdigest()
    for name, path in (("data_manifest", Path(config["data_root"]) / "manifest.json"),
                       ("cache_manifest", Path(config.get("dataset_kwargs", {}).get("feature_cache", "")) / "manifest.json")):
        if path.is_file():
            provenance[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name == "cache_manifest":
                cache = json.loads(path.read_text())
                if cache["encoder_weights_sha256"] != provenance["weights_sha256"]:
                    raise ValueError("Cached features use a different pretrained encoder checkpoint")
    try:
        provenance["code_git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        provenance["code_git_revision"] = "uncommitted_workspace"
    provenance["implementation_hashes"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("model.py", "train.py", "checkpoint.py", "upstream.py", "data.py")}
    model = ShiftWorldModel(base, base_config, ModelConfig(**config.get("model", {})), means, stds,
                            provenance).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Frozen model cannot be trained; use evaluation CLI")
    train_set = make_dataset(config, "train")
    val_set = make_dataset(config, "val")
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError("Both complete training and held-out validation splits must be nonempty")
    generator = torch.Generator().manual_seed(seed)
    workers = config.get("num_workers", 2)
    kwargs = {"batch_size": config.get("batch_size", 64), "num_workers": workers,
              "pin_memory": device.type == "cuda", "persistent_workers": False}
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    optimizer = torch.optim.AdamW(parameters, lr=config.get("lr", 5e-5),
                                 weight_decay=config.get("weight_decay", 1e-3))
    epochs = int(config.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs,
                                                          eta_min=config.get("min_lr", 1e-6))
    start_epoch, step, best, resume_progress = 0, 0, float("inf"), {}
    if config.get("resume"):
        state = resume_training(config["resume"], model, optimizer, scheduler, generator)
        start_epoch, step, best = state["epoch"], state["step"], state["best_metric"]
        resume_progress = state.get("progress", {})
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported() and config.get("bf16", True)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    metadata = {"config": config, "train_windows": len(train_set), "val_windows": len(val_set),
                "trainable_parameters": sum(p.numel() for p in parameters),
                "total_parameters_including_reference": sum(p.numel() for p in model.parameters()),
                "target_coordinates": "immutable_pretrained_encoder_on_canonical_training_renders",
                "context_protocol": "support_0_to_H-1_query_H_onwards"}
    print(json.dumps({"event": "start", **metadata}), flush=True)
    start_time = time.monotonic()
    stopping = [False]
    def stop_after_batch(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop_after_batch)
    signal.signal(signal.SIGINT, stop_after_batch)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, stop_after_batch)
    for epoch in range(start_epoch, epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start_generator = generator.get_state().clone()
        skip_batches = resume_progress.get("batches_completed", 0) if epoch == start_epoch else 0
        sums = resume_progress.get("sums", {}).copy() if skip_batches else {}
        count = resume_progress.get("count", 0) if skip_batches else 0
        for batch_index, raw_batch in enumerate(train_loader):
            if batch_index < skip_batches:
                continue
            batch = to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(batch)
            if not torch.isfinite(outputs["loss"]):
                raise FloatingPointError(f"Nonfinite training loss at epoch={epoch}, step={step}")
            outputs["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.get("grad_clip", 1.0),
                                                       error_if_nonfinite=True)
            optimizer.step()
            n = batch["actions"].shape[0]
            count += n; step += 1
            for key, value in outputs.items():
                if key.endswith("loss") or key.endswith("context_std"):
                    sums[key] = sums.get(key, 0.) + float(value.detach()) * n
            if step % config.get("log_every", 50) == 0:
                print(json.dumps({"event": "step", "epoch": epoch + 1, "step": step,
                                  "loss": float(outputs["loss"].detach()),
                                  "grad_norm": float(grad_norm),
                                  "elapsed_seconds": time.monotonic() - start_time}), flush=True)
            if stopping[0] or (config.get("max_runtime_seconds") and
                               time.monotonic() - start_time > config["max_runtime_seconds"]):
                # Restore epoch-start sampler state on resume, regenerate the
                # deterministic permutation, and skip exactly completed batches.
                sampler_generator = torch.Generator()
                sampler_generator.set_state(epoch_start_generator)
                save_package(model, output / "last", optimizer=optimizer, scheduler=scheduler,
                             epoch=epoch, step=step, best_metric=best, metadata=metadata,
                             loader_generator=sampler_generator,
                             progress={"batches_completed": batch_index + 1, "sums": sums, "count": count})
                result = {"status": "interrupted_checkpoint_saved", "completed_epochs": epoch,
                          "step": step, "batches_completed_in_epoch": batch_index + 1}
                (output / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result), flush=True)
                return result
        model.eval()
        val_sums, val_count = {}, 0
        with torch.inference_mode():
            for raw_batch in val_loader:
                batch = to_device(raw_batch, device)
                with autocast():
                    outputs = model(batch)
                n = batch["actions"].shape[0]
                val_count += n
                for key, value in outputs.items():
                    if key.endswith("loss") or key.endswith("context_std"):
                        val_sums[key] = val_sums.get(key, 0.) + float(value) * n
        scheduler.step()
        metrics = {"epoch": epoch + 1, "step": step,
                   "train": {k: v/count for k, v in sums.items()},
                   "val": {k: v/val_count for k, v in val_sums.items()},
                   "elapsed_seconds": time.monotonic() - start_time,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
        if not math.isfinite(metrics["val"]["prediction_loss"]):
            raise FloatingPointError("Nonfinite validation prediction loss")
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps({"event": "epoch", **metrics}), flush=True)
        improved = metrics["val"]["prediction_loss"] < best
        best = min(best, metrics["val"]["prediction_loss"])
        save_kwargs = dict(optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1, step=step,
                           best_metric=best, metadata=metadata, loader_generator=generator)
        save_package(model, output / "last", **save_kwargs)
        if improved:
            save_package(model, output / "best", **save_kwargs)
    summary = {"completed_epochs": epochs, "step": step, "best_validation_prediction_loss": best,
               "elapsed_seconds": time.monotonic() - start_time, "status": "completed"}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
