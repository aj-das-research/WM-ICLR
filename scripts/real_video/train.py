#!/usr/bin/env python3
"""Full-epoch, resumable training on audited real-video feature caches.

Checkpoint selection uses all five recursively predicted query frames in fixed
training-standardized coordinates. No test images enter training or selection.
Packages contain tensors/configuration, not pickled models, and load offline.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.checkpoint import atomic_torch_save, rng_state, restore_rng
from shiftwm.real_video.data import RealVideoDataset, sha256, validate_manifest
from shiftwm.real_video.model import RealVideoWorldModel, from_config

PACKAGE_KIND = "shiftwm_real_video_droid_v1"
SELECTION = "mean_all_five_recursive_query_standardized_mse"
PRECISION = "float32 validation; autocast disabled; CUDA TF32 disabled"
PRIMARY_CAMERA = "exterior_image_1_left"


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def save_package(model, directory, *, optimizer, scheduler, epoch, step,
                 best_metric, metadata, generator, history):
    """Commit immutable model/optimizer generation with one atomic pointer swap."""
    directory = Path(directory).absolute()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists() and not directory.is_symlink():
        raise ValueError("Cannot replace a non-generation package directory")
    previous = directory.resolve() if directory.is_symlink() else None
    generations = directory.parent / ("." + directory.name + ".generations")
    generation = generations / uuid.uuid4().hex
    generation.mkdir(parents=True)
    pointer = directory.with_name("." + directory.name + "." + uuid.uuid4().hex)
    committed = False
    try:
        config = {**deepcopy(model.package_config), "package_kind": PACKAGE_KIND,
                  "metadata": deepcopy(metadata)}
        state = {"config": config, "state_dict": model.state_dict(), "epoch": epoch,
                 "step": step, "best_metric": best_metric, "history": deepcopy(history)}
        training = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng": rng_state(), "loader_generator": generator.get_state(),
                    "epoch": epoch, "step": step,
                    "training_identity": metadata["training_identity"]}
        atomic_torch_save(state, generation / "model.pt")
        atomic_torch_save(training, generation / "training_state.pt")
        atomic_json(config, generation / "config.json")
        atomic_json({"format_version": 1, "package_kind": PACKAGE_KIND,
                     "files": {p.name: sha256(p) for p in sorted(generation.iterdir())}},
                    generation / "package_manifest.json")
        os.symlink(os.path.relpath(generation, directory.parent), pointer)
        os.replace(pointer, directory)
        committed = True
        for old in generations.iterdir():
            if old.is_dir() and old not in {generation, previous}:
                shutil.rmtree(old)
        return state
    finally:
        pointer.unlink(missing_ok=True)
        if not committed:
            shutil.rmtree(generation, ignore_errors=True)


def read_package(directory, require_training=False):
    path = Path(directory).resolve(strict=True)
    manifest = json.loads((path / "package_manifest.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("package_kind") != PACKAGE_KIND:
        raise ValueError("Unsupported real-video package")
    names = set(manifest.get("files", {}))
    if not {"model.pt", "config.json"} <= names or names - {"model.pt", "config.json", "training_state.pt"}:
        raise ValueError("Invalid package file set")
    if require_training and "training_state.pt" not in names:
        raise ValueError("Missing optimizer/RNG state")
    for name, expected in manifest["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Package hash differs: {name}")
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    config = json.loads((path / "config.json").read_text())
    if (state.get("config") != config or config.get("package_kind") != PACKAGE_KIND
            or config.get("format_version") != 1):
        raise ValueError("Embedded/external package configuration differs")
    if any(type(state.get(key)) is not int or state[key] < 0 for key in ("epoch", "step")):
        raise ValueError("Invalid package epoch/step")
    if require_training:
        training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        if (any(training.get(key) != state[key] for key in ("epoch", "step"))
                or training.get("training_identity") != config["metadata"]["training_identity"]):
            raise ValueError("Model and continuation states disagree")
    return path, state


def load_package(directory, device="cpu"):
    _, state = read_package(directory)
    model = from_config(state["config"])
    model.load_state_dict(state["state_dict"], strict=True)
    for key, value in model.package_config.items():
        if state["config"].get(key) != value:
            raise ValueError(f"Reconstructed configuration differs: {key}")
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if state["config"]["metadata"].get("parameter_counts") != counts:
        raise ValueError("Reconstructed parameter counts differ")
    return model.to(device).eval(), state


def resume_training(directory, model, optimizer, scheduler, generator, metadata):
    path, state = read_package(directory, require_training=True)
    if state["config"]["metadata"] != metadata:
        raise ValueError("Resume training identity or metadata differs")
    for key, value in model.package_config.items():
        if state["config"].get(key) != value:
            raise ValueError(f"Resume model configuration differs: {key}")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    optimizer.load_state_dict(training["optimizer"])
    scheduler.load_state_dict(training["scheduler"])
    generator.set_state(training["loader_generator"])
    restore_rng(training["rng"])
    return state


def scientific_config(config):
    operational = {"device", "num_workers", "cpu_threads", "max_runtime_seconds",
                   "resume_if_present", "output_dir", "log_every"}
    return {key: value for key, value in config.items() if key not in operational}


def source_files():
    relative = ["scripts/real_video/train.py", "scripts/real_video/evaluate.py",
                "src/shiftwm/real_video/model.py", "src/shiftwm/real_video/data.py",
                "src/shiftwm/real_video/features.py", "src/shiftwm/model.py",
                "src/shiftwm/upstream.py", "src/shiftwm/checkpoint.py",
                "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/NOTICE.json"]
    return {str(ROOT / name): sha256(ROOT / name) for name in relative}


def verify_sources(sources):
    for path, expected in sources.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned source or data identity changed: {path}")


def audit_inputs(config):
    """Audit metadata and identities without opening any test feature payload."""
    cache = Path(config["cache_root"]).resolve()
    audit_path = Path(config["metadata_audit"]).resolve()
    processed_manifest = audit_path.parent / "manifest.json"
    audit = json.loads(audit_path.read_text())
    manifest = json.loads((cache / "manifest.json").read_text())
    validate_manifest(manifest)
    expected = sha256(processed_manifest)
    if audit.get("status") != "passed" or audit.get("dataset_manifest_sha256") != expected:
        raise ValueError("Real-recording metadata audit absent, failed, or stale")
    if manifest.get("identity", {}).get("dataset_manifest_sha256") != expected:
        raise ValueError("Feature cache belongs to a different recording manifest")
    stats_path = cache / "training_statistics.json"
    stats = json.loads(stats_path.read_text())
    if (stats.get("fit_split") != "train" or stats.get("camera") != PRIMARY_CAMERA
            or stats.get("cache_manifest_sha256") != sha256(cache / "manifest.json")
            or stats.get("ddof") != 1 or stats.get("std_floor") != 1e-5):
        raise ValueError("Normalization statistics are not frozen training-only statistics")
    for kind, dim in (("feature", manifest["feature_dim"]), ("action", manifest["action_dim"])):
        for suffix in ("mean", "std"):
            value = np.asarray(stats[kind + "_" + suffix], dtype=np.float64)
            if value.shape != (dim,) or not np.isfinite(value).all() or (suffix == "std" and (value <= 0).any()):
                raise ValueError("Invalid training normalization statistics")
    dependencies = source_files()
    for path in (cache / "manifest.json", stats_path, audit_path, processed_manifest,
                 Path(config["protocol_path"]).resolve()):
        dependencies[str(path)] = sha256(path)
    identity = {"scientific_config": scientific_config(config), "dependencies": dependencies,
                "encoder_and_cache_identity": manifest["identity"],
                "normalization": stats, "selection": SELECTION, "validation_precision": PRECISION,
                "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                            "python": sys.version, "cuda_build": torch.version.cuda}}
    return manifest, stats, identity


def verify_training_statistics(dataset, stats):
    """Independently recompute from training episodes, never rewrite shared stats."""
    for kind, field in (("feature", "features"), ("action", "actions")):
        arrays = [episode[field].astype(np.float64) for episode in dataset.episodes]
        count = sum(len(array) for array in arrays)
        total = sum(array.sum(0) for array in arrays)
        squares = sum(np.square(array).sum(0) for array in arrays)
        mean = total / count
        std = np.maximum(np.sqrt(np.maximum((squares - count * mean**2) / (count - 1), 0)), 1e-5)
        if (stats["counts"][kind] != count
                or not np.allclose(mean, stats[kind + "_mean"], rtol=1e-10, atol=1e-10)
                or not np.allclose(std, stats[kind + "_std"], rtol=1e-10, atol=1e-10)):
            raise ValueError("Statistics do not match complete training episodes")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def metric_rows(directory):
    path = Path(directory) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_history(history, directory):
    target = Path(directory) / "metrics.jsonl"
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("w") as handle:
            for row in history:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def epoch_pass(model, loader, device, optimizer=None, bf16=False, grad_clip=1.):
    training = optimizer is not None
    model.train(training)
    total, elements, batches = 0., 0, 0
    with (nullcontext() if training else torch.inference_mode()):
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            if batch["features"].shape[1] != 8 or batch["actions"].shape[1] != 7:
                raise ValueError("Registered training/validation requires three support plus five query frames")
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=training and bf16 and device.type == "cuda",
                                dtype=torch.bfloat16):
                out = model(batch)
                # Read the fixed-coordinate tensors explicitly: a future regularizer
                # must not silently redefine checkpoint selection.
                mse = (out["standardized_predictions"].float() - out["standardized_targets"].float()).square().mean()
                loss = out["loss"] if training else mse
            if not torch.isfinite(loss) or not torch.isfinite(mse):
                raise ValueError("Nonfinite training/validation loss")
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), grad_clip)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
            count = out["standardized_targets"].numel()
            total += float(mse.detach()) * count
            elements += count
            batches += 1
    if not elements:
        raise ValueError("Empty training/validation population")
    return {"standardized_mse": total / elements, "elements": elements, "batches": batches}


def validate_completed(directory, identity=None, verify_dependencies=True):
    directory = Path(directory)
    summary = json.loads((directory / "training_summary.json").read_text())
    config = json.loads((directory / "training_config.json").read_text())
    _, last = read_package(directory / "last", require_training=True)
    _, best = read_package(directory / "best")
    metadata = last["config"]["metadata"]
    frozen = metadata["identity"]
    actual_identity = digest(frozen)
    if (identity is not None and identity != actual_identity):
        raise ValueError("Completed training identity differs")
    if (metadata["training_identity"] != actual_identity
            or summary.get("training_identity") != actual_identity
            or scientific_config(config) != frozen["scientific_config"]):
        raise ValueError("Inconsistent completed run configuration/identity")
    if verify_dependencies:
        verify_sources(frozen["dependencies"])
    if (summary.get("status") != "completed" or summary.get("completed_epochs") != 30
            or config.get("epochs") != 30 or last["epoch"] != 30
            or summary.get("validation_metric") != SELECTION
            or summary.get("validation_precision") != PRECISION
            or metadata.get("selection") != SELECTION or metadata.get("validation_precision") != PRECISION):
        raise ValueError("Full 30-epoch common validation gate not met")
    counts = metadata.get("parameter_counts", {})
    if (summary.get("parameter_counts") != counts or set(counts) != {"total", "trainable"}
            or any(type(value) is not int for value in counts.values())
            or not 0 < counts["trainable"] <= counts["total"]):
        raise ValueError("Invalid parameter-count evidence")
    rows = metric_rows(directory)
    if rows != last["history"] or [row["epoch"] for row in rows] != list(range(1, 31)):
        raise ValueError("Missing, reordered, or altered training history")
    for row in rows:
        for part in ("train", "val"):
            if (not math.isfinite(row[part]["standardized_mse"]) or row[part]["standardized_mse"] < 0
                    or row[part]["elements"] <= 0 or row[part]["batches"] <= 0):
                raise ValueError("Invalid completed metrics")
    selected = min(rows, key=lambda row: row["val"]["standardized_mse"])
    expected_steps = sum(row["train"]["batches"] for row in rows)
    if (best["epoch"] != selected["epoch"] or best["best_metric"] != selected["val"]["standardized_mse"]
            or last["best_metric"] != best["best_metric"] or summary.get("best_epoch") != best["epoch"]
            or summary.get("best_validation_mse") != best["best_metric"]
            or last["step"] != expected_steps or summary.get("step") != expected_steps
            or best["config"] != last["config"]
            or best["history"] != rows[:best["epoch"]]
            or best["step"] != sum(row["train"]["batches"] for row in rows[:best["epoch"]])):
        raise ValueError("Best checkpoint is not the common validation minimum")
    return summary


def fit(model, config, train_dataset, val_dataset, identity):
    """Internal fit API; CLI additionally constructs/audits the real cache inputs."""
    if config.get("epochs") != 30 or config.get("seed") not in (0, 1, 2):
        raise ValueError("Registered training requires 30 epochs and seed 0, 1, or 2")
    if identity["scientific_config"] != scientific_config(config):
        raise ValueError("Fit configuration differs from audited identity")
    if model.config.mode != config["mode"] or model.config.history_length != 3:
        raise ValueError("Model mode/history differs from registered configuration")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cuda"))
    seed_everything(config["seed"])
    model.to(device)
    metadata = {"training_identity": digest(identity), "identity": identity,
                "selection": SELECTION, "validation_precision": PRECISION,
                "parameter_counts": {"total": sum(p.numel() for p in model.parameters()),
                                     "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=config["min_lr"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader_kw = {"batch_size": config["batch_size"], "num_workers": config.get("num_workers", 0),
                 "pin_memory": device.type == "cuda", "drop_last": False}
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kw)
    val_loader = DataLoader(val_dataset, shuffle=False,
                            generator=torch.Generator().manual_seed(173), **loader_kw)
    history, best, step, start = [], float("inf"), 0, 0
    if (output / "last").exists():
        if not config.get("resume_if_present"):
            raise ValueError("Existing run requires --resume-if-present")
        state = resume_training(output / "last", model, optimizer, scheduler, generator, metadata)
        history, best, step, start = state["history"], state["best_metric"], state["step"], state["epoch"]
        if [row["epoch"] for row in history] != list(range(1, start + 1)):
            raise ValueError("Interrupted checkpoint history is inconsistent")
        write_history(history, output)
    elif any((output / name).exists() for name in ("best", "metrics.jsonl", "training_summary.json")):
        raise ValueError("Partial run lacks an authoritative last checkpoint")
    atomic_json(config, output / "training_config.json")
    began = time.monotonic()
    for epoch in range(start + 1, 31):
        verify_sources(identity["dependencies"])
        train_metrics = epoch_pass(model, train_loader, device, optimizer, config.get("bf16", True), config["grad_clip"])
        val_metrics = epoch_pass(model, val_loader, device)
        step += train_metrics["batches"]
        history.append({"epoch": epoch, "step": step, "lr": optimizer.param_groups[0]["lr"],
                        "train": train_metrics, "val": val_metrics})
        scheduler.step()
        improved = val_metrics["standardized_mse"] < best
        best = min(best, val_metrics["standardized_mse"])
        args = dict(optimizer=optimizer, scheduler=scheduler, epoch=epoch, step=step,
                    best_metric=best, metadata=metadata, generator=generator, history=history)
        if improved:
            save_package(model, output / "best", **args)
        save_package(model, output / "last", **args)
        write_history(history, output)
        summary = {"status": "completed" if epoch == 30 else "interrupted",
                   "completed_epochs": epoch, "step": step, "training_identity": metadata["training_identity"],
                   "best_epoch": min(history, key=lambda row: row["val"]["standardized_mse"])["epoch"],
                   "best_validation_mse": best, "validation_metric": SELECTION,
                   "validation_precision": PRECISION,
                   "parameter_counts": metadata["parameter_counts"],
                   "resume_semantics": "completed epochs; interrupted in-flight epoch replays from previous checkpoint"}
        atomic_json(summary, output / "training_summary.json")
        print(json.dumps({"epoch": epoch, "train_mse": train_metrics["standardized_mse"],
                          "val_mse": val_metrics["standardized_mse"], "best_mse": best}), flush=True)
        if epoch < 30 and time.monotonic() - began >= config.get("max_runtime_seconds", float("inf")):
            return summary
    # Repair a crash after committing epoch 30 but before writing its journal or
    # completion marker. The immutable last generation is authoritative.
    if start == 30:
        atomic_json({"status": "completed", "completed_epochs": 30, "step": step,
                     "training_identity": metadata["training_identity"],
                     "best_epoch": min(history, key=lambda row: row["val"]["standardized_mse"])["epoch"],
                     "best_validation_mse": best, "validation_metric": SELECTION,
                     "validation_precision": PRECISION, "parameter_counts": metadata["parameter_counts"],
                     "resume_semantics": "completed epochs; interrupted in-flight epoch replays from previous checkpoint"},
                    output / "training_summary.json")
    return validate_completed(output, metadata["training_identity"])


def train(config):
    torch.set_num_threads(config.get("cpu_threads", 4))
    manifest, stats, identity = audit_inputs(config)
    if config.get("mode") not in RealVideoWorldModel.MODES:
        raise ValueError("Unknown registered mode")
    camera = config.get("camera", PRIMARY_CAMERA)
    if camera != PRIMARY_CAMERA:
        raise ValueError("Training and selection use only exterior camera one")
    train_data = RealVideoDataset(config["cache_root"], "train", horizon=5, stride=config.get("stride", 2), verify=True)
    if config.get("validation_stride", 5) != 5:
        raise ValueError("Registered validation stride is five stored frames")
    val_data = RealVideoDataset(config["cache_root"], "val", horizon=5, stride=5, verify=True)
    verify_training_statistics(train_data, stats)
    c = {**config.get("model_config", {}), "feature_dim": manifest["feature_dim"],
         "action_dim": manifest["action_dim"], "mode": config["mode"], "history_length": 3}
    seed_everything(config["seed"])
    model = RealVideoWorldModel(c, **{key: stats[key] for key in
                               ("feature_mean", "feature_std", "action_mean", "action_std")})
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fit(model, config, train_data, val_data, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-if-present", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config["resume_if_present"] = args.resume_if_present
    if args.max_runtime_seconds is not None:
        config["max_runtime_seconds"] = args.max_runtime_seconds
    result = train(config)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "completed" else 75


if __name__ == "__main__":
    raise SystemExit(main())
