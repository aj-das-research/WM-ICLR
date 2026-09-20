#!/usr/bin/env python3
"""Two explicit external-predictor objectives, common H10 checkpoint selection."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_spatial.data import SpatialDataset, validate_spatial_manifest, statistics_from_episodes


def load_local(name):
    spec = importlib.util.spec_from_file_location("external_dinowm_train_" + name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


models = load_local("model")
base = load_local("engine").load()
base.RealVideoWorldModel = models.ExternalDinoWM
base.from_config = models.from_config
base.RealVideoDataset = SpatialDataset
base.validate_manifest = validate_spatial_manifest
base.PACKAGE_KIND = "adapted_official_dinowm_raw_droid_v2"
base.SELECTION = "window_mean_all10_shared_channel_standardized_mse"


def source_files():
    registry = load_local("registry")
    record = registry.verify()
    return {str(ROOT / p): h for p, h in record["dependencies"].items()} | {str(registry.REG): registry.sha(registry.REG)}
base.source_files = source_files


def verify_training_statistics(dataset, stats):
    actual = statistics_from_episodes(dataset.episodes)
    for key, value in actual.items():
        if key == "counts":
            if value != stats[key]: raise ValueError("Training sample counts differ")
        elif not np.allclose(value, stats[key], rtol=1e-10, atol=1e-10):
            raise ValueError("Shared training-only normalization differs")
base.verify_training_statistics = verify_training_statistics


def epoch_pass(model, loader, device, optimizer=None, bf16=False, grad_clip=None):
    training = optimizer is not None
    objective = model.config.mode if training else "official_raw_recursive_h10"
    steps = 3 if objective == "official_raw_one_step" else 10
    model.train(training)
    if bf16 or grad_clip is not None:
        raise ValueError("Raw v2 requires FP32 without gradient clipping")
    total = 0.; raw_total = 0.; elements = batches = windows = 0; by_episode = {}
    with (nullcontext() if training else torch.inference_mode()):
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=training and bf16 and device.type == "cuda", dtype=torch.bfloat16):
                out = model(batch, objective=objective)
                errors = (out["standardized_predictions"].float() - out["standardized_targets"].float()).square()
                if errors.shape != (len(batch["features"]), steps, 6144):
                    raise ValueError("Objective returned wrong target slots")
                raw_loss = (out["raw_predictions"].float() - out["raw_targets"].float()).square().mean()
                loss = raw_loss if training else errors.mean()
            if not torch.isfinite(loss) or not torch.isfinite(errors).all(): raise ValueError("Nonfinite objective")
            if training:
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                    raise ValueError("Nonfinite gradient; no implicit clipping or fallback")
                optimizer.step()
            else:
                for eid, value in zip(batch["episode_index"].tolist(), errors.mean((1, 2)).tolist()):
                    cumulative, count = by_episode.get(eid, (0., 0))
                    by_episode[eid] = (cumulative + value, count + 1)
            total += float(errors.detach().mean()) * errors.numel()
            raw_total += float(raw_loss.detach()) * errors.numel()
            elements += errors.numel(); windows += len(errors); batches += 1
    if not elements: raise ValueError("Empty epoch")
    return {"standardized_mse": total / elements, "raw_mse": raw_total / elements,
            "optimized_loss": "raw_visual_mse" if training else None, "precision": "float32", "elements": elements, "batches": batches,
            "windows": windows, "supervised_grids": steps, "objective": objective,
            "target_grid_indices": [1, 2, 3] if steps == 3 else list(range(3, 13)),
            "aggregation": "window_weighted" if training else base.SELECTION,
            **({} if training else {"episodes": len(by_episode),
               "equal_episode_diagnostic_mse": float(np.mean([s / n for s, n in by_episode.values()]))})}
base.epoch_pass = epoch_pass


def validate_completed(directory, identity=None, verify_dependencies=True):
    summary = base.validate_completed(directory, identity, verify_dependencies)
    config = json.loads((Path(directory) / "training_config.json").read_text())
    load_local("registry").validate_config(config)
    if summary.get("parameter_counts") != {"total": 19412420, "trainable": 19412420}:
        raise ValueError("External predictor size differs")
    steps = 3 if config["mode"] == "official_raw_one_step" else 10
    for row in base.metric_rows(directory):
        for part, expected_steps, expected_windows, objective in (
                ("train", steps, 18660, config["mode"]), ("val", 10, 1631, "official_raw_recursive_h10")):
            data = row[part]
            if (data.get("supervised_grids") != expected_steps or data.get("windows") != expected_windows
                    or data.get("elements") != expected_windows * expected_steps * 6144
                    or data.get("objective") != objective
                    or data.get("batches") != (expected_windows + 31) // 32
                    or data.get("target_grid_indices") != ([1, 2, 3] if expected_steps == 3 else list(range(3, 13)))):
                raise ValueError("Completed epoch objective/population differs")
        if (row.get("lr") != 5e-4 or row["train"].get("optimized_loss") != "raw_visual_mse"
                or any(row[part].get("precision") != "float32" for part in ("train", "val"))
                or any(not np.isfinite(row[part].get("raw_mse", float("nan"))) for part in ("train", "val"))):
            raise ValueError("Raw-coordinate training recipe/metric identity differs")
        if row["val"].get("episodes") != 141:
            raise ValueError("Incomplete validation population")
    return summary


def train(config):
    registry = load_local("registry")
    registry.validate_config(config)
    registry.verify()
    torch.set_num_threads(config["cpu_threads"])
    manifest, stats, identity = base.audit_inputs(config)
    if stats.get("normalization") != "shared_per_channel_over_train_frames_and_patches":
        raise ValueError("Wrong normalization contract")
    training = SpatialDataset(config["cache_root"], "train", horizon=10, stride=2)
    validation = SpatialDataset(config["cache_root"], "val", horizon=10, stride=5)
    if len(training) != 18660 or len(validation) != 1631:
        raise ValueError("Existing train/development population differs")
    verify_training_statistics(training, stats)
    base.seed_everything(config["seed"])
    model = models.ExternalDinoWM({**config["model_config"], "mode": config["mode"]},
             **{k: stats[k] for k in ("feature_mean", "feature_std", "action_mean", "action_std")})
    result = base.fit(model, config, training, validation, identity)
    if result["status"] == "completed": validate_completed(config["output_dir"])
    return result


atomic_json = base.atomic_json
read_package = base.read_package
load_package = base.load_package
